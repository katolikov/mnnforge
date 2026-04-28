"""Phase 3 — Python access to MNN's FlatBuffers schema.

MNN ships only C++ generated headers. To read/write `.mnn` files from
Python we run flatc once per session against `<mnn_root>/schema/default/*.fbs`,
generating Python bindings into a cache directory, then import them
dynamically.

Cache layout: <mnn_root>/build_mnnforge/_mnn_py_fbs/
"""
from __future__ import annotations
import importlib.util
import os
import shutil
import subprocess
import sys
from types import ModuleType
from typing import Optional

from .log import Logger


CACHE_SUBDIR = "build_mnnforge/_mnn_py_fbs"


def _have(exe: str) -> Optional[str]:
    return shutil.which(exe)


def _ensure_flatc(mnn_root: str, log: Logger) -> str:
    """Build the vendored flatc if needed, mirroring schema/generate.sh."""
    flatc = os.path.join(mnn_root, "3rd_party", "flatbuffers", "tmp", "flatc")
    if os.path.isfile(flatc) and os.access(flatc, os.X_OK):
        return flatc

    sys_flatc = _have("flatc")
    if sys_flatc:
        log.vinfo(f"using system flatc: {sys_flatc}")
        return sys_flatc

    fbs_root = os.path.join(mnn_root, "3rd_party", "flatbuffers")
    if not os.path.isdir(fbs_root):
        raise SystemExit("3rd_party/flatbuffers/ missing — cannot build flatc")
    tmp = os.path.join(fbs_root, "tmp")
    os.makedirs(tmp, exist_ok=True)
    log.info(f"building vendored flatc in {tmp}")
    subprocess.run(["cmake", ".."], cwd=tmp, check=True)
    subprocess.run(["cmake", "--build", ".", "--target", "flatc", "-j"],
                   cwd=tmp, check=True)
    if not os.path.isfile(flatc):
        raise SystemExit(f"flatc build did not produce {flatc}")
    return flatc


def ensure_python_bindings(mnn_root: str, log: Logger) -> str:
    """Generate (or reuse cached) Python bindings for MNN.fbs et al.

    Returns: absolute path to the cache directory containing them.
    """
    cache = os.path.join(mnn_root, CACHE_SUBDIR)
    sentinel = os.path.join(cache, ".generated")
    schema_dir = os.path.join(mnn_root, "schema", "default")

    fbs_files = sorted(
        os.path.join(schema_dir, f)
        for f in os.listdir(schema_dir)
        if f.endswith(".fbs")
    )
    if not fbs_files:
        raise SystemExit("no .fbs files in schema/default")

    # Cache invalidation: regenerate if any .fbs is newer than the sentinel.
    fresh = os.path.exists(sentinel) and all(
        os.path.getmtime(f) <= os.path.getmtime(sentinel)
        for f in fbs_files
    )
    if fresh:
        log.vinfo(f"reusing cached MNN python bindings: {cache}")
        return cache

    log.info("generating MNN python flatbuffer bindings")
    if os.path.isdir(cache):
        shutil.rmtree(cache)
    os.makedirs(cache, exist_ok=True)

    flatc = _ensure_flatc(mnn_root, log)
    cmd = [flatc, "--python", "--gen-object-api", "-o", cache] + fbs_files
    log.vinfo("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    open(sentinel, "w").close()
    log.ok(f"MNN python bindings ready at {cache}")
    return cache


def import_mnn_module(cache_dir: str, name: str = "MNN") -> ModuleType:
    """Import a generated MNN python package from `cache_dir`.

    flatc emits a directory named `MNN/` (matching `namespace MNN;`) under
    cache_dir — we add cache_dir to sys.path and import normally.
    """
    if cache_dir not in sys.path:
        sys.path.insert(0, cache_dir)
    if name in sys.modules:
        return sys.modules[name]
    return importlib.import_module(name)


def load_mnn(mnn_root: str, mnn_path: str, log: Logger):
    """Parse a .mnn file. Returns (NetT object_api_obj, raw_bytes, MNN_module).

    NetT is the object-API mutable representation; raw_bytes is the original
    flatbuffer (kept around in case we want to reserialize unmodified spans).
    """
    cache = ensure_python_bindings(mnn_root, log)
    MNN = import_mnn_module(cache)
    Net = MNN.Net.Net  # generated reader class
    NetT = MNN.Net.NetT  # object-API class (because we passed --gen-object-api)

    with open(mnn_path, "rb") as fh:
        buf = fh.read()
    raw = bytearray(buf)
    net_reader = Net.GetRootAs(raw, 0)
    netT = NetT.InitFromObj(net_reader)
    log.ok(f"parsed {mnn_path}: {len(netT.oplists or [])} ops, "
           f"{len(netT.tensorName or [])} tensors")
    return netT, bytes(buf), MNN


def save_mnn(MNN, netT, out_path: str, log: Logger) -> None:
    """Serialize a NetT object back to a .mnn file."""
    import flatbuffers
    builder = flatbuffers.Builder(1024)
    root = netT.Pack(builder)
    builder.Finish(root)
    out_buf = bytes(builder.Output())
    with open(out_path, "wb") as fh:
        fh.write(out_buf)
    log.ok(f"wrote {out_path} ({len(out_buf)} bytes)")
