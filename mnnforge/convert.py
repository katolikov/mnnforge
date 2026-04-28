"""Phase 2 — drive MNNConvert.

Builds MNNConvert (CMake) on first invocation into <mnn_root>/build_mnnforge,
then runs:  MNNConvert -f ONNX --modelFile X.onnx --MNNModel X.mnn --bizCode mnn

We do NOT modify any MNN source — this is a stock build of the converter.
"""
from __future__ import annotations
import os
import shutil
import subprocess
from typing import Optional

from .log import Logger


BUILD_DIR_NAME = "build_mnnforge"


def _have(executable: str) -> Optional[str]:
    return shutil.which(executable)


def ensure_converter(mnn_root: str, log: Logger) -> str:
    """Ensure MNNConvert exists in <mnn_root>/build_mnnforge/. Build if not.

    Returns: path to the MNNConvert executable.
    """
    build_dir = os.path.join(mnn_root, BUILD_DIR_NAME)
    converter = os.path.join(build_dir, "MNNConvert")

    if os.path.isfile(converter) and os.access(converter, os.X_OK):
        log.ok(f"MNNConvert present: {converter}")
        return converter

    if not _have("cmake"):
        raise SystemExit("cmake not found in PATH; cannot build MNNConvert")
    if not _have("make") and not _have("ninja"):
        raise SystemExit("neither make nor ninja in PATH")

    os.makedirs(build_dir, exist_ok=True)
    log.info(f"configuring CMake in {build_dir}")
    cmake_args = [
        "cmake", mnn_root,
        "-DMNN_BUILD_CONVERTER=ON",
        "-DMNN_OPENCL=ON",
        "-DMNN_BUILD_TEST=OFF",
        "-DMNN_BUILD_DEMO=OFF",
        "-DMNN_BUILD_LLM=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if _have("ninja"):
        cmake_args.extend(["-G", "Ninja"])

    subprocess.run(cmake_args, cwd=build_dir, check=True)
    log.info("building MNNConvert (this is the slow first-time step)")
    subprocess.run(
        ["cmake", "--build", build_dir, "--target", "MNNConvert", "-j"],
        check=True,
    )

    if not os.path.isfile(converter):
        raise SystemExit(f"MNNConvert did not appear at {converter}")
    log.ok(f"built MNNConvert at {converter}")
    return converter


def convert(converter_bin: str, onnx_path: str, mnn_path: str,
            log: Logger, biz_code: str = "mnn") -> None:
    """Invoke MNNConvert -f ONNX -> .mnn."""
    cmd = [
        converter_bin,
        "-f", "ONNX",
        "--modelFile", onnx_path,
        "--MNNModel", mnn_path,
        "--bizCode", biz_code,
    ]
    log.info("running: " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if log.verbose and res.stdout:
        for line in res.stdout.splitlines():
            log.vinfo(f"  | {line}")
    if res.returncode != 0:
        log.err(res.stdout)
        log.err(res.stderr)
        raise SystemExit(f"MNNConvert failed (rc={res.returncode})")
    if not os.path.isfile(mnn_path):
        raise SystemExit(f"MNNConvert returned 0 but {mnn_path} missing")
    log.ok(f"converted -> {mnn_path}")
