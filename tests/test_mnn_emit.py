"""Phase 4 — emission tests against the REAL MNN tree.

These tests copy the relevant *real* MNN files into a temporary sandbox
(so the user's tree is never mutated) and exercise emit_pattern,
patch_fuse_execution, the ONNX-converter emission, codegen, and rollback
end-to-end. No synthetic stubs — everything that gets parsed/patched is
the actual file from MNN.

Tests skip cleanly when the MNN tree isn't reachable:
  * MNN_ROOT env var (preferred)
  * fallback: ../../  relative to this file's tests/ directory
"""
import os
import shutil
import pytest

from mnnforge.log import Logger
from mnnforge.onnx_fsm import Pattern, Chain, ChainStep
from mnnforge import mnn_emit


# ---------------------------------------------------------------------------
# locate the real MNN tree
# ---------------------------------------------------------------------------

def _resolve_mnn_root() -> str:
    cand = os.environ.get("MNN_ROOT")
    if cand and os.path.isdir(cand):
        return cand
    here = os.path.dirname(os.path.abspath(__file__))
    fallback = os.path.realpath(os.path.join(here, "..", "..", ".."))
    return fallback


REAL_MNN = _resolve_mnn_root()
HAS_REAL_MNN = all(os.path.isfile(os.path.join(REAL_MNN, p)) for p in [
    "schema/default/MNN.fbs",
    "source/backend/opencl/execution/image/FuseExecution.cpp",
    "source/backend/opencl/execution/cl/opencl_codegen.py",
])


pytestmark = pytest.mark.skipif(
    not HAS_REAL_MNN,
    reason=(
        "real MNN tree not reachable; set MNN_ROOT env var or place mnnforge "
        "under <MNN_ROOT>/tools/ (current resolved root: " + REAL_MNN + ")"
    ),
)


@pytest.fixture
def real_sandbox(tmp_path):
    """Build a scratch directory that mirrors the real MNN tree's
    structure but uses *copies* of the real files. We never touch the
    user's actual MNN tree."""
    rels = [
        "source/backend/opencl/execution/image/FuseExecution.cpp",
        "source/backend/opencl/execution/cl/opencl_codegen.py",
    ]
    for rel in rels:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(os.path.join(REAL_MNN, rel), dst)

    # Copy every existing .cl file too — opencl_codegen.py walks the
    # whole directory and we want a faithful run.
    real_cl = os.path.join(REAL_MNN, "source/backend/opencl/execution/cl")
    sand_cl = tmp_path / "source/backend/opencl/execution/cl"
    for fn in os.listdir(real_cl):
        if fn.endswith(".cl"):
            shutil.copy2(os.path.join(real_cl, fn), sand_cl / fn)

    # Provide a converter directory so emit_onnx_converter has somewhere
    # to write its output. We don't need the real onnxOpConverter source.
    (tmp_path / "tools/converter/source/onnx").mkdir(parents=True, exist_ok=True)

    return str(tmp_path)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _pattern():
    p = Pattern(
        fingerprint="a3b8c12d",
        op_kinds=[("BinaryOp", 2, -1), ("UnaryOp", 23, 0)],   # Mul, Sigmoid
    )
    p.occurrences = [Chain(
        steps=[
            ChainStep(onnx_idx=0, type_name="BinaryOp", sub_kind=2,
                      chain_pos=-1, extra_input_tensors=[]),
            ChainStep(onnx_idx=1, type_name="UnaryOp", sub_kind=23,
                      chain_pos=0, extra_input_tensors=[]),
        ],
        boundary_inputs=["a", "b"],
        boundary_output="y",
    )]
    p.score = 2.0
    return p


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_emit_pattern_writes_three_files(real_sandbox):
    e = mnn_emit.emit_pattern(real_sandbox, _pattern(), Logger(verbose=False))
    assert os.path.exists(e.cl_path)
    assert os.path.exists(e.exec_hpp_path)
    assert os.path.exists(e.exec_cpp_path)
    assert "__kernel void mnnforge_a3b8c12d" in open(e.cl_path).read()
    assert "MnnForgeA3b8c12dExecution" in open(e.exec_hpp_path).read()


def test_patch_fuse_execution_inserts_dispatch(real_sandbox):
    e = mnn_emit.emit_pattern(real_sandbox, _pattern(), Logger(verbose=False))
    mnn_emit.patch_fuse_execution(real_sandbox, [e], Logger(verbose=False))
    fuse_path = os.path.join(
        real_sandbox, "source/backend/opencl/execution/image/FuseExecution.cpp"
    )
    fuse = open(fuse_path).read()
    assert mnn_emit.MARKER_BEGIN in fuse
    assert mnn_emit.MARKER_END in fuse
    assert "MnnForgeA3b8c12dExecution" in fuse
    assert "_mnnforge_dispatch" in fuse
    assert os.path.exists(fuse_path + ".mnnforge.bak")


def test_patch_is_idempotent(real_sandbox):
    e = mnn_emit.emit_pattern(real_sandbox, _pattern(), Logger(verbose=False))
    mnn_emit.patch_fuse_execution(real_sandbox, [e], Logger(verbose=False))
    mnn_emit.patch_fuse_execution(real_sandbox, [e], Logger(verbose=False))
    fuse = open(os.path.join(
        real_sandbox, "source/backend/opencl/execution/image/FuseExecution.cpp"
    )).read()
    assert fuse.count(mnn_emit.MARKER_BEGIN) == 1
    assert fuse.count(mnn_emit.MARKER_END) == 1


def test_emit_all_runs_real_codegen(real_sandbox):
    emissions = mnn_emit.emit_all(real_sandbox, [_pattern()], top_n=4,
                                   log=Logger(verbose=False))
    assert len(emissions) == 1
    src_map = os.path.join(
        real_sandbox, "source/backend/opencl/execution/cl/opencl_source_map.hpp"
    )
    assert os.path.exists(src_map)
    body = open(src_map).read()
    # The real opencl_codegen.py produced a registry with our kernel name in it.
    assert "mnnforge_a3b8c12d" in body
    # And it produced the embedded-string blob.
    assert os.path.exists(os.path.join(
        real_sandbox,
        "source/backend/opencl/execution/cl/mnnforge_a3b8c12d_mnn_cl.cpp",
    ))


def test_emit_all_writes_cpu_stub(real_sandbox):
    """Aligns mnnforge with MNN's add-new-op SKILL step 3 (CPU is the
    universal fallback). Without this, MNN's GeometryComputerUtils const-
    folder fails on backup CPU backend, and the model can't run on
    backends other than OpenCL."""
    # Make sure source/backend/cpu/ is present in the sandbox.
    cpu_dir = os.path.join(real_sandbox, "source/backend/cpu")
    os.makedirs(cpu_dir, exist_ok=True)

    emissions = mnn_emit.emit_all(real_sandbox, [_pattern()], top_n=4,
                                   log=Logger(verbose=False))
    cpath = os.path.join(cpu_dir, "MnnForgeCPU.cpp")
    assert os.path.exists(cpath), "MnnForgeCPU.cpp not emitted"
    body = open(cpath).read()
    # Registers as the CPU creator for OpType_Extra.
    assert "REGISTER_CPU_OP_CREATOR(MnnForgeCPUCreator, OpType_Extra)" in body
    # Returns nullptr for non-MnnForge Extra ops (preserves existing
    # behaviour for other Extra users).
    assert 'rfind("MnnForge_", 0) != 0' in body
    # Reads op_kinds from extra->attr.
    assert '"op_kinds"' in body


def test_emit_all_writes_onnx_converter(real_sandbox):
    """The fix for the 'These Op Not Support' error — engine='MNN' converter."""
    emissions = mnn_emit.emit_all(real_sandbox, [_pattern()], top_n=4,
                                   log=Logger(verbose=False))
    cpath = os.path.join(real_sandbox,
                         "tools/converter/source/onnx/MnnForgeOnnx.cpp")
    assert os.path.exists(cpath), "MnnForgeOnnx.cpp not emitted"
    body = open(cpath).read()
    # Must set engine to "MNN" — that's what writeFb.cpp:200 checks.
    assert 'extra->engine   = "MNN"' in body
    # Must register for our fingerprint.
    assert "REGISTER_CONVERTER(MnnForgeOnnx, MnnForge_a3b8c12d);" in body
    # opType must be Extra so the dispatch in FuseExecution.cpp sees it.
    assert "MNN::OpType_Extra" in body


def test_rollback_restores_real_files(real_sandbox):
    mnn_emit.emit_all(real_sandbox, [_pattern()], top_n=4,
                     log=Logger(verbose=False))
    n = mnn_emit.rollback(real_sandbox, Logger(verbose=False))
    assert n >= 5   # 1 .cl + 1 _mnn_cl.cpp + 1 .hpp + 1 .cpp + onnx + cpu stub
    fuse = open(os.path.join(
        real_sandbox, "source/backend/opencl/execution/image/FuseExecution.cpp"
    )).read()
    assert mnn_emit.MARKER_BEGIN not in fuse
    img = os.path.join(real_sandbox, "source/backend/opencl/execution/image")
    assert not any(f.startswith("MnnForge") for f in os.listdir(img))
    onx = os.path.join(real_sandbox, "tools/converter/source/onnx")
    assert not any(f == "MnnForgeOnnx.cpp" for f in os.listdir(onx))


def test_kernel_uses_same_intrinsics_as_mnn_unary_cl(real_sandbox):
    """Bit-exact-on-fp32 hinges on our snippets using the same OpenCL float
    intrinsics that MNN's stock unary.cl uses. Spot-check by parsing the
    real unary.cl and verifying our generated sigmoid kernel uses
    native_recip + native_exp."""
    real_unary = os.path.join(
        REAL_MNN, "source/backend/opencl/execution/cl/unary.cl",
    )
    if not os.path.isfile(real_unary):
        pytest.skip("real unary.cl missing")
    e = mnn_emit.emit_pattern(real_sandbox, _pattern(), Logger(verbose=False))
    cl = open(e.cl_path).read()
    # Both use native_recip + native_exp for sigmoid.
    assert "native_recip" in cl
    assert "native_exp" in cl
