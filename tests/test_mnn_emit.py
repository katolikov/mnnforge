"""Phase 4 — emission tests using a fake MNN tree.

We construct a tiny mock of source/backend/opencl/{cl,image} so that
emit_pattern + patch_fuse_execution can run without depending on a real
MNN clone.
"""
import os
import shutil
import textwrap
import pytest

from mnnforge.log import Logger
from mnnforge.onnx_fsm import Pattern, Chain, ChainStep
from mnnforge import mnn_emit


# --- fake MNN tree fixture ----------------------------------------

FUSE_EXECUTION_STUB = textwrap.dedent("""\
    // fake MNN FuseExecution.cpp for tests
    #include "backend/opencl/execution/image/FuseExecution.hpp"

    namespace MNN {
    namespace OpenCL {

    class FuseCreator : public OpenCLBackend::Creator {
    public:
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                                    const std::vector<Tensor *> &outputs,
                                    const MNN::Op *op,
                                    Backend *backend) const override {
            auto param = op->main_as_Extra();
            if (param && param->type()->str() == "ExtraConvolution2DPrelu") {
                return nullptr;
            }
            return new FuseExecution(inputs, backend, op);
        }
    };
    REGISTER_OPENCL_OP_CREATOR(FuseCreator, OpType_Extra, IMAGE);

    } // namespace OpenCL
    } // namespace MNN
""")


CODEGEN_STUB = textwrap.dedent("""\
    # tiny stand-in: scan *.cl, write _mnn_cl.cpp + opencl_source_map.hpp
    import os, sys
    d = sys.argv[1]
    cls = sorted(f for f in os.listdir(d) if f.endswith('.cl'))
    with open(os.path.join(d, 'opencl_source_map.hpp'), 'w') as fh:
        for name in cls:
            stem = name[:-3]
            fh.write(f'extern const char* {stem};\\n')
        fh.write('namespace MNN { '
                 'const std::map<std::string, const char*> '
                 'OpenCLProgramMap = {')
        for name in cls:
            stem = name[:-3]
            fh.write(f' {{"{stem}", {stem}}},')
        fh.write('}; }\\n')
    for name in cls:
        stem = name[:-3]
        with open(os.path.join(d, f'{stem}_mnn_cl.cpp'), 'w') as fh:
            fh.write(f'const char* {stem} = "...";\\n')
""")


@pytest.fixture
def fake_mnn(tmp_path):
    cl = tmp_path / "source" / "backend" / "opencl" / "execution" / "cl"
    img = tmp_path / "source" / "backend" / "opencl" / "execution" / "image"
    cl.mkdir(parents=True); img.mkdir(parents=True)
    (img / "FuseExecution.cpp").write_text(FUSE_EXECUTION_STUB)
    (cl / "opencl_codegen.py").write_text(CODEGEN_STUB)
    return str(tmp_path)


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


# --- tests --------------------------------------------------------

def test_emit_pattern_writes_three_files(fake_mnn):
    p = _pattern()
    e = mnn_emit.emit_pattern(fake_mnn, p, Logger(verbose=False))
    assert os.path.exists(e.cl_path)
    assert os.path.exists(e.exec_hpp_path)
    assert os.path.exists(e.exec_cpp_path)
    cl = open(e.cl_path).read()
    assert "__kernel void mnnforge_a3b8c12d" in cl
    hpp = open(e.exec_hpp_path).read()
    assert "MnnForgeA3b8c12dExecution" in hpp


def test_patch_fuse_execution_inserts_dispatch(fake_mnn):
    p = _pattern()
    e = mnn_emit.emit_pattern(fake_mnn, p, Logger(verbose=False))
    mnn_emit.patch_fuse_execution(fake_mnn, [e], Logger(verbose=False))
    fuse = open(os.path.join(fake_mnn, "source", "backend", "opencl",
                             "execution", "image", "FuseExecution.cpp")).read()
    assert mnn_emit.MARKER_BEGIN in fuse
    assert mnn_emit.MARKER_END in fuse
    assert "MnnForgeA3b8c12dExecution" in fuse
    assert "_mnnforge_dispatch" in fuse
    # Backup created.
    assert os.path.exists(os.path.join(fake_mnn, "source", "backend", "opencl",
                                       "execution", "image",
                                       "FuseExecution.cpp.mnnforge.bak"))


def test_patch_is_idempotent(fake_mnn):
    p = _pattern()
    e = mnn_emit.emit_pattern(fake_mnn, p, Logger(verbose=False))
    mnn_emit.patch_fuse_execution(fake_mnn, [e], Logger(verbose=False))
    mnn_emit.patch_fuse_execution(fake_mnn, [e], Logger(verbose=False))
    fuse = open(os.path.join(fake_mnn, "source", "backend", "opencl",
                             "execution", "image", "FuseExecution.cpp")).read()
    # Block markers should appear exactly once.
    assert fuse.count(mnn_emit.MARKER_BEGIN) == 1
    assert fuse.count(mnn_emit.MARKER_END) == 1


def test_emit_all_runs_codegen(fake_mnn):
    p = _pattern()
    emissions = mnn_emit.emit_all(fake_mnn, [p], top_n=4,
                                   log=Logger(verbose=False))
    assert len(emissions) == 1
    # Codegen stub should have produced opencl_source_map.hpp
    src_map = os.path.join(fake_mnn, "source", "backend", "opencl",
                           "execution", "cl", "opencl_source_map.hpp")
    assert os.path.exists(src_map)
    body = open(src_map).read()
    assert "mnnforge_a3b8c12d" in body


def test_rollback_restores_and_removes(fake_mnn):
    p = _pattern()
    mnn_emit.emit_all(fake_mnn, [p], top_n=4, log=Logger(verbose=False))
    n = mnn_emit.rollback(fake_mnn, Logger(verbose=False))
    assert n >= 3   # one .cl, one .cpp, one .hpp at minimum
    fuse = open(os.path.join(fake_mnn, "source", "backend", "opencl",
                             "execution", "image", "FuseExecution.cpp")).read()
    assert mnn_emit.MARKER_BEGIN not in fuse
    # Generated files removed.
    img = os.path.join(fake_mnn, "source", "backend", "opencl",
                       "execution", "image")
    assert not any(f.startswith("MnnForge") for f in os.listdir(img))
