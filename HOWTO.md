# HOWTO — Using mnnforge end-to-end

A step-by-step guide. If something looks wrong, jump to
[Troubleshooting](#10-troubleshooting) at the bottom.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Install](#2-install)
3. [First run (one-shot)](#3-first-run-one-shot)
4. [What lands where](#4-what-lands-where)
5. [Build MNN](#5-build-mnn)
6. [Convert the optimized ONNX](#6-convert-the-optimized-onnx)
7. [Verify end-to-end](#7-verify-end-to-end)
8. [Tuning fusion](#8-tuning-fusion)
9. [Rollback](#9-rollback)
10. [Troubleshooting](#10-troubleshooting)
11. [FAQ](#11-faq)

---

## 1. Prerequisites

| Requirement | Why | How to get |
|---|---|---|
| Python ≥ 3.9 | runs `mnnforge` | system / pyenv |
| MNN source tree | mnnforge writes into `source/backend/opencl/...` | `git clone https://github.com/alibaba/MNN` |
| C++ toolchain (CMake + ninja or make) | builds MNN itself afterwards | Xcode CLT / build-essential |
| Python deps | `pip install -r requirements.txt` | onnx, onnxruntime, numpy |

---

## 2. Install

```bash
git clone https://github.com/katolikov/mnnforge
cd mnnforge

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Verify:

```bash
python -m mnnforge --version              # mnnforge 0.2.0
python -m pytest tests/ -q                # 42 passed
```

---

## 3. First run (one-shot)

```bash
python -m mnnforge /path/to/MNN ./model.onnx -v
```

What happens:

| Phase | What's printed | Output |
|---:|---|---|
| 0 | preflight checks | — |
| 1 | `[pass] prelu / fold_unary / negative_axes / dead_init` | `model.canon.onnx` |
| 2 | `pattern <fp>: 12× len=3 score=24 [BinaryOp(2,pos=0) → ...]` | (table) |
| 3 | (kernel synthesis is silent unless `-v`) | (in-memory) |
| 4 | `emitted source/backend/opencl/execution/cl/mnnforge_<fp>.cl` ×N, `patched FuseExecution.cpp` | files in MNN tree |
| 5 | `wrote model.optimized.onnx (18 subgraph(s) replaced)` | `model.optimized.onnx` |
| 6 | `[checker:..] OK`, `[ort_smoke] OK`, `[output_names_match] OK`, `[custom_nodes_well_formed] N custom node(s)` | `model.mnnforge.report.json` |

Exit code 0 if Phase 6 passes.

---

## 4. What lands where

**Next to your ONNX file** (in `--workdir DIR` if you set one):

```
model.canon.onnx
model.optimized.onnx
model.mnnforge.report.json
```

**Inside the MNN tree**, all auto-generated:

```
source/backend/opencl/execution/cl/mnnforge_<fp>.cl                  ← kernel source
source/backend/opencl/execution/cl/mnnforge_<fp>_mnn_cl.cpp          ← string blob
source/backend/opencl/execution/cl/opencl_source_map.hpp             ← regenerated
source/backend/opencl/execution/image/MnnForge<Fp>Execution.hpp      ← class header
source/backend/opencl/execution/image/MnnForge<Fp>Execution.cpp      ← class impl
source/backend/opencl/execution/image/FuseExecution.cpp              ← patched
source/backend/opencl/execution/image/FuseExecution.cpp.mnnforge.bak ← restore copy
```

The `MNNFORGE-BEGIN`/`MNNFORGE-END` markers in `FuseExecution.cpp` make the patch easy
to spot in `git diff` and trivially safe to re-apply on re-runs.

You should also add the generated paths to your local MNN `.gitignore` if you don't want
them tracked. They're regenerated on every `mnnforge` invocation.

---

## 5. Build MNN

A normal MNN build picks up the new files automatically — no extra flags needed.

```bash
cd /path/to/MNN
mkdir -p build && cd build
cmake .. -DMNN_OPENCL=ON -DMNN_BUILD_CONVERTER=ON -DCMAKE_BUILD_TYPE=Release
make -j
```

If the build fails complaining about a generated file, run `mnnforge --rollback` and check
the error against the templates in `mnn_emit.py`.

---

## 6. Convert the optimized ONNX

Use the freshly built `MNNConvert`:

```bash
./MNNConvert \
    -f ONNX \
    --modelFile /path/to/model.optimized.onnx \
    --MNNModel  /path/to/model.mnn \
    --bizCode mnn
```

MNN's default ONNX importer auto-wraps the `MnnForge_<fp>` custom-domain nodes as
`OpType_Extra` with `extra.type = "MnnForge_<fp>"`. The patched `FuseExecution.cpp`
recognizes that prefix and dispatches to your generated `MnnForge<Fp>Execution`.

---

## 7. Verify end-to-end

Run the original `.mnn` (built from the un-optimized ONNX) and the fused `.mnn` against
ONNX Runtime as ground truth:

```bash
# Build a stock .mnn for comparison (one-time)
./MNNConvert -f ONNX --modelFile /path/to/model.canon.onnx \
                     --MNNModel  /path/to/model.original.mnn \
                     --bizCode mnn
```

A simple comparison harness:

```python
import numpy as np, onnxruntime as ort, MNN
import MNN.expr as F

inputs = {"x": np.random.randn(1, 3, 224, 224).astype(np.float32)}

# ORT
sess = ort.InferenceSession("model.canon.onnx",
                            providers=["CPUExecutionProvider"])
ref = sess.run(None, inputs)[0]

# MNN OpenCL
rt  = MNN.nn.create_runtime_manager(({"backend": 3, "precision": "high"},))
net = MNN.nn.load_module_from_file("model.mnn", ["x"], ["y"], runtime_manager=rt)
v = F.placeholder(list(inputs["x"].shape), F.NCHW); v.write(inputs["x"])
out = F.convert(net.forward([v])[0], F.NCHW).read()

print("max abs err:", float(np.abs(ref - out).max()))
```

Tolerable max-abs-err on fp32 is typically `1e-3` to `5e-3` against ORT.

### Bit-exact: optimized.mnn vs original.mnn

The stronger guarantee mnnforge gives is **bit-exact equivalence between
`original.mnn` and `optimized.mnn` when both run on MNN at `precision=high`
(fp32)**. This holds because:

1. The OpenCL float intrinsics in our generated kernels (`native_recip`,
   `native_exp`, `native_sqrt`, GELU polynomial, …) are byte-identical to
   the snippets in MNN's stock `unary.cl`/`binary.cl`. There's a unit test
   that asserts this against the live `unary.cl`
   (`tests/test_mnn_emit.py::test_kernel_uses_same_intrinsics_as_mnn_unary_cl`).
2. We preserve the original op order — chains are extended only forward,
   never reordered.
3. fp32 IEEE-754 ops on the same lane values are deterministic regardless
   of whether the intermediate float4 lives in registers or round-trips
   through Image2D.

Bit-exact verification:

```python
import numpy as np, MNN
import MNN.expr as F

inputs = {"x": np.random.randn(1, 3, 224, 224).astype(np.float32)}

def run(mnn_path):
    rt  = MNN.nn.create_runtime_manager(({"backend": 3,
                                           "precision": "high"},))
    net = MNN.nn.load_module_from_file(mnn_path, ["x"], ["y"],
                                        runtime_manager=rt)
    v = F.placeholder(list(inputs["x"].shape), F.NCHW); v.write(inputs["x"])
    return np.array(F.convert(net.forward([v])[0], F.NCHW).read(), copy=True)

a = run("model.original.mnn")
b = run("model.fused.mnn")

# Bit-exact: every float lane must compare equal under .view('uint32').
assert (a.view(np.uint32) == b.view(np.uint32)).all(), \\
       f"max abs diff: {np.abs(a - b).max()}"
print("bit-exact ✓")
```

Caveats:

* **Precision mode matters.** `precision=low`/`precision=normal` use fp16
  storage; bit-exact is impossible because Image2D fp16 round-trips lose
  precision that registers preserve. Use `precision=high` for the
  guarantee.
* **GPU vendors matter.** Intel and Adreno drivers occasionally differ on
  `native_*` rounding. Bit-exact holds within a single driver; cross-vendor
  comparisons may show ULP-level drift.
* **Approximate intrinsics inside chains.** `native_exp`/`native_log` are
  vendor-implementation-defined. If you need cross-vendor reproducibility
  rather than within-device bit-exact, swap to the slower precise versions
  (`exp`, `log`) by editing your `mnnforge_<fp>.cl` file before building MNN.

---

## 8. Tuning fusion

| Flag | Default | Effect |
|---|---:|---|
| `--top-n N` | 4 | Apply only the N highest-scoring patterns |
| `--max-pattern-size K` | 6 | Don't extend a chain beyond K ops |

```bash
# Aggressive
python -m mnnforge /path/to/MNN model.onnx --top-n 8 --max-pattern-size 8

# Conservative (one pattern, length ≤ 3)
python -m mnnforge /path/to/MNN model.onnx --top-n 1 --max-pattern-size 3

# Analysis only (no MNN tree edits, no ONNX rewrite)
python -m mnnforge /path/to/MNN model.onnx --skip-emit --skip-rewrite
```

---

## 9. Rollback

To revert all MNN-tree edits and remove every `mnnforge_<fp>.cl` /
`MnnForge<Fp>Execution.{hpp,cpp}` file:

```bash
python -m mnnforge /path/to/MNN --rollback
```

This:

* restores `FuseExecution.cpp` from `.mnnforge.bak`
* removes every `mnnforge_*.cl` and `*_mnn_cl.cpp`
* removes every `MnnForge*Execution.{hpp,cpp}`
* re-runs `opencl_codegen.py` so `opencl_source_map.hpp` is consistent

`git status` should now report a clean MNN tree.

---

## 10. Troubleshooting

### `These Op Not Support: ONNX::MnnForge_<fp> | …  Converted Failed!`
This is MNN's `writeFb.cpp` rejecting `OpType_Extra` ops whose
`engine != "MNN"`. Two fixes:

1. **Permanent (recommended)** — re-run `python -m mnnforge /path/to/MNN`,
   which writes
   `tools/converter/source/onnx/MnnForgeOnnx.cpp`. Rebuild MNN. The new
   converter sets `engine="MNN"` so MNNConvert accepts the ops without any
   extra flag. Versions ≥ 0.2.1 emit this file automatically.
2. **One-shot workaround** — pass `--allowCustomOp` to MNNConvert:
   ```bash
   ./MNNConvert --allowCustomOp -f ONNX --modelFile model.optimized.onnx \\
                --MNNModel model.mnn --bizCode mnn
   ```
   This bypasses the engine check globally.

### `cmake not found in PATH`
Install CMake (`brew install cmake` / `apt install cmake`).

### `opencl_codegen.py failed (rc=N)`
The generated `.cl` has invalid OpenCL syntax (rare with v1's snippet library).
Check the offending file in `source/backend/opencl/execution/cl/` and the stderr emitted
above. Open an issue with the report.json attached.

### MNN build fails on a generated file
A generated `Execution` class doesn't compile against your specific MNN revision.
Run `mnnforge --rollback`, file an issue with the MNN commit hash.

### `[ort_smoke] failed to run`
Your *canonical* ONNX is broken — likely a Phase 1 pass changed something it shouldn't
have. Re-run with `--no-ort-verify-canon` to see Phase 1's report and report against the
specific pass.

### `no fusable patterns discovered`
Your model has no repeating elementwise motifs of length ≥ 2 (some classical CNNs).
Try `--max-pattern-size 3` to lower the bar, or accept that there's nothing to do.

### `mnn_root missing required path: source/backend/opencl/...`
The path you passed isn't an MNN tree, or the tree is too old (pre-OpenCL backend).

### Patched `FuseExecution.cpp` doesn't compile
The `MNNFORGE-BEGIN/END` block is bracketed for safety. If your MNN revision changed the
namespace declaration order, the patch's anchor (`namespace MNN`) may be wrong. Run
`mnnforge --rollback` and file an issue with the diff context.

---

## 11. FAQ

**Q: Does mnnforge modify MNN source?**
Yes. Inside `source/backend/opencl/execution/cl/` and
`source/backend/opencl/execution/image/`. `--rollback` reverses every edit. Files are clearly
marked auto-generated and the `FuseExecution.cpp` patch is bracketed by
`MNNFORGE-BEGIN/MNNFORGE-END`.

**Q: Will the resulting `.mnn` file load on stock MNN binaries that don't have my
mnnforge-built MNN?**
**No.** Unlike the post-processor design, the runtime now expects the generated
`Execution` classes to be linked into the MNN binary. You ship your patched MNN alongside
your `.mnn` files.

**Q: Can I commit the generated files into my MNN fork?**
Yes — they're plain C++ and `.cl`. Add them to your fork's tracked files; rebuild as
normal.

**Q: Does the dispatch patch in `FuseExecution.cpp` add overhead at runtime?**
No measurable overhead. It's a string compare per `OpType_Extra` op at session creation
time, then the right `Execution` is cached.

**Q: What if MNN already fuses this pattern internally (e.g. LayerNorm)?**
mnnforge's snippet library covers BinaryOp/UnaryOp/Relu only; it never matches
LayerNorm/Attention/etc. shapes that MNN already fuses via `Fuse*.cpp` passes.

**Q: Will mnnforge handle Vulkan / Metal / CUDA next?**
Roadmap. The runtime fuse path exists in those backends (`MetalFuse.mm`,
`VulkanFuse.cpp`, `FuseExecutionV2.cu`); v1 emits OpenCL only.

**Q: Where do I report bugs?**
Open an issue with the `*.mnnforge.report.json` and the exact command line. If it's an
MNN converter issue, file it against `alibaba/MNN`.
