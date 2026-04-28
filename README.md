# mnnforge

> **Forge custom OpenCL kernels for [MNN](https://github.com/alibaba/MNN) by collapsing repeated subgraphs in your ONNX model.**
> Fewer kernel launches, fewer VRAM round-trips, generated kernels live in the MNN tree.

`mnnforge` is a single-binary CLI that takes a path to an MNN source tree and an `.onnx`
model, mines repeating elementwise op chains, and writes:

* OpenCL kernels into `source/backend/opencl/execution/cl/mnnforge_<fp>.cl`
* matching `Execution` C++ classes into `source/backend/opencl/execution/image/`
* a 3-line dispatch patch into `source/backend/opencl/execution/image/FuseExecution.cpp`
* a rewritten `model.optimized.onnx` whose fused subgraphs are single custom-op nodes

You then build MNN once and run `MNNConvert` on `model.optimized.onnx` — MNN's existing ONNX
importer auto-wraps the unknown nodes as `OpType_Extra`, the patched `FuseExecution.cpp`
dispatches them to your generated `Execution` classes, and the kernels run on the OpenCL
backend.

```
input.onnx ──▶ canonicalize ──▶ FSM ──▶ synth OpenCL ──▶ emit into MNN tree
                                                                 │
                                                                 ├─▶ source/backend/opencl/execution/cl/*.cl
                                                                 ├─▶ source/backend/opencl/execution/image/*Execution.{hpp,cpp}
                                                                 └─▶ FuseExecution.cpp dispatch patch
            ──▶ rewrite ONNX (custom-op nodes) ──▶ model.optimized.onnx
            ──▶ structural verify ──▶ report.json
```

---

## Why this design

Original plan: post-process the `.mnn` file. Final design: **edit the MNN source**, write
`.cl` and `Execution` files, patch one dispatcher. Why? Because:

1. You build MNN once and your custom kernels are baked into the binary — no runtime
   compile cost on session creation.
2. Kernel sources live in version-controlled files inside the MNN tree, easy to inspect
   and tweak.
3. The same `MNNConvert` CLI you'd use anyway handles the conversion. mnnforge never has
   to duplicate that pipeline.
4. The patched `FuseExecution.cpp` change is **tiny and idempotent**: a 3-line `if` chain
   wrapped in `MNNFORGE-BEGIN/END` markers, removable via `--rollback`.

---

## Pipeline

```
                ┌──────────────────────────────────────────┐
                │              your .onnx model            │
                └──────────────────────┬───────────────────┘
                                       │
   Phase 0 preflight ──────────────────▼─────────────  validates inputs;
                                                        refuses to touch
                                                        schema/private/
                                       │
   Phase 1 canonicalize ───────────────▼─────────────  PReLU normalize, fold
                                                        unary on initializers,
                                                        negative-axes, dead init
                                                        →  model.canon.onnx
                                       │
   Phase 2 FSM on ONNX graph ──────────▼─────────────  linear elementwise chains
                                                        fingerprinted by op-kind
                                                        sequence, scored by
                                                        eliminated VRAM hops
                                       │
   Phase 3 synthesize OpenCL kernel ───▼─────────────  per-pattern .cl source
                                                        with float4 body
                                       │
   Phase 4 emit into MNN tree ─────────▼─────────────  - mnnforge_<fp>.cl
                                                        - MnnForge<Fp>Execution.{hpp,cpp}
                                                        - patches FuseExecution.cpp
                                                        - re-runs opencl_codegen.py
                                       │
   Phase 5 rewrite ONNX ───────────────▼─────────────  custom-op nodes for each
                                                        fused chain
                                                        →  model.optimized.onnx
                                       │
   Phase 6 structural verify ──────────▼─────────────  schema + ORT smoke +
                                                        boundary preservation
                                       │
                ┌──────────────────────▼───────────────────┐
                │  optimized.onnx + .mnnforge.report.json   │
                └──────────────────────────────────────────┘
                                       │
                  ──── you build MNN ──┴── you run MNNConvert ────▶ .mnn
```

---

## Install

```bash
git clone https://github.com/katolikov/mnnforge
cd mnnforge
pip install -r requirements.txt
```

Requirements: Python ≥ 3.9, an existing MNN source tree, `onnx`, `onnxruntime`, `numpy`.

---

## Quick start

```bash
python -m mnnforge /path/to/MNN ./model.onnx -v
```

Outputs (next to your ONNX file by default):

```
model.canon.onnx                   ← canonicalized input
model.optimized.onnx               ← fused subgraphs replaced with custom-op nodes
model.mnnforge.report.json         ← structural verification report
```

…and inside your MNN tree (auto-generated, gitignore-able):

```
source/backend/opencl/execution/cl/mnnforge_<fp>.cl
source/backend/opencl/execution/cl/mnnforge_<fp>_mnn_cl.cpp     (regenerated)
source/backend/opencl/execution/cl/opencl_source_map.hpp        (regenerated)
source/backend/opencl/execution/image/MnnForge<Fp>Execution.hpp
source/backend/opencl/execution/image/MnnForge<Fp>Execution.cpp
source/backend/opencl/execution/image/FuseExecution.cpp         (patched)
source/backend/opencl/execution/image/FuseExecution.cpp.mnnforge.bak
tools/converter/source/onnx/MnnForgeOnnx.cpp                    (ONNX→MNN converter)
```

The auto-generated `MnnForgeOnnx.cpp` registers an ONNX→MNN converter for
each `MnnForge_<fp>` op type with `engine="MNN"` so `MNNConvert` accepts
the optimized ONNX without `--allowCustomOp`.

Then **you** build MNN and convert the optimized ONNX:

```bash
cd /path/to/MNN/build
cmake .. -DMNN_OPENCL=ON -DMNN_BUILD_CONVERTER=ON && make -j
./MNNConvert -f ONNX --modelFile ../tools/somewhere/model.optimized.onnx \
                     --MNNModel model.mnn --bizCode mnn
```

That's it — `model.mnn` now uses your generated kernels at runtime.

---

## Common usage

```bash
# Be greedier
python -m mnnforge /path/to/MNN model.onnx --top-n 8 --max-pattern-size 8

# Analysis only (no MNN tree changes)
python -m mnnforge /path/to/MNN model.onnx --skip-emit --skip-rewrite

# Skip the slow ORT verification inside canonicalize
python -m mnnforge /path/to/MNN model.onnx --no-ort-verify-canon

# Roll back all MNN-tree edits and remove generated files
python -m mnnforge /path/to/MNN --rollback
```

See **[HOWTO.md](./HOWTO.md)** for the step-by-step walkthrough, troubleshooting, and FAQ.

---

## What gets fused

Currently `mnnforge` accepts patterns matching **all** of:

* every primitive in the chain is in our snippet library
  * `BinaryOp`: ADD, SUB, MUL, DIV (non-commutative ones honor operand position)
  * `UnaryOp`: ABS, NEG, SQRT, RSQRT, EXP, LOG, RECIPROCAL, TANH, SIGMOID, GELU
  * named: `Relu`
* every intermediate tensor has exactly one consumer **and** is not a graph output
* op output shape is preserved across the chain (no broadcasting, no reductions)
* the chain is at least 2 ops long

The generated kernel signature mirrors MNN's runtime `FuseExecution`:

```c
__kernel void mnnforge_<fp>(
    __read_only image2d_t in0, __read_only image2d_t in1, …,
    __write_only image2d_t out0,
    __private const int W0,
    __private const int W1,
    __private const int W2)
```

So your generated `Execution` class is a near-copy of `FuseExecution::onEncode` — bind
images, set `W0/W1/W2`, autotune local work size, run.

---

## Numerical guarantees

`mnnforge` itself can't run the optimized ONNX through ONNX Runtime (the custom op_types are
unknown to ORT). What it does verify:

1. Canonical ONNX passes `onnx.checker`.
2. Optimized ONNX passes `onnx.checker` (custom-domain ops skipped).
3. ORT can load and run the canonical ONNX without error.
4. Graph output names are unchanged between canonical and optimized.
5. Each emitted custom-op node has the expected attributes and tensor connections.

**Bit-exact within MNN** (precision=high, fp32): `original.mnn` and
`optimized.mnn` produce byte-identical output tensors when run on the
same OpenCL device. This holds because our generated kernels use the
same `native_*` intrinsics MNN's stock `unary.cl`/`binary.cl` use
(verified by a unit test) and we never reorder ops within a chain.
See **[HOWTO.md §7](./HOWTO.md#7-verify-end-to-end)** for the
verification script.

End-to-end numerical equivalence vs ORT (which uses different math
libraries) is in the `1e-3` to `5e-3` max-abs-error range on fp32.

---

## Architecture (one paragraph)

`mnnforge` parses your `.onnx`, runs canonical-ONNX optimizer passes, and walks the graph to
find linear elementwise chains using a tensor→consumer index. Each unique op-kind sequence
(including operand position for non-commutative `BinaryOp`) is a `Pattern`; occurrences claim
ops by score so two patterns can never overlap. For each accepted pattern the kernel
synthesizer emits an OpenCL `__kernel` that matches `FuseExecution`'s arg layout, plus a
matching MNN `Execution` C++ class that loads the kernel by program name and binds Image2D
inputs/outputs. A small **idempotent** patch in `FuseExecution.cpp` adds an `if/else` chain
that dispatches `OpType_Extra` ops with a `MnnForge_<fp>` `extra->type()` to the right
class. Finally the ONNX is rewritten with one custom-op node per fused subgraph; MNN's
default ONNX importer wraps unknown ops as `OpType_Extra` automatically, so your generated
dispatch fires at runtime.

---

## Testing

```bash
python -m pytest tests/ -v
# 42 passed
```

Coverage:

* imports + argparse construction
* primitive-snippet integrity (arity, helper consistency)
* kernel synthesis: commutative / non-commutative chains, signature match,
  helper inclusion, error paths
* ONNX FSM: graph-output preservation, multi-consumer breakage, occurrence
  grouping, non-commutative `chain_pos`, `max_pattern_size` capping,
  custom-domain skipping
* ONNX surgery: chain replacement, output preservation, attribute completeness,
  custom-domain `opset_import` registration, no-op when no patterns,
  overlap resolution by score
* MNN emit: writes 3 files per pattern, patches `FuseExecution.cpp`,
  patch is idempotent, runs `opencl_codegen.py` stub, rollback restores

---

## Project layout

```
mnnforge/
├── README.md                  ← this file
├── HOWTO.md                   ← step-by-step usage guide
├── LICENSE
├── requirements.txt
├── mnnforge/
│   ├── __main__.py            ← `python -m mnnforge ...`
│   ├── cli.py                 ← argparse driver
│   ├── log.py
│   ├── preflight.py           ← Phase 0
│   ├── canonicalize.py        ← Phase 1
│   ├── onnx_fsm.py            ← Phase 2 (mining over ONNX)
│   ├── kernel_synth.py        ← Phase 3 (OpenCL synthesis)
│   ├── primitives.py          ← snippet library
│   ├── mnn_emit.py            ← Phase 4 (writes into MNN tree)
│   ├── onnx_surgery.py        ← Phase 5 (rewrite ONNX)
│   └── verify.py              ← Phase 6 (structural)
└── tests/                     ← 42 unit tests
```

---

## License

Apache 2.0. See [LICENSE](./LICENSE).

`mnnforge` writes auto-generated files into the MNN tree under clearly-marked paths and a
single backed-up dispatcher patch. `--rollback` cleanly reverses every edit.
