# mnnforge

> **Forge custom OpenCL kernels for [MNN](https://github.com/alibaba/MNN) by collapsing repeated subgraphs in your ONNX model.**
> Fewer kernel launches, fewer VRAM round-trips, a numerically-equivalent `.mnn` file.

`mnnforge` is a single-binary CLI that takes a path to an MNN source tree and an `.onnx`
model, mines repeating elementwise op chains, synthesizes a fused OpenCL kernel for each
discovered pattern, and writes a new `.mnn` file in which those op-spans are replaced by a
single runtime-compiled custom op. Numerical equivalence vs. the original model is verified
end-to-end against ONNX Runtime.

```
input.onnx ──▶ canonicalize ──▶ MNNConvert ──▶ FSM ──▶ synthesize OpenCL ──▶ patch .mnn ──▶ verify
```

---

## Why this exists

MNN already does a lot of fusion at conversion time (LayerNorm, GELU, Attention, Conv+BN+ReLU,
…). What it can't fuse is **the long tail of model-specific repeating motifs** — the
2-to-6-op chains that show up dozens of times in transformer/diffusion graphs and burn GPU
time on Image2D round-trips. `mnnforge` discovers those patterns automatically and folds each
into one OpenCL kernel that keeps intermediates in registers.

The trick that makes this practical: **MNN already supports runtime-compiled custom kernels
on every GPU backend**. `OpType_Extra` carries the kernel source as bytes inside the `.mnn`
file, and `FuseExecution.cpp` compiles it via `clBuildProgram` at session creation. So
`mnnforge` is a *pure post-processor*: it never modifies the MNN source tree, never edits a
schema, never runs `flatc` on MNN's headers, never triggers an MNN rebuild for new patterns.

---

## How it works

```
                       ┌──────────────────────────────────────────┐
                       │              your .onnx model            │
                       └──────────────────────┬───────────────────┘
                                              │
   Phase 1  canonicalize ─────────────────────▼─────────────────── delegates to
                                                                   optimize_onnx_for_mnn.py
                                                                   (PReLU normalize, fold
                                                                   unary on initializers,
                                                                   negative-axes, dead init)
                                              │
   Phase 2  MNNConvert (stock) ───────────────▼──────────────────  produces original.mnn
                                              │
   Phase 3  parse FlatBuffer ─────────────────▼──────────────────  flatc --python emits
                                                                   bindings into a cache
                                              │
   Phase 4  frequent subgraph mining ─────────▼──────────────────  linear elementwise chains
                                                                   fingerprinted by op-kind
                                                                   sequence, scored by
                                                                   eliminated-VRAM-bytes
                                              │
   Phase 5  synthesize OpenCL kernel ─────────▼──────────────────  walks each pattern, emits
                                                                   one float4 statement per
                                                                   primitive into a single
                                                                   __kernel
                                              │
   Phase 6  patch .mnn  ──────────────────────▼──────────────────  replace each occurrence
                                                                   with one OpType_Extra op
                                                                   carrying kernel bytes
                                              │
   Phase 7  verify  ──────────────────────────▼──────────────────  ORT(canonical.onnx)
                                                                   vs MNN-OpenCL(fused.mnn)
                                                                   atol=1e-3 by default
                                              │
                       ┌──────────────────────▼───────────────────┐
                       │  fused.mnn  +  *.mnnforge.report.json    │
                       └──────────────────────────────────────────┘
```

---

## Install

Requires Python ≥ 3.9, an existing MNN source tree, and a C++ toolchain (CMake + make/ninja).

```bash
git clone https://github.com/katolikov/mnnforge
cd mnnforge
pip install -r requirements.txt
pip install MNN          # pymnn — needed for the verify phase only
```

That's it. The package is pure Python; `flatc` is built once on demand from the MNN tree's
vendored copy at `<mnn_root>/3rd_party/flatbuffers/`.

---

## Quick start

```bash
# Most basic invocation
python -m mnnforge /path/to/MNN ./resnet50.onnx -v

# Outputs (next to your model by default):
#   resnet50.canon.onnx              ← canonicalized ONNX
#   resnet50.original.mnn            ← stock MNNConvert result (baseline)
#   resnet50.fused.mnn               ← with custom kernels injected
#   resnet50.mnnforge.report.json    ← per-output max abs/rel error
```

Common variations:

```bash
# Tune fusion aggressiveness
python -m mnnforge /path/to/MNN model.onnx --top-n 8 --max-pattern-size 8

# Skip end-to-end verification (faster; useful in CI loops)
python -m mnnforge /path/to/MNN model.onnx --skip-verify

# Convert + verify only (for measuring stock MNNConvert correctness on this model)
python -m mnnforge /path/to/MNN model.onnx --skip-fuse

# Tighten / loosen numerical tolerance
python -m mnnforge /path/to/MNN model.onnx --atol 1e-4 --rtol 1e-4
```

See **[HOWTO.md](./HOWTO.md)** for a full step-by-step walkthrough.

---

## What gets fused

Currently `mnnforge` accepts patterns matching **all** of:

* every primitive in the chain is in our snippet library
  * `BinaryOp`: ADD, SUB, MUL, DIV
  * `UnaryOp`: ABS, NEG, SQUARE, SQRT, RSQRT, EXP, LOG, RECIPROCAL, TANH, SIGMOID, GELU
  * named: `ReLU`, `ReLU6`, `Sigmoid`, `TanH`
* every intermediate tensor has exactly one consumer **and** is not a graph output
* op output shape is preserved across the chain (no broadcasting, no reductions)
* the chain is at least 2 ops long

That's small on purpose: the v1 kernel template inherits exactly the calling convention of
MNN's runtime `FuseExecution` (`source/backend/opencl/execution/image/FuseExecution.cpp`), so
fused kernels run today on every GPU backend without any MNN code changes. The constrained
shape is what makes that possible.

Roadmap items (not in v1): broadcast, `MatMul`/`Conv` fan-in, channel-wise reductions
(LayerNorm-shaped patterns), and Vulkan / Metal kernel mirror generation.

---

## Numerical guarantees

Every run produces a `*.mnnforge.report.json` containing:

```jsonc
{
  "results": [
    { "label": "baseline_ort_vs_mnn_original",
      "ok": true,
      "per_output": [{"output": "y", "shape": [1, 1000],
                      "max_abs_err": 4.7e-7, "max_rel_err": 9.3e-7,
                      "pass": true}] },
    { "label": "ort_vs_mnn_fused",
      "ok": true,
      "per_output": [{"output": "y", "shape": [1, 1000],
                      "max_abs_err": 6.1e-5, "max_rel_err": 1.1e-4,
                      "pass": true}] }
  ]
}
```

Verification compares ONNX Runtime (CPU provider, `float32`) against pymnn on the OpenCL
backend. The tool returns exit code `0` only if both the **baseline** (stock MNNConvert) and
the **fused** comparison pass within `--atol`/`--rtol`. If pymnn or OpenCL aren't available
on your machine, the tool reports inconclusive (exit `2`) and never silently passes.

---

## Architecture (one paragraph)

`mnnforge` works on the FlatBuffer level, not on MNN's C++ API. After stock `MNNConvert`
runs, the `.mnn` file is parsed into a mutable Python representation (via `flatc --python
--gen-object-api` cached in `<mnn_root>/build_mnnforge/_mnn_py_fbs/`). The mining phase walks
the op list, builds a tensor→consumer index, and enumerates maximal linear chains of
supported primitives. Each unique op-kind sequence becomes a `Pattern`; occurrences claim ops
by score so two patterns can never overlap. For each accepted pattern, the kernel synthesizer
emits an OpenCL `__kernel` matching `FuseExecution`'s expected argument layout (input
images first, then output images, then three `int W0/W1/W2` global-work-size args). The
surgery phase rewrites the op list, replacing each chain with one `OpType_Extra` op whose
`Extra.info` byte field carries the kernel source; the FlatBuffer is reserialized. At session
creation time, MNN's existing `FuseExecution` reads `Extra.info`, calls
`buildKernelFromSource`, sets the args, and runs.

---

## Testing

```bash
python -m pytest tests/ -v
# 46 passed in 0.27s
```

Test surface covers:

* imports + argparse construction
* primitive-snippet integrity (arity, helper consistency)
* kernel synthesis: commutative and non-commutative chains, signature matches
  `FuseExecution`, helper inclusion, error paths
* FSM: graph-output preservation, multi-consumer chain breakage, occurrence
  grouping, non-commutative `chain_pos` tracking, `max_pattern_size` capping
* surgery: overlap resolution by score, `Extra.info` bytes encoding, end-to-end
  span replacement, arity mismatches
* verify: tolerance comparisons, shape-recovery via reshape, missing-ORT/MNN
  inconclusive paths, JSON report write

---

## Project layout

```
mnnforge/
├── README.md                  ← this file
├── HOWTO.md                   ← step-by-step usage guide
├── LICENSE
├── requirements.txt
├── mnnforge/                  ← Python package
│   ├── __main__.py            ← `python -m mnnforge ...`
│   ├── cli.py                 ← argparse driver
│   ├── log.py
│   ├── preflight.py           ← Phase 0
│   ├── canonicalize.py        ← Phase 1 (delegates to optimize_onnx_for_mnn.py)
│   ├── convert.py             ← Phase 2 (drives stock MNNConvert)
│   ├── mnn_fbs.py             ← Phase 3 (flatc --python on demand)
│   ├── fsm.py                 ← Phase 4 (frequent subgraph mining)
│   ├── kernel_synth.py        ← Phase 5 (OpenCL kernel generator)
│   ├── primitives.py          ← snippet library
│   ├── surgery.py             ← Phase 6 (.mnn op-span rewrite)
│   └── verify.py              ← Phase 7 (ORT vs pymnn comparison)
└── tests/                     ← 46 unit tests
```

---

## License

Apache 2.0. See [LICENSE](./LICENSE).

`mnnforge` is an independent tool that works alongside MNN; it does not modify MNN source.
The MNN project is licensed by Alibaba under its own Apache 2.0 license.
