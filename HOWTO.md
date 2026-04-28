# HOWTO — Using mnnforge end-to-end

A practical, step-by-step guide. If something here looks wrong, see the
[Troubleshooting](#troubleshooting) section at the bottom — most issues map to a single,
known failure mode.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Install](#2-install)
3. [First run](#3-first-run)
4. [Reading the output](#4-reading-the-output)
5. [Tuning fusion](#5-tuning-fusion)
6. [CI / scripted use](#6-ci--scripted-use)
7. [Verifying that fusion actually ran](#7-verifying-that-fusion-actually-ran)
8. [Performance check (timing the fused .mnn)](#8-performance-check-timing-the-fused-mnn)
9. [Iterating on a model](#9-iterating-on-a-model)
10. [Troubleshooting](#10-troubleshooting)
11. [FAQ](#11-faq)

---

## 1. Prerequisites

| Requirement | Why | How to get |
|---|---|---|
| Python ≥ 3.9 | runs `mnnforge` itself | system / pyenv |
| MNN source tree | `mnnforge` builds `MNNConvert` from it and uses its `flatc` | `git clone https://github.com/alibaba/MNN` |
| C++ toolchain (cmake + ninja or make) | builds `MNNConvert` and `flatc` once | Xcode CLT / build-essential |
| `pip install onnx onnxruntime flatbuffers numpy networkx` | core deps | `pip install -r requirements.txt` |
| `pip install MNN` (pymnn) | only required for Phase 7 verification | `pip install MNN` |
| OpenCL drivers | optional; without them, verify falls back to CPU | platform-specific |

> On macOS Apple Silicon, OpenCL is deprecated. `mnnforge` will gracefully fall back to the
> MNN CPU backend for verification; the fused `.mnn` itself is portable and will run on
> OpenCL on devices that support it.

---

## 2. Install

```bash
git clone https://github.com/katolikov/mnnforge
cd mnnforge

python3 -m venv .venv && source .venv/bin/activate    # optional but recommended
pip install -r requirements.txt
pip install MNN                                       # for verification
```

Verify the install:

```bash
python -m mnnforge --version
# → mnnforge 0.1.0
python -m pytest tests/ -q
# → 46 passed
```

---

## 3. First run

Pick any ONNX model. A small ImageNet classifier or a snippet of a transformer is ideal —
`mnnforge` is most useful on models with repeated motifs.

```bash
python -m mnnforge /path/to/MNN ./model.onnx -v
```

What happens, in order:

| Phase | What's printed | Output produced |
|------:|---|---|
| 0 | preflight checks | — |
| 1 | `[pass] prelu / fold_unary / negative_axes / dead_init` | `model.canon.onnx` |
| 2 | `building MNNConvert` (first run only) → `running MNNConvert` | `model.original.mnn` |
| 3 | `parsed ... ops, ... tensors` | (in-memory NetT) |
| 4 | `pattern <fp>: 12× len=3 score=24 [BinaryOp(2,pos=0) → ...]` | (table) |
| 5 | `synthesized kernel 'mnnforge_<fp>'` | (in-memory) |
| 6 | `fused N occurrence(s); ops 230 -> 195` | `model.fused.mnn` |
| 7 | `baseline ORT vs MNN(original): PASS` then `fused ORT vs MNN(fused): PASS` | `model.mnnforge.report.json` |

Exit code is `0` if both comparisons in Phase 7 pass, `2` otherwise.

---

## 4. Reading the output

```bash
cat model.mnnforge.report.json | python -m json.tool
```

Each entry has:

* `label` — `baseline_ort_vs_mnn_original` (sanity, stock conversion path) or
  `ort_vs_mnn_fused` (the actual test that fusion is correct)
* `ok` — `true` / `false` / `null` (null = run was skipped / unavailable)
* `per_output` — one row per output tensor with `max_abs_err`, `max_rel_err`, `shape`, `pass`

If `baseline_ort_vs_mnn_original.ok` is `false`, your model has a stock-MNNConvert issue —
not an `mnnforge` issue. Re-run with `--skip-fuse` to confirm and report it upstream.

---

## 5. Tuning fusion

```bash
# Default
python -m mnnforge /path/to/MNN model.onnx

# Be greedier:
python -m mnnforge /path/to/MNN model.onnx \
    --top-n 8 \
    --max-pattern-size 8

# Be conservative:
python -m mnnforge /path/to/MNN model.onnx \
    --top-n 1 \
    --max-pattern-size 3
```

| Flag | Default | Effect |
|---|---:|---|
| `--top-n N` | 4 | Apply only the N highest-scoring patterns. Each pattern can have many occurrences. |
| `--max-pattern-size K` | 6 | Don't extend a chain beyond K ops. Larger values → fewer but more aggressive kernels. |
| `--atol` | 1e-3 | Absolute tolerance in Phase 7 comparisons (fp32). |
| `--rtol` | 1e-3 | Relative tolerance. |

---

## 6. CI / scripted use

```bash
python -m mnnforge /path/to/MNN model.onnx \
    --no-ort-verify-canon \
    --workdir build/mnnforge_artifacts \
    --verbose
echo "exit=$?"
```

* `--no-ort-verify-canon` — skip the per-pass ORT verification *inside* canonicalize
  (Phase 1). The end-to-end Phase 7 still runs and protects you.
* `--workdir DIR` — keep all artifacts in one folder (handy for upload).
* `--skip-build` is **not** a flag (we never have a flag named that); the `MNNConvert`
  build is cached in `build_mnnforge/` inside your MNN tree.

Exit codes:

| Code | Meaning |
|---:|---|
| 0 | All Phase 7 comparisons passed within tolerance. |
| 2 | One or more comparisons failed, or all comparisons inconclusive. |
| nonzero, not 2 | Hard error — see stderr (preflight refusal, MNNConvert build crash, parse error). |

---

## 7. Verifying that fusion actually ran

The most reliable signal is the op count change at the end of Phase 6:

```
[ ✓ ] fused 18 occurrence(s); ops 230 -> 195
```

You can also inspect the `.mnn` flatbuffer:

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, "/path/to/MNN/build_mnnforge/_mnn_py_fbs")
from MNN.Net import Net
from MNN.OpType import OpType
import os

raw = open("model.fused.mnn", "rb").read()
net = Net.GetRootAs(bytearray(raw), 0)
n_extra = sum(1 for i in range(net.OplistsLength())
              if net.Oplists(i).Type() == OpType.Extra)
print(f"OpType_Extra count in fused.mnn: {n_extra}")
PY
```

If `n_extra == 0`, no patterns met the score threshold — try `--max-pattern-size 8 --top-n 8`.

---

## 8. Performance check (timing the fused .mnn)

`mnnforge` does not time models itself — that's MNN's `MNNV2Basic.out` job. After a successful
fusion run:

```bash
# Inside the MNN tree:
./build_mnnforge/MNNV2Basic.out model.original.mnn 100 0 0 3 4   # backend=3 (OpenCL)
./build_mnnforge/MNNV2Basic.out model.fused.mnn    100 0 0 3 4
```

Compare the average forward times. Speedups vary widely:

* CNNs with many small `BatchNorm + ReLU` chains: 5–15% wall-clock improvement
* Transformers with repeated `Mul / Add / Sigmoid` patterns: 10–20%
* Models with no repeated elementwise motifs: ~0% (and that's fine — `mnnforge` exits with
  "no patterns discovered")

---

## 9. Iterating on a model

If a fused model fails Phase 7:

1. Check the report JSON. If only one specific output diverges, the offending pattern is
   feeding into that output. Re-run with `--top-n 1` to narrow down.
2. If `baseline_ort_vs_mnn_original` already fails, the divergence is **not** from `mnnforge`
   — it's a stock MNN converter issue. File it against MNN.
3. If `--max-pattern-size 2` passes but `--max-pattern-size 6` fails, a long chain has
   accumulated rounding error past `--atol`. Loosen tolerance with `--atol 5e-3 --rtol 5e-3`
   or shrink the chain limit.
4. If `pymnn` isn't installed or OpenCL isn't available, you'll see `ok: null` results and an
   inconclusive verdict. Install `pip install MNN` and retry.

---

## 10. Troubleshooting

### `cmake not found in PATH`
Install CMake (`brew install cmake` / `apt install cmake`). The first run builds `MNNConvert`.

### `MNN load failed for ... on backend=3`
OpenCL ICD missing or your GPU doesn't expose OpenCL. `mnnforge` automatically retries on
CPU; the report will be labeled accordingly. The fused `.mnn` itself is still correct and
portable to any device with OpenCL.

### `MNNConvert returned 0 but model.original.mnn missing`
You probably ran out of disk space or ran with insufficient permissions in the workdir.

### `flatc build did not produce ...`
The vendored flatbuffers source under `<mnn_root>/3rd_party/flatbuffers/` is incomplete or
its `CMakeLists.txt` is out of sync. Run `<mnn_root>/schema/generate.sh` once manually to
prime it.

### `verification inconclusive`
Either `onnxruntime` or `MNN` (pymnn) couldn't be imported, so we have no reference to
compare against. `pip install onnxruntime MNN` and retry.

### `no fusable patterns discovered — fused = original`
Your model has no repeating elementwise motifs of length ≥ 2. That's normal for very small
models. Try `--max-pattern-size 3` to lower the bar, or accept that there's nothing for
`mnnforge` to do.

### `mnn_root missing required path: source/backend/opencl/...`
The path you passed isn't an MNN source tree, or you cloned a partial checkout. `mnnforge`
also refuses to run if it would need to touch `schema/private/` or `source/internal/`.

---

## 11. FAQ

**Q: Does `mnnforge` modify the MNN source tree?**
No. The only things it writes inside the tree are: a `build_mnnforge/` directory with the
`MNNConvert` binary (built once) and `_mnn_py_fbs/` with auto-generated Python flatbuffer
bindings (cached). Both are listed in `.gitignore`-equivalent paths and can be deleted at
any time.

**Q: Will the fused `.mnn` file load on stock MNN binaries that don't have `mnnforge`
installed?**
Yes. The fused model uses only `OpType_Extra` (= 512), which is a first-class op in the
public schema. Any MNN ≥ 3.0 with the OpenCL backend will load and run it.

**Q: What about Metal / Vulkan / CUDA?**
The runtime fuse path exists in those backends too (`MetalFuse.mm`, `VulkanFuse.cpp`,
`FuseExecutionV2.cu`), but each expects a backend-specific kernel. v1 of `mnnforge`
generates only OpenCL. Multi-backend kernel generation is the highest-priority follow-up.

**Q: Is the FlatBuffer mutation safe across MNN versions?**
`mnnforge` regenerates the Python bindings from `<mnn_root>/schema/default/*.fbs` on every
run (cached by mtime), so it always matches the schema of the MNN tree you point it at. Pin
your MNN version with a git submodule and you're set.

**Q: Why not just write a C++ pass inside MNN's own `tools/converter/source/optimizer/`?**
That was the original plan. We discovered MNN already ships a runtime kernel-compilation
path (`OpType_Extra` + `FuseExecution`) and decided that operating purely on the FlatBuffer
gives you: zero MNN code changes, zero schema collisions, zero rebuild churn when adding new
patterns, and a tool that works against any modern MNN release without forking. The trade-off
is that `mnnforge` only fuses things that MNN's runtime can already execute via `FuseExecution`'s
calling convention — which is precisely what we wanted v1 to scope to anyway.

**Q: Where do I report bugs?**
Open an issue on this repo with the `*.mnnforge.report.json` and the exact command line.
If it's an MNN converter issue (i.e. `baseline_ort_vs_mnn_original.ok == false`), file it
against `alibaba/MNN` instead.
