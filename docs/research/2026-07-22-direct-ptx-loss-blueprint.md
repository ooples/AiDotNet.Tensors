# Direct-PTX loss, objective, logits-to-loss, and loss-backward blueprint

Date: 2026-07-22

Tracking issue: #847

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Verdict

This increment establishes the issue-#847 assembly line and implements one exact
contiguous Ampere FP32 candidate: a warp-per-row **per-sample mean-squared-error
loss**, `loss[i] = mean_j (predictions[i,j] - targets[i,j])^2`. It is **not
promoted**. Each warp fuses the difference, square (via `fma.rn.f32`), butterfly
warp reduction, and the mean scale entirely in registers, so the squared-error
tensor is **never materialized** — zero shared/local, zero global intermediates,
no temporary allocation. Correctness is verified on the local RTX 3080 (SM86)
against a double-precision oracle within a `1e-4 * (|loss| + 1)` relative
tolerance across all admitted shapes.

Production fails closed with `mse-loss-performance-gate-not-met`
(`IsPromotedShape => false`). This draft does not close #847. Its 20-cell manifest
assigns the remaining MAE, Huber, BCE, BCE-with-logits, cross-entropy, NLL,
KL-divergence, fused-linear-cross-entropy, cosine, triplet, and loss-backward
families.

## Formal contiguous ABI

| Tensor | Extent | Access | Alignment |
|---|---:|---|---:|
| predictions | `[rows,columns]` FP32 row-major | read  | 16 B |
| targets     | `[rows,columns]` FP32 row-major | read  | 16 B |
| loss        | `[rows]` FP32 vector             | write | 16 B |

The launch has three 64-bit pointer parameters. Each warp owns a row, loads one
FP32x4/FP32x2 vector from predictions and one from targets, computes the
per-lane differences, accumulates their squares with `fma.rn.f32`, reduces across
the warp, multiplies by the baked `1/columns` immediate, and lane zero performs
one `st.global.f32`. Admitted buckets: `(rows,columns) = (256,128), (2048,64),
(2048,128), (8192,128)`; Ampere only; else exact-reason fallback.

**Reduction semantic:** this specialization computes the per-sample *mean* over
features. A backend whose `mse_loss` uses a different reduction (sum, or global
mean) must be split into its own manifest cell before promotion; the direct path
stays disabled by default so no result changes silently.

## Fair-comparison notes (apples-to-apples)

The resident PyTorch competitor script measures four variants — eager,
CUDA-graph, `max-autotune-no-cudagraphs` compile, and compile+graph — of
`((pred-target)**2).mean(dim=1)`, each scored against a double-precision oracle.
The C# release-gate harness feeds the **strongest** competitor (minimum median
across the AiDotNet `mse_loss` kernel and PyTorch) into `DirectPtxReleaseGate`.

The genuine advantage here is **fusion**: the naive path materializes an
`(pred-target)^2` tensor and then reduces it (two kernels, one extra global
round-trip), while the direct kernel never writes the squared error. That is the
durable win pattern for the loss family — fuse the elementwise objective into its
reduction.

**Protocol hardening (family-wide):** lock GPU clocks (`nvidia-smi -lgc`) and
interleave the C#/Python capture halves to prevent the thermal-drift artifact the
pointwise blueprint already had to exclude a run for.

## Correctness and runtime proof

Focused tests enforce pointer-only PTX with exact pred/target vector loads, four
differences, four fused-multiply-adds, a five-step butterfly reduction, the baked
`0f3C000000` (=1/128) scale, one scalar store, no `.shared`/`.local`/`bar.sync`/
stride/`.param .u32`, a closed unpromoted shape domain, manifest completeness
with exactly one experimental cell and no promoted cell, and (on a validated
Ampere device) mean-MSE parity within relative tolerance with zero local and
static-shared bytes and at least three active blocks per SM.

## Reproduction

```powershell
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-mse-loss 3

python tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_mse_loss_competitors.py --runs 3
```

## Next loss increments

1. Add MAE (abs instead of square) and Huber (baked delta) on the same warp-row
   shape, then BCE-with-logits with a stable `log-sigmoid` lane.
2. Add cross-entropy as fused log-softmax + NLL over classes (one row pass).
3. Extend the fused-linear-cross-entropy path so the logits are never
   materialized (GEMM epilogue into the loss reduction).
4. Add the paired backward kernels (`grad = 2/N (pred-target) gradOut`) and
   FP16/BF16 families; add independently measured Ada, Hopper, and Blackwell.
