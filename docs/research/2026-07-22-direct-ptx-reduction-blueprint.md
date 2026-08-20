# Direct-PTX reductions, scans, arg-reductions, sort, TopK, and histogram blueprint

Date: 2026-07-22

Tracking issue: #843

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Verdict

This increment establishes the issue-#843 assembly line and implements one
exact contiguous Ampere FP32 candidate: a warp-per-row **row-sum** reduction of
a `[rows, columns]` input to a `[rows]` output. It is **not promoted**. Each
warp keeps every lane value in registers from a single vectorized global load
through a butterfly-shuffle add reduction, and lane zero commits one FP32
element — zero shared memory, zero local bytes, zero global intermediates, and
no temporary device allocation. Correctness is verified on the local RTX 3080
(SM86) against a double-precision oracle within a `1e-4 * (|sum| + 1)` relative
tolerance across all admitted shapes.

Production fails closed with `reduction-performance-gate-not-met`
(`IsPromotedShape => false`); an explicit experiment override is required to
measure it. This draft does not close #843. Its 28-cell manifest assigns the
remaining axis-reduction, scan, arg-reduction, sort, TopK, histogram, backward,
segmented, and public-routing families.

## Assembly-line contract

Identical to the parent blueprint. Every scoped public/backend reduction API is
assigned to one manifest cell before a specialization is written; operation
semantics, phase, dtype, architecture, exact shape bucket, physical extent, row
split, alignment, and alias policy are frozen and validated once at admission;
the PTX launch ABI emits only tensor pointers; only resident NVIDIA paths are
compared (current AiDotNet CUDA `sum_axis`, direct PTX, and the strongest
applicable PyTorch/Triton kernel), and CPU MKL/OpenBLAS are ineligible; a cell
is promoted only after three clean runs clear median, P95, allocation,
temporary-VRAM, accuracy, and executed-spill gates.

## Formal contiguous ABI

| Tensor | Exact extent | Access | Alignment |
|---|---:|---|---:|
| input  | `[rows,columns]` FP32 row-major | read  | 16 B |
| output | `[rows]` FP32 vector             | write | 16 B |

The launch has two 64-bit pointer parameters. Each warp owns a row; each lane
loads one FP32x4 (columns=128) or FP32x2 (columns=64) vector, sums its lane
values, butterfly-shuffles the warp accumulator, and lane zero performs one
`st.global.f32`. There are no global intermediates, temporary allocations,
shared memory, dimension parameters, division/remainder instructions, or stride
checks. Admitted buckets: `(rows,columns) = (256,128), (2048,64), (2048,128),
(8192,128)`; Ampere only. All other architectures and shapes fall back with an
exact reason.

## Fair-comparison notes (apples-to-apples)

The resident PyTorch competitor script measures four variants — eager,
CUDA-graph, `max-autotune-no-cudagraphs` compile, and compile+graph — each
scored against a **double-precision** `sum(dim=-1)` oracle, so the direct
graph-captured kernel is compared against PyTorch's graph-captured kernel
(symmetric launch-overhead amortization). The C# release-gate harness feeds the
**strongest** competitor (minimum median across all non-direct methods) into
`DirectPtxReleaseGate`, so a win must beat the best available NVIDIA path, not
merely eager.

**Disclosed asymmetry:** the kernel accumulates in FP32 with warp-shuffle tree
order (`add.rn.f32`), whereas the oracle accumulates in FP64. Any promotion
claim must state that parity is at the FP32 tree-order tolerance above, not
bit-exact against FP64.

**Single-capture diagnostic (RTX 3080, SM86, one run — not promotion evidence).**
Median device times, strongest competitor per shape:

| shape | Direct PTX | AiDotNet `sum_axis` | PyTorch `sum(-1)` | speedup vs strongest | rel err |
|---|---:|---:|---:|---:|---:|
| (256,128)  | 18.7us | 37.0us | 27.7us | 1.48x | 2.5e-6 |
| (2048,64)  | 18.4us | 26.7us | 26.9us | 1.45x | 2.0e-6 |
| (2048,128) | 18.7us | 34.8us | 32.7us | 1.75x | 3.7e-6 |
| (8192,128) | 23.8us | 34.7us | 36.8us | 1.46x (HOLD: P95 52.4>39.7us) | 3.6e-6 |

At these buckets the exact-shape warp-row kernel beats both the generic AiDotNet
`sum_axis` kernel and PyTorch's tuned reduction on the median because both
competitors carry generic-launch/dispatch overhead that the pointer-only ABI
removes. This clears 1.10x on the median at every bucket, but it is **one noisy
capture**: (8192,128) fails the P95 gate, and production requires three clean
runs before promotion. The kernel therefore stays disabled and fails closed.

**Gate-calibration finding.** `DirectPtxReleaseGate.MaximumAbsoluteError` is
`5e-5`, calibrated for `[0,1]` outputs (softmax). A reduction output is
`O(columns * magnitude)`, so its absolute error legitimately exceeds `5e-5`
while relative error is ~1e-6. The reduction family needs a **relative-error**
gate (or a normalized tolerance) before promotion; scoring absolute error here
would reject a numerically-correct kernel. This experiment reports relative
error against the FP64 oracle accordingly.

**Protocol hardening (family-wide):** lock GPU clocks (`nvidia-smi -lgc`) and
interleave the C#/Python capture halves to prevent the thermal-drift artifact
the pointwise blueprint already had to exclude a run for.

## Correctness and runtime proof

Focused tests enforce pointer-only PTX with exact vector-load/store counts, no
`.shared`/`.local`/stride/`.param .u32`/`bar.sync`, a closed unpromoted shape
domain, manifest completeness with exactly one experimental cell and no promoted
cell, and (on a validated Ampere device) driver correctness with zero local and
static-shared bytes and at least three active blocks per SM.

## Reproduction

```powershell
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-reduction 3

python tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_reduction_competitors.py --runs 3
```

## Next reduction increments

1. Prioritize fused reductions that delete a materialized intermediate
   (normalize-L2, variance/std, log-sum-exp) over bare sum/max.
2. Add arg-reductions (argmax/argmin) with paired FP32 value + INT32 index
   register state, then bounded-k TopK and bitonic sort for power-of-two lengths.
3. Add warp/block inclusive-scan (cumsum/cummax) and privatized-histogram cells.
4. Add FP16/BF16 vector families and independently measured Ada, Hopper, and
   Blackwell modules; never infer promotion from Ampere.
