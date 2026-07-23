# Compiled CUDA softmax-family blueprint

Date: 2026-07-22

Tracking issue: #840

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Current verdict

The implementation inventory is complete for the exact SM86 shapes in this
pull request: **10 coverage cells** map the entire in-scope softmax-family NVRTC
inventory to hand-emitted direct-PTX specializations — dense softmax and
softmax-over-rows, log-softmax, log-sum-exp (axis reduction) and its backward,
softmax backward, masked-fill and its backward, sparsemax, and Taylor softmax.

The family divides into two dispatch shapes. The normalizing operators (softmax,
softmax-rows, log-softmax, log-sum-exp, softmax-backward, sparsemax) are
**one-block-per-row reductions**: a block owns one row, its lanes stride the row
for a coalesced partial, and a shared-memory tree-reduce with `bar.sync`
produces the row maximum and row sum (and, for sparsemax, the bisection
threshold). The pointwise operators (masked-fill, masked-fill-backward, Taylor
softmax, log-sum-exp-backward broadcast) are **thread-per-element**: one thread
owns one output cell with no cross-thread reduction.

**No softmax cell is promoted by this pull request.** Every specialization fails
closed: it requires exact-SM86 architecture (`HasValidatedSoftmax`), the opt-in
softmax experiment override, and exact contiguous physical extents, and
`IsPromoted*` is hard-coded `false`. Correctness is checked against fp64 CPU
oracles under GPU-gated driver tests; PTX-structure and manifest-completeness
gates pass on no-GPU CI. The production admission table remains fail-closed and
production behavior is unchanged.

This branch emits and validates **raw PTX**. The compiled-cubin pipeline
(stages 2–9 below) — driver-linked cubin preservation, SASS audit, embedded
`Artifacts/sm86/*.cubin` resources, and Nsight profiling — requires the pinned
CUDA toolkit and an SM86 device, so it runs in the maintainer's environment
alongside the promotion benchmarks. This document freezes the ABIs, shapes, and
optimization contract so that pipeline can be executed deterministically.

## Ten-stage production binary pipeline

| # | Required stage | Implementation and release gate |
|---:|---|---|
| 1 | Generate PTX | Row length, dtype, axis, epsilon, mask/fill value, sparsemax bisection depth, architecture, and physical layout are frozen in each `DirectPtxKernelBlueprint`; hot ABIs contain pointers only. Exact float constants (epsilons, `log2(e)`, Taylor `1/2`) are emitted via `BitConverter` bit patterns. **Done.** |
| 2 | Compile explicitly | `cuLinkCreate` + `cuLinkAddData(CU_JIT_INPUT_PTX)` + `cuLinkComplete` produces the executable before module load. **Pending maintainer SM86 (no toolkit on the CI/dev host).** |
| 3 | Preserve cubin | Returned ELF bytes SHA-256 hashed and exported under the source key; invalid/non-ELF output fails closed. **Pending SM86.** |
| 4 | Disassemble SASS | Pinned `nvdisasm` disassembles the exact preserved cubin and records entry point, registers, instructions, and global/shared/local traffic. **Pending SM86.** |
| 5 | Fail unsafe machine code | CI rejects missing/extra/stale/hash-mismatched artifacts and any final-SASS `LDL`/`STL`. Runtime `DirectPtxResourceBudget.Validate` already rejects nonzero local bytes, excess registers/shared memory, and insufficient occupancy at load time; the repo PTX-discipline guard rejects `.local` emission statically. **Runtime + static gates pass; SASS gate pending SM86.** |
| 6 | Profile exact cubin | A `--direct-ptx-profile-softmax` benchmark target executes every embedded cubin and asserts `EmbeddedCubin` before launch. **Pending SM86 (benchmarks are the maintainer's per the issue gate).** |
| 7 | Embed in NuGet | `Artifacts/sm86/*.cubin` + `softmax-cubins.tsv` embedded resources in `AiDotNet.Tensors`. **Pending SM86 artifact generation.** |
| 8 | Load cubin in production | Resolution order embedded cubin → verified disk cubin → driver-linked cubin. **Pending shared cubin infrastructure merge.** |
| 9 | Restrict PTX JIT | Direct PTX JIT is available only behind the explicit experiment fallback. The softmax path is entirely experiment-gated and unpromoted, so no production JIT occurs. **Pass (fail-closed).** |
| 10 | Cache complete identity | Each `DirectPtxKernelBlueprint` carries a versioned PTX SHA-256, GPU/SM/driver fingerprint, block geometry, resource budget, and occupancy; the `DirectPtxKernelCache` is keyed on the row-length/axis/variant tuple. **Done for the source-side identity; disk sidecars pending SM86.** |

The no-GPU CI verifier already regenerates every current PTX string and blueprint
ID and validates every manifest row and coverage decision. The cubin-hash and
artifact-identity checks activate once stage 3 artifacts exist.

## Ten optimization readiness gates

| # | Production requirement | Current implementation and verdict |
|---:|---|---|
| 1 | Exact contiguous layout | Admission requires contiguous physical views whose logical/physical extents equal the baked row length (`DirectPtxExtentMode.Exact`); axis, epsilon, mask/fill, and bisection depth are removed before the hot launch. Unsupported views fall back to the established NVRTC path. **Pass.** |
| 2 | Coalesced vector memory access | Row reductions stride the row by `blockDim` for coalesced partials and write each normalized value once; pointwise kernels use adjacent per-lane `.f32` loads. **Pass (mechanically); vectorized `v2/v4` widening is a follow-up.** |
| 3 | Shared memory only for reuse | Only the block-per-row reductions stage a 256/128-lane tree-reduce scratch (max, sum, and the sparsemax bisection accumulator). Pointwise masked-fill and Taylor kernels use **zero** shared memory. **Pass.** |
| 4 | Register-resident math | Loaded row values, the running max, exp-sum, backward dot-product, and sparsemax threshold remain in registers until the final store. Every kernel reports **zero local bytes** (asserted in the driver tests) and passes the PTX-discipline `.local` guard. **Pass.** |
| 5 | Combined/fused kernels | Softmax fuses max-reduction, stable exponentiation, sum-reduction, and normalization in one block pass; softmax-backward fuses the `Σ(dY·S)` reduction with the `S·(dY − Σ)` epilogue. **Mechanically pass; performance HOLD.** |
| 6 | Bounded global reductions | Row reductions use a fixed 256/128-lane shared tree-reduce with no output-sized scratch and no atomics; sparsemax runs a fixed 30-step τ-bisection entirely in registers/shared. **Correctness pass; performance HOLD.** |
| 7 | Asynchronous stream ordering | Launches use the backend stream with no host synchronization; the dispatch layer rejects launches during CUDA-graph capture unless prewarmed. `cp.async` is inapplicable to these single-use row loads. **Pass.** |
| 8 | CUDA Graph/lifetime safety | Modules are pinned for capture lifetime and the dispatch shell rejects compilation/cache-miss during capture. **Pass.** |
| 9 | Ahead-of-load binary control | PTX is emitted and blueprint-audited today; linking to cubin, SASS disasm, embedding, and content-addressed caching are staged for the SM86 pipeline. Raw PTX load is experiment-only and unpromoted. **Source-side pass; binary stages pending SM86.** |
| 10 | Promotion evidence | Three independent corrected PyTorch comparisons plus correctness/determinism, zero hot allocation, resource, and tail gates must all pass. **HOLD:** no softmax cell is promoted; the fp64 CPU-oracle correctness harness is wired, but the uncontended three-run performance comparison is the maintainer's SM86 task. |

Tensor Core MMA is not a requirement for this family: softmax normalization is a
row reduction, not a matrix multiply, so cargo-cult MMA instructions that add
conversions or shared-memory traffic are intentionally absent. The softmax that
feeds attention is fused inside the attention kernels tracked by their own
blueprint; this family covers the standalone normalizing operators.

## Kernel and memory contract

All admitted tensors are exact, contiguous, aligned physical views. Row length,
axis, epsilon, mask/fill value, and sparsemax bisection depth are baked into the
PTX and removed before the hot launch; the launch passes buffer pointers only.
Unsupported layouts, architectures, or a disabled experiment gate return to the
established NVRTC path with an exact `DirectPtxLastError` reason. There are no
output-sized temporary device allocations; the only device-side scratch is the
fixed shared reduction buffer inside the block-per-row kernels.

The normalizing kernels assign one block to one row. Lanes stride the row for a
coalesced partial, tree-reduce the row maximum through shared memory with
`bar.sync`, recompute the stable exponentials, tree-reduce the row sum, and
write each normalized (or log-normalized) value once. Softmax-backward reduces
`Σ(dY·S)` the same way and applies `S·(dY − Σ)` in one epilogue pass. Sparsemax
sorts nothing on device: it runs a fixed 30-step threshold bisection over the
row in registers/shared, matching the reference tolerance. The pointwise
kernels (masked-fill, masked-fill-backward, Taylor softmax `1 + x + x²/2`, and
the log-sum-exp-backward broadcast) own one output cell per thread with no
cross-thread communication.

Exponentiation uses the hardware approximate `ex2.approx.f32` on `x·log2(e)`
(the same path the attention softmax uses); `rcp.approx`/`div.rn` produce the
normalization reciprocal. Every constant is emitted from its exact 32-bit
pattern via `BitConverter` so the net471 and net10.0 builds are bit-identical.
