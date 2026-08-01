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
softmax-rows, log-softmax, log-sum-exp, softmax-backward, sparsemax, and Taylor
softmax) are **one-block-per-row reductions**: a block owns one row, its lanes
stride the row for a coalesced partial, and a two-level warp-shuffle reduction
uses only eight shared warp-leader slots. The pointwise operators (masked-fill,
masked-fill-backward, and log-sum-exp-backward broadcast) require no cross-thread
reduction. The masking kernels coarsen each thread across two aligned float4
transactions; log-sum-exp-backward owns one output cell per thread.

**No softmax cell is promoted by this pull request.** Every specialization fails
closed: it requires exact-SM86 architecture (`HasValidatedSoftmax`), the opt-in
softmax experiment override, and exact contiguous physical extents, and
`IsPromoted*` is hard-coded `false`. Correctness is checked against fp64 CPU
oracles under GPU-gated driver tests; PTX-structure and manifest-completeness
gates pass on no-GPU CI. The production admission table remains fail-closed and
production behavior is unchanged. The checked-in head-to-head now measures all
ten cells against the shipped CUDA incumbent on the same backend stream. On the
validated RTX 3080/SM86 shape `[2048,1024]`, eight cells are clear wins and the
two standalone masking passes are exact bandwidth-floor ties. No tie is reported
as a win and no benchmark result opens the production gate automatically.

## Cross-backend parity inventory

The direct-PTX specializations are CUDA-specific, but the public operations do
not silently become CUDA-only. Their peer-backend status is recorded in
`DirectPtxSoftmaxCoverageManifest` and is fail-closed as follows:

- **HIP** implements softmax, softmax-rows, softmax-backward,
  log-sum-exp-backward, masked-fill, and masked-fill-backward natively. The
  missing log-softmax, log-sum-exp-axis, sparsemax, and Taylor-softmax variants
  are tracked by #914.
- **Metal** implements the same six common routes natively; its four missing
  variants are tracked by #915.
- **OpenCL** implements the same six common routes natively; its four missing
  variants are tracked by #916.
- **Vulkan** has native implementations for all ten public routes, so it has no
  parity exception for this family.
- **WebGPU** implements the six common routes natively; its four missing
  variants are tracked by #917.

The four follow-up issues are distinct, open backend-parity requirements. None
of them is treated as implemented coverage, and no CUDA promotion in this pull
request can waive them.

## SM86 head-to-head evidence (2026-07-31)

Command: `--direct-ptx-softmax`. Both sides run on the same backend stream and
are timed with CUDA events. Each side is independently calibrated to a roughly
10 ms batch. One sample is the median of five internally-consistent A/B/B/A
brackets; interrupted brackets are rejected, and three consecutive samples plus
the paired ratio must converge within 5%. Correctness is checked after timing and
the harness asserts that the experimental direct-PTX dispatch counter advanced.

| Operator | Shipped CUDA | Direct PTX | Paired ratio | Max error | Verdict |
|---|---:|---:|---:|---:|---|
| Softmax | 25.9 us | 24.7 us | 1.05x | 0 | direct wins |
| SoftmaxRows | 31.8 us | 24.5 us | 1.27x | 2.33e-10 | direct wins |
| SoftmaxBackward | 1742.7 us | 38.2 us | 45.69x | 5.82e-11 | direct wins |
| LogSoftmax | 717.2 us | 28.3 us | 25.62x | 2.38e-6 | direct wins |
| LogSumExpAxis | 306.1 us | 20.3 us | 14.94x | 2.38e-6 | direct wins |
| LogSumExpBackward | 26.9 us | 25.5 us | 1.09x | 0 | direct wins |
| MaskedFill | 35.5 us | 35.0 us | 1.00x | 0 | tie within noise |
| MaskedFillBackward | 35.7 us | 35.0 us | 1.01x | 0 | tie within noise |
| Sparsemax | 111405.6 us | 157.0 us | 709.90x | 5.96e-8 | direct wins |
| TaylorSoftmax | 835.1 us | 24.9 us | 33.53x | 0 | direct wins |

The two ties have the same irreducible standalone traffic as their incumbents:
two FP32 input reads and one FP32 output write. The vectorized PTX forms already
use read-only loads, L2/write-through stores, zero shared/local memory, and exact
selection semantics. A material end-to-end win therefore requires fusing the
mask predicate into its softmax consumer so the intermediate write and reread
disappear; changing the standalone contract cannot remove those bytes.

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
| 2 | Coalesced vector memory access | Row reductions stride the row by `blockDim` for coalesced partials and write each normalized value once. Masked-fill and its backward use two aligned float4 transactions per thread with read-only loads and L2-only/write-through output policy. **Pass.** |
| 3 | Shared memory only for reuse | Block-per-row reductions use eight float warp-leader slots for the hierarchical reduction; the old 256-element staging tree is gone. Pointwise masked-fill, masked-fill-backward, and log-sum-exp-backward use **zero** shared memory. **Pass.** |
| 4 | Register-resident math | Loaded row values, the running max, exp-sum, backward dot-product, and sparsemax threshold remain in registers until the final store. Every kernel reports **zero local bytes** (asserted in the driver tests) and passes the PTX-discipline `.local` guard. **Pass.** |
| 5 | Combined/fused kernels | Softmax fuses max-reduction, stable exponentiation, sum-reduction, and normalization in one block pass; softmax-backward fuses the `Σ(dY·S)` reduction with the `S·(dY − Σ)` epilogue. **Pass for the eight measured winning cells; standalone mask/softmax fusion is the remaining performance follow-up.** |
| 6 | Bounded global reductions | Row reductions use warp shuffles plus eight shared leaders, two barriers per logical reduction, no output-sized scratch, and no atomics; sparsemax runs a fixed 30-step τ-bisection with the same reducer. **Correctness and SM86 performance pass.** |
| 7 | Asynchronous stream ordering | Launches use the backend stream with no host synchronization; the dispatch layer rejects launches during CUDA-graph capture unless prewarmed. `cp.async` is inapplicable to these single-use row loads. **Pass.** |
| 8 | CUDA Graph/lifetime safety | Modules are pinned for capture lifetime and the dispatch shell rejects compilation/cache-miss during capture. **Pass.** |
| 9 | Ahead-of-load binary control | PTX is emitted and blueprint-audited today; linking to cubin, SASS disasm, embedding, and content-addressed caching are staged for the SM86 pipeline. Raw PTX load is experiment-only and unpromoted. **Source-side pass; binary stages pending SM86.** |
| 10 | Promotion evidence | Three independent corrected competitor comparisons plus correctness/determinism, zero hot allocation, resource, and tail gates must all pass. **HOLD:** the shipped-CUDA head-to-head is now wired and reports eight wins/two ties, but the two masking cells still need fusion evidence and no cell is promoted by this PR. |

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
coalesced partial, reduce within each warp using `shfl.sync.down`, exchange only
the eight warp-leader values through shared memory, recompute or stage the stable
exponentials, reduce the row sum, and write each normalized (or log-normalized)
value once. Softmax-backward reduces
`Σ(dY·S)` the same way and applies `S·(dY − Σ)` in one epilogue pass. Sparsemax
sorts nothing on device: it runs a fixed 30-step threshold bisection over the
row in registers/shared, matching the reference tolerance. The pointwise
kernels (masked-fill, masked-fill-backward, and the log-sum-exp-backward
broadcast) have no cross-thread communication; the masking pair own eight
elements per thread through two aligned float4 transactions.

Exponentiation uses the hardware approximate `ex2.approx.f32` on `x·log2(e)`
(the same path the attention softmax uses); `rcp.approx`/`div.rn` produce the
normalization reciprocal. Every constant is emitted from its exact 32-bit
pattern via `BitConverter` so the net471 and net10.0 builds are bit-identical.
