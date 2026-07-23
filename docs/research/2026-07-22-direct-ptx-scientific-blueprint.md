# Compiled CUDA specialized-scientific blueprint

Date: 2026-07-22

Tracking issue: #854

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Current verdict

The implementation inventory is complete for the exact SM86 shapes in this
pull request: **34 coverage cells** map the entire in-scope NVRTC kernel
inventory for the catch-all specialized-scientific family. Of these, **33 are
hand-emitted direct-PTX specializations** and **1 is an explicit, documented
NVRTC baseline** (`generate_spiral_indices`, a single-threaded `block0`/`thread0`
control-flow graph BFS with insertion-sort and mutable scratch buffers — direct
PTX offers no benefit for a one-thread kernel and it cannot meet the parallel
championship gate, so it is intentionally retained on the established path).

The 33 direct-PTX kernels span ten domains: complex arithmetic
(multiply/conjugate/magnitude/phase/mat-vec), octonion algebra
(add/multiply), Poincaré/Möbius hyperbolic geometry (add/distance/project/exp-map),
distance and similarity (RBF, pairwise L2/squared, cosine), quantum
(measurement, rotation, measurement-forward, normalize-probabilities),
spherical harmonics (forward/backward/softmax), capsule routing
(predictions/transform/weighted-sum/agreement), ANN (compute-distances,
PQ distance-tables, IVF-assign, PQ-ADC-scan), Instant-NGP hash-grid encoding
(forward/backward), and the uniform mesh Laplacian.

**No scientific cell is promoted by this pull request.** Every specialization
fails closed: it requires exact-SM86 architecture (`HasValidatedScientific`),
the opt-in `ScientificExperimentOverride`, and exact contiguous physical extents,
and `IsPromoted*` is hard-coded `false` for every kernel. Correctness is checked
against fp64 CPU oracles under GPU-gated driver tests, and PTX-structure and
manifest-completeness gates pass on no-GPU CI. The production admission table
therefore remains fail-closed and production behavior is unchanged.

This branch emits and validates **raw PTX**. The compiled-cubin pipeline
(stages 2–9 below) — driver-linked cubin preservation, SASS audit, embedded
`Artifacts/sm86/*.cubin` resources, and Nsight profiling — requires the pinned
CUDA toolkit and an SM86 device, so it runs in the maintainer's environment
alongside the promotion benchmarks. This document freezes the ABIs, shapes, and
optimization contract so that pipeline can be executed deterministically.

## Ten-stage production binary pipeline

| # | Required stage | Implementation and release gate |
|---:|---|---|
| 1 | Generate PTX | Shape, dtype, metric/degree/curvature/qubit-count, architecture, physical layout, alias policy, and operation are frozen in each `DirectPtxKernelBlueprint`; hot ABIs contain pointers only. Exact float constants (epsilons, atan/SH coefficients, log2e) are emitted via `BitConverter` bit patterns. **Done.** |
| 2 | Compile explicitly | `cuLinkCreate` + `cuLinkAddData(CU_JIT_INPUT_PTX)` + `cuLinkComplete` produces the executable before module load. **Pending maintainer SM86 (no toolkit on the CI/dev host).** |
| 3 | Preserve cubin | Returned ELF bytes SHA-256 hashed and exported under the source key; invalid/non-ELF output fails closed. **Pending SM86.** |
| 4 | Disassemble SASS | Pinned `nvdisasm` disassembles the exact preserved cubin and records entry point, registers, instructions, and global/shared/local traffic. **Pending SM86.** |
| 5 | Fail unsafe machine code | CI rejects missing/extra/stale/hash-mismatched artifacts and any final-SASS `LDL`/`STL`. Runtime `DirectPtxResourceBudget.Validate` already rejects nonzero local bytes, excess registers/shared memory, and insufficient occupancy at load time; the repo PTX-discipline guard rejects `.local` emission statically. **Runtime + static gates pass; SASS gate pending SM86.** |
| 6 | Profile exact cubin | A `--direct-ptx-profile-scientific` benchmark target executes every embedded cubin and asserts `EmbeddedCubin` before launch. **Pending SM86 (benchmarks are the maintainer's per the issue gate).** |
| 7 | Embed in NuGet | `Artifacts/sm86/*.cubin` + `scientific-cubins.tsv` embedded resources in `AiDotNet.Tensors`. **Pending SM86 artifact generation.** |
| 8 | Load cubin in production | Resolution order embedded cubin → verified disk cubin → driver-linked cubin. **Pending shared cubin infrastructure merge.** |
| 9 | Restrict PTX JIT | Direct PTX JIT is available only behind the explicit experiment fallback. Today the scientific path is entirely experiment-gated and unpromoted, so no production JIT occurs. **Pass (fail-closed).** |
| 10 | Cache complete identity | Each `DirectPtxKernelBlueprint` carries a versioned PTX SHA-256, GPU/SM/driver fingerprint, block geometry, resource budget, and occupancy; the `DirectPtxKernelCache` is keyed on the shape/metric/variant tuple. **Done for the source-side identity; disk sidecars pending SM86.** |

The no-GPU CI verifier already regenerates every current PTX string and blueprint
ID and validates every manifest row and coverage decision (`PtxKernelCoverageTests`
requires an explicit parity decision for every `Ptx*Kernel` in the assembly). The
cubin-hash and artifact-identity checks activate once stage 3 artifacts exist.

## Ten optimization readiness gates

| # | Production requirement | Current implementation and verdict |
|---:|---|---|
| 1 | Exact contiguous layout | Admission requires contiguous physical views whose logical/physical extents equal the baked shape (`DirectPtxExtentMode.Exact`); shapes, strides, metric, degree, curvature, and qubit count are removed before the hot launch. Unsupported views fall back to the established NVRTC path. **Pass.** |
| 2 | Coalesced vector memory access | The interleaved-complex and octonion kernels use widened `v2.f32`/`v4.f32` transactions (complex-multiply loads/stores each `[re,im]` pair as one 8-byte `v2`; octonion-add moves each 4-lane half as one 16-byte `v4`), halving/quartering their memory transactions. Remaining thread-per-element kernels use adjacent per-lane `.f32`/`.u8` loads and a single final store; block-per-row reductions stride the row by `blockDim` for coalesced partials. **Pass; further `v2/v4` widening of the split-buffer kernels is a follow-up.** |
| 3 | Shared memory only for reuse | Only the block-per-row reductions (`normalize-probabilities`, `measurement-forward`, `mobius-add`, `poincare-distance`) stage a 256- or 128-lane tree-reduce scratch. All thread-per-element/row/vector kernels use **zero** shared memory. **Pass.** |
| 4 | Register-resident math | Loaded operands, accumulators, spatial hashes, interpolation weights, SH basis, and reductions remain in registers until the final store. Every kernel reports **zero local bytes** (`Audit.Function.LocalBytesPerThread == 0`, asserted in the driver tests) and passes the PTX-discipline `.local` guard. **Pass.** |
| 5 | Combined/fused kernels | `measurement-forward` fuses the `\|z\|^2` evaluation and row normalization in one block pass; `spherical-softmax` fuses L2-normalize + numerically-stable softmax; the capsule contraction and SH kernels fuse their dot-products with the epilogue. **Mechanically pass; performance HOLD.** |
| 6 | Bounded global reductions | Block-per-row reductions use a fixed 256-lane shared tree-reduce with no output-sized scratch; thread-per-element kernels have no cross-thread reduction. There are no atomics and no output-sized temporaries. **Correctness pass; performance HOLD.** |
| 7 | Asynchronous stream ordering | Launches use the backend stream with no host synchronization; the dispatch layer rejects launches during CUDA-graph capture unless prewarmed. `cp.async` is inapplicable to these single-use loads. **Pass.** |
| 8 | CUDA Graph/lifetime safety | Modules are pinned for capture lifetime and the dispatch shell rejects compilation/cache-miss during capture. **Pass.** |
| 9 | Ahead-of-load binary control | PTX is emitted and blueprint-audited today; linking to cubin, SASS disasm, embedding, and content-addressed caching are staged for the SM86 pipeline. Raw PTX load is experiment-only and unpromoted. **Source-side pass; binary stages pending SM86.** |
| 10 | Promotion evidence | Three independent corrected PyTorch/reference comparisons plus correctness/determinism, zero hot allocation, resource, and tail gates must all pass. **HOLD:** no scientific cell is promoted; the fp64 CPU-oracle correctness harness is wired, but the uncontended three-run performance comparison is the maintainer's SM86 task. |

Tensor Core MMA is not a requirement for this family: the complex/octonion/
hyperbolic/quantum/SH/ANN formulas are element-, vector-, or small-reduction
shaped, not dense matrix multiplies, so cargo-cult MMA instructions that add
conversions or shared-memory traffic are intentionally absent. The one batched
GEMM-like op (`complex-matvec`) uses register-blocked FMA accumulation with a
shared matrix; a Tensor-Core variant is a future specialization, not a
correctness requirement.

## Kernel and memory contract

All admitted tensors are exact, contiguous, aligned physical views. Shapes,
strides, epsilon/clamp constants, metric/degree/curvature/qubit-count, and
level/output strides are baked into the PTX and removed before the hot launch;
the launch passes buffer pointers only. Unsupported layouts, architectures, or
disabled experiment gates return to the established NVRTC path with an exact
`DirectPtxLastError` reason. There are no output-sized temporary device
allocations, and the only device-side scratch is the fixed shared reduction
buffer inside the four block-per-row kernels.

The kernels use four dispatch models, each chosen to match the reference NVRTC
kernel exactly:

- **Thread-per-element** — one thread owns one output cell and walks the
  contracted/feature axis serially in registers: complex arithmetic, octonion
  add, RBF, pairwise distance, ANN compute-distances / PQ-distance-tables /
  PQ-ADC-scan, Instant-NGP hash-encode (8-corner trilinear), and the mesh
  Laplacian (face scan). The octonion multiply is a fully register-resident
  Cayley–Dickson product.
- **Thread-per-vector / row** — one thread owns one vector or row and reduces
  it serially: cosine similarity, capsule contraction/weighted-sum/agreement,
  spherical harmonics forward/backward, ANN IVF-assign (centroid scan with a
  strict-improvement argmin/argmax and int32 output), and the four-pass
  spherical softmax.
- **Block-per-row tree reduction** — one block owns one row; 256 (or 128)
  lanes stride-accumulate a partial into shared memory, tree-reduce with
  `bar.sync`, and rescale in place: `normalize-probabilities`,
  `measurement-forward`, `mobius-add`, and `poincare-distance`.
- **Block-per-batch, barrier-serialized** — `quantum-rotation` copies the
  state, then applies each qubit's Ry gate as a disjoint-pair butterfly
  (`cos.approx`/`sin.approx`, shift/and index math) with a `bar.sync` between
  the unrolled qubit steps; the butterfly pairs are disjoint so no shared
  memory or atomics are needed.

Transcendentals use the hardware approximate instructions where the reference
allows (`ex2.approx`/`lg2.approx` for `exp`/`log`, `tanh.approx`, `rcp.approx`,
`cos.approx`/`sin.approx`), and `sqrt.rn`/`div.rn` where the reference is
IEEE-exact. `atan2` (complex phase) is reconstructed from a minimax `atan`
polynomial plus quadrant folding because PTX has no `atan` primitive. Every
constant is emitted from its exact 32-bit pattern via `BitConverter` so the
net471 and net10.0 builds are bit-identical.
