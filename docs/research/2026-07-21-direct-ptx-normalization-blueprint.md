# Compiled CUDA normalization blueprint

Date: 2026-07-22

Tracking issue: #838

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Current verdict

The implementation inventory is complete for the exact SM86 shapes in this
pull request: 71 row/channel specialization cells map to 66 distinct,
content-addressed cubins. The difference is intentional: the two atomic
parameter kernels have identical PTX for their three runtime grid sizes and
therefore share a compiled artifact. The inventory now includes row-specific
single-pass fused LayerNorm and RMSNorm backward experiments for all three
row buckets.

This is a compiled-cubin pipeline, not a raw-PTX production loader. Every
normal production module load receives an ELF cubin. PTX text is accepted by
the CUDA driver linker only to create a missing artifact; the returned cubin
is the object preserved, hashed, cached, loaded, disassembled, and profiled.
Passing PTX directly to `cuModuleLoadDataEx` is restricted to an explicit
normalization experiment when the linker entry points are unavailable.

No new normalization cell is promoted by this pull request. Correctness,
allocation, repeated graph replay, artifact identity, runtime resource, and
SASS safety gates pass. The latest backward kernels replace destination
memsets with a backend-persistent 2,052-byte accumulator/counter workspace, but
the required clean three-run comparison against compiled PyTorch is still
pending. The production admission table therefore remains fail-closed.

## Ten-stage production binary pipeline

| # | Required stage | Implementation and release gate |
|---:|---|---|
| 1 | Generate PTX | Shape, dtype, epsilon/momentum, architecture, physical layout, alias policy, and operation are frozen in the blueprint; hot ABIs contain pointers only. |
| 2 | Compile explicitly | `cuLinkCreate` + `cuLinkAddData(CU_JIT_INPUT_PTX)` + `cuLinkComplete` produces the executable before module load. Linker failures include both info and error logs. |
| 3 | Preserve cubin | The returned ELF bytes are SHA-256 hashed and exported under their source key. Invalid/non-ELF output fails closed. |
| 4 | Disassemble SASS | Pinned `nvdisasm` 13.3.73 disassembles the exact preserved cubin and records entry point, registers, instructions, global/shared/local traffic, async copies, and Tensor Core instructions. |
| 5 | Fail unsafe machine code | CI rejects a missing/extra/stale/hash-mismatched artifact, multiple direct entries, absent resource metadata, or any final-SASS `LDL`/`STL`. Runtime blueprint budgets independently reject local bytes, excess registers/shared memory, and insufficient occupancy. |
| 6 | Profile exact cubin | `--direct-ptx-profile-normalization` executes every embedded cubin and asserts `EmbeddedCubin` before launch. Nsight Compute is invoked against that target, never against a separately compiled sample. |
| 7 | Embed in NuGet | `Artifacts/sm86/*.cubin` and `normalization-cubins.tsv` are embedded resources in `AiDotNet.Tensors`; the package contains the audited bytes and their release hashes. |
| 8 | Load cubin in production | Resolution order is embedded cubin, verified disk cubin, then driver-linked cubin. `cuModuleLoadData` receives the compiled image. |
| 9 | Restrict PTX JIT | Direct PTX JIT is available only behind the explicit experiment fallback and only when CUDA linker entry points are unavailable. It is not a promotion path. |
| 10 | Cache complete identity | The source key hashes pipeline version + target SM + PTX content. Disk identity also includes CUDA driver/linker version; sidecars preserve source key, cubin hash, target, and driver. |

The no-GPU CI verifier regenerates every current PTX string and blueprint ID,
recomputes source keys, validates every manifest row, checks every cubin hash,
and rejects extra artifacts. Reusing one source key across byte-identical
specializations is allowed only when the cubin filename and hash also match.

## Ten optimization readiness gates

| # | Production requirement | Current implementation and verdict |
|---:|---|---|
| 1 | Exact contiguous layout | Admission requires contiguous, aligned physical views with frozen D=64 shape/stride semantics; unsupported views fall back. **Pass.** |
| 2 | Coalesced vector memory access | Row kernels use adjacent `v2.f32` accesses and whole-tensor L2 uses aligned `v4.f32`; final values are written once. **Pass.** |
| 3 | Shared memory only for reuse | Backward parameter partials are folded in 16 KiB (LayerNorm) or 5 KiB (RMSNorm); L2 uses a 64-byte warp fold. Single-use row inputs are not pointlessly staged. **Pass.** |
| 4 | Register-resident math | Loaded row values, statistics, affine terms, reductions, and grad-input epilogues remain in registers until final stores. Final SASS has zero local loads/stores. **Pass.** |
| 5 | Combined/fused kernels | The large-shape experimental backward paths compute grad-input plus parameter partials in one input pass and one dispatch. Existing residual+BatchNorm+ReLU is also truly fused. **Mechanically pass; performance HOLD.** |
| 6 | Bounded global reductions | Fused backward folds issue one `RED` per block/output into one of four banked accumulators, reducing per-address contention by 4x, plus one completion atomic per block. L2 uses a 512-thread, at-most-128-block grid feeding 16 reusable `RED` banks and a half-warp final fold. The complete persistent workspace is 2,052 bytes; no output-sized scratch exists. **Source/resource pass; current L2 GPU correctness and performance HOLD.** |
| 7 | Asynchronous stream ordering | Launches use the backend stream with no host synchronization. Fused backward/L2 no longer clear outputs; their prewarmed workspace self-resets inside the kernel. `cp.async` is required only for reusable tiles and is inapplicable to these single-use D=64 loads. **Pass.** |
| 8 | CUDA Graph/lifetime safety | Plans are prewarmed and modules pinned for capture lifetime; compilation, tuning, file I/O, and cache misses are rejected during capture. **Pass.** |
| 9 | Ahead-of-load binary control | PTX is linked to cubin, hashed, embedded, loaded, disassembled to SASS, resource-audited, and content-addressed cached. Raw PTX load is experiment-only. **Pass.** |
| 10 | Promotion evidence | Three independent corrected PyTorch comparisons, correctness/determinism, zero hot allocation, bounded workspace, resource, and tail gates must all pass. **HOLD:** the newest topology lacks an uncontended three-run result and previous backward/L2 medians or p95s lost. |

Tensor Core MMA is not a normalization requirement: no matrix multiply exists
in these formulas. The same policy requires MMA for applicable GEMM/attention
PRs and rejects cargo-cult instructions that add conversions or shared-memory
traffic without useful tensor operations.

## Kernel and memory contract

All admitted tensors are exact, contiguous, aligned physical views. Strides,
shapes, epsilon, momentum, dtype, and mode flags are removed before the hot
launch. Unsupported layouts or aliases return to the established CUDA path.
There are no output-sized temporary device allocations. Workspace-requiring
operations share one backend-owned 2,052-byte allocation created during prewarm;
the hot path performs no allocation and its pointer remains stable for graph
capture.

Row forward/input-gradient kernels assign one warp to one 64-value row. Each
lane loads two adjacent half-row values, retains them in registers through
warp reductions and the affine/activation epilogue, then commits the final
values once. Channel kernels use the same register-resident rule and shared
memory only for values reused across threads. Parameter gradients use
coalesced 16-feature tiles: row cohorts accumulate in registers, shuffle
within a warp, fold fixed partials through 1 KiB shared memory, and write the
64 final parameters once.

The experimental fast parameter lane divides rows into 256-row tiles and
uses one final FP32 atomic per tile/output. It is selected only for the
measured 8,192-row RMSNorm fast-mode cell; deterministic mode always retains
the fixed-order kernel. Output zeroing is `cuMemsetD8Async` on the same stream,
so it is ordered, graph-capturable, and has no host synchronization.

The new experimental large-shape backward lane is a single dispatch and a
single global input pass. A 32-warp LayerNorm block retains two adjacent
features per lane in registers and folds 16 KiB of block parameter partials
through shared memory once. RMSNorm uses 20 warps, 5 KiB shared memory, and two
resident blocks per SM. The 64 folding threads issue one atomic per
block/output into one of four accumulator banks, then a last-block completion
protocol folds those banks, copies final parameters, clears every accumulator,
and resets the counter. This removes both destination memsets and a second
global block-partial read pass while cutting atomic contention by 4x. Neither
route is admitted without the explicit experiment override.

The experimental whole-tensor L2 lane uses aligned 16-byte loads, four
register FMAs, 512-thread blocks, a bounded 128-block grid, and a 64-byte
warp/shared fold. Blocks publish through 16 reusable non-returning `RED` banks,
limiting each address to at most eight publishers. The last block uses one warp
with a 16-lane member mask to fold and clear those banks, writes the scalar directly, and resets the
workspace completion counter. This replaces the slower 128-slot second global
fold. Exact shape specializations emit one uniform `LDG.E.128` per thread at
256/2,048 rows and exactly two at 8,192 rows, with no runtime input bounds loop.
It remains unselected until current-source GPU correctness, graph replay,
and clean performance runs all pass. The
deterministic one-block variant also adopts the wider block fold but remains a
fallback-only experiment.

`cp.async` and Tensor Core instructions are requirements only where their
operands are reusable and their arithmetic applies. These D=64 normalization
kernels consume each row value once and perform reductions/elementwise math;
SASS correctly reports zero async copies and zero MMA instructions. Forcing a
global-to-shared copy before a single use would increase traffic. Wider tiled
or GEMM-containing fused cells must revisit both gates with profiler evidence.

## Streaming, graphs, and lifetime

All launches and fallbacks use the backend compute stream. There is no hidden
device synchronization, host copy, host reduction, or hot-path allocation.
The fixed workspace is allocated and zeroed during prewarm; its accumulator
and counter are reset by the completing block before kernel exit. Repeated
launch and CUDA Graph replay tests reject stale accumulation. Modules and plans
are prewarmed before capture and pinned for the graph lifetime so cache
eviction cannot invalidate a retained `CUfunction`. Compilation, file I/O,
tuning, and cache misses are forbidden during capture.

Fused residual + BatchNorm + ReLU is connected through the production fusion
manager. The kernel computes the normalization affine result, adds the
residual while values are resident, applies ReLU, and performs one output
write. Other compositions are not described as fused until their public route
actually dispatches a combined cubin.

## Release gates

A shape/operation/architecture cell may be promoted only when all conditions
hold in three independent runs:

1. Median device time is at least 1.10x faster than the strongest of current
   AiDotNet and PyTorch, including `torch.compile` max-autotune and CUDA Graph.
2. Candidate P95 is no more than 10% worse than that strongest competitor.
3. Identical resident inputs, semantics, numerical mode, warmups, samples,
   launch batch, stream, and graph treatment are used.
4. Correctness and deterministic-mode bit stability pass at declared
   tolerances; fast atomics are explicitly labeled nondeterministic.
5. Managed hot allocation and per-dispatch VRAM allocation are both zero;
   bounded persistent workspace must be declared, graph-stable, self-resetting,
   and independent of tensor extent rather than output-sized.
6. Runtime local bytes and final-SASS `LDL`/`STL` are zero; registers, shared
   memory, and active blocks meet the blueprint budget.
7. The exact cubin is embedded, hash verified, disassembled, and profiled.

The executable screen uses 30 warmups, 101 samples, and 50 resident launches
per device sample. Promotion uses the competitor's strongest measured lane,
not eager PyTorch when compiled PyTorch is faster.

The resident screen reports per-dispatch `tmpB` separately from prewarmed
`persistB`. Advancement requires `tmpB == 0`; nonzero `persistB` is accepted
only when the caller explicitly declares it bounded and reusable and the route
is CUDA-Graph compatible. The Python competitor accepts the same exact row
scopes (`row256`, `row2048`, and `row8192`) as the AiDotNet runner.
`COMPARE` means only that a cell passed this first-stage AiDotNet resident
screen; it is never a production-promotion result without the strongest clean
PyTorch eager/compiled comparison.

Both runners fail closed before and after every cell if any other OS-level
Python process exists or any external Python PID appears in NVIDIA's compute
process table. Checking both closes the observed startup race where a Python
process existed before it registered its CUDA context.

## Evidence collected on RTX 3080 / SM86

- The four-bank backward head passed all 14 focused GPU
  correctness/routing/capture/allocation tests on net10. The current source adds
  a banked-L2 emitter/routing contract that passes on net10 and net471. Exact
  current-source L2 GPU correctness and graph replay, plus the four-bank net471
  GPU replay, wait for an uncontended device.
- All 71 current-source identities and 66 compiled cubins pass the static
  verifier.
- Final SASS passes for all 67 cubins with zero `LDL` and zero `STL`; current
  maximum register use is 48/thread.
- Fused backward and the banked L2 entry contain the intended final
  `RED.E.ADD.F32` publishers; only the integer completion counters use `ATOMG`.
  The banked L2 cubin uses 24 registers/thread and 64 bytes shared memory; its
  hot reduction paths use inline full-warp/half-warp `SHFL` instructions.
- Exact-cubin runtime profiling launches all 71 embedded specializations and
  reports `PROFILED_NORMALIZATION_CUBINS=71`.
- Nsight Compute 2025.4.1 attaches to the exact target, but hardware-counter
  collection is blocked locally by `ERR_NVGPUCTRPERM`. No counter-derived
  bandwidth/occupancy claim is made until an elevated runner or NVIDIA
  non-admin counter access is available.

The corrected three-run PyTorch 2.12.1/CUDA 13 contract uses saved RMS input
for RMS backward and sum-of-squares output for `ReduceNormL2`. At 8,192 rows,
compiled PyTorch medians were 20.46/20.40/20.50 us for LayerNorm backward,
14.38/14.30/14.46 us for RMSNorm backward, and 12.66/12.90/13.15 us for L2.
The exact embedded fused cubins measured 28.84/27.40/19.50 us for LayerNorm
and 19.48/16.51/19.11 us for RMSNorm. Their p95s were 39.67/34.86/36.43 us
and 42.33/22.88/26.34 us respectively, far outside the +10% tail gate. The
superseded single-address/bounded-partial 512-thread L2 experiments measured
about 17.20 us median and 22.18 us p95 in their clean routed screen; the
deterministic embedded variant measured 42.64--43.77 us. The new 16-bank
`RED`/half-warp-fold candidate has not been timed. All three operation families
remain HOLD.

The first workspace topology stored every block partial and folded it in the
last block. A serial 64-thread fold measured 39.44 us LayerNorm / 36.07 us
RMSNorm; an eight-lane-per-feature fold improved that to 36.19 / 19.31 us but
still lost compiled PyTorch. The retained four-bank accumulator topology
removes that second partial read pass and reduces atomic address contention.
The earlier single-bank run cannot be used for promotion because two unrelated
Python CUDA workloads held the GPU at 99% utilization. The four-bank candidate
was not timed when a later `compress_medium.py` workload held utilization near
28%; no contended absolute number is treated as release evidence.

A later low-utilization-start attempt initially reported 7.95 us median /
11.86 us p95 LayerNorm backward and 5.24 / 7.78 us RMSNorm backward. The
`stage_a_uncompressed.py` process already existed at the OS level and registered
CUDA during that run; unchanged forward controls and runs two/three slowed
sharply. The entire attempt is rejected, including those promising first-run
numbers. The new dual process guard reproduces this case and aborts before any
cell can be reported.

## Rejected experiments retained as evidence

- Four outputs per BatchNorm thread increased register pressure and reduced
  useful parallelism. The source was restored exactly and the artifact
  identity verifier confirmed the restoration.
- Atomic parameter gradients regressed 256/2,048-row cells and LayerNorm at
  8,192 rows. Only 8,192-row RMSNorm showed a useful improvement, so routing
  is shape/operation specific.
- Scalar, single-address atomic, and 128-slot block-partial whole-tensor L2
  variants remained slower than current CUDA. The replacement 16-bank `RED`
  variant stays experimental and unpromoted until clean validation.
- Backward sweeps rejected 256-thread blocks, 64/96/128-block common grids,
  `v4` half-warp mapping, `.cg` loads, reciprocal broadcast, duplicate-stat
  shuffles, and 512-row parameter tiles. The retained RMS 20-warp geometry
  improved median occupancy, but not enough to pass PyTorch or tail gates.
- A 24-warp RMS block was rejected by the runtime resource budget: register
  allocation granularity permitted only one block/SM, so the benchmark failed
  closed instead of timing a fallback.
- A 21-warp RMS block was also rejected by the same two-block/SM resource gate;
  20 warps is the largest retained geometry with the required occupancy.
- Last-block workspace folds using one serial feature thread and eight lanes
  per feature were correct but slower than the retained reusable-accumulator
  protocol.

## Follow-up required before production promotion

1. Run the retained four-bank persistent-accumulator topology on an otherwise idle GPU;
   promote it only if all three median and p95 comparisons beat compiled
   PyTorch by the stated gates.
2. Continue reducing RMS reciprocal/reduction and accumulator contention only
   with measured variants that preserve two blocks/SM and the single input
   pass.
3. Validate and measure the 16-bank L2 reduction on an idle device; if it misses
   the gate, continue the hierarchical/cooperative search without output-sized
   scratch or hidden synchronization.
4. Complete three clean independent runs for every remaining forward/channel
   candidate and rerun the corrected residual BatchNorm semantics.
5. Enable NVIDIA performance counters and archive exact-cubin NCU metrics for
   DRAM bytes, shared transactions/conflicts, occupancy, and local/spill
   counters.
6. Promote only the individual cells that pass; keep all other shapes behind
   `normalization-performance-gate-not-met`.

## Reproduction

```powershell
dotnet tests/AiDotNet.Tensors.Benchmarks/bin/Release/net10.0/AiDotNet.Tensors.Benchmarks.dll `
  --verify-direct-ptx-normalization-cubins

dotnet tests/AiDotNet.Tensors.Benchmarks/bin/Release/net10.0/AiDotNet.Tensors.Benchmarks.dll `
  --audit-direct-ptx-normalization-sass <nvdisasm.exe> `
  src/AiDotNet.Tensors/Engines/DirectGpu/CUDA/Ptx/Artifacts/sm86 `
  artifacts/direct-ptx/normalization/sass

dotnet tests/AiDotNet.Tensors.Benchmarks/bin/Release/net10.0/AiDotNet.Tensors.Benchmarks.dll `
  --direct-ptx-normalization 3 all

<ncu.exe> --target-processes all --profile-from-start off `
  dotnet tests/AiDotNet.Tensors.Benchmarks/bin/Release/net10.0/AiDotNet.Tensors.Benchmarks.dll `
  --direct-ptx-profile-normalization
```
