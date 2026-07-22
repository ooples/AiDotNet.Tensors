# Compiled CUDA normalization blueprint

Date: 2026-07-22

Tracking issue: #838

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Current verdict

The implementation inventory is complete for the exact SM86 shapes in this
pull request: 65 row/channel specialization cells map to 61 distinct,
content-addressed cubins. The difference is intentional: the two atomic
parameter kernels have identical PTX for their three runtime grid sizes and
therefore share a compiled artifact.

This is a compiled-cubin pipeline, not a raw-PTX production loader. Every
normal production module load receives an ELF cubin. PTX text is accepted by
the CUDA driver linker only to create a missing artifact; the returned cubin
is the object preserved, hashed, cached, loaded, disassembled, and profiled.
Passing PTX directly to `cuModuleLoadDataEx` is restricted to an explicit
normalization experiment when the linker entry points are unavailable.

No normalization cell is promoted by this pull request yet. Correctness,
allocation, graph capture, artifact identity, runtime resource, and SASS
safety gates pass. Several cells beat current AiDotNet, but the required
three-run comparison against the strongest PyTorch lane is incomplete and
some backward/reduction cells still lose its best compiled result. The
production admission table therefore remains fail-closed.

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

## Kernel and memory contract

All admitted tensors are exact, contiguous, aligned physical views. Strides,
shapes, epsilon, momentum, dtype, and mode flags are removed before the hot
launch. Unsupported layouts or aliases return to the established CUDA path.
There are no output-sized temporary device allocations.

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

The experimental whole-tensor L2 lane uses aligned 16-byte loads, four
register FMAs, a bounded grid, warp/shared block folds, and one atomic per
block. It remains unselected because it still loses the existing CUDA kernel.

`cp.async` and Tensor Core instructions are requirements only where their
operands are reusable and their arithmetic applies. These D=64 normalization
kernels consume each row value once and perform reductions/elementwise math;
SASS correctly reports zero async copies and zero MMA instructions. Forcing a
global-to-shared copy before a single use would increase traffic. Wider tiled
or GEMM-containing fused cells must revisit both gates with profiler evidence.

## Streaming, graphs, and lifetime

All launches, accumulation clears, and fallbacks use the backend compute
stream. There is no hidden device synchronization, host copy, host reduction,
or hot-path allocation. Modules and plans are prewarmed before CUDA Graph
capture and pinned for the graph lifetime so cache eviction cannot invalidate
a retained `CUfunction`. Compilation, file I/O, tuning, and cache misses are
forbidden during capture.

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
5. Managed hot allocation and temporary VRAM are both zero.
6. Runtime local bytes and final-SASS `LDL`/`STL` are zero; registers, shared
   memory, and active blocks meet the blueprint budget.
7. The exact cubin is embedded, hash verified, disassembled, and profiled.

The executable screen uses 30 warmups, 101 samples, and 50 resident launches
per device sample. Promotion uses the competitor's strongest measured lane,
not eager PyTorch when compiled PyTorch is faster.

## Evidence collected on RTX 3080 / SM86

- All 11 focused net10 correctness/routing/capture/allocation tests pass.
- All current-source identities and compiled cubins pass the static verifier.
- Final SASS passes for the complete compiled inventory with zero `LDL` and
  zero `STL`; current maximum register use is 40/thread.
- The atomic parameter entries contain the intended final
  `ATOMG.E.ADD.F32` instructions and no local memory.
- Exact-cubin runtime profiling launches the complete embedded inventory.
- Nsight Compute 2025.4.1 attaches to the exact target, but hardware-counter
  collection is blocked locally by `ERR_NVGPUCTRPERM`. No counter-derived
  bandwidth/occupancy claim is made until an elevated runner or NVIDIA
  non-admin counter access is available.

Preliminary strongest-PyTorch medians show why promotion remains off. At
8,192 rows, compiled PyTorch measured approximately 20.75 us for LayerNorm
backward, 19.37 us for RMSNorm backward, and 14.13 us for whole-tensor L2
reduction. Current direct medians are approximately 44.8 us, 24.2 us, and the
unselected atomic L2 experiment 26.9 us respectively. Several forward/L2-axis
cells beat both competitors, but they still require the complete three-run
tail matrix before independent promotion.

## Rejected experiments retained as evidence

- Four outputs per BatchNorm thread increased register pressure and reduced
  useful parallelism. The source was restored exactly and the artifact
  identity verifier confirmed the restoration.
- Atomic parameter gradients regressed 256/2,048-row cells and LayerNorm at
  8,192 rows. Only 8,192-row RMSNorm showed a useful improvement, so routing
  is shape/operation specific.
- Scalar and `float4` bounded atomic whole-tensor L2 variants both remained
  slower than current CUDA. They stay experimental and unpromoted.

## Follow-up required before production promotion

1. Fuse or otherwise eliminate the second full input read in LayerNorm and
   RMSNorm backward while keeping parameter reduction traffic bounded.
2. Replace whole-tensor L2 with a measured hierarchy that beats compiled
   PyTorch without output-sized scratch or hidden synchronization.
3. Complete three clean independent runs for every candidate forward cell and
   rerun the corrected residual BatchNorm semantics.
4. Enable NVIDIA performance counters and archive exact-cubin NCU metrics for
   DRAM bytes, shared transactions/conflicts, occupancy, and local/spill
   counters.
5. Promote only the individual cells that pass; keep all other shapes behind
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
