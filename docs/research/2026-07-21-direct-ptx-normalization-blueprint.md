# Direct-PTX normalization and fused epilogue blueprint

Date: 2026-07-21

Tracking issue: #838

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Verdict

The first issue-#838 golden slice is a six-input FP32 inference kernel:

```text
gelu_tanh(layernorm(input + residual + bias, gamma, beta))
```

It specializes exact row-major `[rows,64]` tensors for Ampere. The 2,048-row
and 8,192-row cells beat current AiDotNet and the strongest measured PyTorch
CUDA/Triton-compiled path by conservative median factors of 1.35x and 1.79x.
They pass the production median, P95, numerical, allocation, temporary-VRAM,
and driver-JIT local-memory gates. The 256-row cell remains experiment-only:
PyTorch compile reached 1.99 us while the worst direct measurement was 2.17 us.

The route is disabled by default behind
`AIDOTNET_DIRECT_PTX_RESIDUAL_LAYERNORM_GELU=1` or the master direct-PTX gate.
Unsupported layouts, extents, shapes, architectures, and aliases fail closed
to the established CUDA composition. This increment does not close #838. An
executable 41-cell normalization manifest assigns every remaining forward,
backward, FP16, L2, fused activation, and convolution-normalization operation.

## Assembly-line contract

Every later normalization kernel follows the same sequence:

1. Freeze one semantic cell, dtype, architecture, shape bucket, physical
   layout, alignment, alias policy, accuracy mode, and fallback.
2. Establish exact physical extents at the boundary. The kernel never receives
   or dynamically checks strides, dimensions, epsilon, or mode flags.
3. Emit a pointer-only PTX ABI with geometry and constants baked into the
   module identity.
4. Fuse residual, bias, reductions, affine normalization, and activation while
   values are register resident; write only the final tensor to global memory.
5. Prewarm and audit the driver-JIT result before graph capture. Reject local
   memory, excess registers/shared memory, or insufficient occupancy.
6. Benchmark resident operands against current AiDotNet and the strongest
   NVIDIA GPU competitors with identical arithmetic, warmups, sample counts,
   accuracy checks, and device/E2E timing definitions.
7. Promote only a measured cell that clears all release gates in three
   independent runs. Architecture and shape families never inherit promotion.

Contiguity is therefore a boundary invariant, not a hot-loop branch. General
tensor views remain valid elsewhere in AiDotNet, but they must be materialized
or routed to fallback before entering this specialized ABI. A later tensor
layout registry can make those proofs reusable across adjacent fused kernels;
it must not add per-element stride checks.

## Register-resident dataflow

One 256-thread block owns eight rows and one warp owns each row. Every lane
loads two features from each row tensor. Those two values remain lane-owned
through both warp reductions, normalization, affine transform, and tanh-GELU.

```text
coalesced global loads
  input + residual + broadcast bias
        -> two lane registers
        -> warp sum -> mean
        -> warp squared-deviation sum -> inverse standard deviation
        -> broadcast gamma/beta -> tanh-GELU
        -> one final coalesced output store
```

| Allocation | Per output element | Global writes | Lifetime |
|---|---:|---:|---|
| input | one load | 0 | lane registers |
| residual | one load | 0 | lane registers |
| bias | one load | 0 | lane registers |
| gamma | one load | 0 | affine epilogue |
| beta | one load | 0 | affine epilogue |
| output | 0 loads | one store | final commit only |

There is no residual intermediate, bias intermediate, normalization output,
activation output, scratch allocation, shared memory, or temporary VRAM. PTX
emitter tests require exactly ten `ld.global.nc.f32`, two final stores, ten
warp shuffles, six pointer parameters, and no `.shared`, `.local`, scalar
shape parameter, or stride token. Because shared memory is absent, bank
conflicts and `cp.async` staging are inapplicable to this D=64 warp-row cell.
Async shared-memory tiling remains a blueprint option for wider rows and
multi-row Tensor Core cells where a staged panel is actually reused.

## Formal ABI and admission

The launch contains six 64-bit pointers and no scalar parameters:

| Tensor | Layout | Exact extent | Access | Alignment |
|---|---|---|---|---:|
| input | row-major | `[rows,64]` FP32 | read | 16 B |
| residual | row-major | `[rows,64]` FP32 | read | 16 B |
| bias | vector | `[64]` FP32 | read | 16 B |
| gamma | vector | `[64]` FP32 | read | 16 B |
| beta | vector | `[64]` FP32 | read | 16 B |
| output | row-major | `[rows,64]` FP32 | write | 16 B |

Admission checks the feature gate, CUDA availability, exact Ampere
architecture, promoted row bucket, exact byte extents, 16-byte alignment, and
non-aliasing once. Stable fallback reasons distinguish feature-disabled,
CUDA-unavailable, architecture-not-validated, shape-not-implemented,
performance-gate-not-met, and physical-extent-mismatch cases. Capture requires
a prewarmed cache entry and cannot JIT, tune, allocate, perform I/O, or evict.
The captured module is pinned so later specialization churn cannot unload a
`CUfunction` retained by the graph.

## GPU championship evidence

Environment: Windows 10.0.26200, .NET 10.0.10 / SDK 10.0.302, NVIDIA GeForce
RTX 3080, SM86, driver fingerprint
`gpu-79d5ac8ef419a3bce86d78a8222664cd-sm86-drv13030`, PyTorch
2.12.1+cu130, CUDA 13.0, and `triton-windows` 3.7.1.post27. Each cell uses 30
warmups and 101 samples in each of three independent runs. Raw-device samples
average a captured batch of 50 resident launches; E2E samples include one
launch plus synchronization. Uploads, downloads, compilation, allocation, and
graph construction are outside timing.

Useful work is `(19*64+3)*rows`; sqrt, reciprocal, and tanh each count as one
operation. Thus the table reports useful algorithmic GFLOPS rather than an
inflated hardware-instruction estimate. Values are min-max ranges over the
three-run capture. Device columns are `mean/median/P95/P99` microseconds; E2E
columns are `median/P95/P99` microseconds.

### Promoted rows

| Rows | Method | Device us mean/med/P95/P99 | E2E us med/P95/P99 | GFLOPS | managed B/call | temp VRAM B | max error |
|---:|---|---|---|---:|---:|---:|---:|
| 2,048 | AiDotNet CUDA eager | 20.20-22.33 / 20.07-22.30 / 20.52-22.73 / 21.40-29.06 | 42.60-48.80 / 61.80-88.50 / 65.60-255.40 | 111.9-124.4 | 56 | 0 | 5.6e-6 |
| 2,048 | AiDotNet CUDA graph | 20.59-21.80 / 20.36-21.40 / 21.65-22.77 / 25.50-30.90 | 37.00-42.20 / 45.60-59.90 / 54.80-77.00 | 116.7-122.6 | 0 | 0 | 5.6e-6 |
| 2,048 | Direct PTX eager | 2.76-2.84 / 2.58-2.79 / 3.13-3.28 / 3.44-7.00 | 19.30-23.60 / 33.10-42.50 / 40.20-44.60 | 896.3-967.5 | 0 | 0 | 5.6e-6 |
| 2,048 | Direct PTX graph | 2.67-2.78 / 2.58-2.64 / 3.11-3.19 / 3.28-6.45 | 18.40-19.30 / 38.90-40.50 / 40.80-46.10 | 945.0-967.5 | 0 | 0 | 5.6e-6 |
| 2,048 | PyTorch CUDA eager | 12.51-13.62 / 12.43-13.60 / 13.33-14.56 / 13.37-19.37 | 108.00-150.90 / 125.60-195.40 / 138.80-305.10 | 183.6-200.8 | n/a | 540,672 | 6.0e-7 |
| 2,048 | PyTorch CUDA graph | same eager device kernel | 32.30-32.40 / 35.50-41.80 / 42.40-60.80 | 183.6-200.8 | n/a | 1,024 | 6.0e-7 |
| 2,048 | PyTorch compile | 3.74-4.13 / 3.77-4.08 / 3.79-4.42 / 3.95-4.73 | 125.10-177.80 / 140.50-210.40 / 155.40-286.80 | 612.6-662.5 | n/a | 0 | 7.2e-7 |
| 2,048 | PyTorch compile graph | same compiled device kernel | 26.10-31.20 / 42.10-45.70 / 48.90-50.20 | 612.6-662.5 | n/a | 1,024 | 7.2e-7 |
| 8,192 | AiDotNet CUDA eager | 76.43-77.61 / 75.76-76.82 / 83.05-83.89 / 84.34-90.69 | 96.40-97.60 / 99.40-101.50 / 116.90-138.10 | 130.0-131.8 | 56 | 0 | 5.6e-6 |
| 8,192 | AiDotNet CUDA graph | 77.68-78.30 / 76.84-76.94 / 83.78-85.44 / 86.75-93.02 | 93.30-99.00 / 95.10-102.20 / 97.40-109.30 | 129.8-130.0 | 0 | 0 | 5.6e-6 |
| 8,192 | Direct PTX eager | 5.83-5.88 / 5.53-5.80 / 6.10-6.39 / 9.18-10.01 | 22.80-24.20 / 44.80-45.70 / 48.80-50.10 | 1,723.0-1,805.9 | 0 | 0 | 5.5e-6 |
| 8,192 | Direct PTX graph | 5.78-6.17 / 5.59-5.96 / 6.00-7.29 / 9.30-10.22 | 15.90-27.50 / 43.70-46.60 / 44.90-52.30 | 1,675.6-1,786.1 | 0 | 0 | 5.5e-6 |
| 8,192 | PyTorch CUDA eager | 40.52-40.70 / 40.26-40.43 / 40.65-45.61 / 46.32-47.47 | 96.10-96.60 / 118.50-128.30 / 135.80-226.90 | 247.0-248.0 | n/a | 2,162,688 | 7.2e-7 |
| 8,192 | PyTorch CUDA graph | same eager device kernel | 60.00-64.40 / 65.90-77.80 / 87.20-189.30 | 247.0-248.0 | n/a | 1,024 | 7.2e-7 |
| 8,192 | PyTorch compile | 10.85-10.97 / 10.65-10.90 / 11.30-11.49 / 11.61-16.98 | 125.20-126.70 / 142.50-154.40 / 172.20-316.90 | 916.5-937.7 | n/a | 0 | 7.2e-7 |
| 8,192 | PyTorch compile graph | same compiled device kernel | 30.40-30.50 / 31.40-34.10 / 35.40-39.90 | 916.5-937.7 | n/a | 1,024 | 7.2e-7 |

The executable TUI reports TFLOPS implicitly through GFLOPS: the 8,192-row
direct kernel sustains 1.68-1.81 useful TFLOPS. It also prints mean, median,
P95, P99, useful GB/s, managed allocation, temporary device bytes, numerical
error, and registers/shared/local/blocks for every competitor and run.

### Rejected 256-row cell

| Method | Device median us | P95 us | GFLOPS | managed B/call | temp VRAM B |
|---|---:|---:|---:|---:|---:|
| AiDotNet CUDA graph | 5.92-6.73 | 6.41-7.27 | 46.3-52.7 | 0 | 0 |
| Direct PTX | 1.99-2.17 | 2.46-3.03 | 143.8-157.1 | 0 | 0 |
| PyTorch compile | 1.99-2.54 | 2.48-2.56 | 123.0-157.1 | n/a | 0 |
| PyTorch compile graph | 1.99-2.54 | 2.48-2.56 | 123.0-157.1 | n/a | 0-1,024 |

The conservative gate compares the worst candidate evidence with the best
competitor evidence. At 2,048 rows it is `3.77/2.79 = 1.35x`; at 8,192 it is
`10.65/5.96 = 1.79x`. At 256 rows it is only `1.99/2.17 = 0.92x`, so that cell
correctly returns `performance-gate-not-met` in production.

## Resource and spill evidence

All three modules report the same driver-JIT resources:

```text
registers/thread: 18
static shared bytes: 0
local bytes/thread: 0
active 256-thread blocks/SM: 6
PTX/binary version: 86/86
```

| Rows | Blueprint suffix | PTX SHA-256 |
|---:|---|---|
| 256 | `w8-r256` | `04382119ccf04f836805f0839a3d48c6886b613d8ecc95949805f7b127cf0a3f` |
| 2,048 | `w8-r2048` | `030060fca5a18a70f81f52ec922756564deee659b88b0524a5ad833fa69270db` |
| 8,192 | `w8-r8192` | `e250a9dc0990d50e1c1527b69f8415cc7f367b3de4c7f7db863d0698f1fb4adf` |

The loader rejects more than 40 registers/thread, any shared or local bytes,
or fewer than six active blocks/SM before caching. Nsight Compute 2025.4.1 was
installed and attached to the exact 8,192-row kernel, but this host returned
`ERR_NVGPUCTRPERM` before metric collection. Consequently this increment does
not claim executed SASS zero-spill proof. It proves compile-time no-local
allocation through the driver audit, and the CSV verifier is ready to require
zero executed register spills, local loads, local stores, and derived local
spill requests once an administrator enables NVIDIA performance counters.

## Reproduction

```powershell
$env:AIDOTNET_DIRECT_PTX_RESIDUAL_LAYERNORM_GELU = '1'
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-residual-layernorm-gelu 3

powershell -ExecutionPolicy Bypass -File `
  tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target residual-layernorm-gelu -NcuPath <path-to-ncu.exe>
```

## Next normalization increments

1. FP32/FP16 LayerNorm and RMSNorm forward families, including residual and
   bias epilogues, using the exact same layout/admission/audit contract.
2. Normalization backward pairs with fused parameter-gradient reductions and
   no global forward-intermediate materialization beyond declared saved state.
3. Wider hidden dimensions using measured warp/block reduction catalogs and,
   only where panels are reused, async shared-memory pipelines.
4. BatchNorm, GroupNorm, InstanceNorm, and LocalResponseNorm families with
   training/inference semantics split into explicit blueprint cells.
5. Fused convolution-normalization-activation cells and FP16 architecture
   families, each independently benchmarked and promoted.
