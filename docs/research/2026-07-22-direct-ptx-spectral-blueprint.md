# Direct PTX spectral and complex blueprint (#850)

## Status

This change establishes an experimental, disabled-by-default golden slice for
contiguous interleaved FP32 complex multiplication. It does not promote the
kernel and makes no GPU correctness, performance, occupancy, or zero-spill
claim. Those decisions require same-machine release evidence that has not been
run in this worktree.

## First inventory cell

`CudaBackend.ComplexMultiply` accepts three exact `[pairs, 2]` row-major FP32
allocations. Each `(real, imaginary)` pair occupies eight adjacent bytes. The
four admitted pair counts are 65,536, 262,144, 1,048,576, and 4,194,304. The
candidate is admitted only on SM86, requires 16-byte-aligned device pointers,
and rejects an output allocation that overlaps either input. Inputs may alias
each other because both are read-only.

One 256-thread block covers exactly 256 pairs, so every admitted extent has a
full final block. Shape, stride, dtype, layout, and bounds do not travel through
the launch ABI. The PTX module receives only the three device pointers.

Each thread performs this fixed dataflow:

1. Load one `float2` from each input.
2. Keep the four components and two results in registers.
3. Evaluate `(ar * br - ai * bi, ar * bi + ai * br)` with two multiplies, two
   fused multiply-adds, and one negate.
4. Store one final `float2` to the output.

There is no intermediate global-memory tensor, shared-memory staging, local
declaration, dynamic stride branch, or tail branch in the emitted PTX. Static
text is not proof of generated SASS behavior; module audit plus executed Nsight
metrics remain mandatory.

## Runtime contract

The production entry point checks the feature gate, exact SM, exact extent,
device pointers, alignment, and aliasing before module lookup. Unsupported or
unpromoted calls preserve the established NVRTC implementation. Modules use a
bounded cache, explicit prewarm, capture-time cold-load rejection, capture
pinning, and backend disposal. The experiment and benchmark baseline overrides
are thread-local.

The spectral coverage manifest records the current implementation and intended
PTX ownership for complex arithmetic, FFT/RFFT, STFT, spectral transforms,
windows, resampling, mel/MFCC, and public tensor-engine routes. Only this first
cell is experimental; no cell is promoted.

## Release evidence matrix

Results must be produced from one idle, thermally recorded NVIDIA machine with
identical resident inputs/outputs, 30 warmups, 101 samples, and three independent
runs. Device samples use CUDA events and 50 launches per sample. End-to-end
samples measure one host dispatch plus synchronization. GFLOPS counts six real
floating-point operations per complex pair; effective bandwidth counts both
inputs and the output. Managed and temporary device allocations are reported
separately.

| Pair count | Competitor | Device median/P95/P99 | E2E median/P95/P99 | GFLOPS | Effective GB/s | Managed bytes | Temporary device bytes | Max error | Registers/shared/local/occupancy | Status |
|---:|---|---|---|---:|---:|---:|---:|---:|---|---|
| 65,536 | AiDotNet NVRTC | pending | pending | pending | pending | pending | pending | pending | n/a | pending GPU run |
| 65,536 | Direct PTX | pending | pending | pending | pending | pending | 0 expected | pending | pending Nsight/module audit | pending GPU run |
| 65,536 | PyTorch CUDA eager | pending | pending | pending | pending | 0 | pending | pending | n/a | pending GPU run |
| 65,536 | PyTorch CUDA graph | pending | pending | pending | pending | 0 | pending | pending | n/a | pending GPU run |
| 262,144 | AiDotNet NVRTC | pending | pending | pending | pending | pending | pending | pending | n/a | pending GPU run |
| 262,144 | Direct PTX | pending | pending | pending | pending | pending | 0 expected | pending | pending Nsight/module audit | pending GPU run |
| 262,144 | PyTorch CUDA eager | pending | pending | pending | pending | 0 | pending | pending | n/a | pending GPU run |
| 262,144 | PyTorch CUDA graph | pending | pending | pending | pending | 0 | pending | pending | n/a | pending GPU run |
| 1,048,576 | AiDotNet NVRTC | pending | pending | pending | pending | pending | pending | pending | n/a | pending GPU run |
| 1,048,576 | Direct PTX | pending | pending | pending | pending | pending | 0 expected | pending | pending Nsight/module audit | pending GPU run |
| 1,048,576 | PyTorch CUDA eager | pending | pending | pending | pending | 0 | pending | pending | n/a | pending GPU run |
| 1,048,576 | PyTorch CUDA graph | pending | pending | pending | pending | 0 | pending | pending | n/a | pending GPU run |
| 4,194,304 | AiDotNet NVRTC | pending | pending | pending | pending | pending | pending | pending | n/a | pending GPU run |
| 4,194,304 | Direct PTX | pending | pending | pending | pending | pending | 0 expected | pending | pending Nsight/module audit | pending GPU run |
| 4,194,304 | PyTorch CUDA eager | pending | pending | pending | pending | 0 | pending | pending | n/a | pending GPU run |
| 4,194,304 | PyTorch CUDA graph | pending | pending | pending | pending | 0 | pending | pending | n/a | pending GPU run |

No row may be bolded and no winner may be declared until all required rows pass
the same-run correctness and evidence gates. cuFFT is a required peer for future
FFT transform cells, but it is not a complex elementwise multiplication API and
therefore is not presented as a competitor for this cell.

## Commands reserved for the release machine

```powershell
dotnet run --project tests\AiDotNet.Tensors.Benchmarks -c Release -- --direct-ptx-complex-multiply 3
python tests\AiDotNet.Tensors.Benchmarks\BaselineRunners\py\run_direct_ptx_complex_multiply_competitors.py --runs 3
pwsh tests\AiDotNet.Tensors.Benchmarks\Profiling\run-direct-ptx-ncu.ps1 -Target complex-multiply
```

Promotion requires correctness against the double-precision oracle, complete
latency/allocation/resource records for every row, zero executed local
load/store instructions in Nsight, resource-budget and occupancy compliance,
and a statistically repeatable win against every required competitor. Until
then the feature remains explicitly experimental and fail-closed.
