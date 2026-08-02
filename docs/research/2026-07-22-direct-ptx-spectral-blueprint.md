# Direct PTX spectral and complex blueprint (#850)

## Status

This change establishes an experimental, disabled-by-default spectral/complex
family. The branch contains the family inventory and direct emitters, but only
contiguous interleaved FP32 complex multiplication currently owns a production
backend route. No shape is promoted. Live SM86 correctness and module-resource
checks are required for every emitter, and performance evidence remains
fail-closed until the paired oracle, external peers, and Nsight gates all pass.

## First inventory cell

`CudaBackend.ComplexMultiply` accepts three exact `[pairs, 2]` row-major FP32
allocations. Each `(real, imaginary)` pair occupies eight adjacent bytes. The
four admitted pair counts are 65,536, 262,144, 1,048,576, and 4,194,304. The
candidate is admitted only on SM86, requires 16-byte-aligned device pointers,
and rejects an output allocation that overlaps either input. Inputs may alias
each other because both are read-only.

Each 256-thread block uses shape-selected coarsening: two pairs/thread at
65,536 pairs, four at 262,144, and eight at the two larger extents. Every thread
processes aligned 16-byte chunks, and successive chunks are separated by one
block-width so every warp remains contiguous. Every admitted extent has a full
final block. Shape, stride, dtype, layout, and bounds do not travel through the
launch ABI. The PTX module receives only the three device pointers.

Each thread performs this fixed dataflow:

1. Load one `float4` (two complex pairs) from each input.
2. Evaluate each pair as `(ar * br - ai * bi, ar * bi + ai * br)` with the
   incumbent-compatible multiply/FMA association.
3. Store one final `float4` and reuse the same arithmetic registers for the
   next warp-contiguous chunk.

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
windows, resampling, mel/MFCC, and public tensor-engine routes. Every baked loop
whose JIT unrolling would multiply live values emits a statement-level
`nounroll` directive. This is a physical-resource contract: the first live run
found 16 cells failing, including 72-164-register runaway variants; the shared
directive and corrected full-residency budgets brought the complete focused
suite to zero failures without allowing local memory. No cell is promoted.

## Release evidence matrix

Results must be produced from one idle, thermally recorded NVIDIA machine. Raw
diagnostic rows retain 30 warmups, 101 samples, and three independent runs. The
automated AiDotNet head-to-head captures 64 public operations per CUDA graph on
separate backends, calibrates equal exposure, samples AB/BA brackets, and keeps
the unchanged three-consecutive-sample 5% spread gate. An unsupported capture
falls back explicitly to the same calibrated public-launch pairing. Correctness
is checked first; a non-finite or over-tolerance result suppresses the ratio.
Unstable cells emit `not-measurable`, never a speed claim. `--oracle-only` skips
the legacy raw table for focused diagnosis and cannot promote a cell.

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

The current diagnostic screen reports exact direct/incumbent error of
`4.768e-7`. The clean 1,048,576-pair cell improved to approximately 1.06x but is
still a tie under the required 1.10x gate; the 4,194,304-pair cell is also a tie,
and the two smaller cells remain not measurable under desktop interference.
These observations are tuning input, not promotion evidence.

## Commands reserved for the release machine

```powershell
dotnet run --project tests\AiDotNet.Tensors.Benchmarks -c Release -- --direct-ptx-complex-multiply 3
dotnet run --project tests\AiDotNet.Tensors.Benchmarks -c Release -- --direct-ptx-complex-multiply 1 --oracle-only
python tests\AiDotNet.Tensors.Benchmarks\BaselineRunners\py\run_direct_ptx_complex_multiply_competitors.py --runs 3
pwsh tests\AiDotNet.Tensors.Benchmarks\Profiling\run-direct-ptx-ncu.ps1 -Target complex-multiply
```

Promotion requires correctness against the double-precision oracle, complete
latency/allocation/resource records for every row, zero executed local
load/store instructions in Nsight, resource-budget and occupancy compliance,
and a statistically repeatable win against every required competitor. Until
then the feature remains explicitly experimental and fail-closed.
