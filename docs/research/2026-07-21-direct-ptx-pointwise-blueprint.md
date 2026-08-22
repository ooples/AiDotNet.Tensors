# Direct-PTX pointwise, activation, and GLU blueprint

Date: 2026-07-21

Tracking issue: #839

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Verdict

This increment establishes the issue-#839 assembly line and implements exact
contiguous Ampere FP32 candidates for tanh-GeGLU forward, SwiGLU forward, and
tanh-GeGLU backward. None is promoted. The fastest GeGLU-forward cells reach
about 0.66 useful TFLOPS and 0.79 TB/s, but the strongest compiled PyTorch
kernel and current AiDotNet kernel reach the same memory roof. The specialized
GeGLU backward beats compiled PyTorch by 1.22x at `[256,11008]`, but ties the
current AiDotNet CUDA kernel. No cell clears the required 1.10x gate over
every eligible competitor.

Production therefore fails closed with `performance-gate-not-met`; explicit
experiment overrides are required. This draft does not close #839. Its
executable 74-cell manifest assigns the remaining arithmetic, broadcast,
comparison, special-math, activation, derivative, GLU, fused epilogue, and
FP16/public-routing families.

The work also fixes an independent correctness defect in every established
CUDA gated-forward route. The old host launcher derived its grid from
`outerSize` although the kernels index `outerSize * halfDimension`; most
features and rows were never launched. A dedicated gated launcher now uses
the complete element count, with a non-specialized `[3,257]` regression test
covering GLU, GeGLU, ReGLU, and SwiGLU.

## Assembly-line contract

1. Assign every scoped public/backend API to one manifest cell before writing
   a specialization.
2. Freeze operation semantics, phase, dtype, architecture, exact shape bucket,
   physical extent, row split, alignment, and alias policy.
3. Validate those facts once at admission. Emit only tensor pointers in the
   PTX launch ABI; dimensions, strides, modes, and constants are module
   identity, never hot-loop parameters.
4. Load vectorized contiguous transactions, keep values and intermediates in
   lane registers, and issue only final coalesced stores. Shared memory and
   `cp.async` are used only where cross-thread tile reuse can repay staging.
5. Prewarm before CUDA graph capture. Reject nonzero local bytes, excessive
   resources, insufficient occupancy, unmeasured architectures, aliases, and
   physical-extent mismatches.
6. Compare resident NVIDIA paths only: current AiDotNet CUDA, direct PTX, and
   the strongest applicable PyTorch/Triton or NVIDIA library kernel. CPU MKL
   and OpenBLAS are intentionally ineligible for this GPU comparison.
7. Promote an exact architecture/shape/semantic cell only after three clean
   independent runs clear median, P95, allocation, temporary-VRAM, accuracy,
   and executed-spill gates. A rejected cell remains behind its experiment
   override and grants no promotion to neighboring shapes.

## Formal contiguous ABIs

Forward rows use `[value | gate]` storage:

| Tensor | Exact extent | Access | Alignment |
|---|---:|---|---:|
| input | `[outer,2D]` FP32 row-major | read | 16 B |
| output | `[outer,D]` FP32 row-major | write | 16 B |

The launch has two 64-bit pointer parameters. Each thread loads two FP32x4
vectors, evaluates four independent gates, and commits one FP32x4 vector.
There are no global intermediates, temporary allocations, shared memory,
dimension parameters, division/remainder instructions, or stride checks.

Backward adds `grad-output [outer,D]` and writes
`grad-input [outer,2D]`. Its launch has three pointers. Each thread performs
three FP32x4 loads and two final FP32x4 stores. `tanh(gate)` is computed once
per lane and reused for both `GELU(gate)` and `GELU'(gate)`; both derivatives
remain registers until the final split-row stores.

The admitted buckets are `(outer,D) = (1,4096), (32,4096), (256,4096), and
(256,11008)`. Ampere is the only implemented architecture family. All other
architectures and sizes fall back with an exact reason.

## Correctness and runtime proof

Focused tests enforce:

- pointer-only PTX, exact vector-load/store counts, no `.shared`, `.local`,
  stride, division, remainder, or scalar shape parameters;
- high-precision forward and derivative oracles, output/input alias rejection,
  exact physical extents, and established CUDA fallback parity;
- disabled-by-default admission, experiment-only override, bounded cached
  modules, deterministic disposal, prewarm, graph capture with captured modules
  pinned until backend disposal, and zero managed allocation across 40 hot calls;
- driver-JIT resource budgets and PTX SHA/device-fingerprint audits.

The maximum observed direct-vs-double-oracle error is `1.4e-5`, within the
`3e-4` derivative and `2e-4` forward tolerances.

## GPU screening evidence

Environment: Windows 10.0.26200, .NET 10.0.10 / SDK 10.0.302, NVIDIA GeForce
RTX 3080, SM86, driver fingerprint
`gpu-79d5ac8ef419a3bce86d78a8222664cd-sm86-drv13030`, PyTorch
2.12.1+cu130, CUDA 13.0, and `triton-windows` 3.7.1.post27. Each screening
cell uses 30 warmups and 101 samples. Raw-device samples average a captured
batch of 50 resident launches; E2E samples include one launch plus
synchronization. Uploads, downloads, compilation, allocation, and graph
construction are outside timing.

Forward counts 10 useful FLOPs and 12 useful bytes per output; backward counts
26 useful FLOPs and 20 useful bytes. Transcendentals count as one useful
operation. Device values below are `mean/median/P95/P99` microseconds from the
clean screening run. This is rejection evidence, not production-promotion
evidence; no three-run claim is made.

### GeGLU forward

| outer,D | Method | Device us mean/med/P95/P99 | GFLOPS | GB/s | managed B/call | temp VRAM B | max error | R/S/L/blocks |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 256,4096 | AiDotNet CUDA eager | 16.61/15.95/21.95/24.62 | 657.3 | 788.7 | 0 | 0 | 1.1e-5 | n/a |
| 256,4096 | Direct PTX eager | 18.42/15.95/23.80/48.74 | 657.3 | 788.7 | 0 | 0 | 1.1e-5 | 20/0/0/6 |
| 256,4096 | PyTorch compile | 17.12/16.71/21.22/25.15 | 627.5 | 752.9 | n/a | 0 | 4.8e-7 | n/a |
| 256,11008 | AiDotNet CUDA eager | 45.09/43.09/53.07/59.70 | 653.9 | 784.7 | 0 | 0 | 1.1e-5 | n/a |
| 256,11008 | Direct PTX eager | 44.12/42.68/51.26/61.93 | 660.3 | 792.3 | 0 | 0 | 1.1e-5 | 20/0/0/6 |
| 256,11008 | PyTorch compile | 45.99/44.52/53.64/58.68 | 632.9 | 759.5 | n/a | 0 | 4.8e-7 | n/a |

At `[256,11008]`, direct PTX is only `43.09/42.68 = 1.01x` faster than
the strongest current-AiDotNet median. At `[256,4096]` it ties. Smaller cells
lose to or tie compiled PyTorch launch-specialized kernels.

### GeGLU backward

| outer,D | Method | Device us mean/med/P95/P99 | GFLOPS | GB/s | managed B/call | temp VRAM B | max error | R/S/L/blocks |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 256,4096 | AiDotNet CUDA eager | 60.72/58.04/81.88/87.78 | 469.7 | 361.3 | 0 | 0 | 1.4e-5 | n/a |
| 256,4096 | Direct PTX eager | 58.20/55.11/79.01/81.41 | 494.7 | 380.5 | 0 | 0 | 1.4e-5 | 30/0/0/6 |
| 256,4096 | PyTorch compile | 57.81/55.91/72.60/77.37 | 487.6 | 375.1 | n/a | 0 | 3.6e-7 | n/a |
| 256,11008 | AiDotNet CUDA graph | 155.03/147.80/199.33/210.90 | 495.7 | 381.3 | 0 | 0 | 1.3e-5 | n/a |
| 256,11008 | Direct PTX eager | 152.06/147.93/193.31/202.71 | 495.3 | 381.0 | 0 | 0 | 1.3e-5 | 30/0/0/6 |
| 256,11008 | PyTorch compile | 183.89/180.00/218.48/240.41 | 407.1 | 313.1 | n/a | 524,288 | 3.6e-7 | n/a |

The large backward cell beats PyTorch median by `1.22x` and P95 by `1.13x`,
but the current AiDotNet graph median is fractionally faster. The release gate
correctly rejects it.

A later three-run forward attempt was excluded because throughput changed
mid-run: the C# half measured about 0.40 TB/s while the immediately following
PyTorch half measured about 0.76 TB/s. That violates the clean-run requirement
and demonstrates why promotion consumes conservative repeated evidence rather
than selecting favorable samples.

## Resource and spill status

| Kernel | Registers/thread | Static shared B | Local B/thread | 256-thread blocks/SM |
|---|---:|---:|---:|---:|
| GeGLU/SwiGLU forward FP32x4 | 20-24 | 0 | 0 | 6 |
| GeGLU backward FP32x4 | 30 | 0 | 0 | 6 |

The driver audit proves zero allocated local bytes and enforces the resource
budget before caching. Since no specialization is promoted, this increment
does not claim Nsight executed-spill proof. Deterministic forward and backward
targets and the repository CSV verifier are wired for Nsight Compute; a future
promotion must record all executed spill/local counters as zero.

## Reproduction

```powershell
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-geglu 3

dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-geglu-backward 3

powershell -ExecutionPolicy Bypass -File `
  tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target geglu-backward -NcuPath <path-to-ncu.exe>
```

## Next pointwise increments

1. Favor fusions that remove an otherwise materialized global intermediate;
   isolated memory-roof pointwise replacements cannot reliably produce 1.10x.
2. Build the bounded expression vocabulary for bias/residual/activation and
   paired forward/backward consumers, with exact saved-state contracts.
3. Add FP16/BF16 vector families and independently measured Ada, Hopper, and
   Blackwell modules; never infer promotion from Ampere.
4. Complete all 74 manifest cells with edge-case matrices, determinism,
   numerical modes, and promotion evidence per exact semantic bucket.
