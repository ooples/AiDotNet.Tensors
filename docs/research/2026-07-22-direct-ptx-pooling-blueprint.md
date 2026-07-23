# Direct-PTX pooling and spatial-transform blueprint

Date: 2026-07-22

Tracking issue: #842

Parent epic: #833

Foundation: `2026-07-20-fused-attention-championship-blueprint.md`

## Status

This increment establishes the issue-#842 assembly line and implements one
disabled, unpromoted candidate: exact contiguous FP32 global average pooling.
No GPU execution, correctness, timing, occupancy, or executed-spill result is
claimed by this branch. Those cells remain pending until the controlled GPU
campaign is run on the exact commit.

The candidate computes:

`output[batch, channel] = mean(input[batch, channel, :, :])`

The canonical contiguous NCHW allocation is viewed once at the host boundary
as `[batch * channels, height * width]`. Dimensions and strides are baked into
the module identity, and the launch ABI contains only input and output pointers.

## Golden-slice choice

Global average pooling is the smallest reduction that exercises the spatial
family's essential blueprint mechanics: exact layout admission, vectorized
coalesced reads, warp reduction, a register-resident scale, single final write,
bounded module caching, graph prewarm, and deterministic fallback. It is not
presented as a fusion victory: the existing CUDA global-average-pool operation
is already a single-pass kernel. Windowed pooling plus activation and
pool-to-linear fusion remain the higher-value follow-up candidates.

## Formal tensor ABI

| Tensor | Logical and physical extent | Type/layout | Access | Alignment | Extent policy |
|---|---|---|---|---:|---|
| `input` | `[rows, spatial]` | FP32 row-major | read | 16 B | exact allocation |
| `output` | `[rows]` | FP32 vector | write | 16 B | exact allocation |

`rows = batch * channels` and `spatial = height * width`. The initial closed
shape set is `(256,128)`, `(2048,64)`, `(2048,128)`, and `(8192,128)`. The
candidate is admitted only for SM86; SM80, SM89, Hopper, and Blackwell require
independent implementations and evidence.

Host admission rejects disabled/unpromoted routes, unsupported architecture or
shape, non-positive/overflowing dimensions, null buffers, null device pointers,
non-exact byte extents, less-than-16-byte alignment, and input/output overlap.
Every rejection records a human-readable fallback reason and returns to the
existing CUDA implementation.

## Fused dataflow and memory boundaries

One warp owns one row.

1. Each lane performs one aligned FP32x2 or FP32x4 global load.
2. Values remain in registers and are accumulated locally.
3. The accumulator is bit-reinterpreted through `.b32` registers for five
   `shfl.sync.bfly.b32` reduction steps, then returned to FP32 arithmetic.
4. Lane zero multiplies by the baked reciprocal and writes one FP32 result.

There is no shared memory, local-memory declaration, global intermediate,
runtime division, runtime stride, or shape parameter. Driver admission rejects
any JIT result with nonzero local bytes or a resource budget violation. Final
acceptance additionally requires Nsight evidence that executed local loads,
local stores, and spilling instructions are all zero.

## Runtime and graph-capture contract

- The feature gate is `AIDOTNET_DIRECT_PTX_GLOBAL_AVGPOOL` and is off by default.
- `IsPromotedShape` is false for every shape on this branch.
- The benchmark-only unpromoted override is thread-local.
- Modules live in the bounded direct-PTX LRU cache and are disposed with the
  owning backend.
- Cold graph capture fails closed and instructs the caller to prewarm.
- A cached module used during capture is pinned until backend disposal so LRU
  churn cannot unload a `CUfunction` retained by a graph executable.

## Executable coverage inventory

`DirectPtxPoolingCoverageManifest` assigns every currently inventoried pooling,
interpolation, padding, grid-sample, ROI, spatial-transform, pixel-shuffle, and
public routing boundary to exactly one existing, experimental, or planned PTX
lane. Only global average pooling is experimental in this increment; nothing is
marked promoted.

## Benchmark and profiler protocol

The resident harness is prepared for 30 warmups, 101 samples, and three clean
runs. CPU MKL and OpenBLAS are excluded because they are not NVIDIA GPU
competitors. Final comparison rows must include current AiDotNet CUDA, Direct
PTX, and resident PyTorch/CUDA for each admitted shape.

| Shape `(rows, spatial)` | Current AiDotNet | Direct PTX | PyTorch CUDA | Winner |
|---|---:|---:|---:|---|
| `(256,128)` | Pending GPU run | Pending GPU run | Pending GPU run | Pending |
| `(2048,64)` | Pending GPU run | Pending GPU run | Pending GPU run | Pending |
| `(2048,128)` | Pending GPU run | Pending GPU run | Pending GPU run | Pending |
| `(8192,128)` | Pending GPU run | Pending GPU run | Pending GPU run | Pending |

Each final row must report device and end-to-end mean/median/P95/P99, effective
GB/s, managed bytes per call, temporary device bytes, maximum error, launch
count, registers/thread, static/dynamic shared bytes, local bytes, occupancy,
and the environment fingerprint. The deterministic Nsight target emits exactly
one launch per specialization and the common verifier rejects incomplete metric
sets or any nonzero spill/local counter.

```powershell
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-global-avgpool 3

python tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_global_avgpool_competitors.py `
  --runs 3

tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target global-avgpool
```

## Promotion gate

No shape may be promoted until three independent clean runs demonstrate at
least 1.10x median speedup over the strongest eligible GPU competitor, candidate
P95 no worse than competitor P95 plus 10%, zero hot managed allocation, zero
avoidable temporary VRAM, acceptable numerical error, zero driver-reported
local bytes, and complete all-zero executed spill/local evidence.

## Next spatial increments

1. Global max pooling and scaled global-average-pool backward.
2. Predicated-tail variants for common 7x7 and 14x14 spatial extents.
3. Baked-window average/max pooling with fused activation.
4. Pool-to-linear fusion when the graph contract permits eliminating the pooled
   tensor materialization.
5. Interpolation, padding, grid-sample and gradients, then FP16/BF16 and
   independently tuned Ada/Hopper/Blackwell modules.
