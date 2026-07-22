# Direct-PTX recurrent/state-space blueprint: RG-LRU golden slice

Issue: #846

Status: experimental, disabled by default, not promoted

Representative operation: RG-LRU forward, FP32 `[batch=1, sequence=128, recurrentDimension=256]`

Architecture: exact SM86 only

## What this PR proves without a GPU run

This PR establishes the reusable implementation surface for recurrent and
state-space kernels. It does **not** claim numerical correctness, a speed win,
occupancy, or zero executed spills. Those claims require the checked-in GPU
harnesses and Nsight target to run on the admitted device.

RG-LRU is the first slice because it is an actual public AiDotNet recurrent
operation and exposes the defining scan constraint: `h[t]` depends on
`h[t-1]`. A partial LSTM epilogue would not satisfy AiDotNet's existing
full-sequence LSTM ABI. RG-LRU lets the first blueprint demonstrate persistent
register state and exact layout admission without misrepresenting scope.

`DirectPtxRecurrentCoverageManifest` is the executable family inventory. It
assigns the public tensor operations, `IDeviceRnn` surface, and all CUDA backend
LSTM/GRU/GLA/xLSTM/GatedDelta/RG-LRU/RWKV/Mamba entry points. Only the exact
RG-LRU cell is `ExperimentalDirectPtx`; every other cell is explicitly planned.

## Mathematical and physical ABI

For every channel, starting from `h[-1] = 0`:

```text
base = sigmoid(-decay[channel])
a[t] = recurrenceGate[t,channel] * base
s[t] = sqrt(max(0, 1 - a[t]^2))
h[t] = a[t] * h[t-1] + s[t] * inputGate[t,channel] * value[t,channel]
output[t,channel] = h[t]
```

All five allocations are separate, 128-byte aligned, offset zero, contiguous,
and exact extent:

| Tensor | Physical layout | Access | Exact elements | Exact bytes |
|---|---|---:|---:|---:|
| `value` | dense `[B,S,D]` FP32 | read | 32,768 | 131,072 |
| `recurrenceGate` | dense `[B,S,D]` FP32 | read | 32,768 | 131,072 |
| `inputGate` | dense `[B,S,D]` FP32 | read | 32,768 | 131,072 |
| `decay` | dense `[D]` FP32 | read | 256 | 1,024 |
| `output` | dense `[B,S,D]` FP32 | write | 32,768 | 131,072 |

The host admission layer rejects null device pointers, over- or undersized
allocations, misalignment, every pairwise alias, address-range overflow,
unsupported dtype/layout/phase, non-SM86 devices, and every non-exact shape.
The PTX receives pointers only. It has no shape, stride, storage-offset, dtype,
layout, or loop parameters.

## Dataflow and memory proof obligations

One 256-thread block maps one thread to each recurrent channel. The 128 time
steps are emitted as straight-line PTX. Each thread:

1. reads its channel's `decay` once and computes `base`;
2. keeps `base` and recurrent `h` live in registers;
3. reads `value`, `recurrenceGate`, and `inputGate` once per step;
4. writes only the final public output for that step.

There are no global gate/state intermediates and no temporary device
allocation. Adjacent lanes address adjacent channels, so every full warp
touches one aligned 128-byte span per tensor per step. There is no shared
memory, hence no shared-bank conflict surface.

`cp.async` and Tensor Cores are intentionally absent from this specialization.
The inputs have no cross-thread reuse and the recurrence is scalar and
dependency-bound; staging them in shared memory would add synchronization and
traffic. Later projection-fused LSTM/GRU and matrix-state families should use
Tensor Cores and asynchronous tiles where their data reuse makes that choice
measurably beneficial.

The static blueprint budgets at most 32 registers/thread, zero static shared
bytes, zero local bytes/thread, and at least two active blocks/SM. Module load
fails if the CUDA JIT report violates any budget. This is only a JIT resource
gate; executed local/spill counters still require Nsight Compute.

## Runtime, cache, capture, and fallback

The per-kernel rollout switch is:

```text
AIDOTNET_DIRECT_PTX_RECURRENT_STATE=1
```

The master `AIDOTNET_DIRECT_PTX=1` also opts in. Both remain off by default.
Unsupported requests return an exact `rglru-*` reason and immediately execute
the established `rglru_scan_forward` NVRTC kernel.

The Driver-API module is keyed by exact semantics in the bounded LRU owned by
`CudaBackend`. `PrewarmDirectPtxRgLruScan` performs JIT/module creation before
capture. A cold capture is rejected. A captured module is pinned until backend
disposal, so eviction cannot invalidate a retained `CUfunction`. The recurrent
cache is disposed before the shared Driver-API runtime.

## Resident evidence harness

Build and, on a clean SM86 GPU, run:

```powershell
dotnet run --project tests/AiDotNet.Tensors.Benchmarks -c Release -- `
  --direct-ptx-rglru 3
```

The harness keeps all tensors resident and groups these equivalent FP32 lanes:

- direct PTX through a prewarmed CUDA graph;
- direct PTX resident launch;
- current AiDotNet NVRTC `rglru_scan_forward`;
- PyTorch CUDA eager RG-LRU;
- PyTorch `torch.compile(fullgraph=True, mode="max-autotune")` RG-LRU.

There is no cuDNN primitive with RG-LRU semantics, so a cuDNN LSTM/GRU result
would not be apples-to-apples and is deliberately excluded. The later LSTM and
GRU cells must add cuDNN and PyTorch `nn.LSTM`/`nn.GRU` competitors.

Each lane uses 30 warmups, 101 CUDA-event samples, ten resident launches per
sample, and three independent runs. It reports mean/median/P95/P99,
Gupdates/s, estimated GFLOPS, managed bytes/call, temporary or peak device
bytes, maximum error, and candidate registers/shared/local bytes/blocks per SM.
No value is checked into this document until the clean run completes.

## Deterministic Nsight target

```powershell
tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 -Target rglru
```

The target launches exactly `aidotnet_rglru_scan_b1_s128_d256` once. The shared
CSV verifier requires all requested local/spill/resource/occupancy metrics and
fails on nonzero executed local/spill traffic or incomplete evidence.

## Promotion table (pending GPU execution)

| Operation | Competitor | Median us | P95 us | P99 us | Mean us | Gupdates/s | GFLOPS est. | Managed B/call | Temp/peak device B | Max error | Registers | Shared B | Local B | Blocks/SM | Zero executed spills |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RG-LRU B1 S128 D256 | Direct PTX CUDA graph | pending | pending | pending | pending | pending | pending | pending | 0 expected | pending | pending | pending | pending | pending | pending |
| RG-LRU B1 S128 D256 | Direct PTX fused | pending | pending | pending | pending | pending | pending | pending | 0 expected | pending | pending | pending | pending | pending | pending |
| RG-LRU B1 S128 D256 | AiDotNet current NVRTC | pending | pending | pending | pending | pending | pending | pending | pending | pending | n/a | n/a | n/a | n/a | n/a |
| RG-LRU B1 S128 D256 | PyTorch CUDA eager | pending | pending | pending | pending | pending | pending | n/a | pending | pending | n/a | n/a | n/a | n/a | n/a |
| RG-LRU B1 S128 D256 | PyTorch compile max-autotune | pending | pending | pending | pending | pending | pending | n/a | pending | pending | n/a | n/a | n/a | n/a | n/a |

Promotion requires all three clean runs, high-precision/current-AiDotNet
correctness, at least 1.10x median speedup over the fastest eligible competitor,
candidate P95 no worse than competitor P95 +10%, zero hot managed allocation,
zero avoidable temporary VRAM, JIT local bytes zero, and complete Nsight zero
executed spill/local evidence.

## Assembly-line extension order

1. Add a coverage-manifest assignment and exact semantic/physical ABI.
2. Select a high-value exact shape and architecture; keep other cells fail-closed.
3. Emit pointer-only PTX with shape/loop constants baked in.
4. Validate exact extent, alignment, offset, layout, access, and aliasing once.
5. Add bounded cache, prewarm, cold-capture rejection, pinning, and disposal.
6. Route the existing backend API with deterministic established-kernel fallback.
7. Add pure eligibility/emitter/manifest/cache tests.
8. Add resident current/PyTorch/vendor competitors and a deterministic Nsight target.
9. Promote only after the full correctness, tail-latency, allocation, resource,
   zero-spill, and 1.10x championship gates pass.

For LSTM/GRU, the first useful fusion boundary is projection plus gate
activation plus state update; for Mamba/GLA/RWKV, it is a chunk-persistent scan
with an explicit boundary-state ABI. Chunked kernels must prove identical
results at every chunk boundary before they can replace the full-sequence
fallback.
