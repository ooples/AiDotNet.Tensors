# Direct PTX stochastic-kernel blueprint: issue #849

Status: experimental implementation; disabled by default; no GPU promotion evidence attached.

This document defines the reusable assembly-line contract for RNG, dropout,
sampling, and stochastic-noise kernels. The first executable slice is FP32
inverted-dropout forward because it exercises the critical concerns together:
counter ownership, exact layout admission, random generation fused into a
consumer, a backward-visible saved mask, CUDA graph lifetime, and an honest
strongest-peer benchmark.

## Scope and coverage

`DirectPtxRngCoverageManifest` is the executable inventory. It assigns:

- backend uniform, normal, secure-uniform, Gaussian-noise, dropout-mask,
  dropout forward/backward, bias-dropout, Gumbel-softmax, and NeRF importance
  sampling, plus the stochastic DDIM scheduler step;
- tensor random-uniform/range/normal/dropout-mask initialization entry points;
- `IDeviceRng` uniform, normal, and Bernoulli providers;
- categorical, multinomial, relaxed/quantized categorical, continuous,
  discrete, multivariate, composite, transform, and rejection-sampling
  distribution families;
- cryptographic randomness, which is explicitly excluded. Deterministic
  Philox must never be substituted for a secure-random guarantee.

The manifest distinguishes the implemented experimental slice from planned
and baseline-only assignments. Coverage does not imply promotion.

## Versioned RNG ABI

The first ABI is `philox4x32-10-v1`:

| Field | Mapping |
|---|---|
| 64-bit seed | Philox key words `(seed.low32, seed.high32)` |
| 64-bit counter offset | Low 64 bits of the 128-bit Philox counter |
| 64-bit subsequence | High 64 bits of the 128-bit Philox counter |
| float4 group `g` | Counter `(counterOffset + g, subsequence)` with modulo-2^64 low-counter addition |
| element `i` | Word `i mod 4` from group `floor(i / 4)` |
| dropout keep decision | Unsigned word `< floor(keepProbability * 2^32)` |

The public CUDA `Dropout` route maps its existing explicit seed to the Philox
key and uses subsequence/counter offset zero. Repeating the same exact call is
therefore bit-reproducible. Internal launch and oracle APIs expose all three
fields so future schedulers can reserve non-overlapping counter intervals.

Counter consumption is shape-derived and independent of block scheduling:
`ceil(N/4)` counters. Promoted normal, categorical, and rejection algorithms
must document their word consumption before implementation. Variable-length
rejection loops may not silently consume a shared implicit stream.

## Executable v1 physical ABI

The v1 kernel admits only:

- SM86 exactly; Ada, other Ampere SMs, Hopper, and Blackwell fail closed until
  each has independent correctness/resource/performance evidence;
- FP32 exact element counts 4,096, 65,536, or 1,048,576;
- three distinct, exact-size, 16-byte-aligned allocations for input, output,
  and the public saved mask;
- dense vector layout with zero byte offset and no overlap;
- finite dropout rate strictly between zero and one, with a representable
  integer keep threshold and finite inverse-keep scale;
- training forward only. Evaluation, zero-rate, backward, other dtypes,
  views, and all other shapes use the established backend.

Admission occurs once in `DirectPtxRngDropoutAdmission`. Successful admission
creates exact `DirectPtxTensorView` capability tokens. The emitted PTX accepts
no element count, stride, layout, dtype, alignment, alias, or tail parameter.
It contains no tail branch.

## Fused memory dataflow

One 128-thread block processes 512 adjacent FP32 values. Each thread:

1. computes one Philox4x32-10 counter entirely in scalar registers;
2. converts its four random words to four mask decisions using unsigned
   threshold predicates;
3. performs one aligned `ld.global.v4.f32` from input;
4. multiplies in registers by zero or `1 / keepProbability`;
5. performs one aligned float4 output store and one aligned float4 mask store.

There is no random-mask generation launch, no private VRAM intermediate, no
re-read of input, no shared-memory staging, and no temporary device allocation.
Shared memory, `cp.async`, and Tensor Cores are not useful for this streaming,
low-arithmetic-intensity transform. The saved mask is not avoidable under the
current public/backward contract; future fused backward consumers can keep or
regenerate it according to an explicit counter ABI.

The useful-traffic convention is 12 bytes/element: one FP32 input read, one
FP32 output write, and one required FP32 saved-mask write. PyTorch's native
boolean mask is a strong lower-traffic competitor and remains in the matrix.

## Runtime and graph-capture contract

`CudaBackend` owns a bounded shape-keyed LRU of modules. The environment gate
`AIDOTNET_DIRECT_PTX_RNG_DROPOUT=1` is snapshotted at backend construction and
is off by default. Cache misses JIT outside capture. `PrewarmDirectPtxRngDropoutF32`
provides explicit preparation. A cold capture rejects without loading a module;
a warm capture pins the module so LRU eviction cannot invalidate a graph-held
`CUfunction`. Backend disposal unloads every RNG module before its borrowed
CUDA context and stream are destroyed.

Any gate, SM, shape, extent, pointer, alignment, alias, numerical, resource,
JIT, cache, or capture failure returns to the existing two-kernel NVRTC path
with an exact diagnostic in `DirectPtxLastError`. The fallback remains the
source of truth until promotion.

## Resource rejection and evidence

Every loaded specialization produces a `DirectPtxKernelAudit` containing the
blueprint identity, PTX SHA-256, GPU/SM/driver fingerprint, function resource
attributes, block size, active blocks per SM, and JIT log. Runtime rejects:

- any nonzero local bytes per thread;
- more than 56 registers per thread;
- any static shared memory;
- fewer than eight active blocks per SM;
- a JIT max-thread limit below 128.

Driver attributes are necessary but not sufficient. Promotion also requires
the checked-in Nsight target and verifier to show zero executed register-spill,
local-load, and local-store counters for every exact shape. No such result has
been collected in this change.

There is no autotuning search in v1: it has one vector width, block geometry,
and exact specialization per admitted shape. This is a deterministic one-plan
domain, so a tuning cache would add state without alternatives. Operations
with multiple tiles, vector widths, or fusion plans must use the existing
bounded, device/driver/SM/shape/semantics-keyed plan-cache pattern.

## Deterministic non-GPU proof

The focused tests establish without launching a GPU that:

- the CPU Philox reference matches the published all-zero Philox4x32-10 vector;
- seed, subsequence, and counter select deterministic independent domains;
- the PTX emits ten rounds, baked launch geometry, aligned float4 traffic, and
  no shape/stride/tail branch;
- exact SM, shape, extent, pointer, alignment, alias, and rate admission fails
  closed with stable reasons;
- graph-pinned modules cannot be evicted and are disposed with their owner;
- every coverage-manifest family is uniquely assigned and secure randomness is
  excluded.

These tests prove emitter and contract structure, not device correctness.

## Resident benchmark and Nsight commands

After obtaining exclusive use of the admitted GPU:

```powershell
$env:AIDOTNET_DIRECT_PTX_RNG_DROPOUT='1'
dotnet run -c Release -f net10.0 --project tests/AiDotNet.Tensors.Benchmarks -- --direct-ptx-rng-dropout 3
tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 -Target rng-dropout
```

The harness preallocates all tensors/workspaces, prewarms before timing,
requires 30 warmups plus 101 samples for each of three independent runs, and
groups these resident NVIDIA methods per shape:

- Direct PTX fused launch;
- Direct PTX CUDA-graph replay;
- current AiDotNet NVRTC mask generation plus multiply;
- PyTorch `native_dropout` eager;
- PyTorch CUDA graph;
- PyTorch `torch.compile(mode="max-autotune", fullgraph=True)`.

It reports device and synchronized end-to-end mean/median/P95/P99, effective
GB/s, managed bytes/call, temporary/peak device bytes, maximum semantic error,
registers, shared/local bytes, and active blocks/SM. The environment fingerprint
and exact audit JSON must be retained with the run artifacts.

## Promotion table (intentionally pending)

| Shape | Method | Device median | P95 | P99 | GB/s | Managed B/call | Temporary/peak device B | Max error | Regs | Shared B | Local B | Blocks/SM | Spill/local counters |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N=4,096 | Direct PTX / graph / current AiDotNet / PyTorch eager/graph/compile | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| N=65,536 | Direct PTX / graph / current AiDotNet / PyTorch eager/graph/compile | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| N=1,048,576 | Direct PTX / graph / current AiDotNet / PyTorch eager/graph/compile | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

No cell may be filled from a different machine, CPU competitor, partial run, or
screening result. Promotion requires the issue gate: at least 1.10x candidate
median speedup over the strongest eligible peer, candidate P95 within 10% of
that peer, zero hot managed allocation, zero avoidable temporary VRAM,
operation-specific correctness, zero local bytes, and zero executed spill/local
evidence.

## Assembly-line extensions

Each next stochastic PR follows the same order:

1. add every entry point to the manifest before implementation;
2. freeze the mathematical and counter-consumption semantics;
3. choose exact logical/physical layouts and fusion boundaries;
4. add fail-closed admission and a disabled per-kernel gate;
5. emit architecture- and exact-shape-specific PTX with no dynamic layout path;
6. attach a bounded module/plan cache, prewarm, pinning, disposal, and audit;
7. add CPU emitter/admission/determinism tests and high-precision GPU oracles;
8. add resident strongest-peer and deterministic Nsight harnesses;
9. run three clean benchmark suites and populate the grouped evidence table;
10. promote only the winning, zero-spill cells; leave every other contract on
    the established backend with its exact rejection reason.
