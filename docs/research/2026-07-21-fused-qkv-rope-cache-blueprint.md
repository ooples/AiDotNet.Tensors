# Direct-PTX QKV projection, RoPE, and KV-cache blueprint

Date: 2026-07-21

Tracking issue: #835

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Verdict

The first issue-#835 golden specialization is implemented and passes its
production championship gate on an NVIDIA GeForce RTX 3080 (SM86):

```text
one resident FP32 decode token [H*64]
  + output-major packed FP32 QKV weights [3,H,64,H*64]
  + packed FP32 bias [3,H,64]
  + interleaved RoPE tables [capacity,32]
  -> rotated Q [1,H,1,64]
  -> in-place dense K/V cache row [position,H,64]
```

The declared buckets are `H in {4,8,16}`, cache capacity in
`{16,32,64,128}`, D=64, and a baked position inside the selected capacity.
The kernel is GA102/SM86-only and disabled by default behind
`AIDOTNET_DIRECT_PTX_QKV_ROPE_CACHE=1` (or the master direct-PTX gate).

This increment is the reference assembly-line example, not closure of all of
#835. Quantized projection, multi-token prefill, paged-cache writes, standalone
RoPE, training/backward graph fusion, recordable-node substitution, and
Ada/Hopper/Blackwell specializations remain explicitly assigned in the
executable coverage manifest rather than being silently treated as supported.

## Fusion boundary and global-memory ledger

One warp owns one `(head, adjacent-feature-pair)`. Its 32 lanes stream the
input dimension in stride-32 chunks. For the owned feature pair, every lane
accumulates six scalars: Q-even/Q-odd, K-even/K-odd, and V-even/V-odd. Warp
shuffle trees reduce those six accumulators. Lane zero adds the six biases,
loads one cosine/sine pair, rotates Q and K, and commits three two-float stores.

| Allocation | Global reads | Global writes | Notes |
|---|---:|---:|---|
| input | one scalar/lane per unrolled K step and output-pair warp | 0 | warp loads are contiguous 128-byte spans; L1/L2 serve reuse between output warps |
| packed weights | six scalars/lane per unrolled K step | 0 | each warp reads six contiguous output rows; output-major rows are canonical and aligned |
| packed bias | six scalars per output-pair warp | 0 | vector-pair loads after reduction |
| cosine/sine | one scalar each per output-pair warp | 0 | baked position, adjacent-pair index |
| query | 0 | one adjacent FP32 pair | exact dense FlashDecode query ABI |
| key cache | 0 | one adjacent FP32 pair at baked row | untouched cache rows remain live |
| value cache | 0 | one adjacent FP32 pair at baked row | V is not rotated |

No packed projection, split Q/K/V, head-transpose, rotated K/V, scratch, or
metadata tensor is written. All six dot-product accumulators and both rotated
pairs remain in registers. There is no shared memory, so shared-bank conflicts
are impossible. The dominant input/weight transactions are warp-coalesced;
the final 8-byte stores are sparse because only the warp owner commits them,
but they write each final element exactly once.

`cp.async` and Tensor Cores are not used in this FP32 M=1 GEMV specialization.
Shared staging did not solve a material bottleneck at these sizes, and TF32
Tensor Core arithmetic would change the FP32 numerical contract. The selected
warp-reduction dataflow beat the cuBLAS/NVRTC composition across every bucket.
Future FP16/BF16 prefill is a separate tiled Tensor Core blueprint, not a
runtime branch in this kernel.

## Formal physical ABI

The launch takes eight pointers and no scalar arguments. Heads, D, capacity,
position, all byte strides, loop trip counts, and launch geometry are emitted
as constants.

| Tensor | Physical layout | Extent | Access | Extent mode | Alignment |
|---|---|---|---|---|---:|
| input | `Vector` | `[H*64]` | read | exact | 16 B |
| packed QKV weights | `PackedQkvWeights` | `[3,H,64,H*64]` | read | exact | 16 B |
| QKV bias | `PackedQkvBias` | `[3,H,64]` | read | exact | 16 B |
| cosine | `RowMajor2D` | `[capacity,32]` | read | exact | 16 B |
| sine | `RowMajor2D` | `[capacity,32]` | read | exact | 16 B |
| query | `Bhsd` | `[1,H,1,64]` | write | exact | 16 B |
| key cache | `SequenceHeadDim` | `[capacity,H,64]` | read-write | exact | 16 B |
| value cache | `SequenceHeadDim` | `[capacity,H,64]` | read-write | exact | 16 B |

The dispatch boundary checks exact byte extent, dtype/layout capability token,
alignment, supported bucket, cache position, and output aliasing once. The PTX
contains no stride, layout, offset, shape, optional-bias, or bounds mode. An
unsupported contract falls back to the established cuBLAS/NVRTC composition
with an exact diagnostic reason.

The boundary is fail-closed and uses stable reason codes before any PTX module
lookup or launch:

| Rejected contract | Diagnostic reason |
|---|---|
| feature switch off | `qkv-rope-cache-feature-disabled` |
| CUDA backend unavailable | `qkv-rope-cache-backend-unavailable` |
| architecture other than measured SM86 | `qkv-rope-cache-architecture-not-implemented` |
| H outside `{4,8,16}` | `qkv-rope-cache-head-count-not-implemented` |
| capacity outside `{16,32,64,128}` | `qkv-rope-cache-capacity-not-implemented` |
| position outside `[0,capacity)` | `qkv-rope-cache-position-out-of-range` |
| null buffer | `qkv-rope-cache-null-buffer` |
| non-exact physical extent | `qkv-rope-cache-physical-extent-mismatch` |
| null device pointer | `qkv-rope-cache-invalid-device-pointer` |
| non-16-byte-aligned pointer | `qkv-rope-cache-alignment-mismatch` |
| overlapping input/output or output/output ranges | `qkv-rope-cache-alias-not-supported` |

Prewarm and dispatch share the same shape/architecture validator. Tests also
execute the established GPU composition with the feature disabled and with an
unsupported H/capacity pair, proving that rejection is a correct fallback and
not merely a skipped launch.

The public CUDA backend entry point is
`CudaBackend.QkvProjectionRoPECacheD64`. It first attempts the gated PTX path
and otherwise performs packed cuBLAS projection, NVRTC bias and RoPE, then the
three required D2D writes. The PTX output/cache layouts are already the dense
FlashDecode ABI, so the next attention operation needs no split, reshape,
transpose, or cache copy.

## Runtime, cache, and capture contract

The specialization key is `(heads,capacity,position)`. Modules live in the
bounded direct-PTX LRU cache, are disposed with the CUDA backend, and carry a
`DirectPtxKernelAudit` containing the blueprint id, PTX SHA-256, GPU/SM/driver
fingerprint, JIT resource attributes, launch geometry, active blocks/SM, and
JIT log. Prewarm is mandatory before CUDA graph capture; capture never emits
PTX, loads a module, allocates, tunes, performs I/O, or evicts a live module.

Tests cover first/middle/last position, H=4/8/16, exact-extent rejection,
unsupported buckets, public routing, 0 B/call after prewarm, graph
capture/replay, numerical agreement, cache-row preservation, audit lookup, and
module disposal through the established cache owner.

`CudaBackend.EnqueueCapturedGraph` replays an instantiated graph without a host
barrier. The caller owns completion and buffer lifetime. This lets the public
QKV route be captured once and replayed with the same asynchronous contract as
PyTorch CUDA Graph, while the existing `LaunchCapturedGraph` retains its
synchronized whole-training-step contract. Tests warm the native call boundary
and then prove 0 managed B across 32 asynchronous replays.

## Three-run NVIDIA championship evidence

Environment: Windows 10.0.26200, .NET 10.0.10 / SDK 10.0.302, RTX 3080
12 GiB, SM86, driver 610.47, Python 3.14.6, PyTorch 2.12.1+cu130 / CUDA
13.0. The exact clean performance commit is
`3759138f077987b98a61f465f25194beb00832f0`. Every cell uses 30 warmups and
101 samples in each of three independent processes. CUDA-event samples average
ten resident invocations; E2E is one enqueue/launch plus synchronization.
Allocation and all input/output copies are outside timing.
`TFLOPS = 6*(H*64)^2/time`, counting three FP32 projection dot products.
Bias/RoPE FLOPs are not inflated into the rate. Maximum error is measured
against FP64 projection accumulation and FP64 RoPE/cache arithmetic over the
exact FP32 inputs and lookup tables.

The current AiDotNet peer is its actual resident NVIDIA sequence: one cuBLAS
packed projection, NVRTC BiasAdd, NVRTC interleaved RoPE, and three D2D copies.
Its preallocated intermediates total `8*H*64*sizeof(float)`. PyTorch is measured
both eager and as pre-captured CUDA Graph replay. Direct PTX is likewise shown
as asynchronous graph replay and as an uncaptured call through the public
backend entry point. The graph rows are the promotion comparison; both graph
replays have zero temporary allocation.

Aggregation retains all three medians in run order. P95/P99 are the worst run,
mean is the mean of run means, TFLOPS is the middle of the three measured rates,
and allocation/temporary/error/resource values are maxima. Within each shape,
the best timing/rate/error cell is bolded independently rather than assuming
that one method won every statistic.

### H4 / capacity 16 / position 0

| Method | device median R1/R2/R3 us | dev P95 | dev P99 | dev mean | E2E median R1/R2/R3 us | E2E P95 | E2E P99 | E2E mean | TFLOPS | managed B | temp/peak B | max error | resources |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Direct PTX CUDA Graph — WINNER** | **10.65/11.67/10.14** | 35.74 | 82.33 | **13.22** | **21.70/11.00/18.00** | 76.30 | 274.30 | **24.29** | **0.037** | **0** | **0** | 2.178e-8 | 40r/0s/0l/12occ |
| PyTorch CUDA Graph *(strongest external)* | 28.16/29.18/29.08 | 55.50 | 103.01 | 32.23 | 40.30/39.40/88.80 | 102.50 | **227.50** | 53.03 | 0.014 | n/a | **0** | **1.735e-8** | n/a |
| Direct PTX fused, public eager | 15.85/16.18/15.56 | **32.46** | **44.54** | 16.86 | 22.60/24.90/21.50 | **49.90** | 282.00 | 32.37 | 0.025 | **0** | **0** | 2.178e-8 | 40r/0s/0l/12occ |
| AiDotNet cuBLAS+NVRTC | 79.56/73.42/72.60 | 137.42 | 172.13 | 81.30 | 75.80/81.90/80.90 | 122.20 | 264.80 | 82.61 | 0.005 | 0 | 8192 | 2.275e-8 | n/a |
| PyTorch CUDA eager | 540.05/692.12/608.87 | 894.46 | 1048.58 | 640.09 | 501.10/598.60/686.00 | 1345.70 | 1597.30 | 640.15 | 0.001 | n/a | 6144 | **1.735e-8** | n/a |

Conservative per-run gate minimum: **6.29x device / 3.49x E2E over current
AiDotNet**, and **2.50x device / 1.86x E2E over PyTorch CUDA Graph**. Maximum
paired device-P95 ratio: **0.772**.

### H8 / capacity 64 / position 17

| Method | device median R1/R2/R3 us | dev P95 | dev P99 | dev mean | E2E median R1/R2/R3 us | E2E P95 | E2E P99 | E2E mean | TFLOPS | managed B | temp/peak B | max error | resources |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Direct PTX CUDA Graph — WINNER** | **10.96/10.44/10.44** | **30.72** | **37.68** | **12.27** | **22.20/18.80/18.40** | **45.10** | 245.70 | **23.39** | **0.151** | **0** | **0** | **2.561e-8** | 40r/0s/0l/12occ |
| PyTorch CUDA Graph *(strongest external)* | 30.01/28.26/33.38 | 53.66 | 97.08 | 33.34 | 43.70/47.90/49.40 | 71.90 | 240.40 | 50.59 | 0.052 | n/a | **0** | 3.953e-8 | n/a |
| Direct PTX fused, public eager | 16.28/14.94/14.95 | 31.13 | 43.32 | 16.72 | 26.60/26.10/27.10 | 47.90 | **211.40** | 32.83 | 0.105 | **0** | **0** | **2.561e-8** | 40r/0s/0l/12occ |
| AiDotNet cuBLAS+NVRTC | 78.54/74.24/75.17 | 111.00 | 133.43 | 80.29 | 91.70/85.10/79.40 | 247.00 | 320.60 | 97.85 | 0.021 | 0 | 16384 | 4.428e-8 | n/a |
| PyTorch CUDA eager | 574.87/646.25/553.78 | 950.99 | 1147.60 | 623.82 | 519.40/631.70/513.10 | 1130.80 | 2480.60 | 610.82 | 0.003 | n/a | 12288 | 3.953e-8 | n/a |

Conservative per-run gate minimum: **7.11x device / 4.13x E2E over current
AiDotNet**, and **2.71x device / 1.97x E2E over PyTorch CUDA Graph**. Maximum
paired device-P95 ratio: **0.669**.

### H16 / capacity 128 / position 127

| Method | device median R1/R2/R3 us | dev P95 | dev P99 | dev mean | E2E median R1/R2/R3 us | E2E P95 | E2E P99 | E2E mean | TFLOPS | managed B | temp/peak B | max error | resources |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Direct PTX CUDA Graph — WINNER** | **18.64/17.92/17.82** | **29.39** | 60.93 | **19.34** | **38.60/33.40/32.30** | 108.80 | 267.00 | 44.65 | **0.351** | **0** | **0** | **4.458e-8** | 40r/0s/0l/12occ |
| PyTorch CUDA Graph *(strongest external)* | 39.94/40.96/40.55 | 63.08 | 132.92 | 43.33 | 56.80/66.00/56.50 | 93.10 | 319.10 | 65.32 | 0.155 | n/a | **0** | 8.104e-8 | n/a |
| Direct PTX fused, public eager | 26.83/18.94/18.84 | 47.21 | **59.08** | 23.21 | 50.30/37.20/33.80 | **69.50** | **254.60** | **44.08** | 0.332 | **0** | **0** | **4.458e-8** | 40r/0s/0l/12occ |
| AiDotNet cuBLAS+NVRTC | 113.77/73.42/67.89 | 163.33 | 205.62 | 89.78 | 126.60/69.60/71.50 | 413.60 | 471.10 | 108.39 | 0.086 | 0 | 32768 | 7.281e-8 | n/a |
| PyTorch CUDA eager | 528.59/675.53/631.81 | 1092.51 | 1215.80 | 644.42 | 507.40/729.30/717.20 | 1112.50 | 2127.20 | 694.28 | 0.010 | n/a | 24576 | 8.104e-8 | n/a |

Conservative per-run gate minimum: **3.81x device / 2.08x E2E over current
AiDotNet**, and **2.14x device / 1.47x E2E over PyTorch CUDA Graph**. Maximum
paired device-P95 ratio: **0.567**.

The release runner produced no rejected environment attempts. Its manifest
SHA-256 is
`616f9489263828425aef569c825d021d113b79122f41b6209fcef3b6146f888f`;
the manifest contains the SHA-256 of every raw build, .NET, PyTorch, and gate
log. The gate JSON itself is
`fcff002be20d3d361acc1bcbf0f2ddd96fcbd4690cf9dfa913be160cf68d82e9`.

## Resource and spill evidence

All nine clean process-specialization audits JIT to 40 registers/thread, zero
static shared bytes, zero local bytes/thread, and 12 active 128-thread blocks
per SM. Graph and eager rows share the identical module. The resource budget
rejects more than 48 registers, any shared/local allocation, or fewer than
eight active blocks/SM before a module can enter the cache.

The profile target emits exactly one deterministic launch for each distinct
unrolled head body: H4/cap16/pos0, H8/cap64/pos17, and H16/cap128/pos127.
Capacity and position only change baked address immediates; they do not change
the instruction types or register liveness.

Nsight Compute CLI 2026.2.1 profiled exact clean kernel commit `3759138f` from
an elevated release-verification process. The repository verifier required all
three raw rows and every requested metric column; missing rows or metrics fail
closed.

| Family | grid | block | PTX SHA-256 | regs | static/dynamic shared B | JIT local B/thread | executed local inst | local loads | local stores | active warps % |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| H4 | 32 | 128 | `f13e11c2d8ff01b988c1509094aff03b721301b455bc5c11d59aff1231ae4b7f` | 40 | 0/0 | 0 | **0** | **0** | **0** | 8.28 |
| H8 | 64 | 128 | `d805151e7b28d3e3fdcbc65788da6ec74cbc3e6b2d223911e5d5cd618e86cfd5` | 40 | 0/0 | 0 | **0** | **0** | **0** | 8.26 |
| H16 | 128 | 128 | `a6bb887f31d8358361a2b32de87b8678f784af524e99793abb288e8fda7efc75` | 40 | 0/0 | 0 | **0** | **0** | **0** | 15.03 |

Executed counters are `smsp__sass_inst_executed_op_local.sum`,
`smsp__sass_inst_executed_op_local_ld.sum`, and
`smsp__sass_inst_executed_op_local_st.sum`. All are zero on every launch. Raw
CSV SHA-256:
`40f881e26b6d17b5cf7ff9300a4737492c3b1bf1e65a4bdaf7804b06f4b6ea55`.
Normal PTX execution and timing do not require counter elevation.

## Reproduction

```powershell
$evidence = Join-Path $env:TEMP 'aidotnet-qkv-release-evidence'
powershell -ExecutionPolicy Bypass -File `
  tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-release-evidence.ps1 `
  -Runs 3 -Issue835Only -OutputDirectory $evidence
```

The runner creates separate clean .NET and PyTorch processes for each run,
hashes every raw log, and writes `qkv-release-gate.json` only after joining the
rows and enforcing the error/resource/allocation/median/P95 gate against every
competitor.

For deterministic Nsight attachment after the Release build:

```powershell
$env:NSIGHT_COMPUTE_CLI = '<path-to-ncu.exe>'
powershell -ExecutionPolicy Bypass -File `
  tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target qkv-rope-cache -OutputCsv (Join-Path $env:TEMP 'aidotnet-qkv-ncu.csv')
```

The profile target filters `aidotnet_qkv_rope_cache_d64`, emits one launch per
head-codegen family, prints each complete JIT audit, and fails closed unless
all required Nsight counter columns and all three launch rows are present.

## Next assembly-line increments

1. FP16/BF16 multi-token prefill with architecture-specific `cp.async` and
   Tensor Core tiles, direct online-attention handoff, and no packed QKV write.
2. Paged-cache projection/RoPE write with the block-table contract baked into
   its own ABI rather than a dense/paged hot-loop branch.
3. Quantized packed projection specializations with explicit scale/zero-point
   tensor contracts and accumulation tolerance.
4. Recorded decoder-plan fusion so graph construction selects a prewarmed
   fused node instead of recording projection, RoPE, and cache-copy nodes.
5. Separate Ada, Hopper, and Blackwell emitters and evidence. No architecture
   is promoted from Ampere measurements by assumption.
