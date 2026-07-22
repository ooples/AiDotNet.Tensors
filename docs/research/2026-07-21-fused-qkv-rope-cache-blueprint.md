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

## Three-run NVIDIA championship evidence

Environment: Windows 10.0.26200, .NET 10.0.10 / SDK 10.0.302, RTX 3080
12 GiB, SM86, driver 610.47, Python PyTorch 2.12.1+cu130. Each cell uses 30
warmups and 101 samples. CUDA-event samples average ten resident invocations;
E2E is one public launch plus synchronization. Allocation and all input/output
copies are outside timing. `TFLOPS = 6*(H*64)^2/time`, counting three FP32
projection dot products. Bias/RoPE FLOPs are not inflated into the rate.
Maximum error is measured against FP64 projection accumulation and FP64
RoPE/cache arithmetic over the exact FP32 inputs and lookup tables.

The current AiDotNet peer is its actual resident NVIDIA sequence: one cuBLAS
packed projection, NVRTC BiasAdd, NVRTC interleaved RoPE, and three D2D copies.
Its preallocated intermediates total `8*H*64*sizeof(float)`. PyTorch is measured
both eager and as pre-captured CUDA Graph replay. The graph row is the strongest
PyTorch competitor and has zero replay allocation.

| Shape | Method | Device median us | Device P95 us | Device P99 us | E2E median us | E2E P95 us | E2E P99 us | TFLOPS | B/call | tmp MiB | max error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H4/cap16/pos0 | Direct PTX | 13.01-15.16 | 19.56-20.17 | 23.24-28.47 | 19.30-32.10 | 36.30-52.20 | 45.60-56.90 | 0.026-0.030 | 0 | 0 | 1.043e-7 |
| H4/cap16/pos0 | AiDotNet cuBLAS+NVRTC | 72.09-92.47 | 96.15-130.15 | 130.46-163.02 | 74.50-83.90 | 101.50-125.10 | 106.20-157.50 | 0.004-0.005 | 0 | 0.008 | 1.043e-7 |
| H4/cap16/pos0 | PyTorch CUDA Graph | 28.67-30.32 | 34.30-38.29 | 40.03-46.38 | 41.90-43.30 | 50.00-58.20 | 63.60-65.50 | 0.013-0.014 | n/a | 0 | 2.896e-8 |
| H8/cap64/pos17 | Direct PTX | 13.62-16.06 | 19.14-22.73 | 25.29-30.62 | 24.00-26.60 | 47.00-54.90 | 50.60-192.30 | 0.098-0.115 | 0 | 0 | 2.533e-7 |
| H8/cap64/pos17 | AiDotNet cuBLAS+NVRTC | 76.20-77.21 | 93.90-107.11 | 105.16-160.54 | 87.60-92.80 | 121.00-152.70 | 130.00-306.10 | 0.020-0.021 | 0 | 0.016 | 2.682e-7 |
| H8/cap64/pos17 | PyTorch CUDA Graph | 32.25-32.67 | 38.60-39.95 | 43.62-50.69 | 45.50-47.20 | 46.70-48.40 | 50.80-57.40 | 0.048-0.049 | n/a | 0 | 3.953e-8 |
| H16/cap128/pos127 | Direct PTX | 19.46-19.97 | 23.45-25.60 | 25.29-41.57 | 35.00-40.20 | 37.10-52.70 | 41.50-72.60 | 0.315-0.323 | 0 | 0 | 3.874e-7 |
| H16/cap128/pos127 | AiDotNet cuBLAS+NVRTC | 76.19-79.56 | 94.00-104.55 | 109.26-120.73 | 74.40-78.10 | 120.30-125.50 | 138.60-280.20 | 0.079-0.083 | 0 | 0.031 | 4.172e-7 |
| H16/cap128/pos127 | PyTorch CUDA Graph | 40.35-42.29 | 41.47-44.44 | 44.75-58.56 | 58.40-60.80 | 66.30-68.60 | 72.80-85.10 | 0.149-0.156 | n/a | 0 | 8.104e-8 |

PyTorch eager device medians are 612.86-633.46 us (H4), 605.20-619.32 us
(H8), and 598.63-620.44 us (H16); its per-call peak temporary storage is
0.006, 0.012, and 0.023 MiB respectively. The executable TUI prints all eager
and graph rows with mean, median, P95, P99, allocation, throughput, and error.

The conservative minimum paired wins across the two final three-run validation
captures were:

| Shape | vs AiDotNet device | vs AiDotNet E2E | vs strongest PyTorch device | vs strongest PyTorch E2E |
|---|---:|---:|---:|---:|
| H4 | 4.68x | 2.40x | 1.69x | 1.33x |
| H8 | 4.81x | 3.25x | 2.03x | 1.74x |
| H16 | 3.45x | 1.79x | 2.07x | 1.51x |

Every direct device P95 is also below the corresponding competitor P95, so
the `competitor P95 + 10%` gate passes without invoking the allowance.

## Resource and spill evidence

All nine measured direct cells JIT to 40 registers/thread, zero static shared
bytes, zero local bytes/thread, and 12 active 128-thread blocks per SM. The
resource budget rejects more than 48 registers, any shared/local allocation,
or fewer than eight active blocks/SM before a module can enter the cache.

The profile target emits exactly one deterministic launch for each distinct
unrolled head body: H4/cap16/pos0, H8/cap64/pos17, and H16/cap128/pos127.
Capacity and position only change baked address immediates; they do not change
the instruction types or register liveness. The H16 audit recorded:

```text
BlueprintId: qkv-projection-rope-cache-d64-v1-Ampere-decode-fp32-h16-capacity128-position127
DeviceFingerprint: gpu-79d5ac8ef419a3bce86d78a8222664cd-sm86-drv13030
PTX SHA-256: ffd494a26a8e3fc8c1e44ff3c71058a8df0a415cae48ab73c30e42eaa87bc859
registers/thread: 40
static shared: 0
local bytes/thread: 0
active blocks/SM: 12
PTX/binary version: 86/86
```

Nsight Compute 2026.2.1 successfully attaches to
`aidotnet_qkv_rope_cache_d64`, but this Windows host still returns
`ERR_NVGPUCTRPERM` before hardware-counter collection. Consequently the PR
does not claim executed SASS spill counters. Driver-JIT local size zero is an
enforced admission proof; the issue remains experimental until an
administrator enables counters and the existing four-counter CSV verifier
accepts zero register-spill instructions, local loads, local stores, and local
spilling requests.

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
