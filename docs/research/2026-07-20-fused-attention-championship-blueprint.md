# Direct-PTX online-attention kernel blueprint

Date: 2026-07-20; productionization pass 2026-07-21
Baseline: `b386d35` plus the working-tree experiment described here

## Verdict

The forward-inference experiment is implemented end to end for NVIDIA SM80+
and the declared shape family:

```text
FP16 Q/K/V, dense BHSD, S in {16,32,64,128}, D=64
  -> scaled attention, unmasked or causal
  -> optional head-local LayerNorm(D=64) + affine + tanh-GELU
  -> FP32 BHSD output
```

The final wide benchmark is GPU-only. It compares the direct PTX kernel with
the current AiDotNet CUDA/NVRTC implementation, a direct cuBLAS Tensor Core
composition, and PyTorch SDPA with its exposed Flash and Math backends forced
one at a time. Intel MKL and OpenBLAS are deliberately absent: they are CPU
BLAS libraries, not NVIDIA GPU kernels. The NVIDIA analogue for this
experiment is cuBLAS.

On the RTX 3080 used here, the fused direct-PTX path wins steady-state median
in every tested attention-plus-epilogue cell from decode through saturation.
The attention-only path wins every cell in the representative capture below,
but its saturated unmasked BH512 result is a 1% near tie and has reversed in
repeated one-launch host captures on this display GPU. The fused experiment
therefore clears the competition gate; the attention-only saturation case
remains an optimization lane rather than a robust universal win. These claims
apply only to this GPU, dtype, shape family, and semantic contract—not other
GPUs, dimensions, dropout, arbitrary masks, or backward.

## Apples-to-apples contracts

Two lanes are reported separately.

| Lane | Required result | FLOP accounting |
|---|---|---:|
| Attention | FP16 Q/K/V -> FP32 `softmax(QK^T * scale + mask)V` | `4 * BH * S^2 * D` |
| Attention + epilogue | The same FP32 attention, then row-wise LayerNorm over D=64, affine, and tanh-GELU | Attention FLOPs only, labeled effective TFLOPS |

All steady-state rows use resident GPU inputs and outputs. Setup, allocation,
host/device copies, module JIT, and PyTorch/AiDotNet construction are outside
the timed region. Each invocation includes its framework or Driver-API launch
and a completion synchronization. PyTorch SDPA's native FP16 result is
converted to FP32 inside the timed action. TorchSharp 0.106 does not expose
GELU's approximation selector, so the PyTorch epilogue spells out the same
tanh-GELU equation with PyTorch GPU tensor operations.

The inference PTX specialization does not emit log-sum-exp statistics. The
production AiDotNet inference route uses the API-compatible specialization,
which does emit the existing API's FP32 statistics. The current AiDotNet benchmark
therefore performs a small additional documented side effect. No public
PyTorch SDPA API in this package returns matching LSE output.

`TFLOPS` counts the two attention matrix products. It does not invent FLOPs
for softmax, mask predicates, normalization, or activation. `B/call` is
managed allocation on the calling thread. `tmp MiB` is known intermediate
VRAM, excluding inputs and final outputs.

## Exact championship cell

System: NVIDIA GeForce RTX 3080, SM 8.6, driver 610.47, Windows display GPU.
Shape: `[BH=128,S=128,D=64]`, FP16 Q/K/V, FP32 output.
Samples: 101 after 30 warmups. CUDA-event device samples average ten launches;
end-to-end samples are one synchronized invocation each.

```text
Mode      Method                              median us     p95 us     p99 us    mean us    TFLOPS   B/call  tmp MiB    max err  regs  local B
-----------------------------------------------------------------------------------------------------------------------------------------------
unmasked  Direct PTX attention [device]           33.69      34.82      79.67      35.51    15.936        0    0.000  7.081e-6     66        0
unmasked  Direct PTX attention [E2E]              46.10      50.50     251.10      51.69    11.646        0    0.000  7.081e-6     66        0
unmasked  cuBLAS+PTX softmax [device]             51.30      80.28      93.90      54.39    10.465        0   12.000  1.052e-5    n/a      n/a
unmasked  cuBLAS+PTX softmax [E2E]                69.60     118.20     153.70      78.93     7.714        0   12.000  1.052e-5    n/a      n/a
unmasked  Direct PTX + LN + GELU [device]         38.60      66.05      89.91      41.23    13.907        0    0.000  5.491e-4     66        0
unmasked  Direct PTX + LN + GELU [E2E]            52.90      63.70     308.50      62.78    10.149        0    0.000  5.491e-4     66        0
unmasked  AiDotNet NVRTC attention [E2E]        1175.30    1578.20    1788.40    1261.20     0.457        0    0.062   2.980e-8    n/a      n/a
unmasked  AiDotNet NVRTC + LN + GELU [E2E]      1259.90    1614.70    2068.30    1321.81     0.426       40    8.188   5.603e-6    n/a      n/a
unmasked  PyTorch Flash-SDPA FP32 [E2E]           64.10     132.20     138.80      72.38     8.376       96      n/a   1.214e-5    n/a      n/a
unmasked  PyTorch Flash + LN + GELU [E2E]        406.00    1681.90    2115.90     811.50     1.322      704      n/a   9.871e-4    n/a      n/a
causal    Direct PTX attention [device]           25.60      53.04      59.90      28.84    20.972        0    0.000  2.855e-5     80        0
causal    Direct PTX attention [E2E]              39.80     129.40     326.60      64.62    13.489        0    0.000  2.855e-5     80        0
causal    cuBLAS+PTX softmax [device]             51.51      96.15     157.49      61.22    10.423        0   12.000  3.869e-5    n/a      n/a
causal    cuBLAS+PTX softmax [E2E]                71.30     242.60     293.70     104.13     7.530        0   12.000  3.869e-5    n/a      n/a
causal    Direct PTX + LN + GELU [device]         26.73      36.76      46.80      28.25    20.088        0    0.000  5.491e-4     80        0
causal    Direct PTX + LN + GELU [E2E]            41.20      43.70      45.40      41.42    13.031        0    0.000  5.491e-4     80        0
causal    AiDotNet NVRTC attention [E2E]         912.30    1137.60    1527.30     943.25     0.588        0    0.062   2.980e-8    n/a      n/a
causal    AiDotNet NVRTC + LN + GELU [E2E]      1025.10    1280.00    1529.30    1061.36     0.524       40    8.188   5.722e-6    n/a      n/a
causal    PyTorch Flash-SDPA FP32 [E2E]          102.30     119.50     131.60     102.53     5.248       96      n/a   6.071e-5    n/a      n/a
causal    PyTorch Flash + LN + GELU [E2E]        371.50    1598.10    1794.40     766.39     1.445      704      n/a   1.247e-3    n/a      n/a
```

The direct kernel's device median is 1.52x faster than the decomposed cuBLAS
floor unmasked and 2.01x faster causal. Its synchronized attention median is
1.39x/2.57x faster than forced PyTorch Flash-SDPA. The fused median is
7.67x/9.02x faster than PyTorch's equivalent tanh-GELU composition and
23.82x/24.88x faster than current AiDotNet.

The RTX 3080 is also driving the desktop. Occasional Windows GPU scheduling
outliers are visible in host p99 (for example, the unmasked direct-attention
E2E row), while its CUDA-event p99 remains below the cuBLAS device p99. Tail
values are reported rather than removed.

## Wide shape matrix

The matrix covers six occupancy regimes, both mask modes, and both semantic
lanes. The following is the compact championship view from a representative
serialized capture; `best competitor` is the lowest median among cuBLAS composition,
current AiDotNet, forced PyTorch Flash-SDPA, and forced PyTorch Math-SDPA.

| Shape | Mode | Lane | Direct median us | Direct p95 | Direct p99 | Direct TFLOPS | Best competitor | Competitor median us | Speedup |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| BH12 S16 | plain | attention | 21.40 | 41.95 | 44.25 | 0.037 | cuBLAS+PTX | 42.50 | 1.99x |
| BH12 S16 | causal | attention | 17.80 | 19.35 | 21.70 | 0.044 | cuBLAS+PTX | 45.50 | 2.56x |
| BH12 S32 | plain | attention | 19.20 | 35.40 | 44.10 | 0.164 | cuBLAS+PTX | 43.50 | 2.27x |
| BH12 S32 | causal | attention | 17.80 | 33.65 | 41.85 | 0.177 | PyTorch Math | 52.10 | 2.93x |
| BH12 S64 | plain | attention | 21.20 | 40.20 | 73.40 | 0.594 | cuBLAS+PTX | 49.40 | 2.33x |
| BH12 S64 | causal | attention | 20.20 | 39.70 | 45.60 | 0.623 | PyTorch Math | 52.00 | 2.57x |
| BH12 S128 | plain | attention | 35.40 | 56.15 | 97.15 | 1.422 | cuBLAS+PTX | 48.00 | 1.36x |
| BH12 S128 | causal | attention | 27.70 | 48.75 | 107.45 | 1.817 | cuBLAS+PTX | 46.80 | 1.69x |
| BH128 S128 | plain | attention | 48.00 | 51.80 | 54.75 | 11.185 | PyTorch Math | 65.20 | 1.36x |
| BH128 S128 | causal | attention | 39.00 | 56.70 | 79.65 | 13.766 | PyTorch Flash | 64.20 | 1.65x |
| BH512 S128 | plain | attention | 130.60 | 133.40 | 138.05 | 16.443 | PyTorch Flash | 132.10 | 1.01x |
| BH512 S128 | causal | attention | 91.40 | 262.80 | 350.90 | 23.495 | PyTorch Math | 118.00 | 1.29x |
| BH12 S16 | plain | attn+epi | 18.50 | 31.05 | 39.75 | 0.043 | AiDotNet NVRTC | 55.80 | 3.02x |
| BH12 S16 | causal | attn+epi | 22.40 | 48.90 | 66.55 | 0.035 | AiDotNet NVRTC | 55.30 | 2.47x |
| BH12 S32 | plain | attn+epi | 18.90 | 39.05 | 40.50 | 0.166 | AiDotNet NVRTC | 82.60 | 4.37x |
| BH12 S32 | causal | attn+epi | 23.00 | 49.70 | 57.60 | 0.137 | AiDotNet NVRTC | 80.90 | 3.52x |
| BH12 S64 | plain | attn+epi | 21.50 | 38.05 | 45.30 | 0.585 | AiDotNet NVRTC | 129.20 | 6.01x |
| BH12 S64 | causal | attn+epi | 20.80 | 41.55 | 44.30 | 0.605 | AiDotNet NVRTC | 132.90 | 6.39x |
| BH12 S128 | plain | attn+epi | 32.10 | 32.85 | 35.20 | 1.568 | AiDotNet NVRTC | 238.90 | 7.44x |
| BH12 S128 | causal | attn+epi | 35.40 | 69.50 | 163.15 | 1.422 | AiDotNet NVRTC | 241.70 | 6.83x |
| BH128 S128 | plain | attn+epi | 52.40 | 56.55 | 59.65 | 10.246 | PyTorch Math | 304.60 | 5.81x |
| BH128 S128 | causal | attn+epi | 39.00 | 40.10 | 43.70 | 13.766 | PyTorch Math | 275.20 | 7.06x |
| BH512 S128 | plain | attn+epi | 135.70 | 182.35 | 320.05 | 15.825 | PyTorch Math | 804.70 | 5.93x |
| BH512 S128 | causal | attn+epi | 103.00 | 114.25 | 255.30 | 20.849 | PyTorch Flash | 809.50 | 7.86x |

The executable prints every competitor row with median, p95, p99, mean,
effective TFLOPS, managed allocation, and known temporary VRAM. The table
above is intentionally a compact winner view; it does not replace the raw TUI.
Repeated 101-sample, one-launch host captures under desktop load occasionally
reverse the closest attention-only cells because scheduler phase changes by
more than their margin. They did not erase the fused lane's 2.47x--7.86x
median advantage. Use the CUDA-event championship cell for raw device work,
and an otherwise idle GPU for release-gate host distributions.

## Kernel dataflow

```text
dense aligned FP16 BHSD Q/K/V in global memory
                     |
                     v
          cp.async 16-byte transactions
                     |
                     v
  warp-private Q tiles + CTA-shared double-buffered K/V tiles
                     |
                     v
       mma.sync.m16n8k16 QK fragments in FP32 registers
                     |
                     v
 compile-time causal mask + online max/sum recurrence
 (fully masked future tiles are skipped by each causal warp)
                     |
                     v
       mma.sync.m16n8k16 PV; 16x64 output stays in registers
                     |
                     +--> optional row LayerNorm + affine + tanh-GELU
                     |
                     v
              one final FP32 output store
```

For S=128, all eight 16-row query warps for one head share a CTA. The K/V tile
is fetched once per CTA, double buffered, and reused by all eight warps. The
first online tile omits a provably useless zero-accumulator rescale; later row
rescaling is predicated on the running maximum actually changing.

"Avoid VRAM traffic" means avoiding intermediate global-memory traffic. Q,
K, V must be read from global memory and the final output must be committed.
The kernel never materializes an SxS score or probability tensor in global or
shared memory. Its online max, sum, score fragments, and 16x64 output fragment
remain in registers. The inference specialization writes only the output; the
API-compatible inference specialization additionally writes the required LSE stats.

## Runtime and feature gate

The custom path uses `nvcuda.dll` Driver API calls directly:

```text
C# shape/fusion specialization -> emitted PTX
  -> cuModuleLoadDataEx (driver JIT PTX to GPU-specific SASS)
  -> cuModuleGetFunction
  -> cuLaunchKernel on AiDotNet's existing stream/context
```

It does not use CUDA Runtime, NVRTC, cuBLAS, or cuDNN for the custom kernel.
It cannot bypass the NVIDIA driver: the driver owns context creation, PTX JIT,
memory, stream submission, and execution. PTX is a virtual ISA close to the
hardware, not a hand-encoded SASS binary.

Production integration is fail-closed:

- The default is off. Set `AIDOTNET_DIRECT_PTX_ATTENTION=1` to opt in.
- SM < 8.0, unsupported shape/dtype/layout, bias, active autodiff tape, stream
  capture, JIT failure, or launch failure falls back to the existing path.
- The runtime borrows the existing `CudaBackend` context and stream, so
  resident AiDotNet buffers require no copy or context bridge.
- Modules are cached by BH, S, mask, fusion signature, scale bits, and epsilon
  bits, and unloaded before the backend destroys its context/stream.
- The kernel cache and launches use the backend's existing dispatch locks.
- Stream capture stays on the established resident NVRTC path because lazy
  PTX JIT and eager scratch allocation are not capture safe. A later graph API
  should prewarm the specialization and provide stable output buffers first.

## Canonical physical-layout policy

Stride checks should not exist inside a machine-code hot loop. They cannot be
eliminated from an ergonomic public tensor API altogether: someone must prove
that the pointer satisfies the kernel contract. The correct boundary is one
dispatch-time proof followed by a stride-free capability token.

The implemented token is `DirectPtxTensorView`:

- dense row-major `[batch,head,sequence,dimension]` (`BHSD`);
- zero logical storage offset and exact logical backing extent at the tensor
  boundary;
- FP16 physical Q/K/V and FP32 output/stats/gamma/beta;
- at least 16-byte device-pointer alignment;
- sufficient byte extent and dtype-compatible extent;
- no stride fields and no shape fields passed to PTX.

Shape, offsets, trip counts, mask mode, scale, and fusion choices are baked
into the PTX specialization. Public `Tensor<float>` input is currently
resolved to persistent/resident FP16 storage once at the dispatch boundary.
The longer-term tensor design should preserve logical views for usability but
attach an immutable physical-layout proof (`layout id`, dtype, alignment,
storage generation, extent) to resident allocations. View/transpose changes
invalidate the proof. A compiled plan can then propagate the proof without
rechecking every invocation; unsupported views are materialized once or sent
to a general fallback.

## Zero-spill proof

Module admission calls `cuFuncGetAttribute` after the driver has JIT-compiled
PTX to the installed GPU. If `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` is nonzero,
`GetFunction` rejects the specialization and AiDotNet falls back. It is not a
benchmark-only assertion.

The final S=128 functions on this RTX 3080 report:

| Specialization | Registers/thread | Static shared | Local bytes/thread |
|---|---:|---:|---:|
| Unmasked | 66 | 25,088 bytes | 0 |
| Causal | 80 | 25,088 bytes | 0 |

`local bytes/thread = 0` is direct runtime evidence that this JIT result has
no local stack or register-spill allocation. Tests carry the value through
the borrowed production runtime and assert zero. Structural tests also assert
the expected `cp.async`, `mma.sync`, online tiling, absence of score/probability
pointers, and absence of stride parameters.

`ptxas`, `nvdisasm`, and `cuobjdump` are not installed on this machine, so the
report does not claim a separately inspected SASS listing. Nsight Compute
2025.4.1 was installed as a user-local portable tool and successfully attached
to the deterministic target, but the driver returned `ERR_NVGPUCTRPERM` because
performance-counter access is disabled for this Windows user. No executed
hardware-counter result is claimed until that OS/driver permission is enabled
and the checked-in profile command is rerun. PerfView is useful for managed
host allocation and scheduling, but it cannot prove GPU spills, tensor-pipe
use, shared-bank conflicts, or DRAM transactions. The Driver-API local-memory
attribute remains the available zero-spill admission proof for this capture.

## Edge-case dispatch table

| Condition | Direct PTX behavior |
|---|---|
| S = 16, 32, 64, or 128; D=64 | Shape-specialized kernel |
| Other S or D | Existing CUDA fallback; no semantic padding |
| Unmasked / causal | Separate compile-time specializations |
| Arbitrary attention bias or padding mask | Fallback |
| Dropout | Fallback |
| FP16 physical Q/K/V, FP32 output | Accepted |
| Noncontiguous, sparse, nonzero-offset, or aliased larger logical view | Materialize/general fallback; never enter stride-free PTX |
| Misaligned or undersized device pointer | Capability-token rejection and fallback |
| Active autodiff tape / backward | Existing recorded implementation |
| CUDA graph capture | Backend direct launch is allowed only after explicit prewarm; the high-level route retains its resident NVRTC fallback because FP16 scratch/output materialization is not capture-safe yet |
| Non-Ampere architecture | Gate disabled until a separately tuned Ada/Hopper/Blackwell family exists |
| Driver JIT reports local bytes | Module rejected and fallback |
| Causal future K/V tile | Compute skipped per owning query warp |
| Inference output-only | No score, probability, or LSE global store |
| Training-forward / backward | Existing recorded CUDA implementation; direct PTX is inference-only |

## Reusable kernel assembly line

Each future low-level kernel should be delivered as one repeatable unit:

1. Freeze the semantic contract: layouts, physical dtypes, accumulation,
   outputs, approximation, masking, and exact FLOP accounting.
2. Add a scalar oracle before writing PTX.
3. Define a canonical physical-layout capability token. Validate once at the
   dispatch boundary; never put stride or dtype branches in the hot loop.
4. Select explicit shape buckets and a correct general fallback. Never pad a
   mask-sensitive operation merely to enter a fast bucket.
5. Write a memory ledger for inputs, register state, shared tiles, optional
   outputs, and forbidden global intermediates.
6. Emit architecture/fusion/shape-specialized PTX and cache it by every bit
   that can change code generation.
7. Pipeline global-to-shared movement asynchronously, reuse shared tiles
   across warps, and keep recurrence/output state in registers.
8. Fuse epilogues only when their semantic axis and approximation are exact.
   Here LayerNorm is per `[B,H,S]` row over D=64, with shared D64 gamma/beta.
9. Query registers, shared memory, and local bytes from the compiled function.
   Reject spills as a runtime admission rule.
10. Validate every supported bucket, mask, and fusion signature, including
    layout rejection and fallback behavior.
11. Benchmark only semantic and hardware peers: current AiDotNet GPU,
    NVIDIA vendor composition, and framework GPU primitives with backends
    forced where the API permits. Keep CPU libraries in a separate study.
12. Report median, p95, p99, mean, effective throughput, managed allocation,
    known temporary VRAM, numerical error, registers, local bytes, and JIT
    setup separately.
13. Promote behind an opt-in gate only after the declared cell wins; retain a
    safe fallback and expand one shape/fusion cell at a time.

The reusable code layers are:

- `DirectPtxRuntime`: Driver API, owned/borrowed context, JIT, module, launch,
  events, allocation, and zero-local-memory admission.
- `DirectPtxFeatureGate` / `DirectPtxTensorView`: opt-in and physical contract.
- `DirectPtxKernelBlueprint`: logical/physical ABI, architecture family,
  semantic manifest, resource budget, PTX identity, and profiler evidence.
- `DirectPtxKernelCache` / `DirectPtxAttentionAutotuner`: bounded LRU modules,
  persisted device-specific winners, and prewarm-only graph capture.
- `PtxOnlineFusedAttention128x64Kernel`: shape emitter, async online dataflow,
  causal specialization, optional LSE, and fused epilogue.
- `CudaBackend.DirectPtx`: production cache, existing stream/context, audit,
  fallback, and teardown.
- `DirectPtxWmmaTests`: structural, scalar-oracle, layout, gate, borrowed-
  context, feature-route, and zero-spill tests.
- `DirectPtxGpuMatrixExperiment`: NVIDIA-only wide competitor harness.

## Productionization pass: all seven improvements

The experiment has now been extracted into a reusable system rather than
copied into another one-off emitter.

| Improvement | Implemented result | Remaining evidence boundary |
|---|---|---|
| Formal layout ABI | Logical and physical extents, layout identity, dtype, alignment, access, padding, byte offset, and allocation extent are validated once in `DirectPtxTensorContract`; the PTX has no stride path | Packed QKV and paged KV have ABI identities but no kernel yet |
| Autotuned dispatch | Attention tries valid query-warp variants, persists the winner through the existing autotune cache, keys it by GPU UUID/SM/driver and semantics, and bounds loaded modules with an LRU | A clean production release run still requires three independent captures |
| Stronger spill proof | Runtime admission enforces register/shared/local budgets and records PTX SHA-256, JIT attributes, occupancy, GPU fingerprint, and log; an Nsight CSV parser and deterministic profile target enforce executed spill/local counters | Nsight attached, but `ERR_NVGPUCTRPERM` blocks counter access until enabled by the host administrator |
| Hardened performance gate | Policy requires >=1.10x median, bounded p95, zero hot managed/device temporary bytes, numerical tolerance, zero local bytes, and three independent runs | Windows display scheduling can hold an otherwise fast candidate |
| Expanded correctness | Explicit eligibility reasons cover dtype, non-square Q/KV, D, GQA/MQA, mask, dropout, phase/backward, ragged and paged requests; unsupported cells fail closed | Those semantics remain implementation work, not silently advertised support |
| Real second fusion | `residual + RMSNorm(D=64)` is one warp-owned reduction/fusion using the same ABI, audit, gate, cache, graph-prewarm, tests, and benchmark framework | QKV projection/RoPE remains the next tensor-core fusion |
| Architecture families | Ampere, Ada, Hopper, and Blackwell are distinct dispatch domains | Only Ampere SM86 was executed here; other families deliberately reject rather than inherit Ampere tuning |

### Physical layout rule

AiDotNet tensors are not globally forced contiguous. Views remain legal for
general operations. A direct kernel instead requires an immutable capability:

```text
logical tensor/view
  -> graph-boundary materialize or existing canonical resident allocation
  -> DirectPtxTensorContract validation
     (layout + logical/physical extent + dtype + offset + alignment + access)
  -> stride-free pointer capability
  -> specialized PTX
```

This supports future padding, packed QKV, ragged offsets, and paged KV as
different physical ABIs rather than runtime stride branches.

### Second-blueprint result

RTX 3080, FP32 resident inputs/output, fused `RMSNorm(input + residual)` over
D=64, 101 synchronized samples after 30 warmups:

```text
Rows  Method                       median us  p95 us  p99 us  GB/s    B/call  tmp MiB  max err    regs  local B
32    Direct PTX fused                 17.50    25.10   34.40    1.43       0    0.000  4.768e-7   18       0
32    AiDotNet add + RMSNorm           20.30    44.60   53.90    1.23      56    0.008  4.768e-7
256   Direct PTX fused                 18.00    21.20   28.30   10.99       0    0.000  4.768e-7   18       0
256   AiDotNet add + RMSNorm           20.70    50.30   61.30    9.56      56    0.062  4.768e-7
2048  Direct PTX fused                 19.10    51.00   60.60   82.79       0    0.000  7.153e-7   18       0
2048  AiDotNet add + RMSNorm           24.40    42.20   74.90   64.81      56    0.500  7.153e-7
8192  Direct PTX fused                 21.50    43.80   45.50  294.16       0    0.000  5.960e-7   18       0
8192  AiDotNet add + RMSNorm           43.90    68.80  257.50  144.07      56    2.000  7.153e-7
```

Every median clears the 10% threshold (1.15x--2.04x). The diagnostic gate
passes rows 32, 256, and 8192. Rows 2048 is correctly held because its p95
exceeded the allowed competitor tail by 4.58 us on this capture. Production
promotion additionally requires three clean independent runs.

### Strong GPU competitors

The in-process suite still includes current AiDotNet CUDA, direct cuBLAS
composition, and forced TorchSharp Flash/Math SDPA. The new external harness
adds explicitly forced PyTorch cuDNN, Flash, Efficient, and Math SDPA, each in
eager and `torch.compile(max-autotune)` form, plus the official
`flash-attn` package when installed. It reports CUDA-event median/p95/p99/mean,
effective TFLOPS, and peak device allocation.

The external matrix was subsequently executed in an isolated Python 3.12
environment with PyTorch 2.12.1+cu130 on the same RTX 3080. The Windows wheel
does not contain PyTorch Flash-SDPA, and the third-party `flash-attn` package is
not installed, so those lanes report explicit skips. cuDNN, Efficient, and Math
were forced independently; no candidate was allowed to silently fall back.
The best eager external median for the two largest shapes is shown below. Peak
bytes include the candidate output allocation and are not directly comparable
to the direct kernel's `temporary VRAM` column.

| Shape | Mode | Lane | Forced backend | Median us | P95 us | P99 us | Mean us | TFLOPS | Peak bytes | Max error |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| BH128 S128 | plain | attention | cuDNN | 196.61 | 381.95 | 577.54 | 231.98 | 2.731 | 6,291,456 | 3.052e-5 |
| BH128 S128 | causal | attention | cuDNN | 161.76 | 330.75 | 485.38 | 189.69 | 3.319 | 6,291,456 | 1.221e-4 |
| BH128 S128 | plain | attn+epi | cuDNN | 399.36 | 594.94 | 677.89 | 431.90 | 1.344 | 20,971,520 | 2.632e-3 |
| BH128 S128 | causal | attn+epi | cuDNN | 393.22 | 494.78 | 728.06 | 407.97 | 1.365 | 20,971,520 | 2.451e-3 |
| BH512 S128 | plain | attention | Efficient | 160.77 | 297.98 | 345.09 | 191.96 | 13.358 | 25,165,824 | 3.052e-5 |
| BH512 S128 | causal | attention | Efficient | 94.21 | 332.80 | 438.27 | 124.45 | 22.795 | 25,165,824 | 1.221e-4 |
| BH512 S128 | plain | attn+epi | Efficient | 886.78 | 1,180.67 | 1,576.96 | 927.18 | 2.422 | 83,886,080 | 2.528e-3 |
| BH512 S128 | causal | attn+epi | Efficient | 904.19 | 1,482.75 | 1,938.43 | 1,001.07 | 2.375 | 83,886,080 | 2.614e-3 |

`torch.compile(max-autotune)` correctly remains an unavailable lane on this
Windows installation because the official wheel has no working Triton
installation. The harness records that failure rather than relabeling eager
execution as compiled execution.

### Resource evidence workflow

The runtime JIT audit remains a mandatory first gate. Release profiling adds:

```powershell
tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target attention -OutputCsv attention-ncu.csv

tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target residual-rmsnorm -OutputCsv residual-rmsnorm-ncu.csv
```

The script profiles a deterministic kernel target and fails unless executed
register-spill instructions, local loads, local stores, and local spill
requests are all zero. PerfView remains appropriate for managed allocation
and ETW; it is not accepted as GPU spill evidence.

## Reproduction

```powershell
dotnet build tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release

dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj `
  -c Release -f net10.0 --filter DirectPtxWmmaTests

dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release --no-build -- --direct-ptx-online-attention

dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release --no-build -- --direct-ptx-gpu-matrix

dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release --no-build -- --direct-ptx-residual-rmsnorm

dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release --no-build -- --direct-ptx-external-gpu-baselines

$env:AIDOTNET_DIRECT_PTX_ATTENTION='1'
$env:AIDOTNET_DIRECT_PTX_RESIDUAL_RMSNORM='1'
# Or enable all admitted direct kernels:
$env:AIDOTNET_DIRECT_PTX='1'
```

Run on an otherwise idle system. The exact values are hardware, driver,
clock, thermal, and scheduler dependent; the contract and pass/fail process
are the reusable artifacts.
