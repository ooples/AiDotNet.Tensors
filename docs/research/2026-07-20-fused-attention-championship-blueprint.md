# Direct-PTX online-attention kernel blueprint

Date: 2026-07-20; productionization pass 2026-07-21
Baseline: `b386d35` plus the working-tree experiment described here

## Verdict

The forward-inference experiment is implemented end to end for the measured
NVIDIA SM86 target and the declared shape family. Deterministic FP32
materialized-probability and Flash/LSE-recomputation backward families are
also implemented for SM86:

```text
FP16 Q/K/V, dense BHSD, Sq and Skv independently in {16,32,64,128}, D=64
  -> MHA, GQA, or MQA with baked query-head-to-KV-head mapping
  -> scaled attention, unmasked or causal
  -> optional head-local LayerNorm(D=64) + affine + tanh-GELU
  -> FP32 BHSD output
```

The final wide benchmark is GPU-only. It compares the direct PTX kernel with
the current AiDotNet CUDA/NVRTC implementation, a direct cuBLAS Tensor Core
composition, and PyTorch SDPA. The in-process TorchSharp 0.106 lane can only
toggle Flash and Math preferences, so it is labeled as a preference lane; the
external Python harness is the authoritative forced-backend comparison for
cuDNN, Flash, Efficient, and Math. Intel MKL and OpenBLAS are deliberately absent: they are CPU
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
GPUs, dimensions, dropout, arbitrary masks, or undeclared backward variants.

The issue-#834 v3 expansion generalizes the previously square/equal-head
emitter to rectangular attention and shared KV heads. Its correctness and
zero-local-memory tests are complete on the RTX 3080; its championship matrix
is deliberately not inferred from the older square-MHA numbers below. Each new
rectangular/GQA cell remains experimental until its own three-run performance
and Nsight evidence passes the release gate.

## Executable #834 coverage inventory

`DirectPtxAttentionCoverageManifest` assigns every scoped entry point to an
existing implementation and a direct-PTX work lane. A focused test fails on a
missing or duplicate assignment.

| API | Existing CUDA implementation | Direct-PTX assignment |
|---|---|---|
| ScaledDotProductAttention | `scaled_dot_product_attention` | v3 weight-free dense FP16 inference implemented; weights/masks/softcap planned |
| ScaledDotProductAttentionBackward | GEMM + softmax-backward composition | v1 deterministic materialized-probability FP32 D64 MHA backward implemented; recomputation planned |
| FlashAttention / FlashAttentionV2 | `flash_attention_v2` | v3 rectangular dense FP16, LSE, no-bias route implemented |
| FlashAttentionBackward | atomic and deterministic NVRTC variants | v1 deterministic FP32 D64 LSE-recomputation MHA implemented for unmasked/causal with optional broadcast/per-batch additive bias; remaining semantics planned |
| GroupedQueryAttention | `grouped_query_attention` | v3 weight-free GQA/MQA inference implemented; weights mode planned |
| GroupedQueryAttentionBackward | atomic and deterministic NVRTC variants | v1 deterministic materialized-probability FP32 D64 GQA/MQA backward implemented; recomputation planned |
| FlashDecode | `flash_decode_partial` + `flash_decode_reduce` | v1 register-online D64 FP32 MHA/GQA/MQA decode implemented for S=16/32/64/128; explicit split hints fall back |
| PagedAttention decode, MHA/GQA | two `paged_attention_decode*` kernels | v1 block-table D64 FP32 decode implemented for block size 16/32 |
| PagedAttention prefill, MHA/GQA | two `paged_attention_prefill*` kernels | v1 causal block-table D64 FP32 MHA/GQA/MQA prefill implemented for Q=2/4/8/16/32, total KV<=128, block size 16/32 |

The machine-readable manifest is authoritative; this compact table is its
human-readable summary.

## Apples-to-apples contracts

Two lanes are reported separately.

| Lane | Required result | FLOP accounting |
|---|---|---:|
| Attention | FP16 Q/K/V -> FP32 `softmax(QK^T * scale + mask)V` | `4 * BH * S^2 * D` |
| Attention + epilogue | The same FP32 attention, then row-wise LayerNorm over D=64, affine, and tanh-GELU | Attention FLOPs only, labeled effective TFLOPS |

The physical-kernel championship lane uses FP16 Q/K/V for direct PTX, cuBLAS,
and PyTorch. Current AiDotNet NVRTC has no equivalent FP16 entry point; its
separately labeled framework baseline receives the same half-rounded numerical
values in FP32 storage. It is required coverage for the original implementation,
but is not presented as dtype-identical to the physical-kernel lane.

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

`GFLOPS` and `TFLOPS` count the same two attention matrix products at different
scales. They do not invent FLOPs
for softmax, mask predicates, normalization, or activation. `B/call` is
managed allocation on the calling thread. `tmp MiB` is known intermediate
VRAM, excluding inputs and final outputs. Correctness rows report maximum
absolute error and a stable symmetric relative error
`2*abs(actual-expected)/(abs(actual)+abs(expected)+1e-3)` so values near zero
do not turn the relative column into noise.

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
unmasked  TorchSharp flash-preferred SDPA [E2E]   64.10     132.20     138.80      72.38     8.376       96      n/a   1.214e-5    n/a      n/a
unmasked  TorchSharp flash-preferred + epilogue  406.00    1681.90    2115.90     811.50     1.322      704      n/a   9.871e-4    n/a      n/a
causal    Direct PTX attention [device]           25.60      53.04      59.90      28.84    20.972        0    0.000  2.855e-5     80        0
causal    Direct PTX attention [E2E]              39.80     129.40     326.60      64.62    13.489        0    0.000  2.855e-5     80        0
causal    cuBLAS+PTX softmax [device]             51.51      96.15     157.49      61.22    10.423        0   12.000  3.869e-5    n/a      n/a
causal    cuBLAS+PTX softmax [E2E]                71.30     242.60     293.70     104.13     7.530        0   12.000  3.869e-5    n/a      n/a
causal    Direct PTX + LN + GELU [device]         26.73      36.76      46.80      28.25    20.088        0    0.000  5.491e-4     80        0
causal    Direct PTX + LN + GELU [E2E]            41.20      43.70      45.40      41.42    13.031        0    0.000  5.491e-4     80        0
causal    AiDotNet NVRTC attention [E2E]         912.30    1137.60    1527.30     943.25     0.588        0    0.062   2.980e-8    n/a      n/a
causal    AiDotNet NVRTC + LN + GELU [E2E]      1025.10    1280.00    1529.30    1061.36     0.524       40    8.188   5.722e-6    n/a      n/a
causal    TorchSharp flash-preferred SDPA [E2E]  102.30     119.50     131.60     102.53     5.248       96      n/a   6.071e-5    n/a      n/a
causal    TorchSharp flash-preferred + epilogue  371.50    1598.10    1794.40     766.39     1.445      704      n/a   1.247e-3    n/a      n/a
```

The direct kernel's device median is 1.52x faster than the decomposed cuBLAS
floor unmasked and 2.01x faster causal. Its synchronized attention median is
1.39x/2.57x faster than the TorchSharp flash-preferred lane. The fused median is
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
current AiDotNet, and the TorchSharp Flash-preferred/Flash-disabled preference lanes.

| Shape | Mode | Lane | Direct median us | Direct p95 | Direct p99 | Direct TFLOPS | Best competitor | Competitor median us | Speedup |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| BH12 S16 | plain | attention | 21.40 | 41.95 | 44.25 | 0.037 | cuBLAS+PTX | 42.50 | 1.99x |
| BH12 S16 | causal | attention | 17.80 | 19.35 | 21.70 | 0.044 | cuBLAS+PTX | 45.50 | 2.56x |
| BH12 S32 | plain | attention | 19.20 | 35.40 | 44.10 | 0.164 | cuBLAS+PTX | 43.50 | 2.27x |
| BH12 S32 | causal | attention | 17.80 | 33.65 | 41.85 | 0.177 | TorchSharp Flash-disabled | 52.10 | 2.93x |
| BH12 S64 | plain | attention | 21.20 | 40.20 | 73.40 | 0.594 | cuBLAS+PTX | 49.40 | 2.33x |
| BH12 S64 | causal | attention | 20.20 | 39.70 | 45.60 | 0.623 | TorchSharp Flash-disabled | 52.00 | 2.57x |
| BH12 S128 | plain | attention | 35.40 | 56.15 | 97.15 | 1.422 | cuBLAS+PTX | 48.00 | 1.36x |
| BH12 S128 | causal | attention | 27.70 | 48.75 | 107.45 | 1.817 | cuBLAS+PTX | 46.80 | 1.69x |
| BH128 S128 | plain | attention | 48.00 | 51.80 | 54.75 | 11.185 | TorchSharp Flash-disabled | 65.20 | 1.36x |
| BH128 S128 | causal | attention | 39.00 | 56.70 | 79.65 | 13.766 | TorchSharp Flash-preferred | 64.20 | 1.65x |
| BH512 S128 | plain | attention | 130.60 | 133.40 | 138.05 | 16.443 | TorchSharp Flash-preferred | 132.10 | 1.01x |
| BH512 S128 | causal | attention | 91.40 | 262.80 | 350.90 | 23.495 | TorchSharp Flash-disabled | 118.00 | 1.29x |
| BH12 S16 | plain | attn+epi | 18.50 | 31.05 | 39.75 | 0.043 | AiDotNet NVRTC | 55.80 | 3.02x |
| BH12 S16 | causal | attn+epi | 22.40 | 48.90 | 66.55 | 0.035 | AiDotNet NVRTC | 55.30 | 2.47x |
| BH12 S32 | plain | attn+epi | 18.90 | 39.05 | 40.50 | 0.166 | AiDotNet NVRTC | 82.60 | 4.37x |
| BH12 S32 | causal | attn+epi | 23.00 | 49.70 | 57.60 | 0.137 | AiDotNet NVRTC | 80.90 | 3.52x |
| BH12 S64 | plain | attn+epi | 21.50 | 38.05 | 45.30 | 0.585 | AiDotNet NVRTC | 129.20 | 6.01x |
| BH12 S64 | causal | attn+epi | 20.80 | 41.55 | 44.30 | 0.605 | AiDotNet NVRTC | 132.90 | 6.39x |
| BH12 S128 | plain | attn+epi | 32.10 | 32.85 | 35.20 | 1.568 | AiDotNet NVRTC | 238.90 | 7.44x |
| BH12 S128 | causal | attn+epi | 35.40 | 69.50 | 163.15 | 1.422 | AiDotNet NVRTC | 241.70 | 6.83x |
| BH128 S128 | plain | attn+epi | 52.40 | 56.55 | 59.65 | 10.246 | TorchSharp Flash-disabled | 304.60 | 5.81x |
| BH128 S128 | causal | attn+epi | 39.00 | 40.10 | 43.70 | 13.766 | TorchSharp Flash-disabled | 275.20 | 7.06x |
| BH512 S128 | plain | attn+epi | 135.70 | 182.35 | 320.05 | 15.825 | TorchSharp Flash-disabled | 804.70 | 5.93x |
| BH512 S128 | causal | attn+epi | 103.00 | 114.25 | 255.30 | 20.849 | TorchSharp Flash-preferred | 809.50 | 7.86x |

The executable prints every competitor row with median, p95, p99, mean,
effective TFLOPS, managed allocation, and known temporary VRAM. The table
above is intentionally a compact winner view; it does not replace the raw TUI.
Repeated 101-sample, one-launch host captures under desktop load occasionally
reverse the closest attention-only cells because scheduler phase changes by
more than their margin. They did not erase the fused lane's 2.47x--7.86x
median advantage. Use the CUDA-event championship cell for raw device work,
and an otherwise idle GPU for release-gate host distributions.

## Decode championship slice

The first #834 follow-on specialization covers FP32 D=64 single-token decode
for MHA/GQA/MQA, dense sequence lengths 16/32/64/128, and paged block sizes
16/32. The table below summarizes three independent clean runs; every run has
30 warmups and 101 synchronized end-to-end samples. Ranges retain run-to-run
Windows display-GPU scheduling variation rather than selecting one favorable
capture.

```text
Shape       Direct median us  Direct P95 us  Direct P99 us   GFLOPS    AiDotNet median us  Min speedup  B/call  tmp MiB  max error  regs/shared/local  blocks/SM
--------------------------------------------------------------------------------------------------------------------------------------------------------------
dense-mha       20.7-21.8       41.8-43.2      43.4-189.9   3.01-3.17      64.3-64.7            2.96x        0    0.000   2.608e-8      40 / 0 / 0          12
dense-gqa       22.5-26.2       42.7-46.5      43.8-64.4    5.00-5.83      64.7-65.7            2.48x        0    0.000   2.980e-8      40 / 0 / 0          12
dense-mqa       26.5-27.8       43.1-50.9      47.7-231.8   9.43-9.89      92.1-93.5            3.31x        0    0.000   2.794e-8      40 / 0 / 0          12
paged-gqa       25.1-27.6       43.4-56.1      47.8-128.0   4.75-5.22     189.3-190.7           6.86x        0    0.000   2.608e-8      40 / 0 / 0          12
paged-mqa       28.2-30.9       32.4-39.1      43.1-55.1    8.48-9.30     350.9-354.6          11.48x        0    0.000   2.049e-8      46 / 0 / 0          10
```

The established dense AiDotNet split-K path uses 0.016 MiB of partial
max/sum/accumulator storage per measured cell; the direct path has no temporary
device allocation. The paged comparison has no temporary tensor on either
side, but the direct path removes the general dynamic kernel's address and
launch overhead. All five cells pass the paired-run production policy in all
three captures: at least 1.10x median, P95 within competitor P95 +10%, zero hot
managed bytes, zero temporary device bytes, error below 5e-5, and zero local
bytes.

The exact FP32 PyTorch public-SDPA competitor run forced each backend rather
than allowing fallback. This Windows PyTorch 2.12.1+cu130 wheel has no eligible
FP32 Flash or cuDNN lane. Efficient-SDPA accepts only dense MHA; native GQA/MQA
uses Math-SDPA. Across the same three-run/101-sample protocol, the eligible
competitor ranges were:

| Shape | Forced PyTorch backend | Median us | P95 us | P99 us | Mean us | GFLOPS | Peak device bytes | Max error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dense MHA | Efficient | 53.2-85.7 | 76.2-212.8 | 168.0-429.5 | 59.42-121.57 | 0.76-1.23 | 2,048 | 3.725e-8 |
| dense GQA | Math | 371.9-385.7 | 532.7-667.9 | 601.7-1159.3 | 390.51-425.03 | 0.34-0.35 | 397,312 | 0 |
| dense MQA | Math | 337.6-385.1 | 491.8-548.2 | 585.3-745.8 | 371.10-398.86 | 0.68-0.78 | 792,576 | 0 |

PyTorch peak bytes include its required result allocation; public SDPA has no
caller-supplied output. PyTorch exposes no equivalent paged block-table SDPA
operation, so the paged cells use AiDotNet's existing NVIDIA kernel as their
semantic and hardware peer. The runtime local-memory attribute proves zero JIT
local allocation for all promoted decode cells. Executed Nsight spill counters
remain externally blocked by `ERR_NVGPUCTRPERM`; this slice remains experimental
until that host permission is available and the required CSV evidence is attached.

## Paged-prefill championship slice

The next #834 specialization covers causal FP32 D=64 paged prefill with
canonical query/output `[Q,Hq,D]`, paged K/V
`[poolBlock,blockPosition,Hkv,D]`, and an exact Int32 block table. Each range
below comes from three independent clean runs with 30 warmups and 101
synchronized end-to-end samples per run.

```text
Shape          Direct median us  Direct P95 us  Direct P99 us   GFLOPS     AiDotNet median us  Min speedup  B/call  tmp MiB  max error  regs/shared/local  blocks/SM
------------------------------------------------------------------------------------------------------------------------------------------------------------------
prefill-mha        22.7-25.1       42.5-46.5      50.0-92.6    4.73-5.23       85.2-89.2            3.55x        0    0.000   2.980e-8      25 / 0 / 0          12
prefill-gqa        26.6-27.8       44.7-49.4      47.4-59.8   16.80-17.55     159.9-161.7           5.75x        0    0.000   2.980e-8      29 / 0 / 0          12
prefill-mqa        33.2-35.0       38.8-48.3      42.5-64.5   52.90-55.76     380.6-381.5          10.87x        0    0.000   3.353e-8      29 / 0 / 0          12
prefill-long       50.2-53.0       60.7-129.2     83.9-362.2  74.50-78.66     795.7-807.7          15.11x        0    0.000   2.980e-8      29 / 0 / 0          12
```

All four cells pass the paired-run policy in all three captures. The direct
kernel keeps online max, normalization sum, and PV accumulators in registers,
emits one final vector store, and allocates neither a score matrix nor split
scratch. Driver-JIT admission reports zero shared and local bytes. Nsight
attached to the deterministic target, but counter collection remains blocked
by the host's `ERR_NVGPUCTRPERM` policy.

The exact FP32 PyTorch comparison forced eligible SDPA backends and gathered
the logical K/V sequence from pages before timing, deliberately excluding
page translation in PyTorch's favor. Public PyTorch SDPA has no paged ABI.
Flash and cuDNN rejected FP32; Efficient accepted MHA only, while GQA/MQA used
the independently forced Math backend.

| Shape | Forced PyTorch backend | Median us | P95 us | P99 us | Mean us | GFLOPS | Peak device bytes | Max error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| prefill MHA | Efficient | 120.2-122.6 | 172.7-266.9 | 223.3-395.9 | 127.31-147.24 | 0.97-0.99 | 8,704 | 4.470e-8 |
| prefill GQA | Math | 402.1-529.6 | 671.2-739.6 | 755.9-1438.5 | 455.58-546.90 | 0.88-1.16 | 222,208 | 2.980e-8 |
| prefill MQA | Math | 405.2-466.6 | 601.1-708.8 | 684.1-993.2 | 448.25-498.09 | 3.97-4.57 | 462,848 | 2.980e-8 |
| prefill long | Math | 404.2-454.7 | 600.4-677.3 | 623.8-911.6 | 440.45-493.54 | 8.68-9.77 | 892,928 | 2.980e-8 |

## Deterministic attention-backward championship slice

The v1 backward specialization accepts the exact FP32 probabilities already
produced by `ScaledDotProductAttention` or `GroupedQueryAttention`. Its formal
ABI is dense canonical BHSD for `dO/Q/K/V/dQ/dK/dV` and dense BHSqSkv for the
probabilities, with D=64, batch <=16, and Sq/Skv independently in
`{16,32,64,128}`. MHA, GQA, and MQA use a baked query-head-to-KV-head map.

The deterministic dataflow has three entry points. A row-owned warp computes
`delta = sum(dP * P)` and writes one scalar per query row into the final dQ
allocation as temporary output workspace. Key-owned warps then write each dK
and dV element exactly once, without atomics. Query-owned warps finally
overwrite the complete dQ allocation. This avoids the current implementation's
five SxS/transpose temporaries and makes no additional allocation. It does not
claim recomputation backward: the API-provided probability tensor is read once
per derivative pass and remains part of this v1 contract.

RTX 3080, 30 warmups and 101 synchronized E2E samples per cell, three
independent runs. `GFLOPS = 8 * B * Hq * Sq * Skv * D / time`, counting the
four derivative matrix products. The ranges below are from the final captured
three-run gate:

```text
Shape          Direct median us  Direct P95 us  Direct P99 us  Direct mean us   GFLOPS   AiDotNet median us  Min speedup  B/call  tmp MiB  max error  regs d/dq/dkv  shared/local  blocks/SM
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
backward-mha       43.0-60.7       99.0-109.5     118.6-228.1     56.68-68.94   17.27-24.39     433.4-481.6          7.14x        0    0.000  2.235e-8      22/25/29          0/0       12/12/12
backward-gqa       69.9-73.4      105.4-184.2     200.5-308.8     79.76-94.39   28.57-30.00    2109.4-2344.9        28.74x        0    0.000  4.470e-8      22/25/29          0/0       12/12/12
backward-mqa      114.2-118.2     138.1-155.8     193.9-276.3    122.72-123.76  17.74-18.36    4028.6-4236.9        35.13x        0    0.000  1.490e-7      22/25/29          0/0       12/12/12
backward-long     132.4-139.6     156.8-167.8     196.4-338.0    137.85-145.77 120.18-126.72  17410.9-17441.2      124.94x        0    0.000  3.725e-8      22/25/29          0/0       12/12/12
```

Every paired run passes the policy against AiDotNet's established deterministic
CUDA implementation. The exact external PyTorch composition uses resident
CUDA tensors and the same materialized probabilities. Its best three-run
median ranges are 426.3-439.3 us (MHA), 603.2-820.4 us (GQA), 532.0-927.0 us
(MQA), and 554.9-589.4 us (long), so the conservative direct-PTX median wins
are 7.02x, 8.22x, 4.50x, and 3.97x using the slowest direct and fastest external
median from this capture. PyTorch peak allocation ranges from 115,200 to
722,944 bytes; the direct route reports zero managed bytes and zero device
temporary bytes. `torch.compile(max-autotune)` was attempted and records an
explicit skip because this Windows PyTorch environment has no working Triton
backend.

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

Decode is a distinct FP32 ABI: Q and output are `[Hq,64]`; dense K/V are
canonical `[S,Hkv,64]`; paged K/V are `[poolBlock,blockPosition,Hkv,64]` plus
an exact-length Int32 block table. One warp owns one query head and each lane
owns two output dimensions. QK reduction, online max/sum, and PV recurrence
remain in registers; there is one final vector output store. There is no SxS
score/probability tensor, split-K scratch, shared-memory allocation, or dynamic
stride parameter. `cp.async` and Tensor Cores are deliberately not used in
this FP32 GEMV-like decode cell: warp shuffles and coalesced vector loads were
the measurably winning dataflow. Paged address translation has its own formal
ABI and resource domain rather than a runtime layout branch in the dense PTX.

Paged prefill assigns one warp to each `(query,queryHead)` pair. Its causal
key bound and query-to-KV-head mapping are compile-time constants; block-table
translation is the only address indirection. Each lane streams its two D=64
components from paged global K/V, performs warp reductions for QK and the
online recurrence, retains PV in registers, and writes output once. Because
this FP32 cell is GEMV-like at each query, measured coalesced vector loads beat
adding shared-memory staging or Tensor Core packing; those mechanisms remain
appropriate for the existing FP16 tile-matrix family.

Backward is intentionally a different blueprint rather than a branch in the
forward hot loop. The v1 SDPA/GQA public backward APIs provide materialized
probabilities, so that family consumes the probability tensor while eliminating
derivative-score matrices, transpose buffers, atomics, dynamic strides, and
intermediate allocation.

`FlashAttentionBackward` instead uses the actual Flash pattern: it recomputes
each probability from Q/K and the forward LSE statistic. Two deterministic
non-atomic entry points assign one warp to a dQ row or dK/dV row. QK, dO.V,
softmax-gradient reduction, dQ/dK accumulation, and dV accumulation stay in
registers; only final dQ/dK/dV vectors are written. It has no SxS score or
probability buffer, temporary allocation, stride argument, shared/local memory,
or atomics. The currently admitted family is FP32 BHSD, D=64, MHA,
Sq/Skv independently in {16,32,64,128}, unmasked or top-left causal, with
optional canonical `[H,Sq,Skv]` broadcast or `[B,H,Sq,Skv]` additive bias.

### Flash/LSE recomputation-backward evidence

The in-process championship uses CUDA events for the raw production gate and
also reports one-launch synchronized E2E latency. Device samples average ten
back-to-back launches. Across three independent RTX 3080 runs, every paired
AiDotNet gate passed (>=1.10x median, bounded p95, zero allocation, numerical
tolerance, and zero local bytes).

| Shape | Direct worst device median / p95 (us) | Current best device median / p95 (us) | Minimum paired median win | Direct peak GFLOPS | Direct worst E2E median / p95 (us) | PyTorch Efficient best E2E median / p95 (us) | Direct B/call | PyTorch peak bytes | Max direct error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H8, Sq16, Skv16 | 30.92 / 56.63 | 141.31 / 145.72 | 4.59x | 39.40 | 43.50 / 99.70 | 369.40 / 586.00 | 0 | 263,168 | 3.725e-8 |
| H8, Sq16, Skv32 | 29.59 / 55.91 | 230.71 / 263.78 | 7.80x | 75.57 | 49.80 / 98.90 | 352.90 / 628.50 | 0 | 328,704 | 1.863e-8 |
| H8, Sq16, Skv32, bias | 30.62 / 46.49 | 236.24 / 261.63 | 7.82x | 76.70 | 40.00 / 79.30 | 335.30 / 579.60 | 0 | 328,704 | 2.235e-8 |
| H8, Sq32, Skv32, causal | 38.40 / 62.57 | 337.72 / 365.06 | 8.95x | 135.63 | 45.10 / 90.30 | 328.60 / 517.40 | 0 | 394,752 | 1.490e-7 |
| H8, Sq64, Skv64, causal | 46.69 / 66.97 | 1,010.48 / 1,064.86 | 22.29x | 376.64 | 63.50 / 82.50 | 336.20 / 557.20 | 0 | 659,456 | 1.192e-7 |

“Worst” and “best” are conservative extrema across the three captures, while
the minimum win is calculated within paired runs. GFLOPS is the best observed
direct device throughput and counts `8*H*Sq*Skv*D` useful backward FLOPs. The
forced PyTorch cuDNN and Flash lanes explicitly skipped: FP32 is unsupported by
cuDNN attention here and this Windows wheel has no FlashAttention kernel.
Efficient and Math ran natively; no fallback was relabeled as a competitor.

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

- Every family defaults off. Set `AIDOTNET_DIRECT_PTX_ATTENTION=1`,
  `AIDOTNET_DIRECT_PTX_FLASH_DECODE=1`, or
  `AIDOTNET_DIRECT_PTX_PAGED_DECODE=1`, or
  `AIDOTNET_DIRECT_PTX_PAGED_PREFILL=1`, or
  `AIDOTNET_DIRECT_PTX_ATTENTION_BACKWARD=1`, or
  `AIDOTNET_DIRECT_PTX_FLASH_ATTENTION_BACKWARD=1` to admit only that family.
- The process-level gates are resolved once when a CUDA backend is constructed,
  steady-state dispatch does not reread environment variables.
- SM other than the executed SM86 target, unsupported shape/dtype/layout,
  bias, active autodiff tape, stream
  capture, JIT failure, or launch failure falls back to the existing path.
- The runtime borrows the existing `CudaBackend` context and stream, so
  resident AiDotNet buffers require no copy or context bridge.
- Modules are cached by BH, S, mask, fusion signature, scale bits, and epsilon
  bits, and unloaded before the backend destroys its context/stream.
- The kernel cache and launches use the backend's existing dispatch locks.
- Capture may use an already prewarmed attention or decode specialization with
  stable caller-owned buffers. A capture-time JIT/cache miss fails closed;
  capture never tunes, allocates, performs file I/O, or evicts a live module.
- The allocating high-level route stays on the established resident NVRTC path
  during stream capture; only the prewarmed physical backend route with stable
  caller-owned buffers is graph-capturable.

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
- materialized backward uses exact-extent FP32 BHSD tensors and FP32 BHSqSkv
  probabilities; Flash backward uses exact-extent FP32 BHSD plus FP32 BHSq LSE;
- additive Flash-backward bias is a separate baked ABI: exact FP32
  `[H,Sq,Skv]` batch broadcast or `[B,H,Sq,Skv]` per batch;
- at least 16-byte device-pointer alignment;
- sufficient byte extent and dtype-compatible extent;
- no stride fields and no shape fields passed to PTX.

Shape, offsets, trip counts, mask mode, scale, and fusion choices are baked
into the PTX specialization. Public `Tensor<float>` input reuses an existing
resident FP16 activation when one is available. Otherwise the current
high-level bridge creates and owns an FP16 conversion buffer for that call;
the zero-temporary-VRAM claim applies to the prevalidated physical FP16
backend launch, not that conversion fallback.
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
| Flash backward dQ, no bias / bias | 29 / 33 | 0 bytes | 0 |
| Flash backward dK/dV, no bias / bias | 33 / 35 | 0 bytes | 0 |

`local bytes/thread = 0` is a mandatory JIT resource-admission result, but it
is not by itself sufficient proof that no executed spill/local-memory traffic
occurred. Tests carry the value through
the borrowed production runtime and assert zero. Structural tests also assert
the expected `cp.async`, `mma.sync`, online tiling, absence of score/probability
pointers, and absence of stride parameters.

`ptxas`, `nvdisasm`, and `cuobjdump` are not installed on this machine, so the
report does not claim a separately inspected SASS listing. Nsight Compute
2026.2.1 was extracted as a user-local portable tool and successfully attached
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
| Sq and Skv independently = 16, 32, 64, or 128; D=64 | Shape-specialized kernel |
| MHA / GQA / MQA with `Hq % Hkv == 0` | Baked batch/head mapping; no expanded K/V tensor |
| Other Sq, Skv, D, or non-divisible head mapping | Existing CUDA fallback; no semantic padding |
| Unmasked / causal | Separate compile-time specializations |
| Arbitrary attention bias or padding mask | Fallback |
| Dropout | Fallback |
| FP16 physical Q/K/V, FP32 output | Accepted |
| FP32 D64 decode; S=16/32/64/128; no explicit split hint | Dense warp-online decode specialization |
| FP32 D64 paged decode; block size 16/32 | Separate paged ABI; MHA/GQA/MQA head map baked into PTX |
| Other decode dtype, D, sequence bucket, block geometry, or explicit split-K request | Existing CUDA fallback with an exact reason |
| Paged table/pool byte extent mismatch | Rejected before launch; page indices are trusted device metadata under the same caller precondition as the established paged kernel |
| Noncontiguous, sparse, nonzero-offset, or aliased larger logical view | Materialize/general fallback; never enter stride-free PTX |
| Misaligned or undersized device pointer | Capability-token rejection and fallback |
| Active autodiff tape / forward recording | Existing recorded implementation |
| CUDA graph capture | Caller-owned-buffer direct launches are allowed only after explicit prewarm; capture-time JIT/cache misses fail closed, and allocating high-level routes retain their established behavior |
| SM other than 8.6 | Gate disabled until that SM target is separately tuned and proven |
| Driver JIT reports local bytes | Module rejected and fallback |
| Rectangular causal | Separate baked domains: FlashAttentionV2 top-left (`offset=0`) and SDPA bottom-right (`offset=Skv-Sq`); negative offsets emit zero output and `-inf` LSE for fully masked leading rows |
| Causal future K/V tile | Compute skipped per owning query warp |
| Inference output-only | No score, probability, or LSE global store |
| Materialized-probability backward, FP32 D64, supported shape/head map | Three-entry deterministic direct PTX; final dQ allocation supplies the row-delta workspace |
| Flash/recomputation backward, FP32 D64 MHA, Sq/Skv=16/32/64/128, optional canonical broadcast/per-batch bias | Two-entry deterministic direct PTX; probabilities recomputed from Q/K/bias/LSE; no scratch |
| Flash/recomputation backward with noncanonical bias, unsupported dtype/D/shape, or GQA | Existing CUDA implementation with an exact rejection reason |

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
   Reject nonzero local-byte admission, then require executed Nsight spill/local
   counters to be zero before release.
10. Validate every supported bucket, mask, and fusion signature, including
    layout rejection and fallback behavior.
11. Benchmark only semantic and hardware peers: current AiDotNet GPU,
    NVIDIA vendor composition, and framework GPU primitives with backends
    forced where the API permits. Keep CPU libraries in a separate study.
12. Report median, p95, p99, mean, effective GFLOP/s and TFLOP/s, managed allocation,
    known temporary VRAM, numerical error, registers, static/dynamic shared
    memory, local bytes, occupancy, and JIT
    setup separately.
13. Promote behind an opt-in gate only after the declared cell wins; retain a
    safe fallback and expand one shape/fusion cell at a time.
14. Capture at least three clean independent processes with the checked-in
    release-evidence script. A concurrent compute process or thermally invalid
    start aborts the run instead of producing a publishable row.
15. Run Nsight on every admitted specialization and size. Zero JIT local bytes
    is necessary but release still requires zero executed spill, local-load,
    and local-store counters plus recorded launch resources/occupancy.

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
- `PtxFusedDecodeAttentionD64Kernel`: separate dense/paged physical ABIs,
  warp-owned online decode, block-table translation, and single final store.
- `PtxFusedPagedPrefillAttentionD64Kernel`: query-head warp ownership,
  causal paged traversal, online recurrence, and single final vector store.
- `PtxFusedAttentionBackwardD64Kernel`: deterministic row-delta, key-owned
  dK/dV, and query-owned dQ entry points with output-workspace reuse.
- `PtxFlashAttentionBackwardD64Kernel`: deterministic LSE-recomputation dQ and
  dK/dV entry points with no materialized probability or scratch tensor.
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
| Formal layout ABI | Logical and physical extents, layout identity, dtype, alignment, access, padding, byte offset, and allocation extent are validated once in `DirectPtxTensorContract`; dense sequence-head-dimension and paged KV decode/prefill now have distinct executable ABIs; the PTX has no stride path | Packed QKV and ragged-offset kernels remain |
| Autotuned dispatch | Attention tries valid query-warp variants, persists the winner through the existing autotune cache, keys it by GPU UUID/SM/driver and semantics, and bounds loaded modules with an LRU | A clean production release run still requires three independent captures |
| Stronger spill proof | Runtime admission enforces register/shared/local budgets and records PTX SHA-256, JIT attributes, occupancy, GPU fingerprint, and log; an Nsight CSV parser and deterministic profile target enforce executed spill/local counters | Nsight attached, but `ERR_NVGPUCTRPERM` blocks counter access until enabled by the host administrator |
| Hardened performance gate | Policy requires >=1.10x median, bounded p95, zero hot managed/device temporary bytes, numerical tolerance, zero local bytes, and three independent runs | Windows display scheduling can hold an otherwise fast candidate |
| Expanded correctness | Rectangular Sq/Skv D64 FP16 attention, D64 FP32 dense/paged decode, causal D64 FP32 paged prefill, deterministic materialized-probability D64 FP32 backward, and deterministic optionally biased Flash/LSE-recomputation D64 FP32 backward admit their tested buckets; explicit reasons cover dtype, D, invalid head maps, page geometry, mask, bias, dropout, phase, and ragged requests | BF16, arbitrary-mask prefill/backward, Flash GQA, and ragged semantics remain implementation work |
| Real second fusion | `residual + RMSNorm(D=64)` is one warp-owned reduction/fusion using the same ABI, audit, gate, cache, graph-prewarm, tests, and benchmark framework | QKV projection/RoPE remains the next tensor-core fusion |
| Architecture families | Ampere, Ada, Hopper, and Blackwell are distinct dispatch domains | Only SM86 was executed and is admitted; every other SM target rejects rather than inheriting its tuning |

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
composition, and explicitly labeled TorchSharp Flash-preferred/Flash-disabled
SDPA lanes. The external harness
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

tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target decode -OutputCsv decode-ncu.csv

tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target paged-prefill -OutputCsv paged-prefill-ncu.csv

tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target attention-backward -OutputCsv attention-backward-ncu.csv

tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target flash-attention-backward -OutputCsv flash-attention-backward-ncu.csv
```

The script profiles a deterministic kernel target and fails unless executed
register-spill instructions, local loads, and local stores are all zero. Its
raw CSV also records registers/thread, static/dynamic shared memory, theoretical
occupancy, and achieved occupancy. PerfView remains appropriate for managed allocation
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

$env:AIDOTNET_DIRECT_PTX_FLASH_DECODE='1'
$env:AIDOTNET_DIRECT_PTX_PAGED_DECODE='1'
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release --no-build -- --direct-ptx-decode 3

python `
  tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_decode_competitors.py --runs 3

$env:AIDOTNET_DIRECT_PTX_PAGED_PREFILL='1'
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release --no-build -- --direct-ptx-paged-prefill 3

python `
  tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_paged_prefill_competitors.py --runs 3

$env:AIDOTNET_DIRECT_PTX_ATTENTION_BACKWARD='1'
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release --no-build -- --direct-ptx-attention-backward 3

$env:DIRECT_PTX_PY\Scripts\python.exe `
  tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_attention_backward_competitors.py --runs 3

$env:AIDOTNET_DIRECT_PTX_FLASH_ATTENTION_BACKWARD='1'
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release --no-build -- --direct-ptx-flash-attention-backward 3

$env:DIRECT_PTX_PY\Scripts\python.exe `
  tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_flash_attention_backward_competitors.py --runs 3

# Mandatory release capture: three separate processes per suite, guarded
# against concurrent compute, with SHA-256-addressed raw logs.
tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-release-evidence.ps1

$env:AIDOTNET_DIRECT_PTX_ATTENTION='1'
$env:AIDOTNET_DIRECT_PTX_RESIDUAL_RMSNORM='1'
$env:AIDOTNET_DIRECT_PTX_FLASH_DECODE='1'
$env:AIDOTNET_DIRECT_PTX_PAGED_DECODE='1'
$env:AIDOTNET_DIRECT_PTX_PAGED_PREFILL='1'
$env:AIDOTNET_DIRECT_PTX_ATTENTION_BACKWARD='1'
$env:AIDOTNET_DIRECT_PTX_FLASH_ATTENTION_BACKWARD='1'
# Or enable all admitted direct kernels:
$env:AIDOTNET_DIRECT_PTX='1'
```

Run on an otherwise idle system. The exact values are hardware, driver,
clock, thermal, and scheduler dependent; the contract and pass/fail process
are the reusable artifacts.
