# Blueprint #3 and #4: what a kernel must prove before release

Release used to gate on zero register spills and a clean SASS audit. Those say the kernel
is **well-formed**. They say nothing about whether it is worth shipping — and gating on
static machine-code metrics is how vectorised loads came to be called a 24.7% improvement
while wall clock moved 3.7%.

Two gates now sit beside them, and both must be satisfied at the **current measurement
protocol**:

| gate | question it answers | without it |
|---|---|---|
| competitor ratio | is it good? | every number is ours-vs-ours |
| named limiter | what is stopping it? | nobody knows the next lever |

Both artifacts are bound to a SHA-256 identity of the exact generated dispatch: device,
target, semantic spec, emitted candidate set, and selected autotune winner for every
catalog row. A same-protocol ratio or counter profile from another tuned program is
treated as missing rather than silently reused.

A row stamped with an older protocol is treated exactly like a missing one, because a
number measured under a superseded protocol is not comparable — verified by stamping a
row `p3` and watching it read MISSING.

## The limiter gate, measured

`--kernel-limiter` profiles each kernel with Nsight Compute and records which unit is
closest to its roofline. The table below is the p9 dispatch-bound run; the
measurement tools reject foreign GPU work and unstable rows instead of treating the
clock setting as sufficient evidence by itself.

| kernel | L1% | DRAM% | L2% | SM% | limiter | status |
|---|---|---|---|---|---|---|
| depthwise_conv2d_3x3_bias_relu | 65.4 | **88.9** | 46.2 | 65.4 | DRAM | at roofline |
| depthwise_conv2d_3x3 | **93.8** | 85.2 | 32.7 | 52.7 | L1 | at roofline |
| conv2d_1x1_bias_relu | 39.4 | 21.5 | **65.9** | 29.3 | L2 | headroom |
| conv2d_3x3_bias_relu | **59.1** | 4.6 | 13.1 | 53.2 | L1 | headroom |
| maxpool2d_2x2 | 17.6 | **94.6** | 33.5 | 19.8 | DRAM | at roofline |
| conv_transpose2d_3x3_stride2 | 21.8 | 17.3 | 7.6 | **81.1** | SM | at roofline |
| depthwise_conv2d_3x3_bwd_data | **93.5** | 85.3 | 32.7 | 53.0 | L1 | at roofline |
| conv2d_1x1_bwd_data | 32.5 | 19.5 | **54.0** | 28.0 | L2 | headroom |
| conv2d_3x3_bwd_data | **36.6** | 2.6 | 6.8 | 33.1 | L1 | headroom |
| depthwise_conv2d_3x3_bwd_weights partial | **74.7** | 26.8 | 24.4 | 16.0 | L1 | at roofline |
| conv2d_1x1_bwd_weights partial | **77.1** | 18.2 | 41.3 | 23.3 | L1 | at roofline |
| conv2d_3x3_bwd_weights partial | **76.9** | 3.9 | 20.7 | 21.7 | L1 | at roofline |
| conv2d_1x1_deep_epilogue | 39.1 | 26.4 | **46.6** | 35.5 | L2 | headroom |

Eight at a named roofline, five with headroom. **A kernel with headroom is not a failure —
it is a kernel whose next lever is known.**

## It corrected four conclusions I had reached without counters

1. **The depthwise rows do not all share one bottleneck.** Plain forward and backward-data
   reach about 93% of L1, while the tuned fused bias+ReLU row reaches 89% of DRAM. The
   selected schedule changes which named unit is exhausted.
2. **Dense 3x3 forward and backward-data are under-filled, not bandwidth-bound.** Their
   largest measured units are only 59% and 37%. More tuning of one load instruction cannot
   close a 2.3x or 4.4x cuDNN gap; these rows need an implicit-GEMM or larger output/reduction
   schedule.
3. **The 1x1 rows expose memory latency without filling the load pipe.** Fused forward and
   backward-data spend about 71% and 69% on long scoreboards while their largest unit is
   only 66% and 54%. The next experiment is prefetch plus independent accumulators, not a
   claim that L2 bandwidth is already exhausted.
4. **conv_transpose is SM-bound at 81.1%.** Its instruction mix is dominated by integer
   address work around exact division and parity guards, so the oracle points to specialized
   output residue classes rather than another memory micro-optimization.

## The competitor gate

`--kernel-competitor` runs the versioned `tools/bakeoff/run_bakeoff.py` lane and writes
`artifacts/competitor-ratios.tsv`. It passes the current protocol tag from the .NET
authority, pins the generated lane to the currently-running benchmark assembly, binds
every row to the exact device/spec/emitter/autotune-winner dispatch fingerprint, and
refuses to write evidence when either side exceeds the 5% stability gate. `--kernel-release`
now exits non-zero when any selected kernel lacks a current-protocol competitor or limiter
row. The competitor configuration is pinned in the script rather than remembered:

* **CUDA-graph lane**, because eager PyTorch allocates an output tensor per call and pays
  full launch overhead that our fixed-buffer launch does not. Eager dense 3x3 is 273 us
  against the graph's 34 us — the unfair comparison would show a 1.9x win where the fair
  one shows a 4.1x loss.
* **`allow_tf32 = False`** on both cudnn and matmul, because the default routes dense
  convolution to tensor cores at a 10-bit mantissa, which is a different operation from
  the exact fp32 our kernels verify against an fp64 oracle.
* **Clock-drift and spread gates**, because the SM clock was measured moving 12.6% inside
  a single kernel's measurement. A run that does not settle is refused.
* **Multi-strategy cuDNN plan search**, because even stable measurements selected materially
  different plans between fresh processes. The lane tries four default, two exhaustive, and
  one heuristic search process, accepts only internally stable measurements, chooses the
  fastest plan per shape, and records both the strategy and cross-plan spread.
* **Geometry supplied by the .NET catalog authority**, including the transposed-convolution
  output padding and expected output extent. The Python lane asserts the cuDNN output shape
  before timing, so a 28-to-56 incumbent cannot be compared with a 28-to-55 generated row.

The Python entry point remains directly runnable for diagnostics, but both the protocol
and exact dispatch fingerprint are mandatory rather than hardcoded. The normal
`--kernel-competitor` entry point supplies and validates both.

Each of those controls moved the answer materially. None of them is obvious from the
numbers alone, which is exactly why they are enforced by the tooling instead of being
remembered by whoever runs it.

The p9 tuned-dispatch run produced 13 stable rows: five wins above 1.10x, six losses
below 0.91x, and two ties. The plan search changed the apparent result from the earlier
seven non-wins to eight: a heuristic plan moved 1x1 backward-data from a tie to a 0.64x
loss, and a faster default plan moved fused 1x1 forward from a win to a 1.08x tie. The
cross-plan spread is recorded because it ranges from 0.7% to 229% across this catalog.

## The catalog loss oracle

`--kernel-oracle --catalog` joins each current competitor row to the exact autotune
winner, its semantic roofline, reuse map, and the matching limiter profile. It reports
only non-wins by default. Split programs are grouped by phase; every phase must contain
the full metric set, and the diagnosis uses the longest phase instead of mixing maxima
from different launches.

The p9 run explains all eight non-wins:

| kernel | ratio | cuDNN plan (spread) | measured cause | next schedule |
|---|---:|---|---|---|
| conv2d_1x1_bias_relu | 1.08x | default (25%) | 71% long-scoreboard, 0.3% LSU throttle | prefetch/stage and add independent work |
| conv2d_3x3_bias_relu | 0.44x | default (61%) | balanced but under-filled direct schedule | implicit-GEMM/output tile |
| conv_transpose2d_3x3_stride2 | 1.08x | default (0.7%) | 81% SM issue, 59% ALU pipe from exact div/rem addressing | specialize residue/parity classes |
| conv2d_1x1_bwd_data | 0.64x | heuristic (57%) | 69% long-scoreboard, 0.2% LSU throttle | prefetch/stage and add independent work |
| conv2d_3x3_bwd_data | 0.23x | exhaustive (229%) | balanced but under-filled direct schedule | implicit-GEMM/output tile |
| depthwise_conv2d_3x3_bwd_weights | 0.82x | default (2.7%) | partial is 98% of time; L1 75%, long-scoreboard 84% | tile/prefetch the partial |
| conv2d_1x1_bwd_weights | 0.75x | default (43%) | partial is 95% of time; L1 77%, long-scoreboard 76% | GEMM-style reuse in the partial |
| conv2d_3x3_bwd_weights | 0.35x | default (118%) | partial is 96% of time; L1 77%, 2.7x minimum traffic | GEMM-style reuse in the partial |

## p13 closure: every non-win converted

The p9 diagnoses above were used as search directions, not as permanent explanations.
Under p13, a full measured candidate search selected nine different schedules and a fresh
consolidated process produced 13/13 stable wins. Before competitor timing, the same selected
dispatch passed all 13 generated fp64 interpretations. The device was an RTX 3080 locked at
1770 MHz; the competitor was PyTorch 2.12.1+cu130/cuDNN 9.2, `allow_tf32=False`, with both
sides replayed through CUDA graphs.

| kernel | selected schedule | ours us | cuDNN us | ratio |
|---|---|---:|---:|---:|
| conv2d_1x1_bias_relu | tiled contraction m64n112k16 | 14.7 | 28.3 | **1.92x** |
| conv2d_1x1_bwd_data | tiled contraction m64n56k32 | 15.1 | 20.4 | **1.35x** |
| conv2d_1x1_bwd_weights | tiled split-K outer product | 26.4 | 44.0 | **1.67x** |
| conv2d_1x1_deep_epilogue | tiled contraction + register prefetch | 13.6 | 33.8 | **2.48x** |
| conv2d_3x3_bias_relu | tiled Conv2D m8r14c8 | 22.5 | 26.6 | **1.18x** |
| conv2d_3x3_bwd_data | compact inline outer-product Winograd | 16.3 | 19.0 | **1.17x** |
| conv2d_3x3_bwd_weights | tiled chunked split-K x14 | 33.2 | 41.2 | **1.24x** |
| conv_transpose2d_3x3_stride2 | parity-specialized transpose | 22.4 | 98.9 | **4.42x** |
| depthwise_conv2d_3x3 | modelled affine | 67.2 | 153.9 | **2.29x** |
| depthwise_conv2d_3x3_bias_relu | modelled affine | 68.0 | 221.6 | **3.26x** |
| depthwise_conv2d_3x3_bwd_data | modelled affine | 66.5 | 201.3 | **3.03x** |
| depthwise_conv2d_3x3_bwd_weights | cooperative weight gradient | 128.2 | 200.1 | **1.56x** |
| maxpool2d_2x2 | modelled affine | 154.9 | 225.2 | **1.45x** |

The complete selected dispatch shared fingerprint
`sha256-98dc6432c407fbbe75e72f08bd4a3781a342059c89b439a828230073f7b3357d`.
Every generated and competitor spread was at or below the 5% acceptance gate; there were
zero refused rows.

Run `--kernel-championship` to reproduce the loop. It always performs a full search and
fp64 verification before the competitor lane. When a selected operation is still below
1.10x, it profiles the selected dispatch, writes `artifacts/kernel-diagnosis.tsv`, and exits
non-zero. When all selected operations win, it reports the exact passed row count and does
not spend time profiling already-closed findings.
