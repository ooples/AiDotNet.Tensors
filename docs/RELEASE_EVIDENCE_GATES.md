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

A row stamped with an older protocol is treated exactly like a missing one, because a
number measured under a superseded protocol is not comparable — verified by stamping a
row `p3` and watching it read MISSING.

## The limiter gate, measured

`--kernel-limiter` profiles each kernel with Nsight Compute and records which unit is
closest to its roofline. Clocks locked at 1770 MHz.

| kernel | L1% | DRAM% | L2% | SM% | limiter | status |
|---|---|---|---|---|---|---|
| depthwise_conv2d_3x3_bwd_data | **93.5** | 79.7 | 37.4 | 52.6 | L1 | at roofline |
| depthwise_conv2d_3x3_bias_relu | **93.0** | 78.8 | 37.5 | 53.0 | L1 | at roofline |
| depthwise_conv2d_3x3 | **92.9** | 77.7 | 38.4 | 52.0 | L1 | at roofline |
| maxpool2d_2x2 | 46.2 | **94.2** | 50.7 | 11.3 | DRAM | at roofline |
| conv_transpose2d_3x3_stride2 | 20.9 | 16.5 | 7.0 | **82.5** | SM | at roofline |
| conv2d_1x1_bias_relu | 39.5 | 24.0 | **66.3** | 29.5 | L2 | headroom |
| conv2d_3x3_bias_relu | **53.4** | 3.9 | 10.1 | 44.4 | L1 | headroom |
| conv2d_1x1_bwd_data | 32.7 | 20.2 | **53.5** | 28.3 | L2 | headroom |
| conv2d_3x3_bwd_data | **38.4** | 2.9 | 6.2 | 33.5 | L1 | headroom |

Five at a named roofline, four with headroom. **A kernel with headroom is not a failure —
it is a kernel whose next lever is known.**

## It corrected three conclusions I had reached without counters

1. **The depthwise family is L1-bound at 93%, not DRAM-bound.** I had classified it as "at
   the DRAM roofline" from computed GB/s (93% of 760 GB/s). The counters say DRAM is 79%
   and L1 is 93%, so L1 is the tighter constraint. Both are near saturation, which is why
   the conclusion "no headroom" survived — but the *named* unit was wrong, and the name is
   what tells you what to try.
2. **conv2d_3x3's L1 pressure fell from 89.99% to 53.4%** after reuse tiling. That is the
   tiling working exactly as designed, and it means nothing is saturated now (max 53%) —
   the kernel is latency- or occupancy-bound, which matches the model going from 1.02x to
   0.68x accurate on it. Fixing the bottleneck moved the bottleneck, and the gate says
   where it moved to.
3. **conv2d_1x1 is now L2-bound at 66%**, not load-issue bound as the static model
   predicts. Its next lever is L2 traffic, not fewer load instructions.
4. **conv_transpose is SM-bound at 82.5%** — issue-limited, not memory-limited at all.
   That explains why the static model was 3x optimistic about it: the model has no
   issue-rate term, and this kernel is the one kernel where that term dominates.

## The competitor gate

`tools/bakeoff/run_bakeoff.py` writes `artifacts/competitor-ratios.tsv`, and release
refuses to call a kernel proven without a current-protocol row in it. The competitor
configuration is pinned in the script rather than remembered:

* **CUDA-graph lane**, because eager PyTorch allocates an output tensor per call and pays
  full launch overhead that our fixed-buffer launch does not. Eager dense 3x3 is 273 us
  against the graph's 34 us — the unfair comparison would show a 1.9x win where the fair
  one shows a 4.1x loss.
* **`allow_tf32 = False`** on both cudnn and matmul, because the default routes dense
  convolution to tensor cores at a 10-bit mantissa, which is a different operation from
  the exact fp32 our kernels verify against an fp64 oracle.
* **Locked clocks**, because the SM clock was measured moving 12.6% inside a single
  kernel's measurement.

Each of those three moved the answer materially. None of them is obvious from the
numbers alone, which is exactly why they are enforced by the tooling instead of being
remembered by whoever runs it.
