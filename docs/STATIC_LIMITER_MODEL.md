# Predicting a kernel's bottleneck from its index maps, without a GPU

## Why this exists

The release gate was static machine-code metrics: SASS instruction count, LDG count,
registers, spills. Those said vectorised loads were a 24.7% improvement. Wall clock moved
3.7%. **The gate measured what was easy to measure, not what decides whether a kernel is
competitive**, and that mistake cost a full optimisation cycle.

Profiling eventually showed dense convolution pinned at 89.99% of L1 request throughput
while issuing 158x more load instructions than cuDNN for identical arithmetic. That was
derivable from the specification alone: the index maps say how many loads each output
costs, and the machine says how many it can retire.

This model does that arithmetic up front. A kernel that cannot win is identified before
anyone writes a lowering for it, and it costs nothing per kernel — the only property that
matters when there are hundreds.

Run it with `--kernel-predict`. It never touches the GPU.

## What it computes

| quantity | source |
|---|---|
| output elements, MACs | iteration space |
| unique bytes | binding shapes |
| dynamic loads per thread | emitter, counting loop bodies once per trip |
| warp load instructions | thread loads / 32 |
| **loads per MAC** | the headline diagnostic |
| time if load-issue bound | warp loads / device load-issue rate |
| time if DRAM bound | unique bytes / device bandwidth |
| time if compute bound | MACs / device FMA rate |
| **predicted limiter** | whichever of the three is slowest |

The load count comes *from the emitter* rather than being re-derived. Re-deriving would
duplicate every lowering decision — coarsening, operand sharing, vectorisation,
strip-mining — and the copy would drift, which is exactly the defect class the index-map
IR exists to prevent. Emission is pure string building, so asking costs nothing.

One calibrated constant: **0.293 warp-level global load instructions per SM per cycle**.
Derived, not guessed — dense 3x3 executed 4,017,216 warp loads in 126.85 us on 68 SMs at
1770 MHz with Nsight Compute reporting l1tex at 89.99% of peak, giving
`(4.017e6 / 126.85e-6) / (68 x 1.77e9) / 0.8999 = 0.293`.

## Validation against measured hardware

Locked clocks at 1770 MHz, true fp32 on both sides.

| kernel | loads/MAC | predicted | limiter | measured | ratio |
|---|---|---|---|---|---|
| conv2d_3x3_bwd_data | 1.250 | 128.1 us | LoadIssue | 127.9 | **1.00x** |
| conv2d_3x3_bias_relu | 1.251 | 128.1 us | LoadIssue | 125.9 | **1.02x** |
| maxpool2d_2x2 | 1.000 | 169.0 us | DramBandwidth | 156.2 | 1.08x |
| depthwise_conv2d_3x3_bwd_data | 1.250 | 67.6 us | DramBandwidth | 72.3 | 0.94x |
| depthwise_conv2d_3x3 | 1.250 | 67.6 us | DramBandwidth | 72.6 | 0.93x |
| depthwise_conv2d_3x3_bias_relu | 1.278 | 67.6 us | DramBandwidth | 73.0 | 0.93x |
| conv2d_1x1_bwd_data | 0.500 | 22.8 us | LoadIssue | 47.5 | 0.48x |
| conv2d_1x1_bias_relu | 0.316 | 14.4 us | LoadIssue | 45.8 | 0.31x |
| conv_transpose2d_3x3_stride2 | 1.250 | 32.0 us | LoadIssue | 99.2 | 0.32x |

**Every limiter prediction is correct** wherever hardware evidence exists:

* dense 3x3 predicted LoadIssue — Nsight Compute measured l1tex 89.99%, DRAM 2.41%;
* the depthwise family predicted DramBandwidth — measured at 93-94% of the roofline;
* maxpool predicted DramBandwidth — measured at 108% of it.

**Six of nine runtimes land within 8%**, including both dense 3x3 kernels at 1.00x and
1.02x — the two the model was built to diagnose.

## Where it is wrong, and why that is recorded rather than hidden

Three kernels are predicted **2-3x too fast**. The model does not pretend otherwise: a
test pins them as known-optimistic and fails if one silently becomes accurate.

* **conv2d_1x1 (0.31x, 0.48x)** — `ow` is 28, not a multiple of the warp width. With
  4-wide coarsening seven threads cover a row, so a 32-thread warp straddles about 4.6
  rows and its requests span many cache lines. The model counts *instructions* and
  assumes each is one efficient transaction; it has no sector-efficiency term.
* **conv_transpose (0.32x)** — 78 registers per thread limits occupancy, and its
  exact-division predicates mean three quarters of issued loads are predicated off. The
  model has no occupancy term.

So the model is an **honest lower bound**: how fast a kernel could run if the counted
loads were the only constraint. When measurement lands far above the prediction, that gap
is itself the signal — it says coalescing or occupancy is the next thing to look at, not
load count.

## How to use it on a new kernel

1. Run `--kernel-predict` before writing any lowering.
2. If `loads/MAC >= 1.0` the kernel **cannot** reach the compute roofline. Fix the
   iteration space before tuning anything else.
3. Read the reuse axes it prints. An operand invariant in an axis can share one load
   across every position of that axis; those are the axes worth tiling.
4. Compare measured against predicted. Close means the model understands the kernel;
   measured much slower means coalescing or occupancy, not load count.

For the current catalog the reuse analysis reports exactly the structure that explains the
dense-convolution loss:

```
conv2d_3x3_bias_relu   input     invariant in {k}          <- the 64x redundancy
conv2d_3x3_bias_relu   weights   invariant in {n, oh, ow}
```

The input is independent of the output-channel axis, so tiling K shares one input load
across every output channel in the tile. That is the next change, and the model already
predicts what it is worth.
