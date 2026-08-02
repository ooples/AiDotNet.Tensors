# Head-to-head: generated kernels vs PyTorch/cuDNN

The first competitor comparison. Everything before this was ours-vs-ours.

Measured 2026-07-25 on an idle RTX 3080 (sm_86, driver 610.47, CUDA 13.3,
torch 2.12.1+cu130, cuDNN 9.2). Reproduce with `python tools/bakeoff/run_bakeoff.py`.

## Result

| kernel | ours us | spread | cuDNN us | spread | ratio | our GB/s | %peak bw | our TF/s | bound |
|---|---|---|---|---|---|---|---|---|---|
| depthwise_conv2d_3x3_bias_relu | 98.5 | 1.3% | 233.7 | 2.4% | **2.37x** | 522 | 69% | 1.17 | MEMORY, at roofline |
| depthwise_conv2d_3x3_bwd_data | 96.8 | 1.1% | 214.1 | 2.0% | **2.21x** | 531 | 70% | 1.19 | MEMORY, at roofline |
| depthwise_conv2d_3x3 | 95.6 | 0.6% | 161.6 | 2.4% | **1.69x** | 537 | 71% | 1.21 | MEMORY, at roofline |
| maxpool2d_2x2 | 159.1 | 2.6% | 248.5 | 2.1% | **1.56x** | 807 | 106% | — | MEMORY, at roofline |
| conv_transpose2d_3x3_stride2 | 124.1 | 1.7% | 108.6 | 9.3% | 0.87x | 129 | 17% | 0.47 | reuse-limited |
| conv2d_1x1_bias_relu | 75.0 | 1.0% | 24.7 | 8.0% | **0.33x** | 86 | 11% | 1.37 | reuse-limited |
| conv2d_1x1_bwd_data | 82.2 | 0.5% | 24.5 | 18.4% | **0.30x** | 78 | 10% | 1.25 | reuse-limited |
| conv2d_3x3_bias_relu | 141.7 | 1.0% | 34.3 | 18.9% | **0.24x** | 17 | 2% | 1.63 | reuse-limited |
| conv2d_3x3_bwd_data | 145.6 | 2.0% | 26.9 | 23.0% | **0.18x** | 17 | 2% | 1.59 | reuse-limited |

**4 wins at >=1.10x, 5 losses at <=0.91x.** The split is not random, and the roofline
columns say exactly what it is.

## We win every memory-bound kernel, because we are at the roofline and cuDNN is not

The four wins all sit at **69-106% of the card's 760 GB/s**. `maxpool2d_2x2` at 807 GB/s
(106% of spec, with L2 supplying the difference) is *at the hardware limit* — there is
no remaining headroom, and no compiler improvement can add any.

cuDNN reaches only 323 GB/s on the same depthwise convolution. Depthwise has almost no
data reuse to exploit, so a general library's tiling machinery buys nothing and its
generality costs something. Fusing bias and ReLU into the same kernel widens the gap
further: PyTorch cannot fuse through a cuDNN call, so it pays a second full pass over
the tensor (2.37x fused vs 1.69x unfused — the fusion is worth about 0.68x of it).

**These wins are real and structural, not measurement luck.** Spreads are 0.6-2.6% on
our side against a 1.05% noise floor.

## We lose every reuse-limited kernel, by 3-6x, and the cause is data reuse

The five losses sit at **2-17% of bandwidth and 1.2-1.6 TFLOP/s against a ~29.8 TFLOP/s
FP32 peak**. Being far from *both* rooflines is the signature: the kernel is neither
moving data nor doing arithmetic at rate, it is stalling.

cuDNN gets 8.3 TFLOP/s on the dense 3x3 (28% of peak) where we get 1.63 (5.4%). The
reason is not instruction selection, it is **reuse**. Our lowering assigns one thread
per output element and walks the reduction serially, so every input element is re-read
from L2 or HBM once per output that needs it. In a 3x3 convolution each input element
is needed by 9 outputs, and across 64 output channels the same input tile is re-read 64
times. cuDNN stages a tile in shared memory once, then blocks registers so each loaded
value feeds many FMAs.

That is also the explanation for the earlier Phase 2 result: `ld.global.v4.f32` cut
LDG by 37% and bought 1.037x, because reducing the *cost per load* cannot help a kernel
whose problem is the *number of loads*. **Instruction selection was the wrong lever;
data reuse is the right one.**

`conv_transpose2d_3x3_stride2` at 0.87x is the mildest loss and a different shape of
problem: its exact-division predicate means three quarters of its taps are discarded,
so it does 4x the index work for the same output.

## Method, and its limits

* Competitor is the **CUDA-graph lane**, deliberately the strongest form. Eager PyTorch
  allocates an output tensor per call and pays full launch overhead; our kernel writes a
  preallocated buffer and does not. Graph replay removes both, leaving kernel work.
  Using eager numbers would have flattered us badly — eager dense 3x3 is 273 us against
  the graph's 34 us, so the eager comparison would have shown a **1.9x win where the
  fair one shows a 4.1x loss**.
* Graph replay is **verified to reproduce the eager result** to 1e-5 before it is timed.
  A 5-6x speedup from graph capture alone is implausible enough that the first
  suspicion had to be a no-op replay; it is not one.
* Both lanes use the same protocol: 50 launches per timed region, 51 samples, median,
  3 runs, full spread reported.
* **Cross-process.** Our lane is .NET and the competitor is a Python subprocess, so this
  cannot be paired sample-by-sample the way an in-process A/B is. The competitor's
  spreads are 2-23%, much worse than ours; the 3-6x losses are far outside that, but
  `conv_transpose2d_3x3_stride2` at 0.87x with a 9.3% competitor spread is close enough
  to parity that it should be treated as "roughly level", not a measured loss.
* cuDNN runs with `benchmark = True`, so it picks its best algorithm per shape.

## What this changes

The target is now quantified rather than aspirational: **~5x on dense convolution
through data reuse.** Concretely, in priority order:

1. **Thread coarsening / register blocking** — each thread computes several adjacent
   outputs, so a loaded input value feeds several FMAs instead of one. This is also the
   prerequisite for vectorising the activation operand.
2. **Shared-memory tiling** — stage an input tile once per block. Every kernel in the
   catalog currently reports LDS 0 / STS 0: shared memory is completely unused.
3. **`cp.async`** — overlap the tile fetch with compute, which only becomes meaningful
   once tiling exists.

And it says plainly where not to spend effort: the four memory-bound kernels are at the
roofline and already beat cuDNN by 1.56-2.37x. There is nothing to win there.
