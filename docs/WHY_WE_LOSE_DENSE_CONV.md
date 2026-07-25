# Why we lose dense convolution (and only dense convolution)

Same hardware, same numerics: we beat cuDNN by 2.1-3.0x on depthwise and pooling while
losing 1.8-4.1x on dense convolution. This is the analysis of that one gap.

It is not a measurement artefact. The losses are 3-6x against a harness noise floor of
1.05%, they reproduce across every run, and they hold with steady clocks and best-of-3
reporting.

Hardware counters were unavailable (`ncu` returns `ERR_NVGPUCTRPERM` without
administrator rights), so the diagnosis rests on two things that need no profiler: exact
arithmetic over the index maps, and controlled sweeps where the code generator varies one
knob at a time.

## 1. The exact cause: input loads amplified 576x, of which 64x is pure waste

Dense 3x3 bench shape, N8 / C32 -> K64 / 28x28:

| quantity | value |
|---|---|
| unique input bytes | 0.80 MB |
| input load traffic our kernel issues | 462 MB |
| **amplification** | **576x** |
| ...inherent to a 3x3 stencil | 9x (each element genuinely feeds 9 outputs) |
| ...redundant across the K axis | **64x** |
| product | 9 x 64 = 576x, matching the measured traffic exactly |

The 64x comes straight from the index maps:

```
input   [n, c, oh + kh - 1, ow + kw - 1]    <- no k
weights [k, c, kh, kw]                       <- has k
output  [n, k, oh, ow]                       <- has k
```

**The input index does not depend on k.** Our iteration space makes `k` a parallel axis,
so each of the 64 output channels is a separate thread and each independently re-loads the
same input element. The arithmetic is right; the loads are 64x more numerous than the data
requires.

## 2. Spatial coarsening cannot fix it, and the sweep shows the lever is spent

Coarsening along the contiguous spatial axis shares the *weight* load across lanes but
leaves the *input* load per lane. So loads per MAC is `(Tw + 1)/Tw`, whose **asymptote is
1.0 however wide the tile gets**. cuDNN needs roughly 0.03.

Measured, varying only the coarsening factor (min of 3 attempts, steady clock):

| coarsen | loads/MAC | dense 3x3 | blocks | dense 1x1 | blocks |
|---|---|---|---|---|---|
| 1 | 2.000 | 143.1 us | 1,568 | 74.8 us | 3,136 |
| 2 | 1.500 | 109.6 us | 784 | 80.8 us | 1,568 |
| 4 | 1.250 | 130.3 us | 392 | **43.5 us** | 784 |
| 7 | 1.143 | **100.0 us** | 224 | 51.2 us | 448 |
| 14 | 1.071 | 146.9 us | 112 | 108.5 us | 224 |

**Non-monotonic, with a shallow optimum, then collapse.** Two forces cross: the load
saving from a wider tile shrinks fast (2.00 -> 1.50 is a 25% cut, 1.14 -> 1.07 is 6%),
while occupancy falls linearly with block count. At coarsen=14 there are 112 blocks for 68
SMs, under two per SM, and there is no longer enough concurrency to hide memory latency.

Even the best point, 100.0 us at coarsen=7, is still **3.3x behind cuDNN's 30.6 us**.

## 3. The ordering across the catalog confirms the mechanism

If loads-per-MAC drives the gap, the kernel with the fewest should sit closest to cuDNN.
It does:

| kernel | loads/MAC | vs cuDNN |
|---|---|---|
| conv2d_1x1 — input vectorises across lanes (1 v4 + 1 weight per 4 MACs) | **0.50** | 0.56x |
| conv2d_3x3 — input is a gathered window, cannot vectorise | **1.25** | 0.25x |

The 1x1's input is `input[n, c, oh, ow]`, unit-stride in the coarsened axis, so four lane
values arrive in one `ld.global.v4.f32`. The 3x3's input is
`input[n, c, oh+kh-1, ow+kw-1]`, a gathered window with no alignment guarantee, so it stays
on four scalar loads. That one structural difference accounts for 2.5x in loads per MAC and
tracks 2.2x in competitive position.

## 4. What fixes it: a 2D register tile over (K x spatial)

The input is independent of `k` and the weights are independent of `ow`, so a tile over
**both** axes lets each load feed several MACs in both directions:

```
loads per MAC for a Tk x Tw tile = (Tw + Tk) / (Tk * Tw)
```

| tile | loads/MAC | vs today's 1.25 |
|---|---|---|
| Tk=1, Tw=4 (today) | 1.250 | — |
| Tk=2, Tw=4 | 0.750 | 1.7x fewer |
| Tk=4, Tw=4 | 0.500 | 2.5x fewer |
| Tk=4, Tw=8 | 0.375 | 3.3x fewer |
| Tk=8, Tw=8 | 0.250 | 5.0x fewer |

The asymmetry is the point: with `Tk = 1` the expression collapses to `(Tw+1)/Tw` and can
never go below 1.0. **Only a second tiled axis breaks the barrier.** That is what cuDNN's
implicit-GEMM kernels do, and why they reach 8.3 TFLOP/s where we reach 1.9.

Registers are the constraint — `Tk * Tw` accumulators plus `Tk + Tw` operands. An 8x8 tile
needs about 64 accumulators; the transposed-convolution kernel already runs at 78 registers
with zero spills, so it is within budget on paper.

**A prediction to test, not a claim:** a 4x4 tile cuts load instructions 2.5x. If the
kernel is load-issue bound, dense 3x3 should land near 50-60 us (from 130) and dense 1x1
near 25-30 us. Landing well short would mean load issue is not the binding resource, and
the next suspect is L2 bandwidth for the input re-reads — which shared-memory staging
addresses instead.

## 5. Ruled out

* **Measurement.** 3-6x gaps against a 1.05% noise floor, reproducing across runs with
  steady clocks.
* **Numerics or a shortcut.** All 9 kernels verify at exactly 0.000E+000 against the fp64
  interpretation of their own spec, so the full arithmetic is being done.
* **Register spilling.** 9/9 report zero spill loads and zero spill stores in SASS.
* **Instruction selection.** Phase 2 cut LDG by 37% with `ld.global.v4.f32` and bought
  1.037x. Reducing cost per load cannot help when the problem is the number of loads.
* **HBM bandwidth.** The dense kernels sit at 3% of peak DRAM bandwidth. The 0.80 MB input
  fits in the 5 MB L2, so we are not starved of memory — we issue far too many requests
  for it.
* **Fusion or the epilogue.** Those are wins: they are why depthwise beats cuDNN by up to
  3.03x.

The gap is one thing: **we issue roughly 30x more load instructions per MAC than a tiled
implementation, because our iteration space assigns one thread per output element and so
cannot reuse an operand across the axis it does not depend on.**
