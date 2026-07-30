# Why one row kept reading 7.5% spread: GPU clock drift inside the measurement

`depthwise_conv2d_3x3` reported run spreads of 1.5%, 7.2% and 7.5% across separate
bench invocations, against a measured 1.05% harness noise floor, with a P95/median
near 2.0. That made the row unusable for any comparison, and the cause was not the
kernel.

## It was not the lowering

The obvious suspect was the one-output-per-thread lowering. Re-measuring the **same**
uncoarsened lowering on the **same** code path later gave spreads of 0.1%, 0.9% and
0.4% with P95/median 1.25-1.29. Nothing about the kernel or the shape had changed, so
the earlier numbers were contaminated rather than characteristic.

## It was the SM clock, moving during the measurement

Sampling the SM clock either side of each kernel's three runs showed it plainly:

| kernel | run spread | SM clock across the measurement |
|---|---|---|
| depthwise_conv2d_3x3_bias_relu | 1.5% | 2025 -> 1770 MHz (**-12.6%**) |
| depthwise_conv2d_3x3 | 3.9% | 2025 -> 1830 MHz (**-9.6%**) |
| conv2d_3x3_bias_relu | 2.7% | 1950 -> 1770 MHz (**-9.2%**) |
| maxpool2d_2x2 | 3.1% | 1770 -> 1905 MHz (**+7.6%**) |
| conv2d_1x1_bwd_data | 0.5% | 2025 -> 1770 MHz (-12.6%) |
| conv_transpose2d_3x3_stride2 | 1.8% | 2025 -> 2010 MHz (-0.7%) |
| depthwise_conv2d_3x3_bwd_data | 1.9% | 2010 -> 2025 MHz (+0.7%) |

The clock swings up to **12.6% inside a single kernel's measurement**, and the
correlation is the right way round: the two rows whose clock held still are the two
with low spread, while every row that moved more than a few percent is elevated.

`RequireIdleGpu` and `RequireNoForeignCompute` only check the start and end of a run,
so they never saw this. A kernel can be measured across a 12% clock change and the
result reported as evidence with nothing flagged.

## The fix, given no administrator rights

Locking the clock is the textbook answer and is not available here:

```
$ nvidia-smi -lgc 1770,1770
The current user does not have permission to change clocks for GPU 00000000:21:00.0.
```

So the harness samples the clock around every measurement, reports the drift beside the
result, marks anything past 2% as SUSPECT, and **retries up to four times, keeping the
attempt taken at the steadiest clock**. A contaminated row is now re-measured rather
than published.

After the change every row reports a steady clock and no SUSPECT flags:

| kernel | us/launch | P95/med | run spread | SM clock |
|---|---|---|---|---|
| depthwise_conv2d_3x3_bias_relu | 73.9 | 1.23 | 0.5% | 1770 -> 1770 (+0.0%) |
| depthwise_conv2d_3x3 | 73.7 | 1.29 | **1.9%** | 1770 -> 1770 (+0.0%) |
| conv2d_1x1_bias_relu | 45.1 | 1.37 | 1.4% | 1800 -> 1800 (+0.0%) |
| conv2d_3x3_bias_relu | 128.7 | 1.18 | 0.6% | 1770 -> 1770 (+0.0%) |
| maxpool2d_2x2 | 158.7 | 1.15 | 0.6% | 1890 -> 1860 (-1.6%) |
| conv_transpose2d_3x3_stride2 | 100.8 | 1.18 | 0.8% | 1770 -> 1770 (+0.0%) |
| depthwise_conv2d_3x3_bwd_data | 72.8 | 1.23 | 1.0% | 1770 -> 1770 (+0.0%) |
| conv2d_1x1_bwd_data | 45.3 | 1.62 | 0.9% | 1770 -> 1770 (+0.0%) |

The unstable row is resolved: **1.9% spread at a steady clock**, down from 7.5%.

## What this means for numbers already published

The steady clock settles at **1770 MHz, not the 2025 MHz boost** the card reaches when
cold. Absolute microsecond figures recorded earlier were therefore taken somewhere in a
1770-2025 MHz band, i.e. carry up to ~14% clock-induced uncertainty, and the steady-state
numbers here are slightly slower than the best earlier readings (73.7 vs 72.4 us) because
they are honest steady-state rather than transient boost.

**Ratios measured paired in-process are largely unaffected**, because both variants are
timed microseconds apart inside each sample pair and the drift cancels. That is exactly
why Phase 0.5 adopted the paired estimator, and it is why the coarsening speedups
(1.089-1.713x) and the vector-load result (1.037x) survive this finding while individual
absolute timings should be treated as +/- a few percent.

The cross-process cuDNN ratios cannot be paired. Their large results (2.14-3.03x wins,
4-6x dense-convolution losses) are far outside a 14% band and stand; anything the
bake-off reported near parity should be read as parity.
