# Phase 3 spike: how much is a resident program actually worth?

A resident program runs a chain of operators as ONE device program: no launch
boundary between stages, no intermediate tensor pushed to HBM and pulled back. It is
a large amount of compiler work (device-side work queues, grid-wide dependencies,
halo handling), so the first question is not how to build it but how much there is
to win.

Measured 2026-07-25, idle RTX 3080 (sm_86, driver 610.47). Chain: dense 3x3 conv
+ ReLU, N8/C32->K64 at 28x28, then 2x2 max pool. Run it with `--resident-spike`.

## The device, characterised rather than sampled

| elements | traffic | us/launch |
|---|---|---|
| 4,096 | 0.03 MiB | 24.52 |
| 16,384 | 0.12 MiB | 24.84 |
| 65,536 | 0.50 MiB | 21.47 |
| 262,144 | 2.00 MiB | 17.40 |
| 1,048,576 | 8.00 MiB | 21.26 |
| 4,194,304 | 32.00 MiB | 47.14 |
| 16,777,216 | 128.00 MiB | 175.35 |

* **launch floor** (median of sub-L2 sizes, where traffic is negligible): **21.5 us**
* **HBM bandwidth** (fit over the >L2 sizes, R^2 = 1.0000): **785 GB/s** against a
  ~760 GB/s card spec -- i.e. the copy kernel is running at the hardware limit

## The prize

| measurement | us/launch |
|---|---|
| conv 3x3 + ReLU alone | 141.8 |
| chain (conv then pool) | 147.9 |
| **marginal cost of stage 2, paired** | **4.3** |
| stage-1 write of the intermediate (fitted) | 2.0 |

**Upper bound on the fusion win: 4.3% of the chain.** And that is a ceiling that
cannot be reached: it credits fusion with stage 2's arithmetic and final write, which
a fused kernel still pays, and charges it nothing for recomputing halo values.

**Verdict: do not build resident programs for compute-heavy chains.**

## The result that decides it

A launch costs **21.5 us** in isolation, but adding the pool stage to the chain costs
only **4.3 us**. Launches into a stream are asynchronous, so the second kernel's setup
overlaps the first kernel's execution: the launch is already hidden behind 142 us of
convolution. Fusion cannot remove a cost that is not being paid.

This inverts the premise. The launch floor matters only when ops are too small to hide
it -- a network of many small ops pays ~21.5 us *per op* with nothing to overlap. That
is where a resident program earns its keep, and it is the case to measure next. This
chain is the wrong test because it is compute-bound by a wide margin.

## Method, and two corrections it forced

The first version point-estimated both costs: one tiny-kernel median for the launch
floor, one copy median for the traffic. It reported a **31.2%** bound. Re-running it
reported **14.3%**. Neither was a measurement.

**Correction 1 -- fit, do not sample.** The tiny kernel moved 13.2 -> 32.6 us between
runs. Sweeping the copy kernel across four orders of magnitude and fitting the
relationship averages the per-point noise instead of inheriting it.

**Correction 2 -- two regimes, not one line.** The first fit reported 814 GiB/s, above
this card's ~708 GiB/s peak and therefore impossible for HBM traffic. Every buffer up
to ~8 MiB of traffic is largely resident in the 5 MB L2 and moves faster per byte than
memory can. Fitting one line through cache-resident and memory-bound points models
neither. The launch floor now comes from the sub-L2 end and the bandwidth from the
points that exceed L2, which lands at 785 GB/s against a 760 GB/s spec.

**Correction 3 -- pair the subtraction.** Subtracting two independently-measured
medians inherits both their errors; that is how the pool stage read 28.9 us on one run
and 16.7 on the next. Interleaving "conv" against "conv then pool" and taking the
median per-sample *difference* measures the marginal cost directly, with drift
cancelling inside each pair.
