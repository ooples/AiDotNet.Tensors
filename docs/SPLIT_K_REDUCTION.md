# Split-K: parallelism from the reduction

## The gap

The emitter maps parallel axes to threads and reduction axes to loops, always. When the
parallel axes are small and the reduction is enormous, the machine sits idle:

| | `depthwise_conv2d_3x3_bwd_weights` |
|---|---|
| output elements (dW is `[C,3,3]`) | 576 |
| reduction length (`n x oh x ow`) | 100,352 |
| MACs | 57.8 M |
| threads at one output per thread | 576 |
| blocks on a 68-SM device | **3** — 4% of one wave |
| measured | **4063.5 us** |
| compute roofline | 3.8 us |
| ratio | **1081x** |

The autotuner tried every candidate lowering — `no-tile`, `tile2`, `lanes4`,
`no-staging`, `no-vector` — and none of them moved it. No tile *can*: tiling redistributes
work among threads that exist, and the problem is that only 576 exist.

This is not specific to depthwise. Every weight gradient has this shape, as does every
norm, every loss, and every global pooling. It also outranks the dense-convolution gap,
which is 1.5x in a battle the targeting map says not to pick.

## The transform

Promote a reduction axis to a parallel axis. Its extent becomes threads instead of loop
trips, and the kernel writes one partial result per position of it. A second kernel sums
over the new dimension.

Every index map is untouched — the axis keeps its index, only its *role* changes. The
output gains the promoted axis as its last dimension, which is also the fastest-varying
one in the thread decomposition, so the partial pass's stores stay coalesced.

`CodegenSplitReduction.TryPlan(spec)` returns the two kernels plus the size of the
temporary between them, or `null` when the kernel already fills the device.

## Two passes, not atomics

An `atomicAdd` combine needs no temporary and no second launch. It was rejected: fp32
atomic addition is order-nondeterministic, so the result changes run to run and the exact
`0.000E+000` agreement gate would have to become a tolerance. That gate has caught four
real defects in this project that the structural gates passed. The second launch was
measured at about 10 us against a 4063 us kernel.

## What it bought (p4, locked clocks, best-of-3)

| kernel | promoted | unsplit | partial | combine | total | gain | max rel. dev |
|---|---|---|---|---|---|---|---|
| `depthwise_conv2d_3x3_bwd_weights` | `oh(56)` | 4063.5 | 232.4 | 10.5 | **236.4** | **17.19x** | 5.3E-004 |
| `depthwise_conv2d_3x3_bwd_weights` | `oh(56)+ow(56)` | 4075.5 | 210.3 | 119.9 | 334.7 | 12.18x | 5.2E-004 |
| `conv2d_1x1_bwd_weights` | `oh(28)` | 2133.7 | 54.5 | 12.2 | **58.2** | **36.67x** | 0.0E+000 |
| `conv2d_3x3_bwd_weights` | `oh(28)` | 240.7 | 114.1 | 10.0 | **117.4** | **2.05x** | 0.0E+000 |

Two of the three agree with the unsplit kernel to the last bit. The depthwise deviation is
the pre-existing fp32 accumulation-order difference over 100,352 terms that this kernel
already showed before the split (`5.589E-004` on the conveyor); it is within the 2E-003
tolerance, and the split sums in a *shallower* order than the serial loop it replaces.

Even at 17.19x, depthwise is still 62x off its roofline. Split-K removed the structural
idleness; what remains is an ordinary tuning gap on a kernel that now has work to tune.

## The model was wrong again, in the direction it always is

One axis reached 126 blocks on 68 SMs — under two per SM — so the blocks-per-SM model said
a second axis should help again. It did not:

| | partial | combine | total |
|---|---|---|---|
| one axis | 235.9 | 11.0 | **240.8** |
| two axes | 209.9 | 119.1 | 334.7 |

The partial pass *did* get faster, exactly as predicted. The combine pass is **itself** a
small-output long-reduction kernel — 576 threads, 3 blocks — and promoting a second axis
grew its reduction from 56 to 3136, so it inherited the precise problem being fixed. The
model had no term for that, because the model does not know the combine exists.

Reproduced across two independent runs (240.8 / 334.7, then 236.4 / 334.7).

So `ChooseAxes` returns a **candidate ranking, not a decision**; `TryPlan` defaults to the
one axis that was measured to win; and `--kernel-splitk` measures every prefix and records
what the hardware said. This is the fourth time on this project that a model-chosen
lowering lost to a measured one — see `RELEASE_EVIDENCE_GATES.md`.

## Reproducing

```
nvidia-smi -lgc 1770,1770
dotnet run --project tests/AiDotNet.Tensors.Benchmarks -c Release -f net10.0 -- --kernel-splitk all
```

Writes `artifacts/splitk.tsv`. Every row carries the measured deviation from the unsplit
kernel's own output; a row is only meaningful if that column is at or near zero.

## Not yet done

The split is emitted, executed from a graph, and now chosen by the autotuner — see the
sections below. What remains is that the conveyor's single-kernel stages cannot *launch* a
split winner, so a kernel the tuner measured as 17× faster still verifies and benches under
its single-kernel lowering. Teaching those stages the two-kernel program is the next step.

Remaining order: per-dimension staging (predicted ~15 us on
dense 3x3 against cuDNN's 41.0), then `cp.async` and tensor cores — the last two still
aimed by the targeting map at depthwise, pooling and memory-bound fusion chains, not at
dense convolution with large channel counts.

## Chunking: promoting an axis whole is not always a split

Measured on an idle device (SM clock 1770→1770 MHz, +0.0%), the first version LOST on
every graph the front end could produce:

| graph | single | split | launch config |
|---|---|---|---|
| matmul 128x96x64 | 11.5 | 29.9 | 32blk → 3072+32blk |
| matmul A-transposed | 9.3 | 35.3 | 32blk → 3072+32blk |
| matmul B-transposed | 18.3 | 28.1 | 32blk → 3072+32blk |
| linear 256x128x64 | 14.4 | 49.7 | 16blk → 8192+64blk |
| reduce-sum [512,256] | 175.1 | 186.6 | 1blk → 512+1blk |

The launch-config column carries the diagnosis: **the combine pass's block count never
improves on the original's.** Every one of those graphs has a single reduction axis, and
promoting it whole leaves the partial pass reducing nothing while the combine performs the
entire reduction with the original kernel's thread count. The combine *is* the original
kernel, and the partial is a wasted copy. That is structural, not a tuning miss.

`SplitChunked` fixes it: the axis is cut into `splitFactor` chunks, the chunk index becomes
the parallel axis, and the work *within* a chunk stays a loop — so the partial keeps real
reduction and the combine's reduction is genuinely shorter than the one it replaced. The
chunk index folds into operand maps as an extra term rather than a substitution, which
keeps compound maps (a convolution window is `oh*stride + kh - pad`) correct without
unfolding them. A factor that does not divide the extent is refused, because a partial
chunk would need a bounds guard on a reduction axis and that reads outside the operand.

Re-measured with chunking:

| graph | single | split | outcome |
|---|---|---|---|
| reduce-sum [512,256] | 174.6 | **90.3** | **1.93× faster** |
| matmul 128x96x64 | 14.2 | 36.5 | 2.57× slower |
| matmul A-transposed | 14.9 | 29.1 | 1.95× slower |
| matmul B-transposed | 21.3 | 35.1 | 1.65× slower |
| linear 256x128x64 | 16.1 | 30.5 | 1.89× slower |

Chunking turned the reduction from 1.07× slower into **1.93× faster**. The matmuls still
lose, and that is not a defect in the split — it is a size effect.

## Why the split cannot be chosen statically

The obvious gates do not separate the winners from the losers.

**Not block count.** `conv2d_3x3_bwd_weights` won 2.05× at 64 blocks; the linear layer lost
1.89× at 16 blocks. More blocks, bigger win — backwards.

**Not arithmetic volume.** The reduce-sum that won and the linear layer that lost perform
*the same* 131,072 multiply-accumulates per block. One wins by 1.93×, the other loses by
1.89×.

What actually separates them is how long the unsplit kernel runs:

| | unsplit time |
|---|---|
| winners | 174.6, 240.7, 2133.7, 4063.5 µs |
| losers | 14.2, 14.9, 16.1, 21.3 µs |

A launch on this harness costs about 12 µs — the 16K-element ReLU, which is pure overhead,
measures 12–15 µs. A second launch cannot pay for itself on a kernel that finishes in about
one launch's time. The gap between the two groups is an order of magnitude, but landing on
the right side of it needs the unsplit *runtime*, which needs either a model or a
measurement — and the model has been wrong every time it was checked on this branch.

**So `TryPlan` is advisory and `LastSplitProgram` is a candidate, not a recommendation.**
It is always correct; whether it is faster is measured per shape. This is the same
conclusion the tile search reached, for the same reason — see `RELEASE_EVIDENCE_GATES.md`.

## Measurement hygiene

`--frontend-check` reports timings only on an idle GPU and blanks them otherwise, with the
reason printed. `--force-timing` overrides that for one specific case — a compute process
that has been externally *suspended*, so it holds a context and its memory but occupies no
SMs — and prints what it overrode, so the caveat travels with the numbers. Every timing
above was taken with the SM clock locked and verified unchanged across the run.

## The autotuner decides it now

Since the split cannot be chosen statically, it became a candidate in
`--kernel-autotune` alongside `no-tile`, `tile2`, `lanes4`, `no-staging` and `no-vector`.
It is timed last, against the reference an established single-kernel candidate produced —
running it first would let a wrong split become the reference and reject the correct
kernels — and it must agree numerically before it is timed at all.

Measured across the catalog:

| kernel | modelled | best | winner | gain |
|---|---|---|---|---|
| `depthwise_conv2d_3x3_bwd_weights` | 4059.9 | 237.1 | `split:4` | **17.12×** |
| `conv2d_1x1_bwd_weights` | 2126.9 | 60.6 | `split:3` | **35.09×** |
| `conv2d_3x3_bwd_weights` | 239.5 | 118.2 | `split:5` | **2.03×** |
| the other ten | — | — | a tile knob or `modelled` | ≤ 1.013× |

The split wins on exactly the three weight gradients and is rejected everywhere else, which
is the FE-6 conclusion now enforced by measurement rather than asserted by a rule.

### Two bugs this run exposed

**An absolute tolerance rejected a correct split.** The agreement check compared candidates
with an absolute `2e-3`, and rejected the depthwise split over a deviation of `8.575`. That
is `5.6E-004` *relative* — the ordinary fp32 accumulation-order difference across 100,352
summed terms, and the same figure that kernel already reports on the conveyor. An absolute
tolerance is an fp32-epsilon test, not an agreement test, and it silently tightens as the
reduction lengthens. The false negative cost a measured 17×. The check is now relative.

**The autotune cache was keyed two different ways.** The autotuner *writes* rows under the
catalog entry name (`depthwise_conv2d_3x3_bwd_weights`) while `ApplyTuned` *read* them by
the spec's own name (`dwconv2d_3x3_bwd_weights`). Those differ for every depthwise entry, so
the lookup silently found nothing and those kernels ran the modelled lowering while the
cache reported them as tuned. Pre-existing, and invisible precisely because a miss is
indistinguishable from "the modelled choice already won". Now keyed by the catalog name at
both ends.

### What the conveyor does with a split winner

Nothing — and it says so:

```
    note: depthwise_conv2d_3x3_bwd_weights measured fastest as split:4, a two-kernel
          split this stage cannot launch; running the single-kernel lowering instead
```

A split winner is not a knob but a different *program*: two kernels and a temporary. The
single-kernel verify path cannot launch it, and silently falling through would report a
kernel as tuned while running the lowering the tuner measured as 17× slower.
