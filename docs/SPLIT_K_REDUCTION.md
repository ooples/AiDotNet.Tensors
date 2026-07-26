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

The dispatcher does not launch split plans — it allocates no temporary and issues one
kernel. `TryPlan` gives a consumer everything it needs, and wiring it lands with the
`CodegenLowering` work that makes reductions reachable from the front end at all.

Remaining order: lowering reductions, then per-dimension staging (predicted ~15 us on
dense 3x3 against cuDNN's 41.0), then `cp.async` and tensor cores — the last two still
aimed by the targeting map at depthwise, pooling and memory-bound fusion chains, not at
dense convolution with large channel counts.
