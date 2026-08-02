# Where the weight gradients actually stand

## The claim that needed checking

Split-K took the three weight gradients from 4060/2127/240 µs to 245.9/61.0/121.8 µs —
17.1×, 35.1× and 2.0×. Those are real numbers, and they were measured against **our own
prior lowering**. The evidence table read `MISSING` in the competitor column for all three,
which meant nobody could tell whether 17× arrived at *competitive* or merely at *less
catastrophic*.

Now measured (p4, locked clocks, CUDA-graph competitor lane, true fp32):

| kernel | ours µs | cuDNN µs | ratio | limiter |
|---|---|---|---|---|
| `depthwise_conv2d_3x3_bwd_weights` | 248.8 | 195.1 | **0.78×** | L1 74% |
| `conv2d_1x1_bwd_weights` | 69.7 | 64.3 | **0.92×** | L1 77% |
| `conv2d_3x3_bwd_weights` | 125.8 | 42.1 | **0.33×** | L1 78% |

**We are behind cuDNN on all three.** The 17× recovered a pathology of our own making — a
kernel running 3 blocks on 68 SMs — and landed us short of the competitor rather than past
it. The release gate was right to withhold these, and the split was necessary rather than
sufficient.

Stating it the other way round, which is the honest framing: without the split we were
**21× behind** cuDNN on the depthwise weight gradient. With it we are 1.3× behind.

## What the limiter says to do next

All three are now **L1-bound at 74–78%**, where before the split they profiled at 1–3% on
every unit — a kernel too idle to be limited by anything.

That is the same diagnosis the dense-convolution analysis reached: too many loads per MAC.
The split fixed *occupancy*; it did nothing about *traffic*. cuDNN's weight-gradient kernels
stage operands through shared memory, and shared-memory staging is exactly the lever
`SHARED_MEMORY_STAGING.md` documents and `EnableInputStaging` half-implements. The next
move on these kernels is known and it is not another parallelism trick.

## A profiling bug this exposed

The limiter drove `--kernel-once`, which still launched the **untuned** lowering after
verify, bench and release had all been taught the tuned program. So it profiled a kernel we
no longer ship:

| | L1 | DRAM | L2 | SM | verdict |
|---|---|---|---|---|---|
| profiling the unsplit kernel | 3.0 | 1.5 | 0.6 | 0.9 | HEADROOM |
| profiling the split we ship | 74.5 | 26.7 | 24.3 | 16.0 | AT ROOFLINE |

Same kernel name, opposite conclusion. "Headroom everywhere" would have sent the next round
of work at a lowering that no longer existed. Fixed by routing `Once` through the same
`ResolveTuned` the other stages use — the lesson being that adding a lowering the tuner can
choose obliges *every* stage that measures, not just the ones producing headline numbers.

## An argument-parsing bug this exposed

The kernel tools picked their selector with `args.FirstOrDefault(a => !a.StartsWith("--"))`,
which cannot distinguish a positional argument from a flag's value. So

```
--kernel-limiter --ncu "C:\...\ncu.exe"
```

took the *path* as the kernel name, matched nothing, and printed its header plus
`0 at a named roofline` — a gate reporting a clean result while measuring an empty set. The
same shape applied to `--out` on the autotune and split tools and to `--coarsen` on every
conveyor stage.

`KernelToolArgs` now names the value-taking flags, and an empty selection throws instead of
being reported as a pass.

## Evidence status

`13 of 13 carry both` a competitor ratio and a limiter verdict, up from `0 of 13`. Six
kernels win at ≥1.10×, four lose at ≤0.91×, and the losses are a coherent family: dense 3×3
at large channel counts, forward (0.65×), data gradient (0.56×) and weight gradient (0.33×).
That is cuDNN's home ground and the targeting map already said not to pick that fight.
