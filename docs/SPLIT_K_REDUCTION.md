# The gap the autotuner found: reductions with tiny outputs

FE-4 was planned as per-dimension staging and then tensor cores. Measurement re-aimed it.

## The finding

`depthwise_conv2d_3x3_bwd_weights` runs at **4052.6 us against a 3.8 us compute
roofline — 1081x off.** The autotuner tried every candidate lowering and none of them
moved it.

The reason is structural, not a tuning miss:

| quantity | value |
|---|---|
| output elements (dW is `[C,3,3]`) | **576** |
| reduction length (`n x oh x ow`) | 100,352 |
| MACs | 57.8 M |
| threads at one output per thread | 576 |
| **blocks on a 68-SM device** | **3** |
| fraction of one wave | **4%** |

The kernel is correct and it is idle. 96% of the GPU has nothing to do, because the
emitter maps **parallel axes to threads and reduction axes to loops, always**. When the
parallel axes are tiny and the reduction is enormous, there is no tile that helps — and
the autotuner proved that empirically by failing to find one.

## Why this outranks the dense-convolution gap

Dense 3x3 sits at 0.67x of cuDNN, a 1.5x tuning gap in a battle the targeting map says
not to pick. This is a **1081x structural gap** in a kernel that is required for
training. It is also not specific to depthwise: every weight gradient has a small output
and a long reduction, and so does any norm, any loss, any global pooling.

## What it needs: split the reduction

Partition the reduction into `S` chunks and give each chunk to a different block, so
parallelism comes from the reduction rather than from the output. Two ways to combine:

**Atomics.** Each block `atomicAdd`s its partial sum. Simple, one launch, no temporary.
But fp32 atomic addition is order-nondeterministic, so the result varies run to run and
the exact `0.000E+000` verify — which has caught four real bugs in this project — would
have to be relaxed to a tolerance. That is a bad trade.

**Two-pass.** Each block writes its partial to a temporary `[S, ...]` buffer; a second
kernel sums over `S`. Deterministic, so the correctness bar is unchanged. Costs one
temporary allocation and one extra launch, which the resident-program spike measured at
about 4.3 us of marginal cost — negligible against 4052.

**Two-pass is the right choice**, because the exact-agreement gate is the thing that has
repeatedly caught defects the structural gates missed, and trading it for one launch is
not worth it.

## Order of work

1. **Split-K, two-pass** — the 1081x gap, and it unblocks training.
2. Per-dimension staging — predicted ~15 us on dense 3x3 against cuDNN's 41.0.
3. `cp.async`, then tensor cores.

The targeting map still applies to 2 and 3: aim them at depthwise, pooling and
memory-bound fusion chains, not at dense convolution with large channel counts.
