# Reductions reach the front end

## What was missing

FE-1 connected `CodegenGraph` to `CodegenKernelSpec`, but the translator recognised only
elementwise chains: `ReLU`, `Add`-as-bias, `Mul`-as-product, `LoadInput`. Every other op
declined.

That excluded every **reduction** — so every `MatMul`, so every linear layer. The PTX path
was reachable end to end only for fusion chains, and it declined on the shape that carries
most of the arithmetic in a real model.

The exclusion was not a missing feature in the spec. `CodegenKernelSpec` already expresses

```
out = activation( reduce(product of operands) + bias ) * scale
```

and `C[m,n] = sum_k A[m,k] * B[k,n]` is exactly that: a product of two operands summed over
one axis. What a matmul needs beyond a pointwise chain is only the **index maps** — which
is the thing the spec was built to carry. The translator simply never derived them.

## What now translates

| graph form | spec form |
|---|---|
| `MatMul`, `MatMulTransposeA`, `MatMulTransposeB` | axes `m, n` parallel + `k` reduce, `Sum` |
| `BatchMatMul` | axes `b, m, n` parallel + `k` reduce, `Sum` |
| `ReduceSum` / `ReduceMax` over an axes attribute | reduced axes become `Reduce`, kept stay `Parallel` |
| any of the above + `Add` bias + `ReLU` | one fused spec |

A transpose variant changes one index map and nothing else — the point of expressing
operands as maps rather than baking strides into the emitter.

**Broadcast is derived, not guessed.** An operand right-aligned against the output reads
the output's own axis where extents match and `Const(0)` where its extent is one. That is
what lets a `[N]` bias fuse into an `[M,N]` matmul instead of declining. A dimension that
is neither matching nor one is refused, because stretching it would be an invention.

## What still declines, and why

| form | reason |
|---|---|
| `ReduceMean` | the spec scales by a **tensor**, not by `1/n`; returning a sum would be silently wrong |
| `ReduceMin` | `CodegenReduceKind` has `Sum` and `Max` only |
| full reduction to a scalar | every axis reduced leaves no parallel axis at all |
| operand shapes that do not contract | refused rather than emitted against a wrong `k` |
| `Add` of two computed values | that is an elementwise add of two subgraphs, not a bias |

Declining with a reason is the rule the emitter already follows for index maps: a
translator that quietly mis-lowers is worse than one that refuses.

## How it is checked

A wrong translation is *easy to make and hard to see*: swap two axes in an index map and
the kernel computes `A · Bᵀ`, at full speed, silently. Emitting PTX proves nothing about
that. So there are two gates, and they check different things.

**`CodegenGraphReductionTests`** — the translated spec's own fp64 interpretation against an
independent hand-written triple loop, for every matmul variant, batched matmul, the fused
linear layer, and both reduction kinds. This is what catches a swapped axis.

**`--frontend-check`** — graph → PTX → device, compared against a reference. The `ref`
column says which reference, and the distinction is load-bearing:

```
graph                                  elements       rel dev   ref   result
relu (LowerUnaryPointwise)                16,384    0.000E+000  cpu     PASS
mul (LowerBinaryPointwise)                16,384    0.000E+000  cpu     PASS
mul+add+relu (hand-built chain)           16,384    0.000E+000  cpu     PASS
matmul 128x96x64                           8,192    0.000E+000  fp64    PASS
matmul A-transposed 128x96x64              8,192    0.000E+000  fp64    PASS
matmul B-transposed 128x96x64              8,192    0.000E+000  fp64    PASS
linear: matmul+bias+relu 256x128x64       16,384    0.000E+000  fp64    PASS
reduce-sum [512,256] over axis 1             512    0.000E+000  fp64    PASS
reduce-max [512,256] over axis 1             512    0.000E+000  fp64    PASS

front end: 9 passed, 0 failed
```

- `cpu` — the CPU emitter on the *same graph*. It shares nothing with the PTX translator,
  so it checks translation and emission together.
- `fp64` — the translated spec's own interpretation. It checks the **emitter against the
  spec**, and *not* the spec against the graph. A translator that swapped two axes would
  pass this. The unit tests above are what cover that gap.

The tool prints the reason the stronger reference was unavailable rather than falling back
silently:

```
    no CPU reference from CpuAvx512: AVX-512F is not available on this CPU.
    no CPU reference from CpuDotNetJit: Phase B CPU emitter does not yet handle Matmul ops (found MatMul).
```

Which surfaces something worth knowing: **no CPU emitter in this codebase handles matmul or
reductions at all.** For these forms the PTX path is not the fast path, it is the only
path — and there is consequently no independent same-graph reference to check it against
on any machine, AVX-512 or not.

## Still not done

- **Convolution has no op kind in this IR.** `CodegenOpKind` has no `Conv2D`; convolutions
  arrive as `Opaque` or not at all. So the 13 catalog kernels — which carry the measured
  wins — remain hand-built specs that no graph can produce. Reaching them needs a
  convolution op in the IR and `CodegenLowering` producing it, not another case in this
  translator.
- **Split plans are not launched.** `CodegenSplitReduction.TryPlan` returns the two
  kernels and the temporary size, and nothing calls it from the execution path. Matmul and
  reduction graphs are exactly the shapes that need it — see `SPLIT_K_REDUCTION.md`.
- **`ReduceMean` and `ReduceMin`** need a scalar-scale epilogue and a `Min` reduce kind
  respectively; both are small spec additions rather than translator work.

## The split route is reachable from a graph

`PtxGraphEmitter.LastSplitProgram` carries the two-kernel route when the single kernel
would leave the device idle: both PTX texts, both launch configurations, and the size of
the temporary between them. It is null when the single kernel already fills the device.

Failure to build a split is silent and returns null. That is deliberate — the single
kernel has already emitted and is correct, so a split that cannot be built costs
performance, not correctness, and turning it into a decline would throw away a working
kernel to report a missed optimisation.

The epilogue moves to the combine pass. A partial pass carrying the bias would add it once
per promoted position; refusing epilogues outright — which the first version did — would
have excluded every linear layer, the most common reduction in a model. Bias and scale
index maps are rewritten onto the combine's own axes, and anything that cannot be
translated exactly is refused rather than reused under the wrong numbering.

`--frontend-check` runs the split route on the device and holds it to the same reference
as the single kernel, because a two-kernel path through a temporary is exactly the shape
that produces a fast wrong answer. All routes agree at `0.000E+000`.

## An open question: is the split faster at THESE sizes?

It is not yet known, and the emitter should not be trusted on it.

`ChooseAxes` offers a split whenever the kernel occupies fewer than four blocks per SM.
That test says the device is **idle**; it does not say the kernel runs long enough for the
extra launch and the temporary's round-trip to pay for themselves. The catalog kernels
where the split won were 240–4063 µs unsplit. The front-end graphs here are far smaller,
and for a kernel whose runtime is comparable to a launch, an extra launch is a guaranteed
loss.

A first attempt to measure this was **void**: another process held 84–98% of the SMs
throughout, which produced a 64 µs 16K-element ReLU and a 466 µs 512-element reduction.
Those numbers are discarded, not reported.

The tool now separates the two concerns — correctness runs on a busy box, timings are
suppressed with a printed reason:

```
TIMINGS SUPPRESSED - [frontend-check] Foreign GPU workload detected: pid=... sm=98%
Correctness still runs; contention changes speed, not answers.
```

Re-measure on an idle GPU, and if the split does lose at these sizes, derive the size gate
from the measured crossover rather than a guessed constant. Tracked as FE-6.
