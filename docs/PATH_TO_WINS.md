# Path to wins: every kernel, every competitor

One table per kernel, three measured columns, and a named lever with a prediction that can
be wrong. Nothing here is a plan on its own — each row says what is saturated, what the
next change is, and what result would falsify it.

All numbers: protocol p4, RTX 3080 locked at 1770 MHz, true fp32 (`allow_tf32=False`),
competitor in its CUDA-graph lane, best of 3.

These p4 results are historical. The current p14 protocol additionally requires every
applicable promotable candidate to stabilize, preserves the last identity-valid artifact when
a search is incomplete, and refuses evidence under material foreign host CPU load.

## The board

| kernel | vs cuDNN | limiter | ld/MAC | staged | verdict |
|---|---|---|---|---|---|
| `depthwise_conv2d_3x3_bias_relu` | **2.99×** | L1 93% | 1.278 | none | won, at roofline |
| `depthwise_conv2d_3x3_bwd_data` | **2.75×** | L1 93% | 1.250 | none | won, at roofline |
| `depthwise_conv2d_3x3` | **2.08×** | L1 93% | 1.250 | none | won, at roofline |
| `maxpool2d_2x2` | **1.41×** | DRAM 95% | 1.000 | none | won, at roofline |
| `conv2d_1x1_deep_epilogue` | **1.35×** | L2 54% | 0.133 | none | won, structural |
| `conv2d_1x1_bias_relu` | **1.25×** | L2 65% | 0.129 | none | won, headroom |
| `conv2d_1x1_bwd_data` | 1.02× | L2 53% | 0.312 | none | neutral, headroom |
| `conv_transpose2d_3x3_stride2` | 1.00× | SM 82% | 1.250 | none | neutral, SM-bound |
| `conv2d_1x1_bwd_weights` | 0.92× | L1 77% | 0.125 | none | **loss** |
| `depthwise_conv2d_3x3_bwd_weights` | 0.78× | L1 74% | 1.250 | none | **loss** |
| `conv2d_3x3_bias_relu` | 0.65× | L1 59% | 0.258 | weights | **loss** |
| `conv2d_3x3_bwd_data` | 0.56× | L1 37% | 0.257 | weights | **loss** |
| `conv2d_3x3_bwd_weights` | 0.33× | L1 78% | 0.768 | dOut | **loss** |

Six wins, two neutral, five losses.

## The one fact that organises all five losses

**Every loss is L1-bound, and no loss stages its activation operand.** The `staged` column
reads `none`, `weights` or `dOut` — never `input`. `EnableInputStaging` exists in the
emitter and defaults to **false**.

That is not a coincidence, and it is the same diagnosis the dense-convolution analysis
reached from counters: we issue far more loads per MAC than the arithmetic needs, and L1
saturates before anything else does. cuDNN's implied figure on dense 3×3, reverse-derived
from its shape, is about **0.03 loads/MAC**:

```
one input tile staged once per block   0.250 / 196  = 0.0013
one weight tile staged once per block  0.250 /   9  = 0.0278
                                       total       ≈ 0.029
```

We are at 0.258 on that kernel — **8.9× more loads than the competitor for identical
arithmetic.** The ratio 0.65× is what that buys.

Meanwhile every *win* is either at a hardware roofline the code generator cannot move
(`L1 93%`, `DRAM 95%`) or wins structurally rather than numerically — `conv2d_1x1_deep_epilogue`
beats cuDNN 1.35× because cuDNN cannot fuse through its own convolution call, so each
epilogue stage costs it a launch and a full tensor round trip while costing us one
instruction in a loop we already run.

**So the path to wins on all thirteen is one lever applied five times, plus two
special cases.**

## FALSIFIED: lever 1 was wrong, and so was its replacement

**The section below is kept as written, because it was the pre-registered prediction and it
failed.** Implemented, verified correct, and measured:

| | L1 | DRAM | SM | dense 3x3 time |
|---|---|---|---|---|
| baseline | 64.08% | 4.31% | 53.24% | 64.2 µs |
| per-dimension staging | **77.45%** | 3.36% | 44.43% | slower than `tile2` |

**L1 went UP.** Falsifier (a) fired. The cause invalidates the premise: on NVIDIA hardware
shared memory *is* L1TEX, so `ld.shared` and `st.shared` are counted by the very metric the
lever was meant to relieve. Staging moves traffic *within* the saturated unit instead of out
of it. "L1-bound" never meant "too many global loads"; it means "too much L1TEX traffic of
any kind", and DRAM at 3–4% says we were never close to memory-bound.

The obvious replacement — more register reuse, which bypasses L1TEX entirely — is **also
falsified**. L1 is insensitive to it:

| coarsening | 2 | 4 | 8 |
|---|---|---|---|
| L1 % | 64.27 | 64.11 | 64.20 |
| SM % | 53.20 | 53.84 | 53.18 |

So L1% was not the binding constraint at all. The warp-stall breakdown says what is:

| stall reason | % of warp-active cycles |
|---|---|
| **wait** — fixed-latency / FMA dependency | **17.12** |
| long_scoreboard — global memory dependency | 10.35 |
| short_scoreboard — shared / MIO dependency | 5.13 |
| mio_throttle — load-pipe queue full | **3.03** |
| no_instruction | 0.48 |

`mio_throttle` at 3% is the decisive number: **the load pipe was never the bottleneck**, so
no amount of load reduction could have helped, and the two dead levers were dead before they
were written. Warps issue roughly 64% of the time with no unit above 64% — the kernel is
*balanced*, not starved.

### What that means for dense 3x3

There is no single saturated unit to attack, and closing 62.4 µs → under cuDNN's 41.3 µs is a
1.5× gap with no scheduling lever pointing at it. A 1.5× gap with a balanced profile is the
signature of a **better algorithm**, not a better schedule: cuDNN is running an implicit-GEMM
or Winograd formulation that does less arithmetic, and F(2,3) Winograd alone cuts multiplies
by 2.25×.

That is outside the index-map layer, and this branch already carries Winograd kernels. So the
honest conclusion is that **dense 3x3 is not winnable by the code generator**, which is what
`COMPETITOR_TARGETING_MAP.md` said before this attempt: do not pick that fight. Three losses
in the dense-3x3 family (0.65×, 0.56×, 0.33×) should be recorded as *algorithmically* out of
reach for this layer and routed to the Winograd path instead of retried here.

### What survived

Per-dimension staging is **correct** — 13/13 at `0.000E+000` with both operands staged,
including the 2D block that previously returned 5.277 — and it wins marginally on
`conv_transpose2d_3x3_stride2` (1.016×). It stays as a measured autotuner candidate, chosen
where it wins and ignored where it does not. That is the mechanism working as intended: the
lever was wrong about *why*, and the tuner caught it instead of a document asserting it.

---

## Lever 1 — stage the activation operand (addresses all five losses)

The reuse analysis already prints, per operand, which axes it does *not* reference — the
axes along which one load can serve every position:

```
conv2d_3x3_bias_relu       input    invariant in {k}
conv2d_3x3_bwd_data        dOut     invariant in {o1}
conv2d_3x3_bwd_weights     input    invariant in {o0}
conv2d_1x1_bwd_weights     input    invariant in {o0}
depthwise_..._bwd_weights  dOut     invariant in {o1, o2}
```

Every one of those is an operand re-read across an axis it does not depend on. Staging it
in shared memory once per block removes exactly that redundancy.

**Prediction for `conv2d_3x3_bias_relu`**, the cleanest case and the current 0.65×:

| | value |
|---|---|
| MACs at the bench shape (8,32→64,28,28) | 115,605,504 |
| loads now, at 0.258/MAC | 29.8 M |
| loads with input+weight staging, at ≈0.029/MAC | 3.4 M |
| measured now | 63.9 µs |
| compute roofline | 7.8 µs |
| cuDNN | 41.3 µs |

Cutting L1 traffic ~8.9× on a kernel that is L1-bound at 59% should land it in the
15–25 µs band, i.e. **past cuDNN's 41.3 µs**. It does not need to reach the roofline to
win; it needs to stop being L1-bound.

**Falsification, stated in advance:** if after staging the L1 percentage does *not* fall,
the staging is not reducing traffic and the lever is wrong for this kernel. This has
already happened once in a weaker form — staging *weights only* on dense 3×3 moved the
ratio very little, which is why the activation half is the work. A second falsifier: if L1
falls but time does not, the kernel has become latency-bound rather than throughput-bound,
and the next lever is `cp.async`, not more staging.

**Risk, from experience:** shared staging returned `5.277` instead of zero the first time
it met a 2D block. Any staging change must clear the 13/13 on-device verify at
`0.000E+000` before any timing is believed.

## Lever 2 — `conv_transpose2d_3x3_stride2` is SM-bound, so cut instructions

At **SM 82%** and 1.00× it is neither a win nor an L1 problem: the machine is busy issuing
instructions. Loads/MAC of 1.250 is not the constraint; the transposed index map costs an
exact-division guard per access, and that arithmetic is the SM work.

The lever is index-map strength reduction — hoisting the division out of the inner loop
where the interval analysis can already prove the quotient constant. **Prediction:** SM%
falls and the ratio moves above 1.10×. **Falsifier:** if SM% stays at 82% after hoisting,
the instruction mix is the FMA itself and this kernel is simply at parity — which is an
acceptable answer for a transposed convolution, and should then be recorded as parity
rather than chased.

## Lever 3 — the two neutral 1×1 kernels have L2 headroom

`conv2d_1x1_bwd_data` (1.02×, L2 53%) and `conv2d_1x1_bias_relu` (1.25×, L2 65%) are
L2-bound with real headroom. The lever is the deep-epilogue exploit that already delivers
1.35× on the same shape: fuse more per pass so the tensor crosses L2 fewer times. This is
the one lever where **we have a structural advantage cuDNN cannot copy**, because it cannot
fuse through its own call.

**Prediction:** any additional fused stage moves these the way it moved
`conv2d_1x1_deep_epilogue` — bias +2.84 µs, relu +8.14, scale +6.42 measured as *marginal*
cost against a 23.75 µs convolution, versus a full launch and round trip for the
competitor.

## What is already finished

`depthwise` forward, `bwd_data` and `maxpool2d_2x2` are at L1 93% and DRAM 95%. **No change
to the code generator can improve them** — only changing what the kernel has to move, which
would be a different operator. They win at 2.08–2.99× and 1.41× and should be left alone.

## Order of work

1. **Activation staging** — five losses, one lever, largest expected movement. Gate on
   13/13 verify at `0.000E+000` before any timing.
2. **Re-run the full evidence sweep** — competitor + limiter + autotune. Staging changes
   which lowering wins, so the recorded winners must be re-measured, not assumed.
3. **Index-map strength reduction** for the transposed convolution.
4. **Deeper epilogue fusion** on the 1×1 family, exploiting what cuDNN structurally cannot do.
5. Only then `cp.async` and tensor cores, aimed by `COMPETITOR_TARGETING_MAP.md` at
   depthwise, pooling and memory-bound chains — *not* at dense convolution with large
   channel counts, where cuDNN is at its strongest and we have lost three times.

## How this stays honest

Every row above is a measured ratio against a real competitor plus a measured saturated
unit, not a model output. A static cost model chose lowerings four times on this project
and lost to the hardware every time it was checked, so the autotuner measures candidates and
records winners, and the release gate refuses to call a kernel releasable without both a
competitor ratio and a limiter verdict. Thirteen of thirteen currently carry both.

When a lever lands, the prediction in its section is either met or it is not, and the
falsifier says which. A lever that fails its falsifier gets recorded in
`mechanism_failures`-style prose in its own doc, not quietly retried.

---

# The measured re-aim (FE-13)

The blueprint above was built on throughput percentages, and two levers derived from them
died. The limiter now records **why a warp is not issuing**, for all thirteen kernels, and
the picture is different from the one the percentages suggested.

| kernel | L1% | DRAM% | SM% | wait% | longSb% | mio% | lever the profile points at |
|---|---|---|---|---|---|---|---|
| `maxpool2d_2x2` | 16.5 | 94.5 | 18.6 | 6.2 | **88.3** | 0.6 | hide global latency |
| `depthwise_conv2d_3x3_bwd_weights` | 74.4 | 26.3 | 16.0 | 9.6 | **84.3** | 0.0 | hide global latency |
| `conv2d_1x1_deep_epilogue` | 32.3 | 20.0 | 24.0 | 1.9 | **80.9** | 0.2 | hide global latency |
| `conv2d_1x1_bwd_weights` | 78.0 | 16.7 | 23.4 | 9.2 | **76.9** | 0.1 | hide global latency |
| `conv2d_1x1_bwd_data` | 31.8 | 18.8 | 28.0 | 2.7 | **73.3** | 0.2 | hide global latency |
| `conv2d_1x1_bias_relu` | 38.7 | 24.3 | 28.9 | 2.2 | **69.6** | 0.3 | hide global latency |
| `conv2d_3x3_bwd_weights` | 77.8 | 37.3 | 21.9 | 5.9 | **66.1** | 0.1 | hide global latency |
| `depthwise_conv2d_3x3_bias_relu` | 93.3 | 77.1 | 53.2 | 6.0 | 22.1 | 3.8 | hide global latency |
| `depthwise_conv2d_3x3` | 87.8 | 79.5 | 52.1 | 6.1 | 22.4 | 3.8 | hide global latency |
| `depthwise_conv2d_3x3_bwd_data` | 93.0 | 78.0 | 52.4 | 5.9 | 20.7 | 3.8 | hide global latency |
| `conv2d_3x3_bias_relu` | 59.5 | 4.4 | 53.6 | 17.1 | 10.3 | 2.9 | **balanced — no codegen lever** |
| `conv2d_3x3_bwd_data` | 36.4 | 3.0 | 33.0 | 16.8 | 13.5 | 0.8 | **balanced — no codegen lever** |
| `conv_transpose2d_3x3_stride2` | 20.9 | 17.5 | 82.5 | 12.3 | 1.1 | 0.3 | **balanced — no codegen lever** |

## What this overturns

**`mio_throttle` never exceeds 3.8%, anywhere.** The load/store pipe is not the bottleneck
in a single kernel of the thirteen. Every lever aimed at issuing *fewer memory
instructions* — shared staging, register reuse, vectorising — was aimed at a queue that was
never full. That is why per-dimension staging raised L1 from 64.08% to 77.45% and moved
nothing: it traded global traffic for shared traffic while the kernel was waiting on
neither.

**The dominant stall is `long_scoreboard` — global memory LATENCY, not throughput.** Seven
kernels sit between 66% and 88%. Latency is hidden by overlapping loads with compute, which
is what `cp.async` and prefetching do, and is precisely the opposite of adding a
`bar.sync`-per-step staging scheme.

**Three kernels are balanced, and they are the three the earlier analysis already reached
by hand.** `conv2d_3x3_bias_relu`, `conv2d_3x3_bwd_data` and `conv_transpose2d_3x3_stride2`
show no dominant stall — the profile the dense-3×3 investigation ran into and correctly
called an algorithm gap. That conclusion now falls out of the counters instead of a
one-off manual profiling session.

## The corrected order of work

1. **`cp.async` / prefetch**, aimed by `longSb%`. Ten of thirteen kernels point at it, and
   this is the lever the earlier blueprint never considered because throughput percentages
   cannot see latency.
2. **Leave the three balanced kernels alone** until a different algorithm (Winograd,
   implicit GEMM) is on the table.
3. **Nothing aimed at `mio_throttle`** until a kernel actually shows one. Three kernels sit
   at 3.8% and the rest below 1%.

## Why the limiter records the lever

Each row now names the lever its own stall profile implies, from a fixed mapping:
`mio ≥ 15%` → fewer LSU instructions; `longSb ≥ 20%` → hide global latency; `shortSb ≥ 15%`
→ reduce shared dependency; `wait ≥ 25%` → more independent accumulators; otherwise
balanced. The point is that a future lever has to be justified by a number in this table
before the work starts, which is exactly what did not happen twice.
