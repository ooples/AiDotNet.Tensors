# Tensor cores in the code generator

The emitter emitted no `wmma` or `mma` instruction of any kind. That is why the blueprint
records dense GEMM as unwinnable at 0.33–0.65×: every generated matmul ran on the FP32
pipes while the competitor ran on the tensor cores, which on this device is roughly a factor
of twenty in arithmetic throughput. No amount of tiling closes that.

This closes the *expressiveness* gap. It does not close the performance gap, and the
measurements below say exactly how much is left and what the next lever is.

---

## What it emits

A warp-collective `wmma` matmul, m16n16k16, fp16 multiplicands, fp32 accumulator. One warp
owns one 16×16 output tile; four warps to a block.

`wmma` was chosen over `mma.sync` deliberately. `mma.sync` gives more scheduling freedom and
is what a hand-tuned library uses, but it requires the emitter to place each lane's operand
elements by hand from the register-layout tables. Get one lane wrong and the kernel still
assembles, still runs, and still produces numbers of the right magnitude — it is simply
wrong. `wmma` hands that mapping to the hardware. Both forms drive the same tensor cores.

This is a separate emitter rather than a flag on `PtxAffineEmitter`, because the two
lowerings disagree about what a thread is. The affine emitter gives each thread its own
output element and its own address arithmetic. `wmma` gives a whole warp one tile and
deliberately does not say which lane holds which element.

---

## Correctness

`--tensorcore-check`, against the spec's own fp64 interpretation over the same quantised
operands the device reads:

| kernel | max rel dev |
|---|---|
| matmul 64×64×64 | `0.000E+000` |
| matmul 256×256×256 | `0.000E+000` |
| matmul 512×512×512 | `0.000E+000` |
| matmul 256×2048×256 (looped K) | `0.000E+000` |
| matmul 256×256×256, B transposed | `0.000E+000` |
| matmul 512×512×512 + relu | `0.000E+000` |
| matmul 512×512×512 + gelu | `4.162E-010` |

Exact agreement here is the *expected* result, not a suspicious one: the operands are dyadic
and the partial sums stay under 2¹³ at 1/64 granularity, so every intermediate is
representable in fp32 and the hardware's accumulation order cannot diverge from the oracle's.
That is what makes the check sharp — a wrong lane mapping would have shown up immediately
rather than hiding inside a rounding allowance.

The oracle is `O(M·N·K)` on the CPU, so shapes at and above 1024³ are timed but not
verified: 4096³ is 68 billion fp64 MACs in a scalar loop. The verified shapes exercise every
path the large ones use — unrolled and looped K, both B layouts, and every epilogue.

---

## Performance, including where it loses

Against **our own scalar lowering of the same spec** — what this repository would otherwise
have shipped:

| shape | wmma | scalar | speedup |
|---|---|---|---|
| 512×512×512 | 29.1 µs | 126.5 µs | 4.35× |
| 256×2048×256 | 70.3 µs | 475.9 µs | 6.77× |
| 256×256×256, Bᵀ | 37.0 µs | 182.6 µs | 4.93× |
| 1024×1024×1024 | 190.1 µs | 389.4 µs | 2.05× |

Against **cuBLAS**, which is the denominator that decides whether this is worth shipping:

| shape | ours | cuBLAS | ratio | ours | cuBLAS |
|---|---|---|---|---|---|
| 1024³ | 190.1 µs | 66.6 µs | **0.35×** | 11.3 TFLOP/s | 32.2 TFLOP/s |
| 2048³ | 1455.4 µs | 333.8 µs | **0.23×** | 11.8 TFLOP/s | 51.5 TFLOP/s |
| 4096³ | 45810.5 µs | 2384.1 µs | **0.05×** | 3.0 TFLOP/s | 57.6 TFLOP/s |

**We lose, and we lose worse as the shape grows.** cuBLAS reaches 57.6 TFLOP/s, about 93% of
this device's fp16-with-fp32-accumulate peak. We plateau near 11.8 and then collapse.

### The collapse is diagnostic, not noise

11.8 → 3.0 TFLOP/s between 2048³ and 4096³ is a capacity cliff. One warp per 16×16 tile with
no staging means each warp streams a full 16-row band of A and a full 16-column band of B
straight from global memory, and nothing is shared between the warps that need the same
bands. Total operand traffic is `O(M·N·K)` rather than `O(M·N·K / tile)`. At 2048 the reused
bands still land in L2; at 4096 the working set passes it and every warp goes to DRAM.

cuBLAS does the opposite: 128×128 block tiles staged through shared memory with `cp.async`
pipelining, so each element of A and B is fetched from global roughly once per block rather
than once per warp.

### The next lever, named (now built — see below)

Shared-memory staging of the A and B bands plus multiple tiles per warp. That is the
difference between `O(M·N·K)` and `O(M·N·K / tile)` operand traffic, and it is what stands
between 11.8 TFLOP/s and the 57.6 the hardware demonstrably delivers.

Note this is **not** the staging lever that failed on dense 3×3 convolution. That one was
refuted because `mio_throttle` sat at 3.03% — the load pipe was never that kernel's
bottleneck. Here the evidence is different in kind: throughput *falls by 4×* when the
working set outgrows L2, which is a locality statement no stall counter is needed to read.

---

## Measurement notes

Two false readings were produced before these numbers, and both are worth recording because
they would have been easy to publish:

1. **A launch-submission floor.** Timing a single launch on this box reported 43–87 µs for
   every shape from 64³ to 1024³, and had the cuBLAS fp16→fp32 lane — which does strictly
   *more* work than fp16→fp16 — coming out faster at every size. Both are signatures of a
   fixed floor swamping the signal. Under WDDM, submission granularity dominates below about
   1024³. Batched timing on both sides fixed the comparison, and small-shape ratios are still
   not reported as findings: 256³ measured 21.7, 38.0 and 77.5 µs across three runs.

2. **Two competitor lanes, because the operations differ.** cuBLAS's native fp16 GEMM stores
   fp16, moving half the output bytes ours does. The fp16→fp32 lane is the same end-to-end
   operation ours performs. Both are reported; the ratios above use the *faster* cuBLAS lane
   for each shape, which is the conservative choice.

---

## What the recogniser refuses

A spec `wmma` cannot express exactly is refused with a reason and falls back to the scalar
emitter, which is always correct. Silent refusal would be indistinguishable from the tensor
cores never helping.

| refused | reason |
|---|---|
| fp32 multiplicands | this `wmma` shape takes fp16 |
| any dimension not a multiple of 16 | a partial tile would read past the operands |
| non-`Sum` reduction | the tensor cores compute a sum of products |
| a pre-reduction transform | needs per-element positions the fragment layout does not expose |
| a secondary output | same reason |
| a non-matmul index map (window, stride, broadcast) | not a matrix operand |
| `sm_70` and below | no tensor cores |

---

## Shared-memory staging — the named lever, measured

A block of four warps owns a 64x64 output tile and stages the 64x16 slab of A and the 16x64
slab of B into shared memory once per K step. Each element is fetched from global once per
BLOCK rather than once per warp:

| | operand traffic | fragment loads per mma |
|---|---|---|
| naive, 16x16 per warp | `M·N·K / 8` halves | 2 |
| staged, 64x64 per block | `M·N·K / 32` halves | 0.5 |

| shape | naive | **staged** | cuBLAS | old ratio | **new ratio** |
|---|---|---|---|---|---|
| 1024³ | 190.1 µs · 11.3 TF | **83.0 µs · 25.9 TF** | 66.6 µs · 32.2 TF | 0.35× | **0.80×** |
| 2048³ | 1455.4 µs · 11.8 TF | **553.1 µs · 31.1 TF** | 333.8 µs · 51.5 TF | 0.23× | **0.60×** |
| 4096³ | 45810.5 µs · 3.0 TF | **4200.8 µs · 32.7 TF** | 2384.1 µs · 57.6 TF | 0.05× | **0.57×** |

**The collapse is gone.** Throughput now RISES with size — 25.9, 31.1, 32.7 TFLOP/s — where
before it fell off a cliff at 4096³. That shape improved 10.9×.

This is still a loss against cuBLAS, and is reported as one: we sit at roughly 53% of device
peak against its 93%. But it is a different kind of loss — 1.75× away rather than 20×.

### Why this lever was legitimate when the convolution one was not

Shared-memory staging was built and REFUTED earlier in this campaign, on dense 3×3
convolution: it raised L1 from 64.08% to 77.45%, because on NVIDIA hardware shared memory
*is* L1TEX, so `ld.shared` is counted by the very metric it was meant to relieve. That lever
died because `mio_throttle` sat at 3.03% — the load pipe was never that kernel's bottleneck.

The justification here never rested on a throughput percentage. Throughput **fell fourfold**
when the working set outgrew L2, which is a locality statement no stall counter is needed to
read. Same lever, different evidence, opposite outcome.

### Double buffering — built, measured, and it under-delivered

Single-buffered staging needs two `bar.sync`s per K step: one after the shared store, one
after the reads, so a fast warp cannot overwrite a slab a slow one is still reading. Between
them the global copy and the arithmetic cannot overlap. With two slabs the copy for step k+1
targets the buffer nobody is reading, so it is *issued before* the mma work for step k. One
barrier per step.

Measured as a **paired** A/B — both lowerings emitted, launched and timed in the same process
and the same thermal window, three runs — because the effect turned out to be small enough
that a cross-run comparison could not have told signal from drift:

| shape | run 1 | run 2 | run 3 | ≈ |
|---|---|---|---|---|
| 1024³ | 1.046× | 1.114× | 1.045× | **1.07×** |
| 2048³ | 1.170× | 1.126× | 1.055× | **1.12×** |
| 4096³ | 1.031× | 1.036× | 1.001× | **1.02×** |

The direction is consistent — every one of nine paired measurements is ≥ 1.0 — but the
magnitude is 2–12%, and it *shrinks* at the largest shape, which is where the argument for it
was strongest. It is kept because it is a real win at no correctness cost, but it is not the
step change the reasoning implied.

### Why it under-delivered: the profile

`ncu` on the staged 4096³ kernel, which is what should have been read *before* building the
lever rather than after:

| metric | value |
|---|---|
| **l1tex throughput** | **87.14%** |
| dram throughput | 52.85% |
| sm throughput | 32.61% |
| **tensor pipe active** | **25.29%** |

| stall | value |
|---|---|
| `long_scoreboard` | 17.85 |
| `short_scoreboard` | 10.22 |
| `mio_throttle` | 7.36 |
| **`barrier`** | **5.82** |
| `wait` | 2.78 |
| `no_instruction` | 0.02 |

**`barrier` is 5.82.** Barriers were never the main cost, so removing one could only ever buy
a few percent — which is exactly what it bought. The lever was justified by a structural
argument ("the two barriers serialise the copy against the arithmetic") and never by a
counter, and the blueprint's §2 rule exists precisely to stop that. The rule was written on
this branch and then not followed on this branch.

**L1TEX sits at 87.14% while the tensor cores idle at 25.29%.** Shared memory *is* L1TEX on
this hardware, so the `wmma.load.*.shared` traffic feeding the fragments is at a roofline. The
kernel is not waiting on global memory — DRAM is at 52.85% — it is waiting on shared-memory
reads.

### The larger warp tile — derived from the profile, then measured

A warp loads `M + N` fragments and issues `M × N` mma instructions from them, so the fragment
loads per unit of arithmetic fall as the tile grows. That is the ratio the profile pointed at.

Swept rather than chosen, because the accumulators are `M × N × 8` fp32 per thread and past
some point ptxas spills. Every candidate verified at `0.000E+000` before being timed:

| shape | 2×2 | 2×4 | 4×2 | 4×4 |
|---|---|---|---|---|
| 512³ | 7.0 TF | 6.8 | **8.8** | 6.4 |
| 1024³ | 26.2 TF | 27.2 | **29.9** | 28.6 |
| 2048³ | 31.4 TF | 29.8 | 37.6 | **40.2** |
| 4096³ | 33.2 TF | 36.7 | 42.8 | **43.9** |

**1.28× at 2048³ and 1.32× at 4096³.** And the mechanism is confirmed rather than assumed —
re-profiling 4096³ at both tiles:

| metric | 2×2 | 4×4 |
|---|---|---|
| **l1tex throughput** | 92.34% | **61.38%** |
| dram throughput | 52.05% | **22.40%** |
| **tensor pipe active** | 26.79% | **35.74%** |
| registers per thread | 79 | 240 (no spill) |

L1TEX came off the roofline exactly as the shared-traffic argument predicted, and the tensor
pipe rose. DRAM fell too, because a bigger block tile also cuts global traffic.

The emitter now picks the largest warp tile whose block tile divides the output. That is a
**ladder derived from measurement**, not a cost model, and it is not optimal everywhere: at
1024³ the measured best was 4×2 at 71.9 µs against 4×4's 75.0 µs, so the rule gives up 4%
there. Closing that needs a per-shape autotune pass, not a cleverer rule — a static model
picked lowerings four times on this branch and lost to the hardware every time it was checked.

## cp.async — the limiter after the warp tile

Re-profiling 4096³ at 4×4 showed the previous limiter was gone and a new one had appeared:

| metric | value |
|---|---|
| **sm warps active** | **16.41%** of peak |
| **occupancy limit (registers)** | **2 blocks/SM** at 240 reg/thread |
| `mio_throttle` | 4.95 |
| `wait` | 3.63 |
| `short_scoreboard` | 1.81 |
| `long_scoreboard` | 1.66 |
| `barrier` | 1.11 |

Total stall fell from ~44 to ~14 — the resident warps barely wait. There simply are not enough
of them, because the register-staged copy holds its words live across the whole mma section.

`cp.async` copies global to shared directly: the words never occupy a register, and one
instruction moves 16 bytes instead of four. That addresses the register pressure and the
largest remaining stall at once.

Swept against register staging, paired in one process, every candidate verified at
`0.000E+000`:

| shape | tile | cp.async | registers |
|---|---|---|---|
| 512³ | **2×2** | **13.8 TF** | 8.9 |
| 1024³ | **4×2** | **34.0 TF** | 29.9 |
| 2048³ | **4×4** | **42.9 TF** | 39.3 |
| 4096³ | **4×2** | **47.0 TF** | 38.4 |
| 4096³ | 4×4 | 41.6 | **45.0** |

**The last row is why the catalog records the tile and the staging form together.** At 4096³
the 4×4 tile is *faster* with register staging than with `cp.async`, while 4×2 is much faster
with it. Choosing the two independently picks 4×4 + `cp.async` — the worst of those four. No
model produced that; the sweep did.

## Standing## Standing## Standing

Expressiveness: **closed.** The generator can emit tensor-core kernels, with fused
element-wise epilogues cuBLAS cannot fuse through its own call boundary.

Performance: **improved and still a loss.** Against cuBLAS, across the campaign:

| shape | naive | staged | +dbl buf | +warp tile | **+cp.async** | cuBLAS |
|---|---|---|---|---|---|---|
| 1024³ | 0.35× | 0.80× | 0.84× | 0.93× | **1.05×** | 66.6 µs |
| 2048³ | 0.23× | 0.60× | 0.63× | 0.78× | **0.83×** | 333.8 µs |
| 4096³ | 0.05× | 0.57× | 0.57× | 0.76× | **0.82×** | 2384.1 µs |

**At 1024³ the generated kernel is now faster than cuBLAS** — 63.2 µs against 66.6 µs. At the
larger shapes it remains a loss at 0.82–0.83×, and that is still reported as a loss.

Promotion: **withheld at 2048³ and above**, where routing a caller who could reach cuBLAS to
us is still a regression. 1024³ is the first shape where a promotion case exists on the
evidence, and it should be made on a per-shape basis with the same per-family discipline
§6 applies to convolution — not by flipping a global flag.

Promotion: still **withheld everywhere**. At 0.57–0.80×, routing a caller who could reach
cuBLAS to us is still a regression — smaller, but a regression. The capability ships; the
promotion does not.
