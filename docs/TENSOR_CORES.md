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

### The next levers, named again

1. **Double-buffer the shared slabs.** Two `bar.sync`s per K step currently serialise the
   global load against the mma work; a second slab would let step k+1's copy overlap step k's
   arithmetic.
2. **A larger block tile.** 128×128 quadruples reuse again; cuBLAS uses that scale.

Neither is measured yet, so neither is claimed.

## Standing

Expressiveness: **closed.** The generator can emit tensor-core kernels, with fused
element-wise epilogues cuBLAS cannot fuse through its own call boundary.

Performance: **improved and still a loss.** 0.80× / 0.60× / 0.57× against cuBLAS after
staging, up from 0.35× / 0.23× / 0.05×, at ~53% of device peak against its 93%. The next two
levers are named above and unmeasured.

Promotion: still **withheld everywhere**. At 0.57–0.80×, routing a caller who could reach
cuBLAS to us is still a regression — smaller, but a regression. The capability ships; the
promotion does not.
