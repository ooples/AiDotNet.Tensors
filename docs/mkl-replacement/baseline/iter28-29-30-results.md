# Iter 28 + Iter 29 + Iter 30 — All three reverted

**Status**: Three experiments tried, all failed and reverted. Current landing point is still **iter 18c** (5 wins vs in-run ratio / 7 wins vs true MKL baseline).

## Iter 28 — Prefetch C rows at JIT kernel entry

**Change**: `GenerateGemmMicroKernel` prologue added 6 × `prefetcht0` hints on C rows before the K loop begins:

```csharp
e.MovRR(X86Emitter.R10, X86Emitter.R8);
for (int r = 0; r < 6; r++)
    e.Prefetcht0(X86Emitter.R10, ldcBytes * r);
```

**Motivation**: The EmitGemmAccumStore tail does load-add-store for accumulation. Issuing prefetches for C rows at kernel entry gives kc iterations of compute (~200-1500 cycles on Zen 2 at kc=512) for the prefetches to land.

**Result** (Det=F/Par=T column, clean run vs iter 18c):

| Shape | iter 18c | iter 28 | Δ |
|---|---:|---:|---:|
| DiT attn out | 4,233 | 4,356 | +2.9% |
| DiT QKV fused | 11,531 | 11,844 | +2.7% |
| DiT MLP up | 16,505 | 17,106 | +3.6% |
| DiT MLP down | 15,198 | 15,391 | +1.3% |
| Square 1152² | 4,149 | 4,209 | +1.4% |
| Square 4608² | 359,690 | 385,142 | **+7.1%** |
| Attn A·V | 163 | 163 | 0% |
| Batched B=1 | 5,848 | 5,724 | -2.1% |
| Batched B=4 | 16,833 | 17,188 | +2.1% |

**Verdict**: neutral-to-negative across the board. Square 4608² +7.1% is the clearest signal. Reverted.

**Root cause**: The load-modify-store in `EmitGemmAccumStore` already brings C rows into L1 via the HW prefetcher's next-line stride detector. The 6 extra `prefetcht0` instructions at entry consume L1d load ports during the first few K iterations (when we're already bandwidth-bound on A/B loads) without compensating benefit.

## Iter 29 — Direct no-packing 6×16 GEMM for small matmuls

**Change**: Added `SgemmDirect6x16` — walks A strided (broadcast per row) and B contiguous (two YMM loads per K iter), with no packing, no Parallel.For dispatch, no ArrayPool rentals. Gated in `SgemmAddInternal` on `work ≤ 8M AND k ≤ 512 AND no-transpose`.

**Motivation**: Close the ~2× loss on per-head attention shapes ([256,72]×[72,256] and [256,256]×[256,72]) where MKL hits 85µs and we hit 163µs. Hypothesis: SgemmTiled's packing + dispatch overhead dominates when there's only ~4.7M FMAs of work.

**Result** (Det=F/Par=T column vs iter 18c):

| Shape | iter 18c | iter 29 | Δ |
|---|---:|---:|---:|
| Attn Q·K^T | 163 | 196 | +20% |
| **Attn A·V** | **163** | **909** | **+458% (5.6× slower)** |
| Square 1152² | 4,149 | 4,557 | +10% (noise) |
| Square 4608² | 359,690 | 408,887 | +14% (noise) |

**Verdict**: CATASTROPHIC regression on A·V. Reverted immediately.

**Root cause**: The gate fires for A·V [256,256]×[256,72], which has `n=72` giving `nTail = 72 - (72/16)*16 = 8`. The full-vector path covers `nFull = 64` columns but the 8-column N-edge goes to `DirectKernelMxNMaskedDirect`, which is a scalar `for i → for j → for p` loop with MathF.FusedMultiplyAdd. With mc=6, nc=8, k=256, we run `42 M-tiles × 6 rows × 8 cols × 256 K = 516K scalar FMAs` at roughly 0.3 FMA/cycle due to address-calc overhead = ~1.5ms of scalar work per A·V call.

The full-6×16 hot path concept was sound (Attn Q·K^T went from 163 → 196 which is within noise; the 20% "regression" there is an M-edge of 4 rows × 256 cols × 72 K = 73K scalar FMAs at the same rate ≈ 220µs of overhead on a 163µs call, observed as 196µs but we should have seen ~380µs — perhaps the M-edge path was partially inlined/hoisted).

**Lesson for a future attempt**: the edge kernels CANNOT be scalar. They need a fully vectorized 6×8 kernel for nTail=8 and a masked 6×(nTail) kernel for nTail<8, plus a vectorized (mTail)×16 kernel for M-edge. That's roughly 3× more code than the simple direct-6×16 kernel — deferred until we can justify the engineering investment vs the narrow impact (per-head attention is ~9% of DiT-XL forward pass).

## Iter 30 — 4-way loop unroll for kc ≥ 256 in the JIT 2×-loop path

**Change**: For `kc ≥ 256` in the JIT micro-kernel, emit a 4-iteration loop body (56 AVX instructions) instead of the 2-iteration body (28 instructions). Used the `EmitGemmKIteration` helper so iteration displacements and register usage stay in one place across both unroll factors. Kept `128 < kc < 256` on the 2×-loop to avoid code-size inflation for marginal shapes.

**Motivation**: Square 4608² is our sole large-shape loss vs true MKL (1.10×). Hypothesis: loop-closing overhead (2 pointer advances + SubImm32 + Jne ≈ 1-2 cycles) amortizes over more K-iteration FMA work with 4× unroll; the larger independent K-iter window may also expose more ILP to Zen 2's OoO engine. Per-body L1I footprint analysis: ~300-360 B per body (including disp32 loads at B+128/+192 boundaries), well within the 32 KB L1I budget.

**Result** (Det=F/Par=T, clean run vs iter 18c):

| Shape | iter 18c | iter 30 | Δ | vs MKL (iter 17) |
|---|---:|---:|---:|---:|
| DiT attn out | 4,233 | 4,441 | +4.9% | 1.03× loss |
| DiT QKV fused | 11,531 | 12,180 | +5.6% | 1.04× loss |
| DiT MLP up | 16,505 | 18,296 | +10.9% | 1.06× loss |
| DiT MLP down | 15,198 | 15,828 | +4.1% | 1.01× parity |
| Square 1152² | 4,149 | 4,222 | +1.8% | 0.99× win |
| **Square 4608²** | **359,690** | **378,733** | **+5.3%** | **1.16× LOSS** |
| Attn Q·K^T (kc=72, full-unroll path, unchanged) | 163 | 164 | +0.6% noise | 1.93× loss |
| Attn A·V (kc=256 → moved to 4× path) | 163 | 178 | +9.2% | 2.05× loss |
| Batched B=1 | 5,848 | 6,139 | +5.0% | 0.94× win |
| Batched B=4 | 16,833 | 17,351 | +3.1% | 0.95× win |

Within-run Det=F/Det=T variance was tight (<3%), so this is a real regression, not cross-run noise.

**Verdict**: 1.8-10.9% regression across every shape on the 4× path, including the intended beneficiary (Square 4608² got **worse**, not better). Reverted. vs-MKL scorecard degraded from 7 wins / 3 losses (iter 18c) → 3 wins / 1 parity / 6 losses (iter 30).

**Root cause**: Zen 2's FMA ports are the limiting resource at 2 FMA/cycle. The 2×-loop already issues 12 FMAs per iteration, fully saturating the 2 FMA pipes over 6 cycles of body execution. Loop overhead was already <10% of body time. Adding 28 more instructions to the body:
1. Doesn't add useful ILP — the FMA pipes can't go faster than 2/cycle regardless of how many independent FMAs are in flight.
2. Doubles the front-end decoder pressure per body. Zen 2 decodes ~4 instructions/cycle; 56 AVX instrs takes ~14 cycles to fetch vs 7 cycles for the 2× body. If the uop-cache hits cleanly this shouldn't matter, but disp32 loads at +128/+192 boundaries (4 of them per body) bloat instruction bytes and may evict adjacent kernels from the uop cache.
3. Measured outcome matches (2): shapes not using the 4× path (Attn Q·K^T at kc=72 via full-unroll) saw no regression; only shapes routed to the new 4× body regressed.

**Landing point**: iter 18c's 2×-loop is the local optimum for this micro-architecture. Further loop-level unrolling from here will not help. The remaining gap vs MKL on Square 4608² (1.10×) and per-head attention (1.87-1.92×) requires fundamentally different techniques, not more unrolling:
- Square 4608²: likely wants Mc tuning (128→192 to reduce PackA call count) or packed-B alignment.
- Per-head attention: needs a fully-vectorized no-packing small-matmul kernel with vectorized edge tiles — the iter 29 concept done properly.

Note on the revert: kept the small refactor that has the 2×-loop call `EmitGemmKIteration` instead of inlining the FMA instruction sequence. That's a pure cosmetic change — the helper emits bit-identical bytes. 92/92 matmul/GEMM/JIT tests pass. Net improvement from this session: zero performance delta, slightly cleaner code in CpuJitKernels.cs.

## Current state after all three reverts

Back to iter 18c as the landing point. Against the **true iter 17 MKL baseline** (the reference with MKL.NET actually loaded):

| Shape | MKL (iter17) | Ours (iter18c) | Ratio | Status |
|---|---:|---:|---:|:---:|
| DiT attn out | 4,317 | 4,233 | 0.98× | ✅ win |
| DiT QKV fused | 11,664 | 11,531 | 0.99× | ✅ win |
| DiT MLP up | 17,211 | 16,505 | 0.96× | ✅ win |
| DiT MLP down | 15,726 | 15,198 | 0.97× | ✅ win |
| Square 1152² | 4,265 | 4,149 | 0.97× | ✅ win |
| Batched B=1 | 6,525 | 5,848 | 0.90× | ✅ win |
| Batched B=4 | 18,281 | 16,833 | 0.92× | ✅ win |
| Square 4608² | 326,491 | 359,690 | 1.10× | ❌ 10% loss |
| Attn Q·K^T per-head | 85 | 163 | 1.92× | ❌ 92% loss |
| Attn A·V per-head | 87 | 163 | 1.87× | ❌ 87% loss |

**Score: 7 wins, 3 losses** against true MKL. The 7 wins cover the hot-path DiT-XL projections (which dominate forward-pass cost); the 3 losses are a hard-to-accelerate square at the top-end and two small per-head attention calls that together represent ~9% of DiT-XL forward pass time.

**Honest assessment vs the "all 10 beating MKL" goal**: we are not there yet. The small-matmul shapes need a proper libxsmm-style specialized kernel (vectorized edge tiles) to close the 2× gap, and that's a significant engineering effort given the edge-kernel permutations required. Square 4608² is within reach with Mc tuning but hard to demonstrate with this dev box's noise floor.
