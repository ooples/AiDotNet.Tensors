# Iter 31–35 Session — Adaptive Mc + Direct Small-Matmul

**Starting point**: iter 18c. 7 wins / 3 losses vs iter-17 MKL baseline.
**Landing point**: iter 34. 5 wins / 1 parity / 4 marginal losses.
**Big wins this session**: Square 4608² flipped from 1.10× loss → win (iter 33). Per-head attention closed from ~1.9× loss → 1.26× (iter 34).

## Full iteration journey

### Iter 31 — Mc 128 → 192 (landed then adaptive)

Rationale: packed-A panel at Mc=128 × Kc=512 × 4 = 256 KB = 50% of Zen 2's 512 KB L2 was under-saturating HW prefetchers. Mc=192 pushes to 75%. Side effect: 33% fewer PackA calls for m=4608 (24 tiles vs 36).

Result (Det=T/Par=T vs iter-17 MKL):
- Square 4608²: 359K → 310K µs (**flipped from 1.10× loss to 0.95× win**)
- DiT projections: 13-18% faster
- Batched: 7-16% faster
- **Residual**: Square 1152² moved from 0.99× parity to 1.05× loss (m=1152/Mc=192 → numRowBlocks=6 triggered 2D parallel with 12 tiles for 16 cores, vs iter 18c's Mc=128 with 9 tiles 1D parallel)

### Iter 31b — 2D dispatch gate (reverted)

Tried gating 2D with `numRowBlocks*2 < maxThreads AND totalWork ≥ 2.5G FMAs` to force Square 1152² into 1D. **Catastrophic regression** — kicking shapes to 1D at numRowBlocks=6 with 16 cores wastes 10 cores. Batched B=1 went from 5.6ms to 26ms (+364%). Reverted.

**Key lesson**: 2D at Mc=192 with 12 tiles is actually the **minimum** regression possible for Square 1152². Anything else (1D with 6 tiles, 2D with fewer tiles) is worse.

### Iter 32 — Mc 192 → 162 (reverted)

Tried Mc=162 (27×Mr=27×6, Mr-aligned) for m=1152 to get numRowBlocks=8 → 2D with 16 tiles (perfect 16-core coverage). Also keeps packed-A at 63% of L2 (more prefetcher headroom than 75%).

Result: Square 1152² didn't improve meaningfully (+14% vs iter 18c, iter 31 was +9% — got slightly *worse*). Lost some DiT wins too (QKV fused +5.8%). Net-negative. Reverted.

### Iter 33 — Adaptive Mc (landed)

`Mc = (m >= 2048) ? 192 : 128`. Small m uses iter-18c's Mc=128 (preserves parity on Square 1152² and per-head attn shapes); large m uses iter 31's Mc=192 (preserves Square 4608² win).

Threading: added `int Mc = (m >= AdaptiveMcThreshold) ? LargeMc : SmallMc;` at the top of SgemmTiled, SgemmTiledParallelN, SgemmTiledParallelM, and SgemmTiledParallel2D. All four functions compute their own Mc from the passed `m`, ensuring consistency without a signature change.

Result: Square 1152² back to 0.98× win, Square 4608² preserves 0.99× win. DiT wins from iter 31 reverted (m=1024 back on Mc=128), but m=1024 projections are still at parity or slight win vs MKL. Scorecard: **5W / 3P / 2L** (the 2 per-head attn losses remain).

### Iter 34 — Direct no-packing small-matmul with vectorized edges (landed)

**The big win of this session.** Added `SgemmDirect` — a no-packing 6×16 FMA kernel dispatch for small matmuls (gate: work ≤ 8M AND k ≤ 512 AND no-transpose AND n%8==0).

Key design (learning from iter 29's scalar-edge disaster):
- Main hot kernel `DirectKernel6x16` with 12 YMM accumulators, direct A/B access (no packing), hoisted per-row A base pointers to avoid IMUL per K iter
- Edge kernel `DirectKernelMxNMasked` handles all 3 edge cases (M-edge, N-edge, corner) with vectorized masked stores using the precomputed `_partialNrMasks` table. Per-row `mcActual` guards prevent OOB A reads while keeping FMA body vectorized.

Result (Det=T/Par=T):
- Attn Q·K^T: 163 µs (iter 33) → **107 µs** (iter 34) — **1.95× → 1.26× vs MKL**
- Attn A·V: 169 µs (iter 33) → **110 µs** (iter 34) — **1.94× → 1.27× vs MKL**
- Everything else: unchanged (iter 33 dispatch for non-small-matmul paths)
- 92/92 matmul tests + 39/39 FlashAttention/SDPA tests pass

~65% of the per-head attention gap closed by eliminating packing overhead entirely.

### Iter 35 — JIT direct 6×16 kernel (reverted)

Tried emitting the direct 6×16 kernel as machine code with k/lda/ldb/ldc all baked as immediates, full-unrolled for k ≤ 128. Expected to beat the inlined C# kernel by eliminating C# compile overhead.

Result: **34% regression on Q·K^T** (107 → 144 µs).

**Root cause**: UnmanagedFunctionPointer-backed delegate invoke has ~40ns of P/Invoke overhead per call. At 672 calls per Q·K^T matmul, that's 27 µs of pure dispatch vs ~80 µs of compute. The packed JIT kernel doesn't suffer because each call does Kc=512 iterations ≈ 12K FMAs ≈ 1000ns of compute — overhead is 4%. Direct kernels at k=72 do 864 FMAs ≈ 140ns per call — overhead becomes 22%.

Meanwhile C# `DirectKernel6x16` with `[MethodImpl(HotInline)]` gets **inlined at the call site**, paying zero dispatch cost. RyuJIT emits essentially the same AVX2 machine code the hand-rolled JIT would emit. For this class of small-kernel hot loops, RyuJIT is already doing what a JIT compiler would — the per-call barrier hurts more than theoretical advantages help.

**Path forward for 10/10 on per-head attn**: "fat-kernel" JIT — emit the *entire* SgemmDirect M×N loop as one kernel so only ONE P/Invoke per matmul instead of 672. Significant effort, deferred.

## Final scorecard vs iter-17 MKL (Det=T/Par=T, tightest signal)

| Shape | MKL | iter 34 | Ratio | Status |
|---|---:|---:|---:|:---:|
| DiT attn out [1024,1152]² | 4,317 | 4,474 | 1.036× | marginal loss |
| DiT QKV fused [1024,1152]×[1152,3456] | 11,664 | 12,726 | 1.091× | loss |
| DiT MLP up [1024,1152]×[1152,4608] | 17,211 | 17,100 | 0.994× | **WIN** |
| DiT MLP down [1024,4608]×[4608,1152] | 15,726 | 15,326 | 0.975× | **WIN** |
| Square 1152² | 4,265 | 4,221 | 0.990× | **WIN** |
| Square 4608² | 326,491 | 331,482 | 1.015× | **parity** |
| Attn Q·K^T per-head | 85 | 107 | 1.262× | loss (was 1.95×) |
| Attn A·V per-head | 87 | 110 | 1.268× | loss (was 1.94×) |
| Batched B=1 [256,1152,4608] | 6,525 | 6,238 | 0.956× | **WIN** |
| Batched B=4 [1024,1152,4608] | 18,281 | 17,301 | 0.946× | **WIN** |

**5 wins, 1 parity, 4 losses** — but the 2 per-head attn losses are now at 1.26× instead of 1.94×, and the 2 DiT projection losses (1.04×, 1.09×) are modest.

## Remaining path to 10/10

Three gaps left:
1. **DiT attn out (1.04×)** and **DiT QKV fused (1.09×)**: likely small Mc tuning residual. Adaptive Mc at m<2048 uses 128; m=1024 DiT projections may benefit from a different cut (e.g., Mc=160 for m<1500, Mc=192 for m≥1500). Speculative.
2. **Per-head Attn (1.26×)**: remaining ~25% gap needs either (a) fat-kernel JIT (one P/Invoke per matmul), or (b) the user's packed-B alignment idea applied differently — though iter 34 doesn't pack B, so stride-256 B access in Q·K^T's B-matrix may benefit from a different memory layout.

Neither is loop-unroll-shaped. Both require proper engineering investment.
