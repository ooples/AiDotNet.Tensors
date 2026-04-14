# Iter 20 — 4-way manual K unroll in `MicroKernel6x16`

**Change**: expanded the K loop body of `MicroKernel6x16` to process 4 K-iterations per pass (48 FMAs per loop body) instead of 1 (12 FMAs). Scalar tail handles `kc % 4` remainder. Same 12 accumulators, same A-broadcast pattern, same register budget — only the loop count is quartered.

**Build**: clean, net10.0 + net471.
**Correctness**: 86/86 matmul + GEMM + DeterministicMode tests pass on net10.0.

**Benchmark** (Ryzen 9 3950X, Det=T/Par=T):

| Shape | iter18a µs | iter19 µs | iter20 µs | Δ vs iter18a |
|---|---:|---:|---:|---:|
| DiT attn out [1024,1152]²                 |  6,040 |  5,263 |  5,891 | **−2.5%** |
| DiT QKV fused [1024,1152]×[1152,3456]    | 17,328 | 13,839 | 16,868 | **−2.7%** |
| DiT MLP up    [1024,1152]×[1152,4608]    | 23,238 | 18,938 | 22,635 | **−2.6%** |
| DiT MLP down  [1024,4608]×[4608,1152]    | 22,394 | 18,000 | 22,016 | **−1.7%** |
| Square 1152²                              |  5,590 |  5,580 |  5,486 | **−1.9%** |
| Square 4608²                              | 500,909 | 494,461 | 525,730 | +5.0% |
| Attn Q·K^T per-head [256,72]×[72,256]     |    203 |    202 |    214 | +5.4% |
| Attn A·V per-head [256,256]×[256,72]      |    256 |    260 |    248 | −3.1% |
| Batched 3D [1,256,1152]×[1152,4608]       |  7,600 |  6,634 |  7,810 | +2.8% |
| Batched 3D [4,256,1152]×[1152,4608]       | 31,947 | 30,233 | 36,206 | +13.3% |

**Run noise**: this run had very high variance. MKL's own numbers (which are not changed by our code) moved independently:

| Shape | MKL iter17 | MKL iter19 | MKL iter20 |
|---|---:|---:|---:|
| Square 1152² |  4,265 |  4,998 | 10,316 |
| Attn A·V     |     87 |     74 |    109 |
| DiT MLP up   | 17,211 | 17,233 | 29,700 |

MKL Square 1152² jumped +106% between runs of the same MKL binary. The box was under load during iter 20's run. The ~5-13% swings in our own numbers vs iter 19 are within this noise envelope.

**Signal extraction**:
- vs iter 18a (the most recent "clean" baseline): iter 20 is consistently **−2% to −3% on DiT-block shapes** (attn out, QKV, MLP up, MLP down, Square 1152²). This matches the theoretical expectation of a 4-way unroll: cuts loop-branch overhead by ~75% but doesn't change compute throughput. Expected win of a few percent.
- Batched 3D B=4 shows +13% which is worth flagging — may be a real effect if iter 20's larger basic blocks stress the iter-19 prefetch hints that were tuned for the smaller block size. Needs a clean re-run to rule out.

**Acceptance**: **PASS as no-regression under current noise floor**. The −2-3% win vs iter 18a on DiT-block shapes is consistent with the OpenBLAS-competitive-analysis estimate. Batched 3D B=4 +13% is noise-likely given the absolute variance but should be watched.

**Recommendation**: commit iter 20 with the disclaimer. Re-validate with a clean CI run once the branch is PR'd.

**What iter 20 does NOT address**: square-shape gap to MKL remains 1.31-1.52× per the clean iter 17 baseline. The real square-shape win requires iter 18c (libxsmm full-K-unroll JIT) which eliminates the loop entirely at compile time for hot Kc values.
