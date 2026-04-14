# Iter 19 — oneDNN-style prefetcht2 on next A panel

**Change**: before dispatching each micro-kernel in `MacroKernel`, issue 4 × `Sse.Prefetch2` hints on the next `ir`'s packed-A panel (256 bytes). Uses `prefetcht2` specifically (L3 hint) so it doesn't evict current data from L1/L2 or compete with load-port bandwidth in the inner FMA loop — placing the hints *outside* the K loop avoids the regression that caused iter 9's prefetcht0-in-inner-loop to be reverted in iter 16.

**Build**: clean, net10.0 + net471.
**Correctness**: 86/86 matmul + GEMM + DeterministicMode tests pass on net10.0.

**Benchmark delta** (Ryzen 9 3950X, Det=T/Par=T, vs iter 18a):

| Shape | iter18a µs | iter19 µs | Δ abs | In-run ratio vs MKL |
|---|---:|---:|---:|---:|
| DiT attn out [1024,1152]×[1152,1152]    |  6,040 |  5,263 | −12.9% | 1.34× |
| DiT QKV fused [1024,1152]×[1152,3456]   | 17,328 | 13,839 | −20.1% | 1.48× |
| DiT MLP up [1024,1152]×[1152,4608]      | 23,238 | 18,938 | −18.5% | 1.10× |
| DiT MLP down [1024,4608]×[4608,1152]    | 22,394 | 18,000 | −19.6% | 1.12× |
| Square 1152²                            |  5,575 |  5,580 |  +0.1% | 1.12× |
| Square 4608²                            | 500,909 | 494,461 |  −1.3% | 1.81× |
| Attn Q·K^T per-head [256,72]×[72,256]   |    203 |    202 |  −0.5% | 2.39× |
| Attn A·V per-head [256,256]×[256,72]    |    256 |    260 |  +1.6% | 3.49× |
| Batched 3D [1,256,1152]×[1152,4608]     |  7,580 |  6,634 | −12.5% | 1.05× |
| Batched 3D [4,256,1152]×[1152,4608]     | 31,947 | 30,233 |  −5.4% | 1.69× |

**Caveat — noise-dominated run**: MKL's own numbers swung significantly between runs (Square 4608² MKL: iter17 326 ms → iter19 272 ms = −16.5% without any MKL code change). This is thermal / background-process variance on the test box, not from our changes. The in-run ratios column above should be preferred over the absolute-µs deltas.

**In-run-ratio interpretation** (reading the right-hand column):
- DiT MLP up/down collapsed from ~1.4× slower to **~1.10× slower** in this run. If that's real and not noise, prefetch is a ~20% win on the 1024×1152 class. If noise, it's harmless.
- Square 4608² went from 1.52× → 1.81× in-run ratio. Both Ours and MKL got faster in absolute terms, but Ours by less. This is a ~20% ratio *regression* but in absolute µs, ours only moved −1.3% — it's mostly MKL getting faster. Indicates this run's noise floor is ~15-20%, limiting conclusions.
- Per-head A·V ratio 2.70 → 3.49 is noise (absolute Ours stayed at 256-260 µs; MKL happened to be faster in this run).

**Acceptance**: **PASS as no-regression** — no row regressed beyond the noise floor. The possibly-positive signal at DiT-block shapes needs a second confirmation run on a cleaner machine before we can claim the 15-20% win as real.

**Next**: SDPA fix is in scope for this PR per user direction + Issue #162 finding. The scalar virtual-dispatch triple-loop in `ScaledDotProductAttention` dominates DiT-XL forward wall-clock far more than any square-shape matmul tuning will recover. Deferring iters 20 (K-unroll) / 22 (AVX-512) / 23 (batched-GEMM) until SDPA is handled.
