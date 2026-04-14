# Iter 24 — Batched-GEMM consolidation for ND × 2D broadcast

**Change**: `TensorMatMulBatched<T>` in `CpuEngine.cs:8544` now tries a single 2D GEMM of shape `[batchSize*m, n] × [n, p]` before falling back to the per-slice `Parallel.For` path. For the ND×2D broadcast pattern (A row-major contiguous `[batch, M, K]`, B shared `[K, P]`), this is mathematically identical to the per-slice loop but eliminates:
- Per-slice BLAS dispatch overhead (~2-3 µs per call × batch)
- Redundant B packing inside MKL (packed once for the unified GEMM, was packed N times)
- Managed-side work-steal / `Parallel.For` overhead on low-core runners

Requires `a.IsContiguous` (caller in `TensorMatMul` already ensures this).

**Build**: clean.
**Correctness**: 85/85 matmul + GEMM tests pass on net10.0.

**Benchmark** (Ryzen 9 3950X, Det=T/Par=T):

| Shape | iter20 µs | iter24 µs | Δ vs iter20 | In-run ratio vs MKL |
|---|---:|---:|---:|---:|
| **Batched 3D [4,256,1152]×[1152,4608]**  | **36,206** | **18,015** | **−50%** | **1.02× (near parity, was 1.75×)** |
| Batched 3D [1,256,1152]×[1152,4608]       |  7,810 |  5,709 | −27% | 0.93× (now beats MKL) |
| DiT attn out [1024,1152]²                  |  5,891 |  4,453 | −24% | 0.94× (now beats MKL) |
| DiT QKV fused [1024,1152]×[1152,3456]     | 16,868 | 12,199 | −28% | 0.98× (near parity) |
| DiT MLP up  [1024,1152]×[1152,4608]       | 22,635 | 17,332 | −23% | 0.93× (now beats MKL) |
| DiT MLP down [1024,4608]×[4608,1152]      | 22,016 | 17,049 | −23% | 1.02× (near parity) |
| Square 1152²                               |  5,486 |  4,330 | −21% | **0.78× (22% faster than MKL)** |
| Square 4608²                               | 525,730 | 389,639 | −26% | 1.13× |
| Attn Q·K^T per-head [256,72]×[72,256]      |    214 |    136 | −36% | 1.47× |
| Attn A·V per-head [256,256]×[256,72]       |    248 |    183 | −26% | 1.75× |

## Signal vs noise

**Iter 24's code change only touches `TensorMatMulBatched`** (the ND×2D path). Only the `Batched 3D` rows are directly affected by the change. The improvements on other shapes are run-to-run variance — this run happened to be less thermally loaded than iter 19 / iter 20 runs.

**Isolating the real iter 24 signal** (controlling for MKL drift between runs):

| Shape | Ours iter20 | Ours iter24 | MKL iter20 | MKL iter24 | Ours Δ | MKL Δ | Net iter-24 effect |
|---|---:|---:|---:|---:|---:|---:|---:|
| Batched 3D B=1 | 7,810 | 5,709 | 6,599 | 6,163 | −27% | −7% | **−20% real improvement** |
| Batched 3D B=4 | 36,206 | 18,015 | 17,624 | 17,574 | **−50%** | 0% | **−50% real improvement** |

The **B=4 case** shows the cleanest signal: MKL moved 0%, we moved −50%. That's the batched-GEMM consolidation doing exactly what was predicted — collapsing 4 per-slice dispatches into 1 big GEMM that's 2× faster.

## Updated competitive position

This run (less noisy than iters 19 / 20) gives a cleaner reading of where we stand on the whole DiT-XL shape set:

| Shape | Ratio vs MKL (this run) | Status |
|---|---:|---|
| Square 1152² | **0.78×** | **Faster than MKL** |
| DiT attn out | **0.94×** | **Faster than MKL** |
| DiT MLP up | **0.93×** | **Faster than MKL** |
| Batched 3D B=1 | **0.93×** | **Faster than MKL** |
| DiT QKV fused | 0.98× | Parity |
| DiT MLP down | 1.02× | Parity |
| Batched 3D B=4 | 1.02× | Parity (was 1.75× slower!) |
| Square 4608² | 1.13× | Remaining gap — largest square |
| Attn Q·K^T per-head | 1.47× | Remaining (K-wide/N-narrow specialization) |
| Attn A·V per-head | 1.75× | Remaining (mc=4 edge — iter 18b landed but needs re-bench) |

**4 of 10 shapes now beat MKL; 3 are at parity; 3 remain gapped.** Given the Phase 0 baseline had us 1.31-1.52× slower on squares and 10.3× worst case, this is substantial progress. A clean re-run on less-loaded hardware would firm up these numbers.

## Gating Phase 3 (flip default)

PLAN.md's trigger for Phase 3 is "benchmarks show ≥ MKL at every shape". We're not there yet — Square 4608² (1.13×) and the two per-head attention shapes (1.47× / 1.75×) still lose. But we're close enough that iter 18c (libxsmm-style JIT) is the remaining unknown that could close the 4608² gap.

## Correctness

85/85 matmul + GEMM tests pass on net10.0 after the TensorMatMulBatched change. The consolidation is mathematically identical to per-slice dispatch (A is row-major contiguous, so `[batch, M, K]` is just `[batch*M, K]`): the outputs match the per-slice path to within floating-point rounding (accumulation order is identical when both paths use the same BLAS backend, so bit-equivalence holds when BLAS is deterministic).
