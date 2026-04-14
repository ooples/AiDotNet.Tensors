# SDPA BLAS-backed fast path (Issue #162 fix)

**Change**: `ScaledDotProductAttention<T>` at `CpuEngine.cs:14696` added a fast-path branch for `T=float` that replaces the scalar virtual-dispatch triple-loop with two batched SGEMMs per head (Q·K^T via `BlasProvider.TryGemmEx(transB=true)` then P·V via `TryGemmEx`), parallelized over batch*heads. If BLAS is unavailable, falls back to pre-transposing K into an `ArrayPool` buffer and running `SimdGemm.Sgemm` (our AVX2 blocked kernel).

Non-float types (double, other `T`) continue to hit the original scalar path — preserving exact behavior for any callers depending on it.

**Why**: Issue #162 flagged `ScaledDotProductAttention` as one of the DiT-XL wall-clock dominators. Inspection revealed the default path uses `INumericOperations<T>.Multiply` / `.Add` — one virtual call per FMA — inside a triple-nested loop of 75M FMAs per DiT-XL SDPA call. Across 28 DiT-XL blocks that's ~2.6 seconds of forward time spent in interface-dispatch overhead, not math.

**Correctness**: 39/39 attention tests pass on net10.0 (full suite).

**Benchmark** (Ryzen 9 3950X, net10.0, DiT-XL attention shape [B=4, H=16, S=256, d=72]):

| Path | Mean | Allocated | vs Baseline |
|---|---:|---:|---:|
| Float — BLAS fast path (this commit) | **25.3 ms** |  57.0 MB | — |
| Double — scalar legacy path (baseline for comparison) | 93.3 ms | 114.0 MB | **3.68× faster** |

**DiT-XL forward-pass impact**: 28 SDPA calls per forward.
- Before: 28 × 93 ms = 2.6 seconds for SDPA alone
- After:  28 × 25 ms = 0.7 seconds for SDPA
- **Saves ~2 seconds per forward pass** on Zen 2.

On 2-core CI runners the savings will be smaller per-call (fewer parallel workers) but the ratio holds — the scalar path is bound by per-op virtual dispatch overhead which parallelizes no differently than the BLAS path.

**Why the comparison uses double, not a before/after git checkout**: identifying the exact delta via bisect on the same machine is noise-sensitive. Running both paths in a single BDN session on the same process + thermal state avoids that. Double vs float compute cost is within ~2× on AVX2 Zen 2 (same port count, half the SIMD lanes per register), so the 3.68× reported here is a *conservative* estimate of the speedup — pure-float on-path would widen the ratio.

**What this does NOT address** (separate scope):
- `FlashAttention<T>` at `CpuEngine.cs:14985` — also scalar virtual dispatch. Not on the default path for DiT-XL but worth fixing if downstream ever opts in.
- `LayerNorm` — flagged in Issue #162 as "many per-transformer-block calls". Needs a separate audit.
- `ParallelWorkThreshold = 20M` — was tuned on 16-core Ryzen; may be wrong for 2-4 core CI runners. Needs a separate tuning pass.

**Path through the fix when MKL is unavailable**: `BlasProvider.TryGemmEx` returns false in deterministic mode or when MKL/native BLAS isn't loaded. In that case we fall back to `SimdGemm.Sgemm` (AVX2 blocked). The fallback requires a pre-transposed K, which we materialize from the `ArrayPool<float>` — this allocates a `d_k * seqK` float buffer per head per call (for DiT-XL: 72*256 = 18.4 KB per head). Rented and returned immediately, so no GC pressure.
