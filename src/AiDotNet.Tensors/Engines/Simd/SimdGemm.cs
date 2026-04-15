using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.CpuJit;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// High-performance General Matrix Multiply (GEMM) using BLIS/GotoBLAS tiled architecture.
/// C[m,n] += A[m,k] * B[k,n] with FMA micro-kernel, panel packing, and cache-level blocking.
/// </summary>
internal static class SimdGemm
{
    // Cache blocking parameters (tuned for typical L1=32KB, L2=512KB, L3=8MB+)
    // Iter 2: Mc lowered 256→128 for more row blocks (better parallel utilization).
    // Iter 3 (reverted): Mc=64 regressed across the board — too small for cache reuse.
    // Iter 8 (reverted): Kc=256 was tried to fit per-tile working set in L2, but the
    // extra pc iterations (4 instead of 2) doubled the number of parallel barriers and
    // regressed 512² by 1.5x. The L2-fit argument was wrong because packed A gets
    // evicted by packed B during the inner loop anyway. Kept at 512.
    //
    // Iter 31: Mc 128→192 attacked Square 4608² (flipped from 1.10× loss to 0.95×
    // win, + 13-18% gains on DiT projections). One residual: Square 1152² moved
    // from 0.99× parity to 1.05× loss because m=1152/Mc=192 gives numRowBlocks=6
    // and dispatches to 2D-parallel with 12 tiles for 16 cores (4 idle).
    //
    // Iter 31b (reverted): tightening the 2D gate forced Square 1152² to 1D which
    // made it 2.2× worse (1D at 6 row blocks leaves 10 cores idle). Confirmed
    // 2D@Mc=192 is the minimum regression for that shape.
    //
    // Iter 32 (reverted): Mc 192→162 for "perfect 16-tile coverage" on m=1152.
    // Didn't help — Square 1152² actually got slightly worse (+14% vs iter 18c
    // baseline, iter 31 was +9%). The partial-tile mask-path overhead and lost
    // DiT win (QKV fused +5.8%) made it net-negative vs iter 31.
    //
    // Iter 33: adaptive Mc — use SmallMc (128, iter 18c value) for m &lt; 2048
    // and LargeMc (192, iter 31 value) for m ≥ 2048. Sacrifices iter 31's DiT
    // projection wins (m=1024 reverts to Mc=128) to buy back Square 1152²
    // parity with MKL. Preserves iter 31's flagship Square 4608² win (m=4608
    // keeps Mc=192). User chose this trade to maximize "beats MKL on every
    // shape" optics over peak DiT-XL forward-pass speed.
    private const int SmallMc = 128;
    private const int LargeMc = 192;
    private const int AdaptiveMcThreshold = 2048;
    private const int Kc = 512;  // Panel depth (fits in L1)
    private const int Nc = 4096; // Panel width for B (fits in L3)

    // Micro-kernel register block: 6 rows x 16 columns
    // 6 rows * 16 cols = 96 floats = 12 Vector256<float> accumulators
    private const int Mr = 6;
    private const int Nr = 16;

    // A/B test toggle: set to false to force sequential SgemmTiled for baseline
    // comparisons. Defaults to true so multi-core systems get parallel execution.
    // Intended for benchmark A/B iteration, not production config.
    internal static bool UseParallelGemm = true;

    // Minimum problem size (m*n*k, count as flops/2) to enable parallel dispatch.
    // Iter 2 (2026-04-11): raised 4M → 20M after measuring that 256² (16.8M) regressed
    // with parallel on a 16-core Ryzen 9 3950X — thread-pool dispatch overhead scales
    // with core count (barrier latency is roughly O(log cores) + per-thread work-steal
    // setup). 512² (134M) and 1024² (1B) are the real winners at 16 cores.
    //
    // Issue #162 context: AiDotNet CI runners have 2 cores (Windows) / 4 cores (Linux).
    // At those core counts, thread-dispatch overhead is much lower, so parallel dispatch
    // becomes beneficial at smaller problem sizes. The fixed 20M threshold leaves 2-core
    // boxes running sequential on medium matmuls that would benefit from parallelism.
    //
    // Scale the threshold linearly with core count, floored at 2M (tiny matmuls never
    // parallelize no matter the core count), capped at 20M (preserves the empirically-
    // tuned iter-2 value at 16+ cores).
    //
    // Example thresholds (computed = 1,310,720 × cores, clamped to [2M, 20M]):
    //   2 cores:  2.62 Mi work-elements  → gates parallel on ~137² and above
    //   4 cores:  5.24 Mi  (CI Linux)
    //   8 cores: 10.48 Mi
    //  16 cores: 20 Mi   (iter 2 cap — unchanged)
    //  32 cores: 20 Mi   (capped)
    private static readonly long ParallelWorkThreshold = ComputeParallelWorkThreshold();

    private static long ComputeParallelWorkThreshold()
    {
        int cores = Math.Max(1, Environment.ProcessorCount);
        long scaled = (20L * 1024 * 1024 / 16) * cores;
        if (scaled < 2L * 1024 * 1024) scaled = 2L * 1024 * 1024;
        if (scaled > 20L * 1024 * 1024) scaled = 20L * 1024 * 1024;
        return scaled;
    }

    /// <summary>
    /// Computes C = A * B where A is [m,k], B is [k,n], C is [m,n].
    /// All matrices are in row-major order. C is cleared before computation.
    /// </summary>
    [MethodImpl(Hot)]
    public static void Sgemm(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m,
        int k,
        int n)
    {
        c.Clear();
        SgemmAdd(a, b, c, m, k, n);
    }

    /// <summary>
    /// Sequential SGEMM — forces the tiled kernel to run without internal parallelism
    /// even when work would normally exceed <see cref="ParallelWorkThreshold"/>.
    ///
    /// Use this inside an outer <c>Parallel.For</c> that already provides parallelism
    /// (e.g. per-head attention where the batch*heads loop parallelizes per slice):
    /// letting SgemmTiled also spawn workers would over-subscribe and create more tasks
    /// than cores, hurting throughput. This overload threads an <c>allowParallel=false</c>
    /// flag through the dispatch instead of mutating the shared <see cref="UseParallelGemm"/>
    /// field, so it is safe to call concurrently from many worker threads.
    /// </summary>
    [MethodImpl(Hot)]
    public static void SgemmSequential(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m, int k, int n)
    {
        c.Clear();
        SgemmAddInternal(a, k, false, b, n, false, c, m, k, n, allowParallel: false);
    }

    /// <summary>
    /// Computes C = op(A) * op(B) with optional transpose on either operand.
    /// op(X) = X when transX=false, op(X) = X^T when transX=true.
    /// lda/ldb are the leading dimensions (row strides) of the source storage.
    /// This enables zero-copy matmul on transposed stride-based views.
    /// </summary>
    [MethodImpl(Hot)]
    public static void Sgemm(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c,
        int m, int k, int n)
    {
        c.Clear();
        SgemmAdd(a, lda, transA, b, ldb, transB, c, m, k, n);
    }

    /// <summary>
    /// Computes C = beta*C + A*B. When beta=0, overwrites C (clears first).
    /// When beta=1, accumulates into C. Matches BLAS sgemm semantics.
    /// </summary>
    [MethodImpl(Hot)]
    public static void Sgemm(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m, int k, int n,
        float beta)
    {
        if (beta == 0f)
            c.Clear();
        else if (beta != 1f)
        {
            for (int i = 0; i < c.Length; i++)
                c[i] *= beta;
        }
        SgemmAdd(a, b, c, m, k, n);
    }

    /// <summary>
    /// Computes C += A * B (accumulates into C without clearing).
    /// </summary>
    [MethodImpl(Hot)]
    public static void SgemmAdd(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m,
        int k,
        int n)
    {
        SgemmAdd(a, k, false, b, n, false, c, m, k, n);
    }

    /// <summary>
    /// Computes C += op(A) * op(B) with stride and transpose support.
    /// </summary>
    [MethodImpl(Hot)]
    public static void SgemmAdd(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c,
        int m, int k, int n)
    {
        SgemmAddInternal(a, lda, transA, b, ldb, transB, c, m, k, n, allowParallel: true);
    }

    /// <summary>
    /// Shared implementation of C += op(A) * op(B) with an explicit parallel gate.
    /// Used by both <see cref="SgemmAdd"/> (allowParallel=true) and
    /// <see cref="SgemmSequential"/> (allowParallel=false). Threading the flag here
    /// avoids races on the global <see cref="UseParallelGemm"/> field when the
    /// caller is itself inside a parallel region.
    /// </summary>
    [MethodImpl(Hot)]
    internal static void SgemmAddInternal(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c,
        int m, int k, int n,
        bool allowParallel)
    {
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported && m >= Mr && n >= Nr)
        {
            // Iter 34: small-matmul fast path — no packing, direct 6×16 FMA
            // with fully vectorized masked edge kernels (proper fix for iter
            // 29's scalar-edge disaster). Targets per-head-attention shapes
            // where SgemmTiled's PackA + PackB + parallel dispatch overhead
            // dominates the compute time (MKL hits ~85µs, we hit ~163µs in
            // iter 18c/31/33 — ~2× MKL gap is pure packing overhead).
            //
            // Gate conditions (all required):
            //   - work ≤ SmallMatmulWorkThreshold (8M FMAs): above this,
            //     SgemmTiled's packing cost amortizes and is competitive
            //   - k ≤ SmallMatmulKThreshold (512): keeps 6-row A panel at
            //     ≤ 12KB in L1d during the inner kernel
            //   - no transpose: direct-stride path only handles row-major
            //   - (n % 8 == 0) OR n >= Nr+8: ensures N-edges are 8-aligned
            //     (handled by DirectKernel6x8 / masked 6×8) or big enough to
            //     fall through to the normal split. For DiT-XL shapes n=256
            //     and n=72 both have n % 8 == 0.
            long directWork = (long)m * k * n;
            if (!transA && !transB
                && directWork <= SmallMatmulWorkThreshold
                && k <= SmallMatmulKThreshold
                && (n % 8 == 0))
            {
                SgemmDirect(a, lda, b, ldb, c, m, k, n);
                return;
            }

            SgemmTiled(a, lda, transA, b, ldb, transB, c, m, k, n, allowParallel);
            return;
        }
#endif
        SgemmScalar(a, lda, transA, b, ldb, transB, c, m, k, n);
    }

    // Iter 34 small-matmul gate. 8M FMAs captures per-head attention
    // [256,72]×[72,256] (4.7M) and [256,256]×[256,72] (4.7M) and excludes
    // 1024²-scale work (>= 1B). K ≤ 512 keeps the 6-row A panel at 12KB,
    // well within Zen 2's 32KB L1d.
    private const long SmallMatmulWorkThreshold = 8L * 1024 * 1024;
    private const int SmallMatmulKThreshold = 512;

    /// <summary>
    /// Scalar GEMM fallback with stride/transpose support.
    /// </summary>
    [MethodImpl(Hot)]
    private static void SgemmScalar(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c,
        int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
        {
            int cRowBase = i * n;
            for (int p = 0; p < k; p++)
            {
                // op(A)[i,p]: if transA, read A[p,i] = a[p*lda+i]; else A[i,p] = a[i*lda+p]
                float aip = transA ? a[p * lda + i] : a[i * lda + p];
                for (int j = 0; j < n; j++)
                {
                    // op(B)[p,j]: if transB, read B[j,p] = b[j*ldb+p]; else B[p,j] = b[p*ldb+j]
                    float bpj = transB ? b[j * ldb + p] : b[p * ldb + j];
#if NET5_0_OR_GREATER
                    c[cRowBase + j] = MathF.FusedMultiplyAdd(aip, bpj, c[cRowBase + j]);
#else
                    c[cRowBase + j] += aip * bpj;
#endif
                }
            }
        }
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// Iter 34: libxsmm-style no-packing GEMM dispatch for small matmuls.
    /// <para>
    /// Walks M in Mr=6 blocks and N in Nr=16 blocks. Full 6×16 tiles use
    /// <see cref="DirectKernel6x16"/> (no masking, fastest). Any edge tile
    /// (mc &lt; 6 OR nc &lt; 16) uses <see cref="DirectKernelMxNMasked"/>
    /// which shares the same 12-accumulator FMA body with per-row mc_actual
    /// guarding and precomputed lane masks for the N partial.
    /// </para>
    /// <para>
    /// Unlike the iter 29 attempt (scalar N-edge fallback → 5.6× regression
    /// on A·V), every edge path here stays vectorized.
    /// </para>
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void SgemmDirect(
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb,
        Span<float> c,
        int m, int k, int n)
    {
        int mFull = (m / Mr) * Mr;
        int nFull = (n / Nr) * Nr;

        // Iter 35 (REVERTED): tried JIT-emitting the 6×16 direct kernel with
        // k/lda/ldb/ldc all baked as immediates (full-unroll for k ≤ 128).
        // Expected to beat the inlined C# kernel by eliminating C# compile
        // overhead. Instead regressed Q·K^T by 34% (107µs → 144µs).
        //
        // Root cause: UnmanagedFunctionPointer-backed delegate invoke has
        // ~40ns P/Invoke overhead per call. The PACKED JIT kernel
        // (CpuJitKernels.GetGemmMicroKernel) absorbs this fine because each
        // call does Kc=512 iterations ≈ 12K FMAs ≈ 1000ns of compute — 4%
        // overhead. The DIRECT kernel at k=72 does only 864 FMAs ≈ 140ns
        // per call — 22% overhead from the 40ns dispatch cost, net-negative
        // vs the inlined [MethodImpl(HotInline)] C# variant.
        //
        // The C# DirectKernel6x16 gets RyuJIT-inlined at the call site
        // here, so its dispatch cost is zero. RyuJIT emits essentially the
        // same AVX2 machine code that the hand-rolled JIT would emit. The
        // JIT kernel's theoretical advantages (full unroll, immediate disps)
        // don't overcome the per-call barrier for small kernels.
        //
        // The right path to beat MKL's 85µs on Q·K^T is a FAT-kernel JIT —
        // emit the entire SgemmDirect M×N loop as one kernel so only ONE
        // P/Invoke per matmul instead of 672. Deferred — significant effort.

        fixed (float* pAroot = a, pBroot = b, pCroot = c)
        {
            // Full-Mr rows: 6 rows at a time.
            for (int i = 0; i < mFull; i += Mr)
            {
                float* pARow = pAroot + i * lda;
                float* pCRow = pCroot + i * n;

                // Full-Nr columns: 16 cols at a time → hot 6×16 kernel.
                int j = 0;
                for (; j + Nr <= n; j += Nr)
                {
                    DirectKernel6x16(pARow, lda, pBroot + j, ldb, pCRow + j, n, k);
                }

                // N-edge (nc ∈ [1, 15]) via the masked kernel.
                int ncTail = n - j;
                if (ncTail > 0)
                {
                    DirectKernelMxNMasked(
                        pARow, lda, pBroot + j, ldb, pCRow + j, n,
                        k, mcActual: Mr, ncActual: ncTail);
                }
            }

            // M-edge (mc ∈ [1, 5]): remaining rows go through the masked kernel.
            int mcTail = m - mFull;
            if (mcTail > 0)
            {
                float* pARow = pAroot + mFull * lda;
                float* pCRow = pCroot + mFull * n;

                int j = 0;
                for (; j + Nr <= n; j += Nr)
                {
                    DirectKernelMxNMasked(
                        pARow, lda, pBroot + j, ldb, pCRow + j, n,
                        k, mcActual: mcTail, ncActual: Nr);
                }

                // Corner: both M and N edges.
                int ncTail = n - j;
                if (ncTail > 0)
                {
                    DirectKernelMxNMasked(
                        pARow, lda, pBroot + j, ldb, pCRow + j, n,
                        k, mcActual: mcTail, ncActual: ncTail);
                }
            }
        }
    }

    /// <summary>
    /// Iter 34 hot kernel: direct 6×16 FMA over native strides. No packing,
    /// no fixed statements inside the inner loop (caller pins root pointers).
    /// Same 12-accumulator layout as <see cref="MicroKernel6x16"/> but reads
    /// A with stride lda (one scalar per row per K iter) and B with stride ldb
    /// (two YMM loads of contiguous 256+256 bits per K iter).
    /// </summary>
    [MethodImpl(HotInline)]
    private static unsafe void DirectKernel6x16(
        float* pA, int lda,
        float* pB, int ldb,
        float* pC, int ldc,
        int k)
    {
        var c00 = Vector256<float>.Zero; var c01 = Vector256<float>.Zero;
        var c10 = Vector256<float>.Zero; var c11 = Vector256<float>.Zero;
        var c20 = Vector256<float>.Zero; var c21 = Vector256<float>.Zero;
        var c30 = Vector256<float>.Zero; var c31 = Vector256<float>.Zero;
        var c40 = Vector256<float>.Zero; var c41 = Vector256<float>.Zero;
        var c50 = Vector256<float>.Zero; var c51 = Vector256<float>.Zero;

        // Hoist per-row A base pointers out of the K loop (avoids an IMUL per iter).
        float* pA0 = pA;
        float* pA1 = pA + lda;
        float* pA2 = pA + lda * 2;
        float* pA3 = pA + lda * 3;
        float* pA4 = pA + lda * 4;
        float* pA5 = pA + lda * 5;

        for (int p = 0; p < k; p++)
        {
            var b0 = Avx.LoadVector256(pB);
            var b1 = Avx.LoadVector256(pB + 8);

            var a0 = Vector256.Create(pA0[p]);
            c00 = Fma.MultiplyAdd(a0, b0, c00); c01 = Fma.MultiplyAdd(a0, b1, c01);

            var a1 = Vector256.Create(pA1[p]);
            c10 = Fma.MultiplyAdd(a1, b0, c10); c11 = Fma.MultiplyAdd(a1, b1, c11);

            var a2 = Vector256.Create(pA2[p]);
            c20 = Fma.MultiplyAdd(a2, b0, c20); c21 = Fma.MultiplyAdd(a2, b1, c21);

            var a3 = Vector256.Create(pA3[p]);
            c30 = Fma.MultiplyAdd(a3, b0, c30); c31 = Fma.MultiplyAdd(a3, b1, c31);

            var a4 = Vector256.Create(pA4[p]);
            c40 = Fma.MultiplyAdd(a4, b0, c40); c41 = Fma.MultiplyAdd(a4, b1, c41);

            var a5 = Vector256.Create(pA5[p]);
            c50 = Fma.MultiplyAdd(a5, b0, c50); c51 = Fma.MultiplyAdd(a5, b1, c51);

            pB += ldb;
        }

        // Accumulate into C (load-add-store). Caller cleared C via Sgemm() so
        // on first call C contains zeros, but repeated MacroKernel-style
        // invocations compose correctly.
        Avx.Store(pC + 0, Avx.Add(c00, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 8, Avx.Add(c01, Avx.LoadVector256(pC + 8)));
        pC += ldc;
        Avx.Store(pC + 0, Avx.Add(c10, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 8, Avx.Add(c11, Avx.LoadVector256(pC + 8)));
        pC += ldc;
        Avx.Store(pC + 0, Avx.Add(c20, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 8, Avx.Add(c21, Avx.LoadVector256(pC + 8)));
        pC += ldc;
        Avx.Store(pC + 0, Avx.Add(c30, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 8, Avx.Add(c31, Avx.LoadVector256(pC + 8)));
        pC += ldc;
        Avx.Store(pC + 0, Avx.Add(c40, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 8, Avx.Add(c41, Avx.LoadVector256(pC + 8)));
        pC += ldc;
        Avx.Store(pC + 0, Avx.Add(c50, Avx.LoadVector256(pC + 0)));
        Avx.Store(pC + 8, Avx.Add(c51, Avx.LoadVector256(pC + 8)));
    }

    /// <summary>
    /// Iter 34 edge kernel: masked 6×N or M×16 or M×N with direct A/B access.
    ///
    /// <para>
    /// Same 12-accumulator FMA body as <see cref="DirectKernel6x16"/>. The
    /// FMA loop always computes all 6×16 lanes (accumulators for inactive
    /// rows get zeroed later). Store phase uses <see cref="_partialNrMasks"/>
    /// for N lane masking and per-row mcActual guards for M.
    /// </para>
    /// <para>
    /// Critical invariant (that iter 29 violated): when mcActual &lt; Mr,
    /// rows past mcActual in pA are OUT OF BOUNDS. We must not deref those
    /// addresses. Guarded via explicit `if (mcActual &gt; N)` branches
    /// inside the K loop. Branches are loop-invariant and branch-predict
    /// perfectly, so they're effectively free after first-iter warmup.
    /// </para>
    /// </summary>
    [MethodImpl(HotInline)]
    private static unsafe void DirectKernelMxNMasked(
        float* pA, int lda,
        float* pB, int ldb,
        float* pC, int ldc,
        int k, int mcActual, int ncActual)
    {
        var c00 = Vector256<float>.Zero; var c01 = Vector256<float>.Zero;
        var c10 = Vector256<float>.Zero; var c11 = Vector256<float>.Zero;
        var c20 = Vector256<float>.Zero; var c21 = Vector256<float>.Zero;
        var c30 = Vector256<float>.Zero; var c31 = Vector256<float>.Zero;
        var c40 = Vector256<float>.Zero; var c41 = Vector256<float>.Zero;
        var c50 = Vector256<float>.Zero; var c51 = Vector256<float>.Zero;

        // Per-row A base pointers (avoid IMUL per K iter). We always compute
        // pA0..pA5 addresses but only *deref* the ones within mcActual.
        float* pA0 = pA;
        float* pA1 = pA + lda;
        float* pA2 = pA + lda * 2;
        float* pA3 = pA + lda * 3;
        float* pA4 = pA + lda * 4;
        float* pA5 = pA + lda * 5;

        for (int p = 0; p < k; p++)
        {
            var b0 = Avx.LoadVector256(pB);
            var b1 = Avx.LoadVector256(pB + 8);

            // Row 0 always active (mcActual >= 1 guaranteed by caller).
            var a0 = Vector256.Create(pA0[p]);
            c00 = Fma.MultiplyAdd(a0, b0, c00); c01 = Fma.MultiplyAdd(a0, b1, c01);

            if (mcActual > 1)
            {
                var a1 = Vector256.Create(pA1[p]);
                c10 = Fma.MultiplyAdd(a1, b0, c10); c11 = Fma.MultiplyAdd(a1, b1, c11);
            }
            if (mcActual > 2)
            {
                var a2 = Vector256.Create(pA2[p]);
                c20 = Fma.MultiplyAdd(a2, b0, c20); c21 = Fma.MultiplyAdd(a2, b1, c21);
            }
            if (mcActual > 3)
            {
                var a3 = Vector256.Create(pA3[p]);
                c30 = Fma.MultiplyAdd(a3, b0, c30); c31 = Fma.MultiplyAdd(a3, b1, c31);
            }
            if (mcActual > 4)
            {
                var a4 = Vector256.Create(pA4[p]);
                c40 = Fma.MultiplyAdd(a4, b0, c40); c41 = Fma.MultiplyAdd(a4, b1, c41);
            }
            if (mcActual > 5)
            {
                var a5 = Vector256.Create(pA5[p]);
                c50 = Fma.MultiplyAdd(a5, b0, c50); c51 = Fma.MultiplyAdd(a5, b1, c51);
            }

            pB += ldb;
        }

        // Build column lane masks from ncActual (same logic as MicroKernelMxNMasked).
        int lane0N = ncActual >= 8 ? 8 : ncActual;
        int lane1N = ncActual >= 8 ? ncActual - 8 : 0;
        var mask0 = _partialNrMasks[lane0N].AsSingle();
        var mask1 = _partialNrMasks[lane1N].AsSingle();

        // Masked accumulate-and-store, row by row, skipping rows past mcActual.
        if (mcActual > 0) StoreMaskedAccumRowDirect(pC,            mask0, mask1, c00, c01);
        if (mcActual > 1) StoreMaskedAccumRowDirect(pC + ldc,      mask0, mask1, c10, c11);
        if (mcActual > 2) StoreMaskedAccumRowDirect(pC + ldc * 2,  mask0, mask1, c20, c21);
        if (mcActual > 3) StoreMaskedAccumRowDirect(pC + ldc * 3,  mask0, mask1, c30, c31);
        if (mcActual > 4) StoreMaskedAccumRowDirect(pC + ldc * 4,  mask0, mask1, c40, c41);
        if (mcActual > 5) StoreMaskedAccumRowDirect(pC + ldc * 5,  mask0, mask1, c50, c51);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void StoreMaskedAccumRowDirect(
        float* row, Vector256<float> m0, Vector256<float> m1,
        Vector256<float> v0, Vector256<float> v1)
    {
        var existing0 = Avx.MaskLoad(row, m0);
        var existing1 = Avx.MaskLoad(row + 8, m1);
        Avx.MaskStore(row, m0, Avx.Add(existing0, v0));
        Avx.MaskStore(row + 8, m1, Avx.Add(existing1, v1));
    }

    /// <summary>
    /// Tiled GEMM with panel packing and FMA micro-kernel.
    /// Follows BLIS architecture: loop order is jc -> pc -> ic -> jr -> ir.
    /// Supports parallelism over both M and N dimensions for different matrix shapes.
    /// </summary>
    /// <summary>
    /// Small-M GEMM: for M &lt;= 64, skip packing and compute directly.
    /// Each row of C is computed as a dot product of a row of A with columns of B,
    /// using SIMD to process 8 columns of B at a time.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void SgemmSmallM(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c,
        int m, int k, int n)
    {
        fixed (float* pA = a, pB = b, pC = c)
        {
            for (int i = 0; i < m; i++)
            {
                int cRow = i * n;
                // Process N in chunks of 8 (AVX2 width)
                int j = 0;
                for (; j + 8 <= n; j += 8)
                {
                    var acc = Vector256<float>.Zero;
                    for (int p = 0; p < k; p++)
                    {
                        float aVal = transA ? pA[p * lda + i] : pA[i * lda + p];
                        var aVec = Vector256.Create(aVal);
                        int bIdx = transB ? (j * ldb + p) : (p * ldb + j);
                        // For transB, elements are strided — need gather or scalar
                        if (!transB)
                        {
                            var bVec = Avx.LoadVector256(pB + bIdx);
                            acc = Fma.MultiplyAdd(aVec, bVec, acc);
                        }
                        else
                        {
                            // Gather from transposed B — each element at stride ldb
                            var bVec = Vector256.Create(
                                pB[j * ldb + p], pB[(j + 1) * ldb + p], pB[(j + 2) * ldb + p], pB[(j + 3) * ldb + p],
                                pB[(j + 4) * ldb + p], pB[(j + 5) * ldb + p], pB[(j + 6) * ldb + p], pB[(j + 7) * ldb + p]);
                            acc = Fma.MultiplyAdd(aVec, bVec, acc);
                        }
                    }
                    Avx.Store(pC + cRow + j, acc);
                }
                // Scalar tail — use FMA to match SIMD rounding behavior
                for (; j < n; j++)
                {
                    float sum = 0;
                    for (int p = 0; p < k; p++)
                    {
                        float aVal = transA ? pA[p * lda + i] : pA[i * lda + p];
                        float bVal = transB ? pB[j * ldb + p] : pB[p * ldb + j];
                        sum = MathF.FusedMultiplyAdd(aVal, bVal, sum);
                    }
                    pC[cRow + j] = sum;
                }
            }
        }
    }

    [MethodImpl(Hot)]
    private static void SgemmTiled(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c,
        int m, int k, int n,
        bool allowParallel = true)
    {
        // Decide parallel vs sequential up front. Parallel dispatches either:
        //   - 1D parallel (SgemmTiledParallelM): row blocks only, when nc is too small
        //     to justify col-sub parallelism OR when numRowBlocks already matches the
        //     target core count.
        //   - 2D parallel (SgemmTiledParallel2D): (row block × col sub) grid, when
        //     either M or N alone doesn't saturate available cores. Handles row-heavy
        //     problems (1024²) by splitting columns too, AND col-heavy problems
        //     (LM-head [64, 128]x[128, 50257]) by parallelizing the 1 row block across
        //     many col subs.
        //
        // allowParallel=false: caller is inside an outer Parallel.For region and is
        // already providing parallelism (e.g. SDPA per-head dispatch). Force sequential
        // to avoid spawning (outer_workers × inner_workers) tasks for (cores) cores.
        int maxThreads = Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        // Adaptive Mc (iter 33): Small m (< 2048) uses SmallMc=128 for cache
        // behavior that matches iter 18c's parity on Square 1152² and keeps
        // 1D dispatch at numRowBlocks=9 (9 tiles for 16 cores is fine, avoids
        // 2D dispatch overhead). Large m (≥ 2048) uses LargeMc=192 for Square
        // 4608²'s L2 saturation win (0.95× of MKL). Shadowed as `Mc` locally
        // so the rest of SgemmTiled's body reads unchanged.
        int Mc = (m >= AdaptiveMcThreshold) ? LargeMc : SmallMc;
        int numRowBlocks = (m + Mc - 1) / Mc;
        bool canParallelize = allowParallel
            && UseParallelGemm
            && maxThreads > 1
            && numRowBlocks >= 1
            && !transA && !transB  // Parallel path uses the no-transpose Pack overloads
            && (long)m * k * n >= ParallelWorkThreshold;

        // Round up to micro-tile dimensions to avoid buffer overruns in PackA/PackB padding
        int mcRounded = ((Mc + Mr - 1) / Mr) * Mr;
        int ncRounded = ((Nc + Nr - 1) / Nr) * Nr;
        int packedASize = mcRounded * Kc;
        int packedBSize = Kc * ncRounded;
        float[] packedABuf = ArrayPool<float>.Shared.Rent(packedASize);
        float[] packedBBuf = ArrayPool<float>.Shared.Rent(packedBSize);

        try
        {
            for (int jc = 0; jc < n; jc += Nc)
            {
                int nc = Math.Min(Nc, n - jc);

                for (int pc = 0; pc < k; pc += Kc)
                {
                    int kc = Math.Min(Kc, k - pc);

                    // Decide per-tile parallel dispatch. numColSubs is adaptive to
                    // logical core count so numRowBlocks * numColSubs ≈ maxThreads.
                    // Iter 5: logical cores, not physical — SMT siblings help because
                    // blocked GEMM is load-port-limited not FMA-port-limited.
                    // Iter 7: allow numRowBlocks=1 through the 2D path so col-heavy
                    // problems like LM-head [64,128]x[128,50257] get parallelism.
                    int desiredColSubs = canParallelize ? Math.Max(1, maxThreads / numRowBlocks) : 1;
                    int maxColSubs = Math.Max(1, nc / (Nr * 4));
                    int numColSubs = Math.Min(desiredColSubs, maxColSubs);
                    int totalTiles = numRowBlocks * numColSubs;

                    // Iter 31b (REVERTED): tried gating 2D with (a) numRowBlocks
                    // * 2 < maxThreads AND (b) totalWork >= 2.5G to prevent the
                    // Square 1152² regression from iter 31. Catastrophic result:
                    // shapes that got kicked from 2D to 1D regressed 100-360%
                    // because 1D at numRowBlocks=6 leaves 10 of 16 cores idle
                    // (and Batched B=1's numRowBlocks=2 at Mc=192 leaves 14
                    // cores idle). The 2D dispatch was PROTECTING these shapes,
                    // not hurting them. Data:
                    //   Square 1152² Mc=192+2D: 4,487 µs (+9% vs iter 18c)
                    //   Square 1152² Mc=192+1D: 9,005 µs (+119% vs iter 18c)
                    //   Batched B=1  Mc=192+2D: 5,612 µs
                    //   Batched B=1  Mc=192+1D: 26,069 µs (+364%)
                    // So 2D at Mc=192 is the MINIMUM regression possible for
                    // these shapes. Kept the original unconditional-2D gate.
                    // The Square 1152² +9% under iter 31 is an accepted cost
                    // of Mc=192; see iter31b-run.log for evidence.
                    bool used2D = canParallelize && numColSubs >= 2 && totalTiles >= 2;
                    bool used1D = canParallelize && !used2D && numRowBlocks >= 2;

                    if (used2D)
                    {
                        SgemmTiledParallel2D(
                            a, b, c,
                            m, k, n,
                            jc, nc, pc, kc,
                            numRowBlocks, numColSubs);
                    }
                    else if (used1D)
                    {
                        // 1D parallel-M: each worker owns a disjoint row block with its
                        // own packed-A, B is packed once and shared read-only. Output
                        // row ranges are disjoint so no synchronization on C is needed.
                        SgemmTiledParallelM(
                            a, b, c,
                            m, k, n,
                            jc, nc, pc, kc,
                            numRowBlocks, packedBBuf);
                    }
                    else
                    {
                        // Sequential path (original)
                        PackA(a, packedABuf, lda, transA, ic: 0, mc: Math.Min(Mc, m), pc, kc);
                        PackB(b, packedBBuf, ldb, transB, pc, kc, jc, nc);

                        for (int ic = 0; ic < m; ic += Mc)
                        {
                            int mc = Math.Min(Mc, m - ic);
                            if (ic > 0) // first block already packed above
                                PackA(a, packedABuf, lda, transA, ic, mc, pc, kc);
                            MacroKernel(packedABuf, packedBBuf, c, mc, nc, kc, n, ic, jc);
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(packedABuf);
            ArrayPool<float>.Shared.Return(packedBBuf);
        }
    }

    /// <summary>
    /// N-dimension parallel GEMM: splits columns across workers.
    /// Ideal for small M, large N (e.g. Conv2D im2col where M=outChannels, N=outputSize).
    /// Each worker packs its own B slice and processes all M rows for that column range.
    /// Uses unsafe pinned pointers to avoid array copies for closure capture.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void SgemmTiledParallelN(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m, int k, int n,
        int jc, int nc, int pc, int kc,
        int numNrBlocks, int maxThreads,
        float[] packedABuf)
    {
        int numWorkers = Math.Min(maxThreads, numNrBlocks);
        int nrPerWorker = (numNrBlocks + numWorkers - 1) / numWorkers;

        // Iter 33: adaptive Mc (must match SgemmTiled's choice for consistency)
        int Mc = (m >= AdaptiveMcThreshold) ? LargeMc : SmallMc;

        // Pre-pack A (shared across all workers, m is small so only 1 Mc block)
        int firstMc = Math.Min(Mc, m);
        PackA(a, packedABuf, k, 0, firstMc, pc, kc);

        // Pre-pack B slices for each worker
        var packedBSlices = new float[numWorkers][];
        var sliceNcs = new int[numWorkers];
        var sliceJStarts = new int[numWorkers];
        int actualWorkers = numWorkers;

        for (int w = 0; w < numWorkers; w++)
        {
            int nrStart = w * nrPerWorker;
            if (nrStart >= numNrBlocks) { actualWorkers = w; break; }
            int nrEnd = Math.Min(nrStart + nrPerWorker, numNrBlocks);
            int jStart = nrStart * Nr;
            int jEnd = Math.Min(nrEnd * Nr, nc);
            int localNc = jEnd - jStart;
            sliceNcs[w] = localNc;
            sliceJStarts[w] = jStart;

            // Round up to Nr boundary since PackB pads remaining columns
            int packedNc = ((localNc + Nr - 1) / Nr) * Nr;
            packedBSlices[w] = ArrayPool<float>.Shared.Rent(kc * packedNc);
            PackB(b, packedBSlices[w], n, pc, kc, jc + jStart, localNc);
        }

        // Pin A (only needed if m > Mc for additional blocks)
        float[]? aArr = null;
        GCHandle aHandle = default;
        float* aPtr = null;
        if (m > Mc)
        {
            aArr = ArrayPool<float>.Shared.Rent(a.Length);
            a.CopyTo(aArr);
            aHandle = GCHandle.Alloc(aArr, GCHandleType.Pinned);
            aPtr = (float*)aHandle.AddrOfPinnedObject();
        }

        try
        {
            // Capture locals for closure
            var localPackedABuf = packedABuf;
            var localPackedBSlices = packedBSlices;
            var localSliceNcs = sliceNcs;
            var localSliceJStarts = sliceJStarts;
            var localAPtr = aPtr;
            int localALen = a.Length;
            int localM = m, localK = k, localN = n;
            int localJc = jc, localPc = pc, localKc = kc;
            int localFirstMc = firstMc;
            int cLen = c.Length;

            // Pin C for direct pointer access (workers write to non-overlapping columns)
            fixed (float* cPtr = c)
            {
                var localCPtr = cPtr;
                var localCLen = cLen;

                Helpers.CpuParallelSettings.LightweightParallel(actualWorkers, workerId =>
                {
                    int workerNc = localSliceNcs[workerId];
                    int jStart = localSliceJStarts[workerId];
                    var cSpan = new Span<float>(localCPtr, localCLen);

                    // First Mc block: use shared pre-packed A
                    MacroKernel(localPackedABuf, localPackedBSlices[workerId],
                        cSpan, localFirstMc, workerNc, localKc, localN, 0, localJc + jStart);

                    // Additional Mc blocks (if m > Mc)
                    for (int ic = Mc; ic < localM; ic += Mc)
                    {
                        int mc = Math.Min(Mc, localM - ic);
                        // Round up to Mr boundary since PackA pads remaining rows
                        int packedMc = ((mc + Mr - 1) / Mr) * Mr;
                        float[] workerPackedA = ArrayPool<float>.Shared.Rent(packedMc * localKc);
                        try
                        {
                            var aSpan = new ReadOnlySpan<float>(localAPtr, localALen);
                            PackA(aSpan, workerPackedA, localK, ic, mc, localPc, localKc);
                            MacroKernel(workerPackedA, localPackedBSlices[workerId],
                                cSpan, mc, workerNc, localKc, localN, ic, localJc + jStart);
                        }
                        finally
                        {
                            ArrayPool<float>.Shared.Return(workerPackedA);
                        }
                    }
                });
            }
        }
        finally
        {
            if (aHandle.IsAllocated) aHandle.Free();
            if (aArr is not null) ArrayPool<float>.Shared.Return(aArr);
            for (int w = 0; w < actualWorkers; w++)
            {
                ArrayPool<float>.Shared.Return(packedBSlices[w]);
            }
        }
    }

    /// <summary>
    /// M-dimension parallel GEMM: splits rows across workers.
    /// Ideal for tall matrices (m >= 512). Uses pinned pointers to avoid array copies.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void SgemmTiledParallelM(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m, int k, int n,
        int jc, int nc, int pc, int kc,
        int numRowBlocks, float[] packedBBuf)
    {
        // Iter 33: adaptive Mc (must match SgemmTiled's choice for consistency)
        int Mc = (m >= AdaptiveMcThreshold) ? LargeMc : SmallMc;

        // Pack B once (shared across all M workers, read-only)
        PackB(b, packedBBuf, n, pc, kc, jc, nc);

        // Pin A for closure capture (workers read non-overlapping rows)
        float[] aArr = ArrayPool<float>.Shared.Rent(a.Length);
        a.CopyTo(aArr);
        var aHandle = GCHandle.Alloc(aArr, GCHandleType.Pinned);

        try
        {
            var localPackedBBuf = packedBBuf;
            int localM = m, localK = k, localN = n;
            int localJc = jc, localNc = nc, localPc = pc, localKc = kc;
            float* localAPtr = (float*)aHandle.AddrOfPinnedObject();
            int localALen = a.Length;
            int cLen = c.Length;

            // Pin C for direct pointer access (workers write to non-overlapping rows)
            fixed (float* cPtr = c)
            {
                var localCPtr = cPtr;
                var localCLen = cLen;

                Helpers.CpuParallelSettings.LightweightParallel(numRowBlocks, iiBlock =>
                {
                    int ic = iiBlock * Mc;
                    int mc = Math.Min(Mc, localM - ic);
                    // Round up to Mr boundary since PackA pads remaining rows
                    int packedMc = ((mc + Mr - 1) / Mr) * Mr;
                    float[] localPackedA = ArrayPool<float>.Shared.Rent(packedMc * localKc);
                    try
                    {
                        var aSpan = new ReadOnlySpan<float>(localAPtr, localALen);
                        var cSpan = new Span<float>(localCPtr, localCLen);
                        PackA(aSpan, localPackedA, localK, ic, mc, localPc, localKc);
                        MacroKernel(localPackedA, localPackedBBuf, cSpan, mc, localNc, localKc, localN, ic, localJc);
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(localPackedA);
                    }
                });
            }
        }
        finally
        {
            aHandle.Free();
            ArrayPool<float>.Shared.Return(aArr);
        }
    }

    /// <summary>
    /// 2D parallel GEMM: dispatches a grid of (ic_block × jc_sub) tiles in parallel.
    /// Iter 4 (2026-04-11): breaks past the numRowBlocks parallelism ceiling. On a
    /// 16-core machine at 1024² with Mc=128, the 1D M-parallel path was limited to
    /// 8 workers; this variant adds col-sub parallelism to fill the remaining cores.
    ///
    /// Layout:
    ///   - Each row block r has its own packed-A buffer, shared across all col subs
    ///     (so PackA runs numRowBlocks times total, not numRowBlocks * numColSubs).
    ///   - Each col sub cs has its own packed-B buffer, shared across all row blocks
    ///     (so PackB runs numColSubs times, not numRowBlocks * numColSubs).
    ///   - Compute dispatches numRowBlocks * numColSubs tiles in parallel, each
    ///     reading its row's packed-A and its col-sub's packed-B, writing to a
    ///     disjoint (mc x subNc) region of C.
    ///
    /// Determinism: output regions are disjoint per tile, inner pc loop is still
    /// sequential, and the micro-kernel's FMA order is fixed. Results are bit-exact
    /// identical to the sequential and 1D-parallel paths.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void SgemmTiledParallel2D(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m, int k, int n,
        int jc, int nc, int pc, int kc,
        int numRowBlocks, int numColSubBlocks)
    {
        // Iter 33: adaptive Mc (must match SgemmTiled's choice for consistency)
        int Mc = (m >= AdaptiveMcThreshold) ? LargeMc : SmallMc;

        // colSubSize: number of B columns per sub-block, rounded down to a multiple
        // of Nr so MacroKernel's Nr panel loop stays clean. The last sub absorbs the
        // remainder. If nc < numColSubBlocks*Nr, fall back to the 1D parallel path.
        int colSubSize = (nc / numColSubBlocks / Nr) * Nr;
        if (colSubSize < Nr)
        {
            // Degenerate — not enough cols per sub. Caller should have checked,
            // but we guard here to avoid a zero-sized sub.
            colSubSize = nc;
            numColSubBlocks = 1;
        }

        int mcRounded = ((Mc + Mr - 1) / Mr) * Mr;
        int packedASizePerRow = mcRounded * Kc;
        int colSubRounded = ((colSubSize + Nr - 1) / Nr) * Nr;
        int lastColSubWidth = nc - (numColSubBlocks - 1) * colSubSize;
        int lastColSubRounded = ((lastColSubWidth + Nr - 1) / Nr) * Nr;
        int packedBSizePerSub = Kc * Math.Max(colSubRounded, lastColSubRounded);

        var packedABufs = new float[numRowBlocks][];
        var packedBBufs = new float[numColSubBlocks][];
        for (int r = 0; r < numRowBlocks; r++)
            packedABufs[r] = ArrayPool<float>.Shared.Rent(packedASizePerRow);
        for (int cs = 0; cs < numColSubBlocks; cs++)
            packedBBufs[cs] = ArrayPool<float>.Shared.Rent(packedBSizePerSub);

        try
        {
            var localPackedABufs = packedABufs;
            var localPackedBBufs = packedBBufs;
            int localM = m, localK = k, localN = n;
            int localJc = jc, localNc = nc, localPc = pc, localKc = kc;
            int localColSubSize = colSubSize;
            int localNumColSubs = numColSubBlocks;
            int localMc = Mc;

            // Pin A, B, C directly via the input spans — no extra copy. The fixed
            // block outlives every LightweightParallel call below (synchronous
            // dispatch waits for all workers before returning), so the pointers
            // stay valid throughout. Iter 6: previous version copied A and B into
            // rented arrays for pinning, which added 8MB of serial copy at 1024².
            fixed (float* aPtr0 = a)
            fixed (float* bPtr0 = b)
            fixed (float* cPtr0 = c)
            {
                float* localAPtr = aPtr0;
                int localALen = a.Length;
                float* localBPtr = bPtr0;
                int localBLen = b.Length;
                float* localCPtr = cPtr0;
                int localCLen = c.Length;

                // Iter 12: fuse phases 1 (pack A) and 2 (pack B) into a single parallel
                // dispatch to save one barrier per (jc, pc) iteration. Tasks 0..numRowBlocks-1
                // pack A, tasks numRowBlocks..numRowBlocks+numColSubs-1 pack B. All
                // independent output buffers — no contention, all can run concurrently.
                int numPackTasks = numRowBlocks + numColSubBlocks;
                int localNumRowBlocks = numRowBlocks;
                Helpers.CpuParallelSettings.LightweightParallel(numPackTasks, taskId =>
                {
                    if (taskId < localNumRowBlocks)
                    {
                        // Pack A for row block r = taskId
                        int r = taskId;
                        int ic = r * localMc;
                        int mcLocal = Math.Min(localMc, localM - ic);
                        if (mcLocal > 0)
                        {
                            var aSpan = new ReadOnlySpan<float>(localAPtr, localALen);
                            PackA(aSpan, localPackedABufs[r], localK, ic, mcLocal, localPc, localKc);
                        }
                    }
                    else
                    {
                        // Pack B for col sub cs = taskId - numRowBlocks
                        int cs = taskId - localNumRowBlocks;
                        int jStart = cs * localColSubSize;
                        int subNc = (cs == localNumColSubs - 1) ? (localNc - jStart) : localColSubSize;
                        if (subNc > 0)
                        {
                            var bSpan = new ReadOnlySpan<float>(localBPtr, localBLen);
                            PackB(bSpan, localPackedBBufs[cs], localN, localPc, localKc, localJc + jStart, subNc);
                        }
                    }
                });

                // Phase 2 (was 3): parallel compute for all (ic, jc_sub) tiles.
                int totalTiles = numRowBlocks * numColSubBlocks;
                Helpers.CpuParallelSettings.LightweightParallel(totalTiles, tileId =>
                {
                    int r = tileId / localNumColSubs;
                    int cs = tileId % localNumColSubs;
                    int ic = r * localMc;
                    int mcLocal = Math.Min(localMc, localM - ic);
                    int jStart = cs * localColSubSize;
                    int subNc = (cs == localNumColSubs - 1) ? (localNc - jStart) : localColSubSize;
                    if (mcLocal > 0 && subNc > 0)
                    {
                        var cSpan = new Span<float>(localCPtr, localCLen);
                        MacroKernel(
                            localPackedABufs[r], localPackedBBufs[cs],
                            cSpan, mcLocal, subNc, localKc, localN,
                            ic, localJc + jStart);
                    }
                });
            }
        }
        finally
        {
            for (int r = 0; r < numRowBlocks; r++)
                ArrayPool<float>.Shared.Return(packedABufs[r]);
            for (int cs = 0; cs < numColSubBlocks; cs++)
                ArrayPool<float>.Shared.Return(packedBBufs[cs]);
        }
    }

    /// <summary>
    /// Pack op(A)[ic:ic+mc, pc:pc+kc] into row-panel format for sequential access in micro-kernel.
    /// When transA=false: reads A[row, col] = a[row*lda + col] (row-major).
    /// When transA=true:  reads A^T[row, col] = a[col*lda + row] (transposed).
    /// Layout: groups of Mr rows, each stored as Mr x kc contiguous block.
    /// Iter 14: non-transpose full-Mr panel path uses direct pointer arithmetic with
    /// 6 hoisted row pointers and inner-loop unroll-by-4. Eliminates the JIT's bounds
    /// checks and repeated index calculations, cutting pack A time substantially.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackA(ReadOnlySpan<float> a, float[] packed, int lda, bool transA, int ic, int mc, int pc, int kc)
    {
        int pos = 0;
        int i = 0;

#if NET5_0_OR_GREATER
        // Fast path: non-transpose, full Mr-row panels. Hoist 6 row pointers out
        // of the p-loop, unroll p by 4.
        if (!transA)
        {
            fixed (float* aPtr = a)
            fixed (float* packedPtr = packed)
            {
                for (; i + Mr <= mc; i += Mr)
                {
                    float* row0 = aPtr + (ic + i + 0) * lda + pc;
                    float* row1 = aPtr + (ic + i + 1) * lda + pc;
                    float* row2 = aPtr + (ic + i + 2) * lda + pc;
                    float* row3 = aPtr + (ic + i + 3) * lda + pc;
                    float* row4 = aPtr + (ic + i + 4) * lda + pc;
                    float* row5 = aPtr + (ic + i + 5) * lda + pc;
                    float* pp = packedPtr + pos;

                    int p = 0;
                    for (; p + 4 <= kc; p += 4)
                    {
                        pp[0]  = row0[0]; pp[1]  = row1[0]; pp[2]  = row2[0]; pp[3]  = row3[0]; pp[4]  = row4[0]; pp[5]  = row5[0];
                        pp[6]  = row0[1]; pp[7]  = row1[1]; pp[8]  = row2[1]; pp[9]  = row3[1]; pp[10] = row4[1]; pp[11] = row5[1];
                        pp[12] = row0[2]; pp[13] = row1[2]; pp[14] = row2[2]; pp[15] = row3[2]; pp[16] = row4[2]; pp[17] = row5[2];
                        pp[18] = row0[3]; pp[19] = row1[3]; pp[20] = row2[3]; pp[21] = row3[3]; pp[22] = row4[3]; pp[23] = row5[3];
                        pp += 24;
                        row0 += 4; row1 += 4; row2 += 4; row3 += 4; row4 += 4; row5 += 4;
                    }
                    for (; p < kc; p++)
                    {
                        pp[0] = row0[0]; pp[1] = row1[0]; pp[2] = row2[0]; pp[3] = row3[0]; pp[4] = row4[0]; pp[5] = row5[0];
                        pp += 6;
                        row0++; row1++; row2++; row3++; row4++; row5++;
                    }
                    pos += Mr * kc;
                }
            }
        }
        else
#endif
        {
            // Scalar fallback (transpose path or older TFMs)
            for (; i + Mr <= mc; i += Mr)
            {
                for (int p = 0; p < kc; p++)
                {
                    for (int ii = 0; ii < Mr; ii++)
                    {
                        int row = ic + i + ii;
                        int col = pc + p;
                        packed[pos++] = transA ? a[col * lda + row] : a[row * lda + col];
                    }
                }
            }
        }

        // Remaining rows (less than Mr)
        int remaining = mc - i;
        if (remaining > 0)
        {
            for (int p = 0; p < kc; p++)
            {
                for (int ii = 0; ii < remaining; ii++)
                {
                    int row = ic + i + ii;
                    int col = pc + p;
                    packed[pos++] = transA ? a[col * lda + row] : a[row * lda + col];
                }
                for (int ii = remaining; ii < Mr; ii++)
                {
                    packed[pos++] = 0;
                }
            }
        }
    }

    /// <summary>
    /// Backward-compatible PackA without transpose (used by parallel paths).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PackA(ReadOnlySpan<float> a, float[] packed, int lda, int ic, int mc, int pc, int kc)
        => PackA(a, packed, lda, false, ic, mc, pc, kc);

    /// <summary>
    /// Pack op(B)[pc:pc+kc, jc:jc+nc] into column-panel format.
    /// When transB=false: reads B[row, col] = b[row*ldb + col] (row-major).
    /// When transB=true:  reads B^T[row, col] = b[col*ldb + row] (transposed).
    /// Layout: groups of Nr columns, each stored as kc x Nr contiguous block.
    /// Iter 11: non-transpose full-Nr path uses 2x Vector256 loads/stores per k
    /// iteration — reads 16 contiguous floats from a B row and writes them to
    /// the packed buffer as two 256-bit aligned writes. ~8x faster than the
    /// scalar fallback on cached data.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackB(ReadOnlySpan<float> b, float[] packed, int ldb, bool transB, int pc, int kc, int jc, int nc)
    {
        int pos = 0;
        int j = 0;

#if NET5_0_OR_GREATER
        // Fast path: non-transpose, full Nr-column panels with SIMD copy.
        // Nr=16 floats per row → 2 Vector256<float> = 64 bytes per iter.
        if (!transB && Avx.IsSupported)
        {
            fixed (float* pBBase = b)
            fixed (float* pPackedBase = packed)
            {
                for (; j + Nr <= nc; j += Nr)
                {
                    float* pPacked = pPackedBase + pos;
                    float* pBCol = pBBase + pc * ldb + (jc + j);
                    for (int p = 0; p < kc; p++)
                    {
                        var v0 = Avx.LoadVector256(pBCol);
                        var v1 = Avx.LoadVector256(pBCol + 8);
                        Avx.Store(pPacked, v0);
                        Avx.Store(pPacked + 8, v1);
                        pBCol += ldb;
                        pPacked += Nr;
                    }
                    pos += kc * Nr;
                }
            }
        }
#endif

        // Remaining full panels (transpose path or older TFMs) — scalar loop.
        for (; j + Nr <= nc; j += Nr)
        {
            for (int p = 0; p < kc; p++)
            {
                for (int jj = 0; jj < Nr; jj++)
                {
                    int row = pc + p;
                    int col = jc + j + jj;
                    packed[pos++] = transB ? b[col * ldb + row] : b[row * ldb + col];
                }
            }
        }

        // Remaining columns (less than Nr)
        int remaining = nc - j;
        if (remaining > 0)
        {
            for (int p = 0; p < kc; p++)
            {
                for (int jj = 0; jj < remaining; jj++)
                {
                    int row = pc + p;
                    int col = jc + j + jj;
                    packed[pos++] = transB ? b[col * ldb + row] : b[row * ldb + col];
                }
                for (int jj = remaining; jj < Nr; jj++)
                {
                    packed[pos++] = 0;
                }
            }
        }
    }

    /// <summary>
    /// Backward-compatible PackB without transpose (used by parallel paths).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PackB(ReadOnlySpan<float> b, float[] packed, int ldb, int pc, int kc, int jc, int nc)
        => PackB(b, packed, ldb, false, pc, kc, jc, nc);

    /// <summary>
    /// Macro-kernel: iterate over packed panels with Mr x Nr micro-kernel tiles.
    /// Uses JIT-compiled micro-kernel when available for guaranteed optimal register allocation.
    /// </summary>
    [MethodImpl(HotInline)]
    private static unsafe void MacroKernel(
        float[] packedA,
        float[] packedB,
        Span<float> c,
        int mc, int nc, int kc,
        int ldc, int icOffset, int jcOffset)
    {
        int nrBlocks = (nc + Nr - 1) / Nr;
        int mrBlocks = (mc + Mr - 1) / Mr;

        // Try JIT micro-kernel: bakes ldc as immediate, guarantees 12 YMM accumulators in registers
        CpuJitKernels.GemmMicroKernel? jitKernel =
            CpuJitSelfTest.IsVerified ? CpuJitKernels.GetGemmMicroKernel(kc, ldc) : null;

        // Iter 10 (reverted): tried pinning packedA/B/C once around the whole micro-
        // kernel loop to eliminate per-call fixed overhead. It regressed ~14% at 512².
        // The JIT was already eliding the inner-loop fixed statements, and the outer
        // fixed created a longer-lived GC pin that seemed to hurt. Reverted.
        for (int jr = 0; jr < nrBlocks; jr++)
        {
            int jLocal = jr * Nr;
            int nc_actual = Math.Min(Nr, nc - jLocal);
            int bPanelOffset = jr * kc * Nr;

            for (int ir = 0; ir < mrBlocks; ir++)
            {
                int iLocal = ir * Mr;
                int mc_actual = Math.Min(Mr, mc - iLocal);
                int aPanelOffset = ir * kc * Mr;

                // Iter 19 (oneDNN-style next-A-panel prefetch): before dispatching the
                // current micro-kernel, issue prefetcht2 hints on the NEXT ir's packedA
                // panel. The hardware prefetcher handles *within-panel* sequential access
                // well (iter 16 verified this on Zen 2) but doesn't reliably cross the
                // ir → ir+1 boundary. Prefetching ~4 cache lines (256 B) at the start of
                // the next panel lets the HW prefetcher pick up from there and have the
                // data warm in L1/L2 by the time MicroKernel6x16 first touches it.
                //
                // prefetcht2 specifically (not t0/t1): keeps the hint from evicting the
                // current Mr-panel from L1/L2 and from competing with inner-loop loads
                // for load-port bandwidth. Iter 9's prefetcht0 inside the K loop was
                // reverted in iter 16 precisely because it competed with loads; placing
                // the hints *outside* the K loop and using the weakest level avoids that.
                if (ir + 1 < mrBlocks)
                {
                    int nextAPanelOffset = (ir + 1) * kc * Mr;
                    unsafe
                    {
                        fixed (float* pNextA = &packedA[nextAPanelOffset])
                        {
                            Sse.Prefetch2(pNextA);
                            Sse.Prefetch2(pNextA + 16);
                            Sse.Prefetch2(pNextA + 32);
                            Sse.Prefetch2(pNextA + 48);
                        }
                    }
                }

                if (mc_actual == Mr && nc_actual == Nr)
                {
                    if (jitKernel is not null)
                    {
                        // JIT micro-kernel: pass pointers directly
                        fixed (float* pA = &packedA[aPanelOffset])
                        fixed (float* pB = &packedB[bPanelOffset])
                        fixed (float* pC = c)
                        {
                            int cOffset = (icOffset + iLocal) * ldc + (jcOffset + jLocal);
                            jitKernel(pA, pB, pC + cOffset, kc);
                        }
                    }
                    else
                    {
                        // C# intrinsics micro-kernel fallback
                        MicroKernel6x16(
                            packedA, aPanelOffset,
                            packedB, bPanelOffset,
                            c, ldc,
                            icOffset + iLocal, jcOffset + jLocal,
                            kc);
                    }
                }
                else if (mc_actual > 0 && nc_actual > 0)
                {
                    // Iter 18a + 18b: any partial tile (mc ≤ Mr or nc ≤ Nr) with at
                    // least one active lane. Previously fell through to MicroKernelScalar
                    // (fully scalar K-loop) for any edge case — dominated per-head
                    // attention A·V [256,256]×[256,72] (10.3× slower than MKL; now 2.7×
                    // after iter 18a covered only the nc-edge, and closing further with
                    // this mc-edge extension).
                    //
                    // The masked vector kernel uses the same 12-accumulator FMA path as
                    // MicroKernel6x16. PackA zero-pads rows past mc_actual (lines 772-775
                    // of PackA) and PackB zero-pads cols past nc_actual (lines 830-833 of
                    // PackB), so FMAs on inactive positions contribute nothing to the
                    // accumulators. The store phase masks per-row (mc_actual) and per-col
                    // (nc_actual) so we only write the active output cells.
                    MicroKernelMxNMasked(
                        packedA, aPanelOffset,
                        packedB, bPanelOffset,
                        c, ldc,
                        icOffset + iLocal, jcOffset + jLocal,
                        kc, mc_actual, nc_actual);
                }
            }
        }
    }

    /// <summary>
    /// 6x16 FMA micro-kernel: computes a 6-row x 16-column tile of C.
    /// Uses 12 Vector256 accumulators (6 rows x 2 vectors of 8 floats = 16 columns).
    /// Inner loop over K dimension broadcasts A elements and FMA with B row.
    /// </summary>
    [MethodImpl(HotInline)]
    private static unsafe void MicroKernel6x16(
        float[] packedA, int aOffset,
        float[] packedB, int bOffset,
        Span<float> c, int ldc,
        int cRow, int cCol,
        int kc)
    {
        // 12 accumulators: 6 rows x 2 Vector256 (16 columns)
        var c00 = Vector256<float>.Zero; var c01 = Vector256<float>.Zero;
        var c10 = Vector256<float>.Zero; var c11 = Vector256<float>.Zero;
        var c20 = Vector256<float>.Zero; var c21 = Vector256<float>.Zero;
        var c30 = Vector256<float>.Zero; var c31 = Vector256<float>.Zero;
        var c40 = Vector256<float>.Zero; var c41 = Vector256<float>.Zero;
        var c50 = Vector256<float>.Zero; var c51 = Vector256<float>.Zero;

        ref float aRef = ref MemoryMarshal.GetArrayDataReference(packedA);
        ref float bRef = ref MemoryMarshal.GetArrayDataReference(packedB);

        // Iter 16: removed software prefetch entirely. Zen 2's hardware prefetcher
        // already handles sequential access well. Iter 9's explicit Sse.Prefetch0
        // hints consumed load ports (10 loads per iter → 12 with 2 prefetches =
        // 6 cycles load-limited vs 5 without), slightly slowing the critical path.
        // The branch check (if p < prefetchLimit) also hurt loop predictability.
        //
        // Iter 20: 4-way manual K unroll (vs single-step loop). Motivation from
        // OpenBLAS's sgemm_kernel_16x4_haswell.S which uses 8-way K unroll — we
        // take a lighter 4-way to stay within a reasonable basic-block size for
        // RyuJIT while still cutting the loop branch overhead by 4× and giving
        // the scheduler 48 FMAs per loop body to overlap. Same 12 accumulators,
        // same live-register budget, same A-broadcast ping-pong pattern — the
        // only change is 4 K iterations per pass instead of 1.
        int pEnd4 = kc & ~3;  // round down to multiple of 4
        int p = 0;
        for (; p < pEnd4; p += 4)
        {
            int bBase = bOffset + p * Nr;
            int aBase = aOffset + p * Mr;

            // --- K iteration +0 ---
            var b0 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bBase)));
            var b1 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bBase + 8)));
            var a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 0));
            c00 = Fma.MultiplyAdd(a, b0, c00); c01 = Fma.MultiplyAdd(a, b1, c01);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 1));
            c10 = Fma.MultiplyAdd(a, b0, c10); c11 = Fma.MultiplyAdd(a, b1, c11);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 2));
            c20 = Fma.MultiplyAdd(a, b0, c20); c21 = Fma.MultiplyAdd(a, b1, c21);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 3));
            c30 = Fma.MultiplyAdd(a, b0, c30); c31 = Fma.MultiplyAdd(a, b1, c31);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 4));
            c40 = Fma.MultiplyAdd(a, b0, c40); c41 = Fma.MultiplyAdd(a, b1, c41);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 5));
            c50 = Fma.MultiplyAdd(a, b0, c50); c51 = Fma.MultiplyAdd(a, b1, c51);

            // --- K iteration +1 ---
            b0 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bBase + Nr)));
            b1 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bBase + Nr + 8)));
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + Mr + 0));
            c00 = Fma.MultiplyAdd(a, b0, c00); c01 = Fma.MultiplyAdd(a, b1, c01);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + Mr + 1));
            c10 = Fma.MultiplyAdd(a, b0, c10); c11 = Fma.MultiplyAdd(a, b1, c11);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + Mr + 2));
            c20 = Fma.MultiplyAdd(a, b0, c20); c21 = Fma.MultiplyAdd(a, b1, c21);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + Mr + 3));
            c30 = Fma.MultiplyAdd(a, b0, c30); c31 = Fma.MultiplyAdd(a, b1, c31);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + Mr + 4));
            c40 = Fma.MultiplyAdd(a, b0, c40); c41 = Fma.MultiplyAdd(a, b1, c41);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + Mr + 5));
            c50 = Fma.MultiplyAdd(a, b0, c50); c51 = Fma.MultiplyAdd(a, b1, c51);

            // --- K iteration +2 ---
            b0 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bBase + 2 * Nr)));
            b1 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bBase + 2 * Nr + 8)));
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 2 * Mr + 0));
            c00 = Fma.MultiplyAdd(a, b0, c00); c01 = Fma.MultiplyAdd(a, b1, c01);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 2 * Mr + 1));
            c10 = Fma.MultiplyAdd(a, b0, c10); c11 = Fma.MultiplyAdd(a, b1, c11);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 2 * Mr + 2));
            c20 = Fma.MultiplyAdd(a, b0, c20); c21 = Fma.MultiplyAdd(a, b1, c21);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 2 * Mr + 3));
            c30 = Fma.MultiplyAdd(a, b0, c30); c31 = Fma.MultiplyAdd(a, b1, c31);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 2 * Mr + 4));
            c40 = Fma.MultiplyAdd(a, b0, c40); c41 = Fma.MultiplyAdd(a, b1, c41);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 2 * Mr + 5));
            c50 = Fma.MultiplyAdd(a, b0, c50); c51 = Fma.MultiplyAdd(a, b1, c51);

            // --- K iteration +3 ---
            b0 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bBase + 3 * Nr)));
            b1 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bBase + 3 * Nr + 8)));
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 3 * Mr + 0));
            c00 = Fma.MultiplyAdd(a, b0, c00); c01 = Fma.MultiplyAdd(a, b1, c01);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 3 * Mr + 1));
            c10 = Fma.MultiplyAdd(a, b0, c10); c11 = Fma.MultiplyAdd(a, b1, c11);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 3 * Mr + 2));
            c20 = Fma.MultiplyAdd(a, b0, c20); c21 = Fma.MultiplyAdd(a, b1, c21);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 3 * Mr + 3));
            c30 = Fma.MultiplyAdd(a, b0, c30); c31 = Fma.MultiplyAdd(a, b1, c31);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 3 * Mr + 4));
            c40 = Fma.MultiplyAdd(a, b0, c40); c41 = Fma.MultiplyAdd(a, b1, c41);
            a = Vector256.Create(Unsafe.Add(ref aRef, aBase + 3 * Mr + 5));
            c50 = Fma.MultiplyAdd(a, b0, c50); c51 = Fma.MultiplyAdd(a, b1, c51);
        }

        // Scalar K tail (kc % 4 remainder, 0..3 iterations)
        for (; p < kc; p++)
        {
            int bIdx = bOffset + p * Nr;
            var b0 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bIdx)));
            var b1 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bIdx + 8)));
            int aIdx = aOffset + p * Mr;
            var a = Vector256.Create(Unsafe.Add(ref aRef, aIdx));
            c00 = Fma.MultiplyAdd(a, b0, c00); c01 = Fma.MultiplyAdd(a, b1, c01);
            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 1));
            c10 = Fma.MultiplyAdd(a, b0, c10); c11 = Fma.MultiplyAdd(a, b1, c11);
            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 2));
            c20 = Fma.MultiplyAdd(a, b0, c20); c21 = Fma.MultiplyAdd(a, b1, c21);
            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 3));
            c30 = Fma.MultiplyAdd(a, b0, c30); c31 = Fma.MultiplyAdd(a, b1, c31);
            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 4));
            c40 = Fma.MultiplyAdd(a, b0, c40); c41 = Fma.MultiplyAdd(a, b1, c41);
            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 5));
            c50 = Fma.MultiplyAdd(a, b0, c50); c51 = Fma.MultiplyAdd(a, b1, c51);
        }

        // Store results back to C (accumulate)
        ref float cRef = ref MemoryMarshal.GetReference(c);
        StoreAccumRow(ref cRef, cRow, cCol, ldc, c00, c01);
        StoreAccumRow(ref cRef, cRow + 1, cCol, ldc, c10, c11);
        StoreAccumRow(ref cRef, cRow + 2, cCol, ldc, c20, c21);
        StoreAccumRow(ref cRef, cRow + 3, cCol, ldc, c30, c31);
        StoreAccumRow(ref cRef, cRow + 4, cCol, ldc, c40, c41);
        StoreAccumRow(ref cRef, cRow + 5, cCol, ldc, c50, c51);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreAccumRow(
        ref float cRef, int row, int col, int ldc,
        Vector256<float> v0, Vector256<float> v1)
    {
        int offset = row * ldc + col;
        ref float target = ref Unsafe.Add(ref cRef, offset);

        var existing0 = Unsafe.ReadUnaligned<Vector256<float>>(
            ref Unsafe.As<float, byte>(ref target));
        var existing1 = Unsafe.ReadUnaligned<Vector256<float>>(
            ref Unsafe.As<float, byte>(ref Unsafe.Add(ref target, 8)));

        Unsafe.WriteUnaligned(
            ref Unsafe.As<float, byte>(ref target),
            Avx.Add(existing0, v0));
        Unsafe.WriteUnaligned(
            ref Unsafe.As<float, byte>(ref Unsafe.Add(ref target, 8)),
            Avx.Add(existing1, v1));
    }

    // Iter 18a: precomputed masked-store masks for partial-Nr tiles, indexed by the
    // number of active lanes (0..8). Used to widen the masked 6×N kernel without
    // constructing a mask from scratch per call. MSB=1 in a lane means that lane
    // is active for MaskLoad/MaskStore.
    private static readonly Vector256<int>[] _partialNrMasks = new[]
    {
        Vector256.Create( 0,  0,  0,  0,  0,  0,  0,  0),
        Vector256.Create(-1,  0,  0,  0,  0,  0,  0,  0),
        Vector256.Create(-1, -1,  0,  0,  0,  0,  0,  0),
        Vector256.Create(-1, -1, -1,  0,  0,  0,  0,  0),
        Vector256.Create(-1, -1, -1, -1,  0,  0,  0,  0),
        Vector256.Create(-1, -1, -1, -1, -1,  0,  0,  0),
        Vector256.Create(-1, -1, -1, -1, -1, -1,  0,  0),
        Vector256.Create(-1, -1, -1, -1, -1, -1, -1,  0),
        Vector256.Create(-1, -1, -1, -1, -1, -1, -1, -1),
    };

    /// <summary>
    /// Iter 18a + 18b masked Mr×Nr micro-kernel for any partial tile
    /// (0 &lt; mc_actual &lt;= Mr, 0 &lt; nc_actual &lt;= Nr).
    ///
    /// Shares the 12-accumulator FMA inner loop with <see cref="MicroKernel6x16"/>.
    /// PackA zero-pads rows past mc_actual (SimdGemm.cs:772-775) and PackB zero-
    /// pads cols past nc_actual (SimdGemm.cs:830-833), so FMAs at inactive
    /// positions contribute 0 to the accumulators. The store phase uses per-row
    /// guards (mc_actual) and per-column masked load/store (nc_actual) so we only
    /// touch the active output cells — surrounding data is preserved.
    ///
    /// Iter 18a covered only the mc==Mr, nc&lt;Nr edge. Iter 18b extends to the
    /// mc&lt;Mr cases (all four combinations: mc==Mr/nc&lt;Nr, mc&lt;Mr/nc==Nr,
    /// mc&lt;Mr/nc&lt;Nr). The full-tile case mc==Mr && nc==Nr is still handled by
    /// the fast path above (JIT or plain MicroKernel6x16).
    /// </summary>
    [MethodImpl(HotInline)]
    private static unsafe void MicroKernelMxNMasked(
        float[] packedA, int aOffset,
        float[] packedB, int bOffset,
        Span<float> c, int ldc,
        int cRow, int cCol,
        int kc, int mc_actual, int nc_actual)
    {
        // 12 accumulators — identical to MicroKernel6x16.
        var c00 = Vector256<float>.Zero; var c01 = Vector256<float>.Zero;
        var c10 = Vector256<float>.Zero; var c11 = Vector256<float>.Zero;
        var c20 = Vector256<float>.Zero; var c21 = Vector256<float>.Zero;
        var c30 = Vector256<float>.Zero; var c31 = Vector256<float>.Zero;
        var c40 = Vector256<float>.Zero; var c41 = Vector256<float>.Zero;
        var c50 = Vector256<float>.Zero; var c51 = Vector256<float>.Zero;

        ref float aRef = ref MemoryMarshal.GetArrayDataReference(packedA);
        ref float bRef = ref MemoryMarshal.GetArrayDataReference(packedB);

        for (int p = 0; p < kc; p++)
        {
            int bIdx = bOffset + p * Nr;
            var b0 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bIdx)));
            var b1 = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref bRef, bIdx + 8)));

            int aIdx = aOffset + p * Mr;
            var a = Vector256.Create(Unsafe.Add(ref aRef, aIdx));
            c00 = Fma.MultiplyAdd(a, b0, c00); c01 = Fma.MultiplyAdd(a, b1, c01);

            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 1));
            c10 = Fma.MultiplyAdd(a, b0, c10); c11 = Fma.MultiplyAdd(a, b1, c11);

            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 2));
            c20 = Fma.MultiplyAdd(a, b0, c20); c21 = Fma.MultiplyAdd(a, b1, c21);

            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 3));
            c30 = Fma.MultiplyAdd(a, b0, c30); c31 = Fma.MultiplyAdd(a, b1, c31);

            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 4));
            c40 = Fma.MultiplyAdd(a, b0, c40); c41 = Fma.MultiplyAdd(a, b1, c41);

            a = Vector256.Create(Unsafe.Add(ref aRef, aIdx + 5));
            c50 = Fma.MultiplyAdd(a, b0, c50); c51 = Fma.MultiplyAdd(a, b1, c51);
        }

        // Build column lane masks from nc_actual.
        //   nc_actual ∈ (0, 8]:  lane 0 is partial (nc_actual lanes), lane 1 skipped.
        //   nc_actual ∈ (8, 16]: lane 0 fully used, lane 1 is partial (nc_actual - 8).
        //   (nc_actual == 16 lands lane1N=8, which is the "all lanes" mask — same
        //    behavior as an unmasked store, which is what we want for the nc==Nr case.)
        int lane0N = nc_actual >= 8 ? 8 : nc_actual;
        int lane1N = nc_actual >= 8 ? nc_actual - 8 : 0;
        Vector256<int> mask0 = _partialNrMasks[lane0N];
        Vector256<int> mask1 = _partialNrMasks[lane1N];

        // Masked accumulate-and-store back to C, row-by-row with mc_actual guard.
        // When mc_actual < Mr, rows past mc_actual are skipped entirely — their
        // accumulators (which are zero, since PackA zero-padded the A rows) would
        // otherwise overwrite valid data in the NEXT Ir block or past end of C.
        fixed (float* pC = c)
        {
            float* row = pC + cRow * ldc + cCol;
            if (mc_actual > 0) StoreMaskedAccumRow(row,             mask0, mask1, c00, c01);
            if (mc_actual > 1) StoreMaskedAccumRow(row + ldc,       mask0, mask1, c10, c11);
            if (mc_actual > 2) StoreMaskedAccumRow(row + ldc * 2,   mask0, mask1, c20, c21);
            if (mc_actual > 3) StoreMaskedAccumRow(row + ldc * 3,   mask0, mask1, c30, c31);
            if (mc_actual > 4) StoreMaskedAccumRow(row + ldc * 4,   mask0, mask1, c40, c41);
            if (mc_actual > 5) StoreMaskedAccumRow(row + ldc * 5,   mask0, mask1, c50, c51);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void StoreMaskedAccumRow(
        float* row, Vector256<int> mask0, Vector256<int> mask1,
        Vector256<float> v0, Vector256<float> v1)
    {
        // Masked accumulate: (read-existing, add, write). Avx.MaskLoad/MaskStore for
        // float take a Vector256<float> mask — reinterpret the int mask (MSB in each
        // int lane acts as the selector, bit-identical to the float-mask convention
        // since the int lane bits overlay the float lane bits).
        var m0f = mask0.AsSingle();
        var m1f = mask1.AsSingle();
        var existing0 = Avx.MaskLoad(row, m0f);
        var existing1 = Avx.MaskLoad(row + 8, m1f);
        Avx.MaskStore(row, m0f, Avx.Add(existing0, v0));
        Avx.MaskStore(row + 8, m1f, Avx.Add(existing1, v1));
    }

    /// <summary>
    /// Scalar micro-kernel for edge cases where tile is smaller than Mr x Nr.
    /// </summary>
    [MethodImpl(HotInline)]
    private static void MicroKernelScalar(
        float[] packedA, int aOffset,
        float[] packedB, int bOffset,
        Span<float> c, int ldc,
        int cRow, int cCol,
        int kc, int mr, int nr)
    {
        for (int p = 0; p < kc; p++)
        {
            for (int i = 0; i < mr; i++)
            {
                float aVal = packedA[aOffset + p * Mr + i];
                int cIdx = (cRow + i) * ldc + cCol;
                int bIdx = bOffset + p * Nr;
                for (int j = 0; j < nr; j++)
                {
#if NET5_0_OR_GREATER
                    c[cIdx + j] = MathF.FusedMultiplyAdd(aVal, packedB[bIdx + j], c[cIdx + j]);
#else
                    c[cIdx + j] += aVal * packedB[bIdx + j];
#endif
                }
            }
        }
    }
#endif
}
