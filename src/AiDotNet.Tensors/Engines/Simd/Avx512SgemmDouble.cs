using System;
using System.Threading.Tasks;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// AVX-512 DGEMM microkernel (FP64). Target: Intel Skylake-X, Ice Lake,
/// Sapphire Rapids, Zen 4 / Zen 5. Uses 8-wide <see cref="Vector512"/> FMA
/// for ~2× throughput vs the Vector256 AVX2 FP64 path on zmm-capable CPUs.
///
/// <para>Stage 7 (#415) scope: feature gate + 8×16 microkernel + tiled
/// driver. Mirrors the FP32 <see cref="Avx512Sgemm"/> structure with the
/// FP64 lane count (8 doubles per Vector512&lt;double&gt; vs 16 floats per
/// Vector512&lt;float&gt;). When the AVX-512 gate is not met (older CPUs /
/// non-x86) the entry point falls through to the AVX2 path in
/// <see cref="SimdGemm.Dgemm"/>, so this is purely additive.</para>
///
/// <para>Geometry: 8 rows × 16 cols per micro-tile. 16 accumulators
/// (8 rows × 2 ZMM vectors), leaving 16 ZMM registers free for the
/// streaming B/A loads. Same accumulator count as the FP32 16×16 kernel
/// but half the row count because Vector512&lt;double&gt; is 8-wide
/// (vs 16-wide float) — matches the BLIS-canonical FP64 AVX-512 layout.
/// </para>
/// </summary>
internal static class Avx512SgemmDouble
{
    /// <summary>
    /// True when the AVX-512 FP64 path is callable: compiled on .NET 8+,
    /// runtime reports AVX-512F support, and the feature isn't gated off.
    /// Single gate consulted by upstream dispatchers.
    /// </summary>
    public static bool CanUse
    {
        get
        {
#if NET8_0_OR_GREATER
            return CpuFeatures.HasAVX512F;
#else
            return false;
#endif
        }
    }

    /// <summary>
    /// Entry point for the AVX-512 blocked DGEMM. Routes to the 8×16
    /// microkernel (<see cref="Run8x16Tile"/>) when the shape qualifies,
    /// falls back to the AVX2 path for small / misaligned problems.
    /// </summary>
    public static void DgemmBlocked(
        ReadOnlySpan<double> a, int lda,
        ReadOnlySpan<double> b, int ldb,
        Span<double> c,
        int m, int k, int n,
        bool allowParallel)
    {
#if NET8_0_OR_GREATER
        // Fast path: aligned dims, enough work per tile. Misaligned m or
        // n falls to AVX2 which handles arbitrary shapes via its own
        // packed/scalar tail logic.
        if (m >= 8 && n >= 16
            && m % 8 == 0 && n % 16 == 0
            && Avx512F.IsSupported)
        {
            RunTiledMnAligned(a, lda, b, ldb, c, m, k, n, allowParallel);
            return;
        }
#endif
        // AVX2/scalar fallback for small or non-aligned shapes. Route to
        // BlasManaged.Gemm directly (skipping the obsolete SimdGemm.Dgemm
        // shim) — bypassing the shim avoids the [Obsolete] error and keeps
        // this call path on the same managed-GEMM target that all other
        // FP64 dispatchers in this branch use. BlasManaged.Gemm handles
        // clearing C internally.
        AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            c, ldc: n,
            m, n, k,
            new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<double>
            {
                PackingMode = AiDotNet.Tensors.Engines.BlasManaged.PackingMode.DisableAutotune
            });
    }

#if NET8_0_OR_GREATER
    /// <summary>
    /// 8 (rows) × 16 (cols) microkernel — 16 Vector512&lt;double&gt;
    /// accumulators (8 rows × 2 col-vectors), inner K loop does two 8-wide
    /// B loads plus 8 broadcast-FMAs into each col-vector per K step
    /// (saturates both AVX-512 FMA ports on Intel SKL-X / Sapphire Rapids
    /// and the single port on Zen 4 — Zen 4 will get half-throughput
    /// vs Intel but still ~2× the AVX2 baseline).
    /// </summary>
    private static unsafe void Run8x16Tile(
        double* aPtr, int lda,
        double* bPtr, int ldb,
        double* cPtr, int ldc,
        int k)
    {
        // 16 ZMM accumulators arranged row-major: acc[row, colVec].
        var c00 = Vector512<double>.Zero; var c01 = Vector512<double>.Zero;
        var c10 = Vector512<double>.Zero; var c11 = Vector512<double>.Zero;
        var c20 = Vector512<double>.Zero; var c21 = Vector512<double>.Zero;
        var c30 = Vector512<double>.Zero; var c31 = Vector512<double>.Zero;
        var c40 = Vector512<double>.Zero; var c41 = Vector512<double>.Zero;
        var c50 = Vector512<double>.Zero; var c51 = Vector512<double>.Zero;
        var c60 = Vector512<double>.Zero; var c61 = Vector512<double>.Zero;
        var c70 = Vector512<double>.Zero; var c71 = Vector512<double>.Zero;

        for (int kk = 0; kk < k; kk++)
        {
            var b0 = Vector512.Load(bPtr + kk * ldb);
            var b1 = Vector512.Load(bPtr + kk * ldb + 8);

            var a0 = Vector512.Create(aPtr[0 * lda + kk]);
            var a1 = Vector512.Create(aPtr[1 * lda + kk]);
            var a2 = Vector512.Create(aPtr[2 * lda + kk]);
            var a3 = Vector512.Create(aPtr[3 * lda + kk]);
            var a4 = Vector512.Create(aPtr[4 * lda + kk]);
            var a5 = Vector512.Create(aPtr[5 * lda + kk]);
            var a6 = Vector512.Create(aPtr[6 * lda + kk]);
            var a7 = Vector512.Create(aPtr[7 * lda + kk]);

            c00 = Avx512F.FusedMultiplyAdd(a0, b0, c00); c01 = Avx512F.FusedMultiplyAdd(a0, b1, c01);
            c10 = Avx512F.FusedMultiplyAdd(a1, b0, c10); c11 = Avx512F.FusedMultiplyAdd(a1, b1, c11);
            c20 = Avx512F.FusedMultiplyAdd(a2, b0, c20); c21 = Avx512F.FusedMultiplyAdd(a2, b1, c21);
            c30 = Avx512F.FusedMultiplyAdd(a3, b0, c30); c31 = Avx512F.FusedMultiplyAdd(a3, b1, c31);
            c40 = Avx512F.FusedMultiplyAdd(a4, b0, c40); c41 = Avx512F.FusedMultiplyAdd(a4, b1, c41);
            c50 = Avx512F.FusedMultiplyAdd(a5, b0, c50); c51 = Avx512F.FusedMultiplyAdd(a5, b1, c51);
            c60 = Avx512F.FusedMultiplyAdd(a6, b0, c60); c61 = Avx512F.FusedMultiplyAdd(a6, b1, c61);
            c70 = Avx512F.FusedMultiplyAdd(a7, b0, c70); c71 = Avx512F.FusedMultiplyAdd(a7, b1, c71);
        }

        Vector512.Store(c00, cPtr + 0 * ldc + 0); Vector512.Store(c01, cPtr + 0 * ldc + 8);
        Vector512.Store(c10, cPtr + 1 * ldc + 0); Vector512.Store(c11, cPtr + 1 * ldc + 8);
        Vector512.Store(c20, cPtr + 2 * ldc + 0); Vector512.Store(c21, cPtr + 2 * ldc + 8);
        Vector512.Store(c30, cPtr + 3 * ldc + 0); Vector512.Store(c31, cPtr + 3 * ldc + 8);
        Vector512.Store(c40, cPtr + 4 * ldc + 0); Vector512.Store(c41, cPtr + 4 * ldc + 8);
        Vector512.Store(c50, cPtr + 5 * ldc + 0); Vector512.Store(c51, cPtr + 5 * ldc + 8);
        Vector512.Store(c60, cPtr + 6 * ldc + 0); Vector512.Store(c61, cPtr + 6 * ldc + 8);
        Vector512.Store(c70, cPtr + 7 * ldc + 0); Vector512.Store(c71, cPtr + 7 * ldc + 8);
    }

    /// <summary>
    /// BLIS-lite tiled driver — parallel over M-tiles; per M-tile we pack
    /// the 8 A rows into a contiguous <c>double[8 * k]</c> panel so the
    /// microkernel's inner K loop reads streamed memory instead of chasing
    /// lda-strided rows. Same approach as the FP32 Avx512Sgemm driver.
    /// </summary>
    private static unsafe void RunTiledMnAligned(
        ReadOnlySpan<double> a, int lda,
        ReadOnlySpan<double> b, int ldb,
        Span<double> c, int m, int k, int n, bool allowParallel)
    {
        int mTiles = m / 8;
        int nTiles = n / 16;
        fixed (double* aBase = a)
        fixed (double* bBase = b)
        fixed (double* cBase = c)
        {
            var aPtr = aBase;
            var bPtr = bBase;
            var cPtr = cBase;
            int kLocal = k, ldaLocal = lda, ldbLocal = ldb, ldcLocal = n;
            int nTilesLocal = nTiles;

            void RunMTile(int mt)
            {
                var packed = new double[8 * kLocal];
                fixed (double* pPtr = packed)
                {
                    PackARowMajor8(aPtr + mt * 8 * ldaLocal, ldaLocal, pPtr, kLocal);
                    for (int nt = 0; nt < nTilesLocal; nt++)
                    {
                        Run8x16Tile(
                            pPtr, kLocal,
                            bPtr + nt * 16, ldbLocal,
                            cPtr + mt * 8 * ldcLocal + nt * 16, ldcLocal,
                            kLocal);
                    }
                }
            }

            if (allowParallel && mTiles >= 2)
            {
                AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, mTiles, (long)m * n * k, RunMTile);
            }
            else
            {
                for (int mt = 0; mt < mTiles; mt++) RunMTile(mt);
            }
        }
    }

    /// <summary>
    /// Pack 8 A rows of length k from row-major [lda]-strided source into
    /// a contiguous [8 * k] buffer such that the microkernel reads
    /// <c>panel[row * k + kk]</c> = original A[row, kk]. Stage 7 first-cut:
    /// scalar pack; OK because PackA cost amortises over 16 N-tiles.
    /// </summary>
    private static unsafe void PackARowMajor8(double* aPtr, int lda, double* panel, int k)
    {
        for (int r = 0; r < 8; r++)
        {
            double* srcRow = aPtr + r * lda;
            double* dstRow = panel + r * k;
            for (int kk = 0; kk < k; kk++) dstRow[kk] = srcRow[kk];
        }
    }
#endif
}
