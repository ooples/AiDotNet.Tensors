using System;
using System.Threading.Tasks;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// AVX-512 SGEMM microkernel (FP32). Target: Intel Skylake-X, Ice Lake,
/// Sapphire Rapids, Zen 4 / Zen 5. Uses 16-wide <see cref="Vector512"/> FMA
/// for ~2× throughput vs the Vector256 AVX2 path on zmm-capable CPUs.
///
/// <para>B1 scope: feature gate + entry point. The actual blocked kernel
/// (B2 / B3) plugs in via <see cref="SgemmBlocked"/> once the microkernel
/// and panel-packing routines are in. Until then the entry point falls
/// through to the AVX2 path so the feature gate is a no-op — no behaviour
/// change, ready for the kernel to land behind it.</para>
/// </summary>
internal static class Avx512Sgemm
{
    /// <summary>
    /// True if the AVX-512F path is callable: compiled on .NET 8+, runtime
    /// reports AVX-512F support, and the feature isn't gated off. Used as
    /// the single gate by every upstream dispatcher.
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
    /// Entry point for the AVX-512 blocked SGEMM. Routes to the 16×16
    /// microkernel (<see cref="Run16x16Tile"/>) when the shape qualifies,
    /// falls back to the AVX2 path for small / misaligned problems.
    /// </summary>
    public static void SgemmBlocked(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c,
        int m, int k, int n,
        bool allowParallel)
    {
#if NET8_0_OR_GREATER
        // Fast path: non-transposed, aligned dims, enough work per tile.
        // Transposed ops or small m/n drop to the AVX2 driver which already
        // handles them well.
        if (!transA && !transB
            && m >= 16 && n >= 16
            && m % 16 == 0 && n % 16 == 0
            && Avx512F.IsSupported)
        {
            RunTiledMnAligned(a, lda, b, ldb, c, m, k, n, allowParallel);
            return;
        }
#endif
        SimdGemm.SgemmAddInternal(a, lda, transA, b, ldb, transB, c, m, k, n,
            allowParallel: allowParallel, clearedOutput: true);
    }

#if NET8_0_OR_GREATER
    /// <summary>
    /// 16 (rows) × 16 (cols) microkernel. Accumulates in 16
    /// <c>Vector512&lt;float&gt;</c>s — one per row. Inner K loop does one
    /// 16-wide B load and 16 broadcast-FMAs per K step, saturating both
    /// FMA ports on Intel AVX-512 silicon.
    /// </summary>
    private static unsafe void Run16x16Tile(
        float* aPtr, int lda,
        float* bPtr, int ldb,
        float* cPtr, int ldc,
        int k)
    {
        // Accumulators start from zero to match the SgemmBlocked dispatch
        // contract (clearedOutput: true → C = A*B, not C += A*B). Reading
        // from C was a latent bug: while C IS zero when the caller honours
        // the contract, any stale content (incomplete clear, future call
        // path that passes clearedOutput: false) would silently produce
        // C_new = C_stale + A*B. Zero-start removes that coupling and saves
        // 16 unnecessary loads per tile.
        var acc00 = Vector512<float>.Zero;
        var acc01 = Vector512<float>.Zero;
        var acc02 = Vector512<float>.Zero;
        var acc03 = Vector512<float>.Zero;
        var acc04 = Vector512<float>.Zero;
        var acc05 = Vector512<float>.Zero;
        var acc06 = Vector512<float>.Zero;
        var acc07 = Vector512<float>.Zero;
        var acc08 = Vector512<float>.Zero;
        var acc09 = Vector512<float>.Zero;
        var acc10 = Vector512<float>.Zero;
        var acc11 = Vector512<float>.Zero;
        var acc12 = Vector512<float>.Zero;
        var acc13 = Vector512<float>.Zero;
        var acc14 = Vector512<float>.Zero;
        var acc15 = Vector512<float>.Zero;

        for (int kk = 0; kk < k; kk++)
        {
            var bVec = Vector512.Load(bPtr + kk * ldb);
            acc00 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[0  * lda + kk]), bVec, acc00);
            acc01 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[1  * lda + kk]), bVec, acc01);
            acc02 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[2  * lda + kk]), bVec, acc02);
            acc03 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[3  * lda + kk]), bVec, acc03);
            acc04 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[4  * lda + kk]), bVec, acc04);
            acc05 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[5  * lda + kk]), bVec, acc05);
            acc06 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[6  * lda + kk]), bVec, acc06);
            acc07 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[7  * lda + kk]), bVec, acc07);
            acc08 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[8  * lda + kk]), bVec, acc08);
            acc09 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[9  * lda + kk]), bVec, acc09);
            acc10 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[10 * lda + kk]), bVec, acc10);
            acc11 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[11 * lda + kk]), bVec, acc11);
            acc12 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[12 * lda + kk]), bVec, acc12);
            acc13 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[13 * lda + kk]), bVec, acc13);
            acc14 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[14 * lda + kk]), bVec, acc14);
            acc15 = Avx512F.FusedMultiplyAdd(Vector512.Create(aPtr[15 * lda + kk]), bVec, acc15);
        }

        Vector512.Store(acc00, cPtr + 0  * ldc);
        Vector512.Store(acc01, cPtr + 1  * ldc);
        Vector512.Store(acc02, cPtr + 2  * ldc);
        Vector512.Store(acc03, cPtr + 3  * ldc);
        Vector512.Store(acc04, cPtr + 4  * ldc);
        Vector512.Store(acc05, cPtr + 5  * ldc);
        Vector512.Store(acc06, cPtr + 6  * ldc);
        Vector512.Store(acc07, cPtr + 7  * ldc);
        Vector512.Store(acc08, cPtr + 8  * ldc);
        Vector512.Store(acc09, cPtr + 9  * ldc);
        Vector512.Store(acc10, cPtr + 10 * ldc);
        Vector512.Store(acc11, cPtr + 11 * ldc);
        Vector512.Store(acc12, cPtr + 12 * ldc);
        Vector512.Store(acc13, cPtr + 13 * ldc);
        Vector512.Store(acc14, cPtr + 14 * ldc);
        Vector512.Store(acc15, cPtr + 15 * ldc);
    }

    /// <summary>
    /// B3 BLIS-lite driver: parallel over M-tiles, per M-tile we pack
    /// the 16 A rows into a contiguous <c>float[16 * k]</c> panel so the
    /// microkernel's inner K loop reads streamed memory instead of chasing
    /// lda-strided rows. That's ~80% of the GotoBLAS cache benefit with
    /// 1/5 the code. Full 5-loop K-block + B-pack remains a future task
    /// when we actually measure GEMMs past 4096×4096.
    /// </summary>
    private static unsafe void RunTiledMnAligned(
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb,
        Span<float> c, int m, int k, int n, bool allowParallel)
    {
        int mTiles = m / 16;
        int nTiles = n / 16;
        fixed (float* aBase = a)
        fixed (float* bBase = b)
        fixed (float* cBase = c)
        {
            var aPtr = aBase;
            var bPtr = bBase;
            var cPtr = cBase;
            int kLocal = k, nLocal = n, ldaLocal = lda, ldbLocal = ldb;
            int nTilesLocal = nTiles;
            if (allowParallel && mTiles >= 2)
            {
                float* aOuter = aPtr; float* bOuter = bPtr; float* cOuter = cPtr;
                Parallel.For(0, mTiles, mt =>
                {
                    var packed = new float[16 * kLocal];
                    fixed (float* pPtr = packed)
                    {
                        PackARowMajor16(aOuter + mt * 16 * ldaLocal, ldaLocal, pPtr, kLocal);
                        for (int nt = 0; nt < nTilesLocal; nt++)
                        {
                            Run16x16Tile(
                                pPtr, kLocal,
                                bOuter + nt * 16, ldbLocal,
                                cOuter + mt * 16 * nLocal + nt * 16, nLocal,
                                kLocal);
                        }
                    }
                });
            }
            else
            {
                var packed = new float[16 * k];
                fixed (float* pPtr = packed)
                {
                    for (int mt = 0; mt < mTiles; mt++)
                    {
                        PackARowMajor16(aPtr + mt * 16 * lda, lda, pPtr, k);
                        for (int nt = 0; nt < nTiles; nt++)
                        {
                            Run16x16Tile(
                                pPtr, k,
                                bPtr + nt * 16, ldb,
                                cPtr + mt * 16 * n + nt * 16, n,
                                k);
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Copies 16 rows of A starting at <paramref name="src"/> (row stride
    /// <paramref name="lda"/>) into <paramref name="dst"/> as a contiguous
    /// <c>[16][k]</c> block. After packing, the microkernel reads each row
    /// at stride <c>k</c> with perfect prefetch behaviour.
    /// </summary>
    private static unsafe void PackARowMajor16(float* src, int lda, float* dst, int k)
    {
        for (int i = 0; i < 16; i++)
        {
            Buffer.MemoryCopy(src + i * lda, dst + i * k, sizeof(float) * k, sizeof(float) * k);
        }
    }
#endif
}
