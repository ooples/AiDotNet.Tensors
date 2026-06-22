using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Scalar reference microkernel for an ARBITRARY mr×nr output tile, strided-B.
///
/// <para>
/// The fixed-size scalar kernels (<see cref="ScalarFp32_4x4"/>, <see cref="ScalarFp64_4x4"/>)
/// only compute a 4×4 tile. <see cref="PackAOnlyStrategy"/> packs A in stripes of the caller's
/// active <c>mr</c> and dispatches <c>mr×nr</c> tiles, so on a host WITHOUT the matching SIMD
/// microkernel (e.g. net471 / no-AVX2, where <c>Avx2Fp32_6x16</c>/<c>Avx2Fp32_8x8</c>/etc. all
/// report unsupported) the dispatch would otherwise fall back to the 4×4 kernel and read packed-A
/// at the wrong stride + compute only a 4×4 corner of the tile — producing wrong results. This
/// kernel handles any <c>mr×nr</c> so the scalar fallback is correct for every tile width.
/// </para>
///
/// <para>
/// Packed-A layout matches the SIMD kernels: <c>[Kc × mr]</c> row-major (mr-contiguous within
/// each k-slice), so A row <c>r</c> at K-step <c>k</c> is <c>packedA[k * mr + r]</c>. B is read
/// directly with stride <c>ldb</c> (transB=false, row-major [K, N]); C is
/// read-modify-write at stride <c>ldc</c>. Accumulation is in FP64 to match the
/// fixed-size scalar kernels' precision strategy (cast back to FP32 only at write-back).
/// </para>
/// </summary>
internal static class ScalarGenericStridedB
{
    /// <summary>FP32 variant: accumulate packedA·B into the C[0..mr, 0..nr] tile over kc K-steps.</summary>
    public static void RunFloat(
        ReadOnlySpan<float> packedA, int mr,
        ReadOnlySpan<float> b, int ldb,
        Span<float> c, int ldc, int nr, int kc)
    {
        // FP64 accumulators for the mr×nr tile (mr,nr are small — 8×16 max in practice).
        Span<double> acc = stackalloc double[mr * nr];
        for (int r = 0; r < mr; r++)
            for (int col = 0; col < nr; col++)
                acc[r * nr + col] = c[r * ldc + col];

        for (int k = 0; k < kc; k++)
        {
            int aBase = k * mr;
            int bBase = k * ldb;
            for (int r = 0; r < mr; r++)
            {
                double ar = packedA[aBase + r];
                int accBase = r * nr;
                for (int col = 0; col < nr; col++)
                    acc[accBase + col] += ar * b[bBase + col];
            }
        }

        for (int r = 0; r < mr; r++)
            for (int col = 0; col < nr; col++)
                c[r * ldc + col] = (float)acc[r * nr + col];
    }

    /// <summary>FP64 variant: accumulate packedA·B into the C[0..mr, 0..nr] tile over kc K-steps.</summary>
    public static void RunDouble(
        ReadOnlySpan<double> packedA, int mr,
        ReadOnlySpan<double> b, int ldb,
        Span<double> c, int ldc, int nr, int kc)
    {
        Span<double> acc = stackalloc double[mr * nr];
        for (int r = 0; r < mr; r++)
            for (int col = 0; col < nr; col++)
                acc[r * nr + col] = c[r * ldc + col];

        for (int k = 0; k < kc; k++)
        {
            int aBase = k * mr;
            int bBase = k * ldb;
            for (int r = 0; r < mr; r++)
            {
                double ar = packedA[aBase + r];
                int accBase = r * nr;
                for (int col = 0; col < nr; col++)
                    acc[accBase + col] += ar * b[bBase + col];
            }
        }

        for (int r = 0; r < mr; r++)
            for (int col = 0; col < nr; col++)
                c[r * ldc + col] = acc[r * nr + col];
    }
}
