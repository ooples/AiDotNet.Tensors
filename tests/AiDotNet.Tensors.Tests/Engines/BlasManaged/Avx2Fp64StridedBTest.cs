using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-D follow-up: <see cref="Avx2Fp64_4x8.RunStridedB"/> correctness.
///
/// <para>
/// Verifies that the new FP64 AVX2 strided-B kernel produces results matching
/// the scalar reference within tight FP64 tolerance. Exercises the kernel via
/// <see cref="BlasManagedLib.Gemm{T}"/> with PackingMode.ForcePackAOnly on a
/// shape where m%4==0 AND n%8==0 (the wire-up gate).
/// </para>
/// </summary>
public class Avx2Fp64StridedBTest
{
    private static void ReferenceGemmFp64(
        ReadOnlySpan<double> a, int lda,
        ReadOnlySpan<double> b, int ldb,
        Span<double> c, int ldc,
        int m, int n, int k)
    {
        c.Clear();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++) sum += a[i * lda + p] * b[p * ldb + j];
                c[i * ldc + j] = sum;
            }
    }

    [Theory]
    [InlineData(32, 64, 32)]   // small aligned: m%4=0, n%8=0
    [InlineData(64, 64, 64)]   // medium aligned
    [InlineData(128, 128, 64)] // bigger aligned
    [InlineData(12, 16, 8)]    // m=12%4=0, n=16%8=0 (smallest practical)
    public void PackAOnly_FP64_AVX2_StridedB_Matches_Reference(int M, int N, int K)
    {
        var rng = new Random(42);
        var a = new double[M * K];
        var b = new double[K * N];
        var cBlas = new double[M * N];
        var cRef = new double[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        // ForcePackAOnly so the new Avx2Fp64_4x8.RunStridedB path is reached.
        BlasManagedLib.Gemm<double>(
            a, K, false, b, N, false, cBlas, N, M, N, K,
            new BlasOptions<double> { PackingMode = PackingMode.ForcePackAOnly });

        ReferenceGemmFp64(a, K, b, N, cRef, N, M, N, K);

        // FP64 tolerance: K * eps * worst-magnitude. With K up to 64, eps=2.22e-16,
        // worst sum magnitude ~K ≈ 64. Allow 1e-12 absolute.
        double maxAbsDelta = 0;
        for (int i = 0; i < cBlas.Length; i++)
            maxAbsDelta = Math.Max(maxAbsDelta, Math.Abs(cBlas[i] - cRef[i]));
        Assert.True(maxAbsDelta < 1e-12,
            $"M={M} N={N} K={K}: AVX2 FP64 strided-B drift {maxAbsDelta:G6} > 1e-12");
    }
}
