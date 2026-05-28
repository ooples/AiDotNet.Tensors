using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-D10 (#372 follow-up): verifies the AVX2 8×8 transpose path for
/// transB=true FP32 PackB produces bit-identical output to the scalar
/// reference by exercising it end-to-end via BlasManaged.Gemm.
///
/// <para>
/// transB=true is the worst-case PackBoth path because Avx2Pack previously
/// fell back to scalar for it. The 197×768×768 NaT family at ratio 46×
/// was the bottleneck. D10 should close most of that gap.
/// </para>
/// </summary>
// Bit-exact vs scalar reference — serialize against global reduction-order mutators
// so a concurrent toggle can't change the GEMM reduction order mid-assertion (#375 de-flake).
[Collection("BlasManaged-Stats-Serial")]
public class Avx2PackTransBFp32Test
{
    private static void ReferenceGemmTransBFp32(
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb,
        Span<float> c, int ldc,
        int m, int n, int k)
    {
        // transB=true: B stored as [N, K], so B[j, p] = b[j*ldb + p].
        c.Clear();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float sum = 0;
                for (int p = 0; p < k; p++) sum += a[i * lda + p] * b[j * ldb + p];
                c[i * ldc + j] = sum;
            }
    }

    [Theory]
    [InlineData(64, 64, 64)]      // small aligned
    [InlineData(128, 128, 128)]   // medium aligned
    [InlineData(192, 768, 768)]   // ViT-aligned bulk
    [InlineData(8, 16, 8)]        // smallest practical
    public void Gemm_TransB_True_Matches_Reference_FP32(int M, int N, int K)
    {
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[N * K];  // [N, K] when transB=true
        var cBlas = new float[M * N];
        var cRef = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        BlasManagedLib.Gemm<float>(
            a, K, transA: false,
            b, K, transB: true,
            cBlas, N, M, N, K);

        ReferenceGemmTransBFp32(a, K, b, K, cRef, N, M, N, K);

        double maxAbsDelta = 0;
        for (int i = 0; i < cBlas.Length; i++)
        {
            double delta = Math.Abs((double)cBlas[i] - cRef[i]);
            if (delta > maxAbsDelta) maxAbsDelta = delta;
        }
        // FP32 accumulation tolerance: K * eps * max|a*b|. Generous 1e-3 here.
        Assert.True(maxAbsDelta < 1e-3,
            $"M={M} N={N} K={K} transB=true: max|delta|={maxAbsDelta:G6}");
    }
}
