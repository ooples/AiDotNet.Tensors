using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-B5 (#370 follow-up): verifies PackBoth's 2D MN-grid parallel path
/// produces bit-correct output vs serial / reference, on shapes where
/// ShouldUse2DGrid returns true (m_blocks small, n_blocks > 1).
/// </summary>
[Collection("BlasManaged-Perf-Serial")]
public class PackBoth2DGridTest
{
    private static void ReferenceGemmFp32(
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb,
        Span<float> c, int ldc,
        int m, int n, int k)
    {
        c.Clear();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float sum = 0;
                for (int p = 0; p < k; p++) sum += a[i * lda + p] * b[p * ldb + j];
                c[i * ldc + j] = sum;
            }
    }

    [Theory]
    [InlineData(192, 768, 768)]   // ViT-bulk; m_blocks=2, n_blocks=2 with mc=nc=128
    [InlineData(64, 1024, 512)]   // shorter M, wider N — definite 2D candidate
    [InlineData(128, 256, 128)]   // m_blocks=1, n_blocks=2 — borderline (but valid)
    public void Gemm_PackBoth_2DGrid_Matches_Reference_FP32(int M, int N, int K)
    {
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var cBlas = new float[M * N];
        var cRef = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Force PackBoth (K large enough that the dispatcher picks ForcePackBoth).
        BlasManagedLib.Gemm<float>(
            a, K, false, b, N, false, cBlas, N, M, N, K,
            new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth, NumThreads = 8 });

        ReferenceGemmFp32(a, K, b, N, cRef, N, M, N, K);

        double maxAbsDelta = 0;
        for (int i = 0; i < cBlas.Length; i++)
        {
            double d = Math.Abs((double)cBlas[i] - cRef[i]);
            if (d > maxAbsDelta) maxAbsDelta = d;
        }
        Assert.True(maxAbsDelta < 1e-3,
            $"M={M} N={N} K={K}: 2D-grid output drift {maxAbsDelta:G6} > 1e-3");
    }

    [Fact]
    public void Gemm_PackBoth_2DGrid_BitExact_Vs_Serial_FP32()
    {
        const int M = 192, N = 768, K = 768;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var cSerial = new float[M * N];
        var c2D = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // NumThreads=1 forces serial path (the workspace check in PackBothStrategy).
        BlasManagedLib.Gemm<float>(
            a, K, false, b, N, false, cSerial, N, M, N, K,
            new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth, NumThreads = -1 });

        // NumThreads=8 with this shape (m_blocks=2 with mc=128) triggers 2D mode.
        BlasManagedLib.Gemm<float>(
            a, K, false, b, N, false, c2D, N, M, N, K,
            new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth, NumThreads = 8 });

        for (int i = 0; i < cSerial.Length; i++)
            Assert.True(
                cSerial[i] == c2D[i],
                $"2D-grid not bit-exact vs serial at [{i / N}, {i % N}]: " +
                $"serial={cSerial[i]:G9} 2d={c2D[i]:G9}");
    }
}
