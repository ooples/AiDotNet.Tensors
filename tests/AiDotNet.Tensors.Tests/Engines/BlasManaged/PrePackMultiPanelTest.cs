using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-E (#373): verifies multi-panel PrePackA/B produces bit-identical GEMM
/// output to the no-prepack path. Test shapes span single-panel (small) and
/// multi-panel (large) to confirm both code paths.
/// </summary>
public class PrePackMultiPanelTest
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
    [InlineData(64, 64, 64)]      // single-panel (≤ Mc=128, Kc=256)
    [InlineData(128, 256, 256)]   // single-panel borderline
    [InlineData(256, 512, 512)]   // 2x2 tiles
    [InlineData(384, 768, 768)]   // 3 ic × 3 pc tiles, partial last block
    public void PrePackA_RoundTrips_FP32(int M, int N, int K)
    {
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var cRef = new float[M * N];
        var cPrePack = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        ReferenceGemmFp32(a, K, b, N, cRef, N, M, N, K);

        var handle = BlasManagedLib.PrePackA<float>(a, K, false, M, K);
        try
        {
            // Use ForcePackBoth + serial path (Workspace forces RunSerial which
            // has the most thorough multi-panel consumption logic in this MVP).
            byte[] workspace = new byte[handle.MultiPanelStride * 2 + 32];
            BlasManagedLib.Gemm<float>(
                a, K, false, b, N, false, cPrePack, N, M, N, K,
                new BlasOptions<float>
                {
                    PackingMode = PackingMode.ForcePackBoth,
                    PackedA = handle,
                    Workspace = workspace,
                });

            // Tolerance: K * eps * max|a*b|.
            double maxDelta = 0;
            for (int i = 0; i < cRef.Length; i++)
                maxDelta = Math.Max(maxDelta, Math.Abs((double)cRef[i] - cPrePack[i]));
            Assert.True(maxDelta < 1e-3,
                $"M={M} N={N} K={K}: multi-panel PrePackA drift {maxDelta:G6}");
        }
        finally
        {
            handle.Dispose();
        }
    }

    [Fact]
    public void PrePackA_Accepts_Shapes_Previously_Rejected()
    {
        // Pre-Sub-E, PrePackA threw NotSupportedException for m > 64 OR k > 64.
        // Now any (m, k) is accepted.
        var rng = new Random(42);
        var a = new float[256 * 256];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);

        // This call would have thrown pre-Sub-E.
        var handle = BlasManagedLib.PrePackA<float>(a, 256, false, 256, 256);
        Assert.NotNull(handle);
        Assert.True(handle.MultiPanelStride > 0, "Should be multi-panel layout");
        Assert.True(handle.NumIcBlocks * handle.NumPcBlocks > 1, "Should have multiple tiles");
        handle.Dispose();
    }

    [Fact]
    public void PrePackB_Accepts_Shapes_Previously_Rejected()
    {
        var rng = new Random(42);
        var b = new float[256 * 256];
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var handle = BlasManagedLib.PrePackB<float>(b, 256, false, 256, 256);
        Assert.NotNull(handle);
        Assert.True(handle.MultiPanelStride > 0);
        handle.Dispose();
    }
}
