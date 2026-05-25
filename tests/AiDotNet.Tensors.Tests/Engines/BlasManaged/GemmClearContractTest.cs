using System;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Regression guard for the BlasManaged.Gemm C-clear contract: the initial
/// zero-fill must touch only the logical [m, n] output tile, never the whole
/// backing span. With an ldc-padded (or offset-based) caller the span extends
/// past the tile; the previous c.Clear() clobbered that caller-owned data.
/// </summary>
public class GemmClearContractTest
{
    [Theory]
    [InlineData(3, 4, 2, 8)]    // ldc=8 > n=4 → 4 padding cols per row
    [InlineData(5, 6, 3, 10)]
    public void Gemm_Fp32_ClearsOnlyLogicalTile_PreservesPadding(int m, int n, int k, int ldc)
    {
        const float sentinel = 999f;
        var rng = new Random(2026);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // C backing span is m*ldc; pre-fill everything with a sentinel so we can
        // detect any clobber of the padding columns [n, ldc).
        var c = new float[m * ldc];
        for (int i = 0; i < c.Length; i++) c[i] = sentinel;

        BlasManagedLib.Gemm<float>(a, k, false, b, n, false, c, ldc, m, n, k);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float expected = 0;
                for (int kk = 0; kk < k; kk++) expected += a[i * k + kk] * b[kk * n + j];
                Assert.True(MathF.Abs(c[i * ldc + j] - expected) < 1e-4f,
                    $"tile[{i},{j}] = {c[i * ldc + j]:G6}, expected {expected:G6}");
            }
            for (int j = n; j < ldc; j++)
                Assert.True(c[i * ldc + j] == sentinel,
                    $"padding col [{i},{j}] was clobbered: {c[i * ldc + j]} (expected sentinel {sentinel})");
        }
    }
}
