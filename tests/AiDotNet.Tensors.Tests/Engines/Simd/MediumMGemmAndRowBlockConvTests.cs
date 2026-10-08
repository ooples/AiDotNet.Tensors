#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// The medium-M / wide-N GEMM path (<c>SgemmDirectParallelN</c>) and the row-block parallel conv built
/// on it. Shapes are chosen to satisfy <see cref="SimdGemm.PrefersParallelN"/> so the new paths actually
/// run (asserted), and are checked against a naive reference (GEMM) and the full im2col path (conv).
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class MediumMGemmAndRowBlockConvTests
{
    [SkippableTheory]
    [InlineData(9, 512, 2048)]     // smallest M on the path
    [InlineData(32, 144, 4096)]    // 16-channel 64x64 3x3 conv GEMM
    [InlineData(63, 300, 2048)]    // largest medium M, K not a multiple of 8
    [InlineData(64, 144, 4096)]    // n >= 16 m
    [InlineData(40, 112, 2000)]    // N not a multiple of the panel width
    [InlineData(17, 241, 2112)]    // ragged M and K
    [InlineData(128, 1024, 1024)]  // B panel packed (k >= 384, >= 24 rows): the 4 KB-stride shape that aliased
    [InlineData(96, 512, 1000)]    // packed, last panel narrower than a 16-wide tile
    public void Sgemm_MediumM_MatchesReference_OverwriteAndAccumulate(int m, int k, int n)
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported, "AVX2/FMA");
        int original = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(4, Environment.ProcessorCount);
            Assert.True(SimdGemm.PrefersParallelN(m, k, n), "shape must take the N-parallel path");
            var rnd = new Random(m * 7919 + k * 31 + n);
            var a = Rand(rnd, m * k); var b = Rand(rnd, k * n);
            var expected = Reference(a, b, m, k, n);

            var c = new float[m * n];
            for (int i = 0; i < c.Length; i++) c[i] = float.NaN;   // overwrite must not read C
            SimdGemm.Sgemm(a, k, false, b, n, false, c, m, k, n);
            AssertClose(expected, c, k);

            var acc = new float[m * n];
            for (int i = 0; i < acc.Length; i++) acc[i] = 1f;
            SimdGemm.SgemmAdd(a, k, false, b, n, false, acc, m, k, n);
            for (int i = 0; i < expected.Length; i++) expected[i] += 1f;
            AssertClose(expected, acc, k);
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = original;
        }
    }

    [SkippableTheory]
    [InlineData(1, 16, 64, 64, 32, 1, 1, 1)]   // benchmark shape
    [InlineData(1, 3, 112, 112, 32, 1, 1, 1)]  // stem, K = 27
    [InlineData(8, 32, 56, 56, 32, 1, 1, 1)]   // batched
    [InlineData(2, 16, 61, 47, 24, 1, 1, 1)]   // odd height / width
    [InlineData(1, 16, 128, 128, 32, 2, 1, 1)] // stride 2
    [InlineData(1, 16, 64, 64, 32, 1, 0, 1)]   // no padding
    [InlineData(1, 16, 64, 64, 32, 1, 2, 2)]   // dilation 2
    public void RowBlockConv_MatchesFullIm2Col(int batch, int cin, int h, int w, int cout, int stride, int pad, int dil)
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported, "AVX2/FMA");
        int original = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(4, Environment.ProcessorCount);
            int oh = (h + 2 * pad - dil * 2 - 1) / stride + 1, ow = (w + 2 * pad - dil * 2 - 1) / stride + 1;
            Assert.True(SimdGemm.PrefersParallelN(cout, cin * 9, oh * ow, requireAlignedN: false), "shape must take the row-block conv route");
            var rnd = new Random(batch * 101 + cin * 7 + h);
            var x = new Tensor<float>(Rand(rnd, batch * cin * h * w), new[] { batch, cin, h, w });
            var k = new Tensor<float>(Rand(rnd, cout * cin * 9), new[] { cout, cin, 3, 3 });
            var engine = new CpuEngine();

            int runsBefore = CpuEngine.RowBlockConvRunsOnThisThread;
            float[] rowBlock = engine.Conv2D(x, k, stride, pad, dil).ToArray();
            Assert.Equal(runsBefore + 1, CpuEngine.RowBlockConvRunsOnThisThread);
            float[] full;
            using (CpuEngine.ForceFullIm2ColScope())
                full = engine.Conv2D(x, k, stride, pad, dil).ToArray();
            Assert.Equal(runsBefore + 1, CpuEngine.RowBlockConvRunsOnThisThread);

            Assert.Equal(full.Length, rowBlock.Length);
            // Same products, same k order per element (both run the direct kernel over a panel of the
            // same column matrix), so the results agree to rounding of the panel split.
            for (int i = 0; i < full.Length; i++)
                Assert.True(Math.Abs(full[i] - rowBlock[i]) <= 1e-4f * (1 + Math.Abs(full[i])),
                    $"[{i}] full {full[i]} row-block {rowBlock[i]}");
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = original;
        }
    }

    [SkippableTheory]
    [InlineData(512, 64, 512)]     // attention Q·Kᵀ
    [InlineData(256, 256, 256)]
    [InlineData(1024, 64, 192)]
    [InlineData(300, 100, 520)]    // ragged M, K; N tail
    public void Sgemm_TransposedB_ColumnPanelRoute_MatchesReference(int m, int k, int n)
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported, "AVX2/FMA");
        int original = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(4, Environment.ProcessorCount);
            Assert.True(SimdGemm.PrefersParallelN(m, k, n), "shape must take the column-panel route");
            var rnd = new Random(m + 13 * k + 7 * n);
            var a = Rand(rnd, m * k);
            var bStoredNk = Rand(rnd, n * k);   // B stored [n x k]; C = A·Bᵀ
            var bKn = new float[k * n];
            for (int r = 0; r < n; r++) for (int col = 0; col < k; col++) bKn[col * n + r] = bStoredNk[r * k + col];
            var expected = Reference(a, bKn, m, k, n);

            var c = new float[m * n];
            for (int i = 0; i < c.Length; i++) c[i] = float.NaN;
            SimdGemm.Sgemm(a, k, false, bStoredNk, k, true, c, m, k, n);
            AssertClose(expected, c, k);

            // Engine entry points that route here.
            var engine = new CpuEngine();
            var tA = new Tensor<float>(a, new[] { m, k });
            AssertClose(expected, engine.TensorMatMulTransposed(tA, new Tensor<float>(bStoredNk, new[] { n, k })).ToArray(), k);
            AssertClose(expected, engine.TensorMatMul(tA, new Tensor<float>(bKn, new[] { k, n })).ToArray(), k);
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = original;
        }
    }

    private static float[] Rand(Random rnd, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rnd.NextDouble() * 2 - 1);
        return a;
    }

    private static float[] Reference(float[] a, float[] b, int m, int k, int n)
    {
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int p = 0; p < k; p++) s += (double)a[i * k + p] * b[p * n + j];
                c[i * n + j] = (float)s;
            }
        return c;
    }

    private static void AssertClose(float[] expected, float[] actual, int k)
    {
        float tol = 1e-5f * k;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol * (1 + Math.Abs(expected[i])),
                $"[{i}] expected {expected[i]} got {actual[i]}");
    }
}
#endif
