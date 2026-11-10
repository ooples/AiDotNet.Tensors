using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// A·Bᵀ for a few rows of A against a B stored [n, k] runs as one dot product per output (SgemmSkinnyTransB). It used
/// to go through the packed BLAS or a per-call transpose of all of B: one row against a [51865, 384] table took 51 ms
/// against 4 ms with B pre-transposed. The kernel must equal the product through a materialized transpose for every
/// row count up to its limit and past it, odd widths included.
/// </summary>
public class SkinnyTransBGemmTests
{
    private static readonly CpuEngine Engine = new();

    [Theory]
    [InlineData(1, 8, 1)]
    [InlineData(1, 384, 1029)]
    [InlineData(5, 13, 300)]
    [InlineData(6, 64, 7)]
    [InlineData(16, 384, 1029)]
    [InlineData(17, 64, 300)]
    [InlineData(64, 384, 300)]
    public void MatMulTransposed_EqualsTheProductWithBTransposed(int m, int k, int n)
    {
        var random = new Random(m * 1000 + k * 10 + n);
        var a = Random(random, m, k);
        var b = Random(random, n, k);

        int skinnyBefore = SimdGemm.t_skinnyTransBRuns;
        var transposedProduct = Engine.TensorMatMulTransposed(a, b);
        int skinnyRuns = SimdGemm.t_skinnyTransBRuns - skinnyBefore;
        var reference = Engine.TensorMatMul(a, Engine.TensorTranspose(b));

        // The results alone would pass if dispatch stopped choosing the kernel; the probe says which route ran.
        if (SimdGemm.QualifiesForSkinnyTransB(false, true, m, k, n))
            Assert.True(skinnyRuns > 0, $"[{m}, {k}]·[{n}, {k}]ᵀ qualifies but did not take SgemmSkinnyTransB");
        else
            Assert.Equal(0, skinnyRuns);

        Assert.Equal(reference.Shape.ToArray(), transposedProduct.Shape.ToArray());
        for (int i = 0; i < reference.Length; i++)
            Assert.Equal(reference[i], transposedProduct[i], 3e-4f);
    }

    [Fact]
    public void KernelAccumulatesIntoC_ForStridedA()
    {
        // A read with a row stride wider than k, and C holding a prior value: the kernel adds, as SgemmAdd's contract says.
        const int m = 3, k = 10, n = 5, lda = 12;
        var random = new Random(4);
        var a = new float[m * lda];
        var b = new float[n * k];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(random.NextDouble() * 2 - 1);
        var c = new float[m * n];
        for (int i = 0; i < c.Length; i++) c[i] = 1f;

        SimdGemm.SgemmSkinnyTransB(a, lda, b, k, c, m, k, n, allowParallel: false);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double expected = 1;
                for (int p = 0; p < k; p++) expected += (double)a[i * lda + p] * b[j * k + p];
                Assert.True(Math.Abs(expected - c[i * n + j]) <= 1e-4 * (1 + Math.Abs(expected)),
                    $"C[{i}, {j}] = {c[i * n + j]:R}, reference {expected:R}");
            }
        }
    }

    [Fact]
    public void KernelAccumulatesIntoC_AcrossParallelChunks()
    {
        // n well above two minimum chunks, so B splits into several work items run in parallel; C holds prior values
        // that every chunk must add to, including the last, partial one.
        const int m = 16, k = 384, n = (SimdGemm.SkinnyTransBMinChunk * 2 * 16) + 5;
        var random = RandomHelper.CreateSeededRandom(9);
        var a = new float[m * k];
        var b = new float[n * k];
        var c = new float[m * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < c.Length; i++) c[i] = (float)(random.NextDouble() * 2 - 1);
        var prior = (float[])c.Clone();
        Assert.True((long)m * k * n >= SimdGemm.ParallelWorkThreshold, "the case must reach the parallel branch");

        int parallelBefore = SimdGemm.t_skinnyTransBParallelRuns;
        SimdGemm.SgemmSkinnyTransB(a, k, b, k, c, m, k, n, allowParallel: true);
        Assert.Equal(parallelBefore + 1, SimdGemm.t_skinnyTransBParallelRuns);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double expected = prior[i * n + j];
                for (int p = 0; p < k; p++) expected += (double)a[i * k + p] * b[j * k + p];
                Assert.True(Math.Abs(expected - c[i * n + j]) <= 1e-4 * (1 + Math.Abs(expected)),
                    $"C[{i}, {j}] = {c[i * n + j]:R}, reference {expected:R}");
            }
        }
    }

    [Theory]
    [InlineData(3, 10, 5, 9, 10)]   // lda shorter than a row
    [InlineData(3, 10, 5, 10, 10)]  // A one element short of its last row
    public void Kernel_RejectsOperandsThatDoNotFitTheirSpans(int m, int k, int n, int lda, int ldb)
    {
        // The kernel walks raw pointers: an operand larger than its span must throw, not read past it.
        var a = new float[(m * lda) - (lda == k ? 1 : 0)];
        var b = new float[n * ldb];
        var c = new float[m * n];
        Assert.Throws<ArgumentException>(() => SimdGemm.SgemmSkinnyTransB(a, lda, b, ldb, c, m, k, n, allowParallel: false));
    }

    private static Tensor<float> Random(Random random, int rows, int columns)
    {
        var tensor = new Tensor<float>(new[] { rows, columns });
        var values = new float[rows * columns];
        for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() * 2 - 1);
        tensor.CopyFromArray(values);
        return tensor;
    }
}
