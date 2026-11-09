using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
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

        var transposedProduct = Engine.TensorMatMulTransposed(a, b);
        var reference = Engine.TensorMatMul(a, Engine.TensorTranspose(b));

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
                for (int p = 0; p < k; p++) expected += a[i * lda + p] * b[j * k + p];
                Assert.Equal(expected, c[i * n + j], 4);
            }
        }
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
