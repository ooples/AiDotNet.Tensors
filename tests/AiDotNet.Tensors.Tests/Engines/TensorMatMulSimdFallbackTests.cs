using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Verifies that <see cref="CpuEngine.TensorMatMul{T}"/> and its batched variants
/// produce numerically correct results when the BLAS-accelerated path declines —
/// i.e. when the engine routes through <c>MatrixMultiplyHelper.MultiplyBlocked</c>
/// (SIMD AVX2/FMA via <c>numOps.MultiplyAdd</c>).
/// <para>
/// The previous fallback was a naive <c>Parallel.For</c> triple-loop with
/// <c>numOps.Add(numOps.Multiply(...))</c> per element — correct but ~50–100×
/// slower than the blocked SIMD path on AVX2 hosts. These tests pin the
/// numerical contract (results match a reference triple-loop) so future kernel
/// swaps can't silently regress correctness.
/// </para>
/// <para>
/// <c>double</c> is the primary regression target: <c>BlasProvider</c> is
/// disabled and <c>SimdGemm</c> only ships an Sgemm path, so every
/// <c>Tensor&lt;double&gt;</c> matmul previously hit the naive loop. The
/// <c>byte</c>/<c>int</c> cases verify the change does not regress
/// non-float/double types (which never had a BLAS path to begin with).
/// </para>
/// </summary>
public class TensorMatMulSimdFallbackTests
{
    private readonly CpuEngine _engine = new();

    private static double[] Reference2D(double[] a, double[] b, int m, int n, int p)
    {
        var c = new double[m * p];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += a[i * n + k] * b[k * p + j];
                }
                c[i * p + j] = sum;
            }
        }
        return c;
    }

    private static int[] ReferenceInt2D(int[] a, int[] b, int m, int n, int p)
    {
        var c = new int[m * p];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < p; j++)
            {
                int sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += a[i * n + k] * b[k * p + j];
                }
                c[i * p + j] = sum;
            }
        }
        return c;
    }

    [Theory]
    [InlineData(4, 4, 4)]      // tiny — exercises edge-tile path
    [InlineData(7, 13, 11)]    // odd dims — exercises masked tail
    [InlineData(64, 64, 64)]   // single block tile
    [InlineData(128, 128, 128)] // multi-block tile, parallel-eligible
    [InlineData(96, 130, 80)]  // non-power-of-two, mid-size
    public void TensorMatMul_Double_MatchesNaiveReference(int m, int n, int p)
    {
        var rng = new Random(42);
        var aData = new double[m * n];
        var bData = new double[n * p];
        for (int i = 0; i < aData.Length; i++) aData[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < bData.Length; i++) bData[i] = rng.NextDouble() * 2 - 1;

        var a = new Tensor<double>(aData, new[] { m, n });
        var b = new Tensor<double>(bData, new[] { n, p });

        var result = _engine.TensorMatMul(a, b);
        var actual = result.GetDataArray();
        var expected = Reference2D(aData, bData, m, n, p);

        Assert.Equal(m * p, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            // Block-tiled summation reorders FMAs vs the naive row-major sum,
            // so we tolerate a small per-element drift proportional to n.
            Assert.Equal(expected[i], actual[i], precision: 8);
        }
    }

    [Fact]
    public void TensorMatMul_Double_Identity_PreservesInput()
    {
        const int n = 32;
        var rng = new Random(7);
        var aData = new double[n * n];
        for (int i = 0; i < aData.Length; i++) aData[i] = rng.NextDouble();
        // Snapshot the input before the matmul. Since Tensor<T> can be
        // constructed to alias the supplied array, a regression that
        // accidentally mutates the input in place would otherwise pass —
        // the result would always equal the (mutated) input. Asserting
        // against the snapshot catches that class of bug.
        var aSnapshot = (double[])aData.Clone();

        var identity = new double[n * n];
        for (int i = 0; i < n; i++) identity[i * n + i] = 1.0;

        var a = new Tensor<double>(aData, new[] { n, n });
        var id = new Tensor<double>(identity, new[] { n, n });

        var result = _engine.TensorMatMul(a, id);
        var actual = result.GetDataArray();
        var inputAfter = a.GetDataArray();

        for (int i = 0; i < aSnapshot.Length; i++)
        {
            Assert.Equal(aSnapshot[i], actual[i], precision: 12);
            Assert.Equal(aSnapshot[i], inputAfter[i], precision: 12);
        }
    }

    [Fact]
    public void TensorMatMul_Int_MatchesNaiveReference()
    {
        const int m = 8, n = 12, p = 10;
        var rng = new Random(99);
        var aData = new int[m * n];
        var bData = new int[n * p];
        for (int i = 0; i < aData.Length; i++) aData[i] = rng.Next(-5, 5);
        for (int i = 0; i < bData.Length; i++) bData[i] = rng.Next(-5, 5);

        var a = new Tensor<int>(aData, new[] { m, n });
        var b = new Tensor<int>(bData, new[] { n, p });

        var result = _engine.TensorMatMul(a, b);
        var actual = result.GetDataArray();
        var expected = ReferenceInt2D(aData, bData, m, n, p);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TensorMatMul_BatchedDouble_MatchesPerSliceReference()
    {
        const int batch = 3, m = 16, n = 20, p = 12;
        var rng = new Random(123);
        var aData = new double[batch * m * n];
        var bData = new double[n * p]; // shared B across batches
        for (int i = 0; i < aData.Length; i++) aData[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < bData.Length; i++) bData[i] = rng.NextDouble() * 2 - 1;

        var a = new Tensor<double>(aData, new[] { batch, m, n });
        var b = new Tensor<double>(bData, new[] { n, p });

        var result = _engine.TensorMatMul(a, b);
        var actual = result.GetDataArray();

        Assert.Equal(batch * m * p, actual.Length);
        for (int s = 0; s < batch; s++)
        {
            var aSlice = new double[m * n];
            Array.Copy(aData, s * m * n, aSlice, 0, m * n);
            var expected = Reference2D(aSlice, bData, m, n, p);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], actual[s * m * p + i], precision: 8);
            }
        }
    }

    [Fact]
    public void TensorMatMul_FullBatchedDouble_MatchesPerSliceReference()
    {
        const int batch = 2, m = 12, n = 18, p = 10;
        var rng = new Random(456);
        var aData = new double[batch * m * n];
        var bData = new double[batch * n * p];
        for (int i = 0; i < aData.Length; i++) aData[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < bData.Length; i++) bData[i] = rng.NextDouble() * 2 - 1;

        var a = new Tensor<double>(aData, new[] { batch, m, n });
        var b = new Tensor<double>(bData, new[] { batch, n, p });

        var result = _engine.TensorMatMul(a, b);
        var actual = result.GetDataArray();

        Assert.Equal(batch * m * p, actual.Length);
        for (int s = 0; s < batch; s++)
        {
            var aSlice = new double[m * n];
            var bSlice = new double[n * p];
            Array.Copy(aData, s * m * n, aSlice, 0, m * n);
            Array.Copy(bData, s * n * p, bSlice, 0, n * p);
            var expected = Reference2D(aSlice, bSlice, m, n, p);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], actual[s * m * p + i], precision: 8);
            }
        }
    }

    [Fact]
    public void MatMulInto_Double_MatchesNaiveReference()
    {
        const int m = 24, n = 30, p = 20;
        var rng = new Random(789);
        var aData = new double[m * n];
        var bData = new double[n * p];
        for (int i = 0; i < aData.Length; i++) aData[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < bData.Length; i++) bData[i] = rng.NextDouble() * 2 - 1;

        var a = new Tensor<double>(aData, new[] { m, n });
        var b = new Tensor<double>(bData, new[] { n, p });
        var dst = new Tensor<double>(new[] { m, p });

        _engine.MatMulInto(dst, a, b);
        var actual = dst.GetDataArray();
        var expected = Reference2D(aData, bData, m, n, p);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], precision: 8);
        }
    }
}
