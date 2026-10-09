#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// A transposed-operand GEMM at training-batch scale copies the transposed operand into row-major scratch and runs
/// the untransposed product. These pin the result to a double-precision reference, for both transposes, across shapes
/// that take the route (a dense layer's backward) and ones that stay on the packed path.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]   // toggles the process-wide UseTransposeMaterialization
public class SimdGemmTransposedMaterializationTests
{
    private static float[] Rand(int length, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }

    // C[i, j] = sum_p op(A)[i, p] * op(B)[p, j] in double, with A stored [k, m] when transA and B stored [n, k] when transB.
    private static double[] Reference(float[] a, bool transA, float[] b, bool transB, int m, int k, int n)
    {
        var c = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++)
                {
                    double av = transA ? a[p * m + i] : a[i * k + p];
                    double bv = transB ? b[j * k + p] : b[p * n + j];
                    sum += av * bv;
                }
                c[i * n + j] = sum;
            }
        return c;
    }

    public static TheoryData<int, int, int, bool, bool> Shapes => new()
    {
        // m, k, n, transA, transB. The first four are a [128, 784] -> 512 -> 256 dense stack's backward.
        { 128, 512, 784, false, true },   // dX = dY . W^T
        { 784, 128, 512, true, false },   // dW = X^T . dY
        { 128, 256, 512, false, true },
        { 512, 128, 256, true, false },
        { 96, 300, 200, true, true },     // both transposed, ragged edges
        { 7, 33, 19, false, true },       // below the parallel threshold: the packed path, unchanged
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void TransposedProduct_MatchesDoubleReference(int m, int k, int n, bool transA, bool transB)
    {
        var a = Rand(m * k, 1 + m);
        var b = Rand(k * n, 2 + n);
        var c = new float[m * n];
        int lda = transA ? m : k, ldb = transB ? k : n;
#pragma warning disable CS0618 // the transposed Sgemm overload is the entry point under test
        SimdGemm.Sgemm(a, lda, transA, b, ldb, transB, c, m, k, n);
#pragma warning restore CS0618

        var expected = Reference(a, transA, b, transB, m, k, n);
        double tolerance = 1e-5 * k;
        for (int i = 0; i < c.Length; i++)
            Assert.True(Math.Abs(c[i] - expected[i]) <= tolerance,
                $"[{m}x{k}x{n} tA={transA} tB={transB}] C[{i}] = {c[i]:G9}, reference {expected[i]:G9}");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void TransposedProduct_IsTheSameWithAndWithoutTheRoute_WithinRounding(int m, int k, int n, bool transA, bool transB)
    {
        var a = Rand(m * k, 3 + m);
        var b = Rand(k * n, 4 + n);
        int lda = transA ? m : k, ldb = transB ? k : n;
        var routed = new float[m * n];
        var packed = new float[m * n];
        bool before = SimdGemm.UseTransposeMaterialization;
        try
        {
#pragma warning disable CS0618
            SimdGemm.UseTransposeMaterialization = true;
            SimdGemm.Sgemm(a, lda, transA, b, ldb, transB, routed, m, k, n);
            SimdGemm.UseTransposeMaterialization = false;
            SimdGemm.Sgemm(a, lda, transA, b, ldb, transB, packed, m, k, n);
#pragma warning restore CS0618
        }
        finally
        {
            SimdGemm.UseTransposeMaterialization = before;
        }

        double tolerance = 1e-5 * k;
        for (int i = 0; i < routed.Length; i++)
            Assert.True(Math.Abs(routed[i] - packed[i]) <= tolerance,
                $"[{m}x{k}x{n} tA={transA} tB={transB}] C[{i}] routed {routed[i]:G9}, packed {packed[i]:G9}");
    }
}
#endif
