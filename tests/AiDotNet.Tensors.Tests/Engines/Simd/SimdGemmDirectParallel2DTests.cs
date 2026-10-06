// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// The small-M direct GEMM partitioned over both output axes (SimdGemm.DirectParallel2D.cs): values against a
/// double-precision reference on full and ragged (masked-tail) shapes, bit-identity across every partition the
/// thread count can produce, and that SimdGemm.Sgemm routes training-batch shapes to it.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]   // varies the process-wide MaxDegreeOfParallelism
public class SimdGemmDirectParallel2DTests
{
    private static float[] Rand(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static double[] Reference(float[] a, float[] b, int m, int k, int n)
    {
        var c = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int p = 0; p < k; p++)
            {
                double av = a[i * k + p];
                for (int j = 0; j < n; j++) c[i * n + j] += av * b[p * n + j];
            }
        return c;
    }

    private static void AssertMatchesReference(double[] expected, float[] actual, int k, string what)
    {
        // Each output is a k-term float FMA chain of O(1) terms: allow a few ulps per term of accumulated error.
        double tol = 4e-7 * k + 1e-6;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= tol,
                $"{what}[{i}] = {actual[i]:G9}, expected {expected[i]:G9} (tol {tol:G3})");
    }

    private static int[] ThreadCounts => new[] { 1, 2, 3, 5, 16, Math.Max(2, Environment.ProcessorCount) };

    [Theory]
    [InlineData(64, 784, 512)]    // parity-MLP layer 1 forward (k > 512: previously the packed path)
    [InlineData(64, 512, 128)]    // parity-MLP layer 2 forward
    [InlineData(37, 300, 77)]     // ragged: last row block has 1 row, last column strip 13 columns (n % 8 != 0)
    [InlineData(192, 1024, 16)]   // the gate's corners: max m, max k, one column strip
    [InlineData(6, 256, 1000)]    // a single row block: all parallelism from columns
    public void MatchesReference_AndIsBitIdenticalAcrossPartitions(int m, int k, int n)
    {
        var a = Rand(m * k, m * 31 + k);
        var b = Rand(k * n, n * 17 + k);
        var expected = Reference(a, b, m, k, n);

        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            float[]? first = null;
            foreach (int threads in ThreadCounts)
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                foreach (bool allowParallel in new[] { true, false })
                {
                    var c = new float[m * n];
                    for (int i = 0; i < c.Length; i++) c[i] = float.NaN;   // every element must be written
                    Assert.True(SimdGemm.TrySgemmDirectParallel2D(a, k, b, n, c, m, k, n, allowParallel));
                    AssertMatchesReference(expected, c, k, $"threads={threads} parallel={allowParallel} C");
                    if (first is null) first = c;
                    else
                        for (int i = 0; i < c.Length; i++)
                            Assert.True(BitConverter.SingleToInt32Bits(first[i]) == BitConverter.SingleToInt32Bits(c[i]),
                                $"threads={threads} parallel={allowParallel}: C[{i}] = {c[i]:G9} differs from {first[i]:G9}");
                }
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }
    }

    [Fact]
    public void Sgemm_RoutesTrainingBatchShapeToTheDirect2DPath()
    {
        const int m = 64, k = 784, n = 512;
        var a = Rand(m * k, 1);
        var b = Rand(k * n, 2);
        var direct = new float[m * n];
        Assert.True(SimdGemm.TrySgemmDirectParallel2D(a, k, b, n, direct, m, k, n, allowParallel: true));

        bool before = SimdGemm.UseDirectParallel2D;
        try
        {
            SimdGemm.UseDirectParallel2D = true;
            var routed = new float[m * n];
            SimdGemm.SgemmAddInternal(a, k, false, b, n, false, routed, m, k, n, allowParallel: true, clearedOutput: true);
            for (int i = 0; i < routed.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(direct[i]) == BitConverter.SingleToInt32Bits(routed[i]),
                    $"routed C[{i}] = {routed[i]:G9}, direct path {direct[i]:G9}");
        }
        finally
        {
            SimdGemm.UseDirectParallel2D = before;
        }
    }

    [Fact]
    public void TooNarrowForTheKernel_ReturnsFalseAndLeavesOutputUntouched()
    {
        var a = Rand(5 * 64, 3);
        var b = Rand(64 * 32, 4);
        var c = new float[5 * 32];
        Array.Fill(c, 7f);
        Assert.False(SimdGemm.TrySgemmDirectParallel2D(a, 64, b, 32, c, 5, 64, 32, allowParallel: true));   // m < Mr
        Assert.All(c, v => Assert.Equal(7f, v));
    }
}
#endif
