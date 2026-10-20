// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// C = A*Bt with B stored [n, k] (SgemmDirectParallelMIntoTransB, a dense layer's input gradient dY*Wt): the column
/// groups of three, two and one compute each element with the same arithmetic, so a column equals itself computed
/// alone bit for bit; an odd n writes its last column; and the values match a double reference.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class SimdGemmTransBTests
{
    private static float[] Rand(int n, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    [Theory]
    [InlineData(128, 512, 784)]   // the parity MLP's first-layer input gradient
    [InlineData(64, 37, 9)]       // k tail of 5; three groups of three
    [InlineData(66, 20, 7)]       // two groups of three and a single column (odd n); two scalar rows
    [InlineData(12, 8, 5)]        // a group of three and one of two
    [InlineData(9, 3, 1)]         // k below one vector: all scalar tail; one column
    public void ColumnsMatchThemselvesAlone_AndADoubleReference(int m, int k, int n)
    {
        var a = Rand(m * k, m + 3 * k);
        var b = Rand(n * k, n + 5 * k);   // stored [n, k]
        var c = new float[m * n];
        Array.Fill(c, float.NaN);
        SimdGemm.SgemmDirectParallelMIntoTransB(a, b, c, m, k, n);

        for (int j = 0; j < n; j++)
        {
            var alone = new float[m];
            SimdGemm.SgemmDirectParallelMIntoTransB(a, b.AsSpan(j * k, k), alone, m, k, 1);
            for (int i = 0; i < m; i++)
                Assert.True(BitConverter.SingleToInt32Bits(alone[i]) == BitConverter.SingleToInt32Bits(c[i * n + j]),
                    $"C[{i},{j}] = {c[i * n + j]:G9} in a group of {n}, {alone[i]:G9} alone");
        }

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double e = 0;
                for (int p = 0; p < k; p++) e += (double)a[i * k + p] * b[j * k + p];
                Assert.True(Math.Abs(e - c[i * n + j]) <= 1e-5 * k + 1e-6, $"C[{i},{j}] = {c[i * n + j]:G9}, expected {e:G9}");
            }
    }
}
#endif
