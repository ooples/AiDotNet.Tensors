// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// SimdGemm.SgemmDirectParallelMIntoTransA (C = Aᵀ·B, A stored [k,m]) -- the dense-layer weight-gradient GEMM.
/// Pins the exact bits: rows below the last multiple of 4 are one FMA chain per element over p = 0..k-1 (what
/// the original 4x8 kernel computed and the 6x16 kernel must reproduce), the remaining rows the scalar
/// multiply-then-add loop. Shapes cover every row-block remainder (m mod 6, m mod 4) and column tail (n mod 16,
/// n mod 8), at several thread counts.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]   // varies the process-wide MaxDegreeOfParallelism
public class SimdGemmTransAKernelTests
{
    private static float[] Rand(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    /// <summary>The exact per-element arithmetic the kernel promises.</summary>
    private static float[] Expected(float[] a, float[] b, int m, int k, int n)
    {
        int mFull = (m / 4) * 4;
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int p = 0; p < k; p++)
                {
                    if (i < mFull) acc = MathF.FusedMultiplyAdd(a[p * m + i], b[p * n + j], acc);
                    else acc += a[p * m + i] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        return c;
    }

    [Theory]
    [InlineData(784, 64, 512)]   // parity-MLP layer-1 dW
    [InlineData(512, 64, 128)]   // parity-MLP layer-2 dW
    [InlineData(128, 64, 10)]    // layer-3 dW: n < 16, whole width is a masked tail
    [InlineData(4, 3, 13)]       // the shape that once corrupted the heap (n % 8 != 0)
    [InlineData(10, 7, 31)]      // mFull = 8: one full 6-row block, one 2-row block; 15-column tail
    [InlineData(23, 5, 24)]      // mFull = 20 (6+6+6+2), 3 scalar rows; 8-column tail (lane 1 empty)
    [InlineData(66, 33, 17)]     // mFull = 64 (10 full blocks + 4 rows), 2 scalar rows; 1-column tail
    [InlineData(3, 9, 40)]       // mFull = 0: all scalar rows
    public void MatchesTheFmaChainBitForBit(int m, int k, int n)
    {
        var a = Rand(k * m, m * 7 + k);
        var b = Rand(k * n, n * 11 + k);
        var expected = Expected(a, b, m, k, n);

        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            foreach (int threads in new[] { 1, 3, 16 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                var c = new float[m * n];
                for (int i = 0; i < c.Length; i++) c[i] = float.NaN;   // every element must be written
                SimdGemm.SgemmDirectParallelMIntoTransA(a, b, c, m, k, n);
                for (int i = 0; i < c.Length; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(c[i]),
                        $"threads={threads}: C[{i / n},{i % n}] = {c[i]:G9}, expected {expected[i]:G9}");
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }
    }

    [Fact]
    public void ColumnTailWritesStayInsideTheRow()
    {
        // C is embedded in a larger buffer: the kernel must not touch anything past m*n.
        const int m = 12, k = 4, n = 13;
        var a = Rand(k * m, 1);
        var b = Rand(k * n, 2);
        var c = new float[m * n + 32];
        for (int i = m * n; i < c.Length; i++) c[i] = 42f;
        SimdGemm.SgemmDirectParallelMIntoTransA(a, b, c.AsSpan(0, m * n), m, k, n);
        for (int i = m * n; i < c.Length; i++) Assert.Equal(42f, c[i]);
    }
}
#endif