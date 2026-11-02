// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET9_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

[Collection("BlasManaged-Stats-Serial")]
public class SimdGemmDirectParallel2DTransBTests
{
    [SkippableTheory]
    [InlineData(128, 512, 784, true)]   // MLP dX, 784 -> 512
    [InlineData(128, 256, 512, true)]   // MLP dX, 512 -> 256
    [InlineData(64, 128, 3136, true)]   // CNN classifier dX
    [InlineData(121, 513, 77, true)]    // ragged rows, k past a kc block and off the 8x8 tiles, partial strip
    [InlineData(121, 513, 77, false)]
    [InlineData(6, 7, 16, false)]       // one row block, k under one tile, one 16-column strip
    [InlineData(13, 1024, 33, true)]    // the deepest packed strip; one column past a full strip
    public void TransB_IsBitIdenticalToTransposeThenNn_AndMatchesReference(int m, int k, int n, bool parallel)
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Fma.IsSupported, "direct GEMM requires FMA");
        var random = RandomHelper.CreateSeededRandom(1573 + m + k + n);
        var a = new float[m * k];
        var b = new float[n * k];   // stored [n, k]
        for (int i = 0; i < a.Length; i++) a[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(random.NextDouble() * 2 - 1);
        var actual = new float[m * n];
        var nn = new float[m * n];
        Array.Fill(actual, float.NaN);

        bool? previousWidth = SimdGemm.s_forceVector512Direct;
        int previousThreads = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            SimdGemm.s_forceVector512Direct = true;
            CpuParallelSettings.MaxDegreeOfParallelism = 3;
            int runs = SimdGemm.t_directParallel2DTransBRuns;
            Assert.True(SimdGemm.TrySgemmDirectParallel2DTransB(a, b, actual, m, k, n, parallel));
            Assert.Equal(runs + 1, SimdGemm.t_directParallel2DTransBRuns);

            var bt = new float[k * n];
            SimdTranspose.Transpose(b, bt, n, k);
            Assert.True(SimdGemm.TrySgemmDirectParallel2D(a, k, bt, n, nn, m, k, n, parallel));
        }
        finally
        {
            SimdGemm.s_forceVector512Direct = previousWidth;
            CpuParallelSettings.MaxDegreeOfParallelism = previousThreads;
        }

        for (int i = 0; i < actual.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(nn[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"C[{i / n},{i % n}] = {actual[i]:R}, transpose + NN {nn[i]:R}");
        for (int i = 0; i < m; i += 5)
            for (int j = 0; j < n; j += 3)
            {
                double expected = 0;
                for (int p = 0; p < k; p++) expected += (double)a[i * k + p] * b[j * k + p];
                Assert.True(Math.Abs(actual[i * n + j] - expected) <= 1e-4 * (1 + Math.Abs(expected)),
                    $"C[{i},{j}] = {actual[i * n + j]:R}, reference {expected:R}");
            }
    }

    [Theory]
    [InlineData(5, 64, 32)]    // fewer rows than one block
    [InlineData(64, 1025, 64)] // deeper than a packed strip
    [InlineData(64, 64, 15)]   // narrower than one tile
    public void TransB_DeclinesShapesItDoesNotTile(int m, int k, int n)
    {
        bool? previousWidth = SimdGemm.s_forceVector512Direct;
        try
        {
            SimdGemm.s_forceVector512Direct = true;
            var c = new float[m * n];
            Assert.False(SimdGemm.TrySgemmDirectParallel2DTransB(new float[m * k], new float[n * k], c, m, k, n, true));
            SimdGemm.s_forceVector512Direct = false;
            Assert.False(SimdGemm.TrySgemmDirectParallel2DTransB(new float[64 * 64], new float[64 * 64], new float[64 * 64], 64, 64, 64, true));
        }
        finally
        {
            SimdGemm.s_forceVector512Direct = previousWidth;
        }
    }
}
#endif
