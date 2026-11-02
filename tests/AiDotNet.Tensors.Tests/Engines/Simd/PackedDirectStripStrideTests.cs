// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET9_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

[Collection("BlasManaged-Stats-Serial")]
public class PackedDirectStripStrideTests
{
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void PackedKBlocksRespectInputStridesAndOverwriteRaggedOutput(bool parallel)
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Fma.IsSupported, "direct GEMM requires FMA");
        // Exceeds three chunks' work floor while retaining ragged rows, columns and K blocks.
        const int m = 121, k = 513, n = 77, lda = k + 5, ldb = n + 7;
        var a = new float[m * lda];
        var b = new float[k * ldb];
        var actual = new float[m * n];
        Array.Fill(a, float.NaN);
        Array.Fill(b, float.NaN);
        Array.Fill(actual, float.NaN);
        var random = RandomHelper.CreateSeededRandom(1644);
        for (int i = 0; i < m; i++)
            for (int p = 0; p < k; p++) a[i * lda + p] = (float)(random.NextDouble() * 2 - 1);
        for (int p = 0; p < k; p++)
            for (int j = 0; j < n; j++) b[p * ldb + j] = (float)(random.NextDouble() * 2 - 1);

        bool? previousWidth = SimdGemm.s_forceVector512Direct;
        bool previousPacking = SimdGemm.s_packStrip512;
        int previousThreads = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            SimdGemm.s_forceVector512Direct = true;
            SimdGemm.s_packStrip512 = true;
            CpuParallelSettings.MaxDegreeOfParallelism = 3;
            Assert.True(SimdGemm.TrySgemmDirectParallel2D(a, lda, b, ldb, actual, m, k, n, parallel));
        }
        finally
        {
            SimdGemm.s_forceVector512Direct = previousWidth;
            SimdGemm.s_packStrip512 = previousPacking;
            CpuParallelSettings.MaxDegreeOfParallelism = previousThreads;
        }

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float expected = 0;
                for (int p = 0; p < k; p++) expected = MathF.FusedMultiplyAdd(a[i * lda + p], b[p * ldb + j], expected);
                Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual[i * n + j]));
            }
    }
}
#endif
