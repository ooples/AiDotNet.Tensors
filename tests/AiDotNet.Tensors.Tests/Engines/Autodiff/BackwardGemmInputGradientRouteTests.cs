// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET9_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// dX = dY·Wᵀ for a linear layer: on AVX-512 the 512-bit direct tiles read strips of W transposed as they are packed;
/// elsewhere, where the route prefers it, W is transposed once and the product runs as a plain NN GEMM, instead of the
/// transposed-B dot-product path.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class BackwardGemmInputGradientRouteTests
{
    [Theory]
    [InlineData(true, 128, 256, true)]   // AVX-512: every width
    [InlineData(true, 64, 128, true)]
    [InlineData(true, 32, 256, false)]   // too few rows to amortise the transpose
    [InlineData(false, 128, 256, false)] // AVX2: depth 128 / 256 keep the transposed-B path
    [InlineData(false, 64, 128, false)]
    [InlineData(false, 128, 512, true)]  // AVX2: depth >= 512
    [InlineData(false, 128, 10, true)]   // AVX2: narrow head
    public void Route_FollowsWidthAndDepth(bool vector512, int m, int n, bool expected)
    {
        bool? previous = SimdGemm.s_forceVector512Direct;
        try
        {
            SimdGemm.s_forceVector512Direct = vector512;
            Assert.Equal(expected, BackwardFunctions<float>.BackwardInputGradientPrefersTransposedW(m, n));
        }
        finally
        {
            SimdGemm.s_forceVector512Direct = previous;
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Route_SkipsTheTransposeForALargeWeight(bool vector512)
    {
        // A weight above 1M elements is not transposed: the per-call copy was not kept by the shared pool between
        // training steps (FasterRCNN's fc6 [12544, 1024] allocated 51 MB per step) and the transposed-B kernel matches it.
        bool? previous = SimdGemm.s_forceVector512Direct;
        try
        {
            SimdGemm.s_forceVector512Direct = vector512;
            Assert.True(BackwardFunctions<float>.BackwardInputGradientPrefersTransposedW(128, 512));
            Assert.True(BackwardFunctions<float>.BackwardInputGradientPrefersTransposedW(128, 512, 784));
            Assert.False(BackwardFunctions<float>.BackwardInputGradientPrefersTransposedW(128, 1024, 12544));
        }
        finally
        {
            SimdGemm.s_forceVector512Direct = previous;
        }
    }

    [SkippableTheory]
    [InlineData(true, 128, 784, 512)]
    [InlineData(true, 128, 512, 256)]
    [InlineData(true, 64, 3136, 128)]
    [InlineData(false, 128, 784, 512)]
    [InlineData(false, 128, 256, 10)]
    [InlineData(false, 128, 512, 256)]   // not routed: the transposed-B path, still correct
    public void InputGradient_MatchesReference_AndTakesTheExpectedRoute(bool vector512, int m, int k, int n)
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Fma.IsSupported, "direct GEMM requires FMA");
        var random = RandomHelper.CreateSeededRandom(1573);
        var dY = new float[m * n];
        var w = new float[k * n];
        for (int i = 0; i < dY.Length; i++) dY[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < w.Length; i++) w[i] = (float)(random.NextDouble() * 2 - 1);
        var dX = new float[m * k];
        var nn = new float[m * k];

        bool? previous = SimdGemm.s_forceVector512Direct;
        bool routed;
        try
        {
            SimdGemm.s_forceVector512Direct = vector512;
            // AVX-512 takes the fused transposed-strip kernel; otherwise the explicit transpose where preferred.
            bool fused = vector512 && m >= 64 && n <= SimdGemm.MaxPackedStripK;
            routed = fused || BackwardFunctions<float>.BackwardInputGradientPrefersTransposedW(m, n, k);
            long transposedBefore = BackwardFunctions<float>.t_transposedWInputGradientRuns;
            int fusedBefore = SimdGemm.t_directParallel2DTransBRuns;
            BackwardFunctions<float>.BackwardGemmInputGradient(dY, w, dX, m, k, n);
            Assert.Equal(fused ? fusedBefore + 1 : fusedBefore, SimdGemm.t_directParallel2DTransBRuns);
            Assert.Equal(routed && !fused ? transposedBefore + 1 : transposedBefore,
                BackwardFunctions<float>.t_transposedWInputGradientRuns);

            var wT = new float[n * k];
            SimdTranspose.Transpose(w, wT, k, n);
            SimdGemm.Sgemm(dY, n, false, wT, k, false, nn, m, n, k);
        }
        finally
        {
            SimdGemm.s_forceVector512Direct = previous;
        }

        if (routed)
            for (int i = 0; i < dX.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(nn[i]) == BitConverter.SingleToInt32Bits(dX[i]),
                    $"dX[{i / k},{i % k}] = {dX[i]:R}, NN product {nn[i]:R}");

        for (int row = 0; row < m; row += 7)
            for (int col = 0; col < k; col += 11)
            {
                double expected = 0;
                for (int p = 0; p < n; p++) expected += (double)dY[row * n + p] * w[col * n + p];
                Assert.True(Math.Abs(dX[row * k + col] - expected) <= 1e-4 * (1 + Math.Abs(expected)),
                    $"dX[{row},{col}] = {dX[row * k + col]:R}, reference {expected:R}");
            }
    }
}
#endif
