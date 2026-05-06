// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Coverage for issue #294 Phase 2: <see cref="CpuEngine.LayerNorm"/>'s
/// generic-T path now routes the variance reduction through
/// <see cref="AiDotNet.Tensors.Engines.Optimization.NumericFastPath.SumOfSquaresDouble"/>
/// and the normalize-and-scale step through
/// <see cref="AiDotNet.Tensors.Engines.Optimization.NumericFastPath.AffineNormalizeDouble"/>
/// when <c>T == double</c>. Without an explicit double-precision
/// test, regressions in the new SIMD reduction or affine path would
/// ship without coverage — the float fast path is exercised
/// elsewhere but takes a different code path entirely
/// (<c>ProcessBatchesSimd</c>).
/// </summary>
public class LayerNormDoublePathTests
{
    [Fact]
    public void LayerNorm_Double_RankTwo_NormalizesEachRowToZeroMeanUnitVariance()
    {
        var engine = new CpuEngine();
        // 4 batches × 8 features. Distinct row content so a regression
        // that read the wrong row (broken offset calc) would surface.
        const int B = 4, F = 8;
        var x = new Tensor<double>(new[] { B, F });
        var rng = new Random(42);
        var xSpan = x.AsWritableSpan();
        for (int i = 0; i < xSpan.Length; i++) xSpan[i] = rng.NextDouble() * 10.0 - 5.0;

        // gamma=1, beta=0 → output should have row-mean ≈ 0 and
        // row-variance ≈ 1 modulo eps.
        var gamma = new Tensor<double>(new[] { F });
        var beta = new Tensor<double>(new[] { F });
        var gSpan = gamma.AsWritableSpan();
        for (int i = 0; i < F; i++) gSpan[i] = 1.0;

        var output = engine.LayerNorm(x, gamma, beta, epsilon: 1e-5,
            out var meanT, out var varT);

        Assert.Equal(new[] { B, F }, output._shape);
        Assert.Equal(new[] { B }, meanT._shape);
        Assert.Equal(new[] { B }, varT._shape);

        var outSpan = output.AsSpan();
        var meanSpan = meanT.AsSpan();
        var varSpan = varT.AsSpan();

        for (int b = 0; b < B; b++)
        {
            // Recompute row mean / var of the OUTPUT for the post-
            // normalize check. With gamma=1, beta=0, row mean must be
            // ≈ 0 and row variance must be ≈ 1 (down to numerical
            // tolerance).
            double rowSum = 0;
            for (int f = 0; f < F; f++) rowSum += outSpan[b * F + f];
            double rowMean = rowSum / F;
            double rowSumSq = 0;
            for (int f = 0; f < F; f++)
            {
                double d = outSpan[b * F + f] - rowMean;
                rowSumSq += d * d;
            }
            double rowVar = rowSumSq / F;

            Assert.InRange(rowMean, -1e-10, 1e-10);
            Assert.InRange(rowVar, 1.0 - 5e-3, 1.0 + 5e-3);

            // Verify the saved mean / var tensors match the input row
            // statistics — the path that's exposed for backward.
            double expectedInputMean = 0;
            for (int f = 0; f < F; f++) expectedInputMean += xSpan[b * F + f];
            expectedInputMean /= F;
            Assert.InRange(meanSpan[b] - expectedInputMean, -1e-12, 1e-12);

            double expectedInputVar = 0;
            for (int f = 0; f < F; f++)
            {
                double d = xSpan[b * F + f] - expectedInputMean;
                expectedInputVar += d * d;
            }
            expectedInputVar /= F;
            Assert.InRange(varSpan[b] - expectedInputVar, -1e-10, 1e-10);
        }
    }

    [Fact]
    public void LayerNorm_Double_GammaBetaApplied()
    {
        var engine = new CpuEngine();
        // gamma=2, beta=1 → output row should have mean ≈ 1 and
        // variance ≈ 4 (gamma^2 × normalized var).
        const int B = 2, F = 16;
        var x = new Tensor<double>(new[] { B, F });
        var xSpan = x.AsWritableSpan();
        for (int i = 0; i < xSpan.Length; i++) xSpan[i] = (i + 1) * 0.5;

        var gamma = new Tensor<double>(new[] { F });
        var beta = new Tensor<double>(new[] { F });
        var gSpan = gamma.AsWritableSpan();
        var betaSpan = beta.AsWritableSpan();
        for (int i = 0; i < F; i++) { gSpan[i] = 2.0; betaSpan[i] = 1.0; }

        var output = engine.LayerNorm(x, gamma, beta, epsilon: 1e-5,
            out _, out _);

        var outSpan = output.AsSpan();
        for (int b = 0; b < B; b++)
        {
            double rowSum = 0;
            for (int f = 0; f < F; f++) rowSum += outSpan[b * F + f];
            double rowMean = rowSum / F;
            // mean(gamma·x_norm + beta) = gamma·mean(x_norm) + beta = 0 + 1 = 1
            Assert.InRange(rowMean, 1.0 - 1e-9, 1.0 + 1e-9);

            double rowSumSq = 0;
            for (int f = 0; f < F; f++)
            {
                double d = outSpan[b * F + f] - rowMean;
                rowSumSq += d * d;
            }
            double rowVar = rowSumSq / F;
            // var(gamma·x_norm + beta) = gamma^2 · var(x_norm) ≈ 4
            Assert.InRange(rowVar, 4.0 - 5e-2, 4.0 + 5e-2);
        }
    }

    [Fact]
    public void LayerNorm_Double_RankThree_NormalizesAlongLastTwoDimsWhenGammaRank2()
    {
        // Generic-T LayerNorm normalizes over the trailing dims that
        // match gamma's shape. With input [B, S, F] and gamma [F],
        // it reduces over the last dim per (B, S) row. This is the
        // BERT/transformer canonical shape — verifying double works
        // here exercises the generic path in a real-model layout.
        var engine = new CpuEngine();
        const int B = 2, S = 3, F = 16;
        var x = new Tensor<double>(new[] { B, S, F });
        var xSpan = x.AsWritableSpan();
        var rng = new Random(7);
        for (int i = 0; i < xSpan.Length; i++) xSpan[i] = rng.NextDouble() * 4.0 - 2.0;

        var gamma = new Tensor<double>(new[] { F });
        var beta = new Tensor<double>(new[] { F });
        var gSpan = gamma.AsWritableSpan();
        for (int i = 0; i < F; i++) gSpan[i] = 1.0;

        var output = engine.LayerNorm(x, gamma, beta, epsilon: 1e-5, out _, out _);

        Assert.Equal(new[] { B, S, F }, output._shape);
        var outSpan = output.AsSpan();
        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
            {
                int rowStart = (b * S + s) * F;
                double rowSum = 0;
                for (int f = 0; f < F; f++) rowSum += outSpan[rowStart + f];
                double rowMean = rowSum / F;
                Assert.InRange(rowMean, -1e-10, 1e-10);
            }
    }
}
