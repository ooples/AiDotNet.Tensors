// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The generated row kernels on the inputs that used to go wrong: large equal logits, signed targets, short optional
/// spans, a sin table shorter than the cos table, and NaN through the bias + ReLU.
/// </summary>
public sealed class GeneratedRowKernelEdgeCaseTests
{
    [Fact]
    public void CrossEntropy_LargeEqualLogits_KeepTheirLogTwo()
    {
        // m + log 2 rounds back to m = 1e8 in float: the loss was 0 and the softmax [1, 1].
        float[] logits = { 1e8f, 1e8f };
        float[] target = { 1f, 0f };
        var rowMax = new float[1];
        var rowLse = new float[1];

        float loss = CompiledFusedKernels.LogSoftmaxCrossEntropyForward(logits, target, rowMax, rowLse, 1, 2, out int active);

        Assert.Equal(1, active);
        Assert.Equal(MathF.Log(2f), loss, 6);
        var grad = new float[2];
        CompiledFusedKernels.LogSoftmaxCrossEntropyBackward(1f, logits, target, rowMax, rowLse, grad, 1, 2);
        Assert.Equal(-0.5f, grad[0], 6);
        Assert.Equal(0.5f, grad[1], 6);
    }

    [Fact]
    public void CrossEntropy_SignedTargetsSummingToZero_KeepTheirGradient()
    {
        // d/dz of -sum(t * logSoftmax(z)) is (sum t) * softmax - t = -t when sum t = 0; the row was cleared.
        float[] logits = { 0.3f, -1.2f, 2.0f };
        float[] target = { 1f, -1f, 0f };
        var rowMax = new float[1];
        var rowLse = new float[1];
        CompiledFusedKernels.LogSoftmaxCrossEntropyForward(logits, target, rowMax, rowLse, 1, 3, out _);

        var grad = new float[3];
        CompiledFusedKernels.LogSoftmaxCrossEntropyBackward(1f, logits, target, rowMax, rowLse, grad, 1, 3);

        Assert.Equal(-1f, grad[0], 6);
        Assert.Equal(1f, grad[1], 6);
        Assert.Equal(0f, grad[2], 6);
    }

    [Fact]
    public void CrossEntropy_LargeUpstreamGradient_IsAppliedOnceAndStaysFinite()
    {
        // The scale was multiplied by the active-row count and divided back: 1e38 * 4 overflowed first.
        float[] logits = { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f };
        float[] target = { 1f, 0f, 1f, 0f, 1f, 0f, 1f, 0f };
        var rowMax = new float[4];
        var rowLse = new float[4];
        CompiledFusedKernels.LogSoftmaxCrossEntropyForward(logits, target, rowMax, rowLse, 4, 2, out _);

        var grad = new float[8];
        CompiledFusedKernels.LogSoftmaxCrossEntropyBackward(1e38f, logits, target, rowMax, rowLse, grad, 4, 2);

        for (int r = 0; r < 4; r++)
        {
            Assert.Equal(-5e37f, grad[2 * r], 1e31f);
            Assert.Equal(5e37f, grad[2 * r + 1], 1e31f);
        }
    }

    [Fact]
    public void CrossEntropy_MaskedClassAtNegativeInfinity_AddsNoNaN()
    {
        float[] logits = { 0f, float.NegativeInfinity };
        float[] target = { 1f, 0f };
        var rowMax = new float[1];
        var rowLse = new float[1];

        float loss = CompiledFusedKernels.LogSoftmaxCrossEntropyForward(logits, target, rowMax, rowLse, 1, 2, out _);

        Assert.Equal(0f, loss);
    }

    [Fact]
    public void LayerNormResidualForward_ShortStatisticSpan_ThrowsBeforeWritingAnything()
    {
        const int rows = 2, cols = 4;
        var x = new float[rows * cols];
        var gamma = new float[] { 1, 1, 1, 1 };
        var beta = new float[cols];
        var output = new float[rows * cols];
        var mean = new[] { 7f, 7f };
        var invStd = new[] { 7f };   // one short
        for (int i = 0; i < output.Length; i++) output[i] = 7f;

        Assert.Throws<ArgumentException>(() => CompiledFusedKernels.LayerNormResidualForward(
            x, x, gamma, beta, output, mean, invStd, rows, cols, 1e-5f));
        Assert.All(output, v => Assert.Equal(7f, v));
        Assert.All(mean, v => Assert.Equal(7f, v));
    }

    [Fact]
    public void LayerNormResidualBackward_ShortBetaGradient_DoesNotClearGammaGradient()
    {
        const int rows = 1, cols = 2;
        var ones = new float[] { 1, 1 };
        var gradGamma = new[] { 5f, 5f };
        var gradBeta = new[] { 5f };   // one short

        Assert.Throws<ArgumentException>(() => CompiledFusedKernels.LayerNormResidualBackward(
            ones, ones, ones, ones, new[] { 0f }, new[] { 1f }, new float[2], gradGamma, gradBeta, rows, cols));
        Assert.All(gradGamma, v => Assert.Equal(5f, v));
    }

    [Fact]
    public void Rope_CosSizedForHeadDimButSinCompact_UsesTheCompactStride()
    {
        const int seqLen = 2, headDim = 4, halfDim = 2;
        var input = new float[seqLen * headDim];
        for (int i = 0; i < input.Length; i++) input[i] = i + 1;
        // cos covers seqLen * headDim, sin only seqLen * halfDim: the headDim stride would read past sin.
        var cos = new float[seqLen * headDim];
        var sin = new float[seqLen * halfDim];
        for (int i = 0; i < cos.Length; i++) cos[i] = 1f;
        var output = new float[input.Length];

        CompiledFusedKernels.RotaryPositionEmbeddingSplitHalf(input, cos, sin, output, 1, seqLen, headDim);

        Assert.Equal(input, output);   // cos = 1, sin = 0 on the compact layout
    }

    [Fact]
    public void BiasRelu_NaNIsZeroInEveryColumn()
    {
        const int cols = 9;   // eight AVX lanes and one scalar tail column
        var input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = float.NaN;
        var output = new float[cols];

        CompiledFusedKernels.BiasActivationRowForward(input, new float[cols], output, 1, cols, activationKind: CompiledFusedKernels.RowActivationKind.ReLU);

        Assert.All(output, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void CudaReductions_ReadTheResultBeforeTheSharedSlotIsReused()
    {
        string source = CompiledFusedKernels.UniversalRowFusedCudaSource;
        Assert.Equal(2, Count(source, "float result = shared[0];\n    __syncthreads();\n    return result;"));
        Assert.DoesNotContain("return shared[0];", source);
    }

    [Fact]
    public void CudaCrossEntropy_SkipsZeroTargetsAndKeepsShiftedStatistics()
    {
        string source = CompiledFusedKernels.UniversalRowFusedCudaSource;
        Assert.Contains("if (t == 0.0f) continue;", source);   // 0 * -inf on a masked class is NaN
        Assert.Contains("localLoss -= t * ((logits[baseIdx + c] - rmax) - lse);", source);
        Assert.Contains("float prob = expf((logits[baseIdx + c] - rmax) - lse);", source);
    }

    private static int Count(string text, string part)
    {
        int count = 0;
        for (int at = text.Replace("\r\n", "\n").IndexOf(part, StringComparison.Ordinal); at >= 0;
             at = text.Replace("\r\n", "\n").IndexOf(part, at + part.Length, StringComparison.Ordinal))
            count++;
        return count;
    }
}
