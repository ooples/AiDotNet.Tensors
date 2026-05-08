// Copyright (c) AiDotNet. All rights reserved.
// Issue #310 regression — BatchNorm4D must accept input tensors whose
// underlying data buffer is SIMD-padded beyond the logical extent (i.e.
// every non-power-of-2 spatial chain in EfficientNet / MobileNetV2 /
// MobileNetV3 224x224 inference).

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class BatchNorm4DPaddedDataTests
{
    private static (Tensor<float> input, Tensor<float> gamma, Tensor<float> beta) MakeInputs(
        int batch, int channels, int height, int width, int seed)
    {
        var rng = new Random(seed);
        var x = new Tensor<float>(new[] { batch, channels, height, width });
        var span = x.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var gamma = new Tensor<float>(new[] { channels });
        var beta = new Tensor<float>(new[] { channels });
        var g = gamma.AsWritableSpan();
        var b = beta.AsWritableSpan();
        for (int c = 0; c < channels; c++) { g[c] = 1f; b[c] = 0f; }
        return (x, gamma, beta);
    }

    /// <summary>
    /// Issue #310 — every non-power-of-2 spatial chain in the canonical
    /// 224 → 112 → 56 → 28 → 14 → 7 EfficientNet / MobileNet sequence
    /// previously threw <c>ArgumentException: Data length (X) must match
    /// shape total (Y)</c> from <c>TensorAllocator.Rent</c> because the
    /// SIMD-padded data buffer (e.g. 524,288 elements for a [1, 32, 112, 112]
    /// tensor) was being wrapped in a tensor whose logical shape totalled
    /// only 401,408. The fix sized the output buffer to the logical
    /// extent — these shapes must now run cleanly.
    /// </summary>
    [Theory]
    [InlineData(1, 32, 112, 112)]
    [InlineData(2, 32, 112, 112)]
    [InlineData(4, 96, 112, 112)]
    [InlineData(1, 16, 56, 56)]
    [InlineData(2, 24, 28, 28)]
    [InlineData(1, 40, 14, 14)]
    [InlineData(1, 320, 7, 7)]
    public void BatchNorm_RunsOnNonPowerOfTwoSpatialDims(int batch, int channels, int H, int W)
    {
        var engine = new CpuEngine();
        var (x, gamma, beta) = MakeInputs(batch, channels, H, W, seed: 42);

        var output = engine.BatchNorm(x, gamma, beta, 1e-5, out var mean, out var variance);

        Assert.Equal(new[] { batch, channels, H, W }, output._shape);
        Assert.Equal(new[] { channels }, mean._shape);
        Assert.Equal(new[] { channels }, variance._shape);
        Assert.Equal(batch * channels * H * W, output.Length);
    }

    /// <summary>
    /// With gamma=1, beta=0 the BatchNorm output of each channel must
    /// have mean ≈ 0 and variance ≈ 1 (modulo eps). This is the
    /// definitional guarantee — it has to hold on padded-data inputs
    /// just like everywhere else.
    /// </summary>
    [Theory]
    [InlineData(1, 32, 112, 112)]
    [InlineData(2, 96, 56, 56)]
    public void BatchNorm_NormalizesEachChannelOnPaddedInputs(int batch, int channels, int H, int W)
    {
        var engine = new CpuEngine();
        var (x, gamma, beta) = MakeInputs(batch, channels, H, W, seed: 7);

        var output = engine.BatchNorm(x, gamma, beta, 1e-5, out _, out _);

        var outSpan = output.AsSpan();
        int spatialSize = H * W;
        for (int c = 0; c < channels; c++)
        {
            double sum = 0, sumSq = 0;
            int count = 0;
            for (int n = 0; n < batch; n++)
            {
                int chBase = (n * channels + c) * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    double v = outSpan[chBase + s];
                    sum += v;
                    sumSq += v * v;
                    count++;
                }
            }
            double mean = sum / count;
            double var = (sumSq / count) - mean * mean;
            Assert.InRange(mean, -1e-3, 1e-3);
            // Variance ≈ 1 modulo the epsilon-shifted invStd (allow 5% slop).
            Assert.InRange(var, 1.0 - 5e-2, 1.0 + 5e-2);
        }
    }

    /// <summary>
    /// Make sure we didn't accidentally break the power-of-2 spatial
    /// path with the padded-buffer fix.
    /// </summary>
    [Theory]
    [InlineData(1, 32, 16, 16)]
    [InlineData(2, 64, 32, 32)]
    public void BatchNorm_StillWorksOnPowerOfTwoSpatialDims(int batch, int channels, int H, int W)
    {
        var engine = new CpuEngine();
        var (x, gamma, beta) = MakeInputs(batch, channels, H, W, seed: 13);

        var output = engine.BatchNorm(x, gamma, beta, 1e-5, out var mean, out var variance);

        Assert.Equal(new[] { batch, channels, H, W }, output._shape);
        Assert.Equal(new[] { channels }, mean._shape);
        Assert.Equal(new[] { channels }, variance._shape);
    }
}
