// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// <see cref="CpuEngine.DepthwiseConv2DBatchNorm{T}"/> applies the inference batch norm and its ReLU/ReLU6 as the 3x3
/// depthwise kernel stores each output. It must equal <see cref="CpuEngine.DepthwiseConv2D{T}"/> followed by
/// <see cref="CpuEngine.BatchNormInference{T}(Tensor{T}, Tensor{T}, Tensor{T}, Tensor{T}, Tensor{T}, double, FusedActivationType)"/>
/// bit for bit, through the width kernels and the channel-blocked ones, at strides 1 and 2, with and without a tail
/// vector, for channel counts that are and are not multiples of eight, and with NaN inputs.
/// </summary>
[Collection("ConvPerfSerial")]
public class DepthwiseConv2DBatchNormTests
{
    public static TheoryData<int, int, int, int, bool> Shapes => new()
    {
        // batch, channels, size, stride, channel-blocked kernels
        { 1, 1, 1, 1, false },
        { 2, 3, 5, 1, false },
        { 2, 3, 17, 2, false },
        { 2, 5, 8, 2, false },
        { 1, 2, 47, 2, false },
        { 3, 8, 7, 1, true },
        { 3, 8, 7, 2, true },
        { 2, 16, 14, 1, true },
        { 2, 24, 16, 2, true },
        { 2, 16, 14, 1, false },
        { 2, 24, 16, 2, false },
        { 16, 96, 16, 2, true },
        { 16, 144, 8, 1, true },
    };

    private static Tensor<float> Random(int[] shape, int seed, float lo, float hi)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return t;
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void EqualsDepthwiseThenBatchNorm_BitForBit(int batch, int channels, int size, int stride, bool channelBlocks)
    {
        int previousMaxWidth = DepthwiseConv3x3.s_channelBlockMaxWidth;
        int previousMaxWidth1 = DepthwiseConv3x3.s_channelBlockMaxWidthStride1;
        DepthwiseConv3x3.s_channelBlockMaxWidth = channelBlocks ? int.MaxValue : 0;
        DepthwiseConv3x3.s_channelBlockMaxWidthStride1 = channelBlocks ? int.MaxValue : 0;
        try
        {
            var engine = new CpuEngine();
            var x = Random(new[] { batch, channels, size, size }, 1, -2f, 2f);
            if (x.Length > 3) { x[1] = float.NaN; x[x.Length - 2] = float.NaN; }
            var w = Random(new[] { channels, 1, 3, 3 }, 2, -1f, 1f);
            var gamma = Random(new[] { channels }, 3, 0.5f, 1.5f);
            var beta = Random(new[] { channels }, 4, -0.5f, 0.5f);
            var mean = Random(new[] { channels }, 5, -0.3f, 0.3f);
            var variance = Random(new[] { channels }, 6, 0.2f, 2f);
            int[] s = { stride, stride }, p = { 1, 1 };
            foreach (var activation in new[] { FusedActivationType.ReLU6, FusedActivationType.ReLU, FusedActivationType.None })
            {
                var expected = engine.BatchNormInference(engine.DepthwiseConv2D(x, w, s, p), gamma, beta, mean, variance,
                    1e-5, activation).ToArray();
                var actual = engine.DepthwiseConv2DBatchNorm(x, w, s, p, gamma, beta, mean, variance, 1e-5, activation)
                    .ToArray();
                Assert.Equal(expected.Length, actual.Length);
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i])
                        || (float.IsNaN(expected[i]) && float.IsNaN(actual[i])),
                        $"{activation} [{i}]: unfused {expected[i]}, fused {actual[i]}");
            }
        }
        finally
        {
            DepthwiseConv3x3.s_channelBlockMaxWidth = previousMaxWidth;
            DepthwiseConv3x3.s_channelBlockMaxWidthStride1 = previousMaxWidth1;
        }
    }

    [Fact]
    public void UnderATape_RecordsTheUnfusedOps()
    {
        var engine = new CpuEngine();
        var x = Random(new[] { 2, 8, 6, 6 }, 7, -1f, 1f);
        var w = Random(new[] { 8, 1, 3, 3 }, 8, -1f, 1f);
        var gamma = Random(new[] { 8 }, 9, 0.5f, 1.5f);
        var beta = Random(new[] { 8 }, 10, -0.5f, 0.5f);
        var mean = Random(new[] { 8 }, 11, -0.3f, 0.3f);
        var variance = Random(new[] { 8 }, 12, 0.2f, 2f);
        using var tape = new GradientTape<float>();
        var y = engine.DepthwiseConv2DBatchNorm(x, w, new[] { 1, 1 }, new[] { 1, 1 }, gamma, beta, mean, variance, 1e-5,
            FusedActivationType.ReLU6);
        var grads = tape.ComputeGradients(engine.ReduceSum(y, null), sources: new[] { w });
        Assert.True(grads.ContainsKey(w), "the depthwise kernel received no gradient: the fused op was not recorded");
    }
}
