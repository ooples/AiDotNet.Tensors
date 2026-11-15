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
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void Rejects_mismatched_normalization_parameters_before_the_fused_path(int bad)
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 1, 2, 3, 3 });
        var kernel = new Tensor<float>(new[] { 2, 1, 3, 3 });
        var parameters = new Tensor<float>[4];
        for (int i = 0; i < 4; i++) parameters[i] = new Tensor<float>(new[] { i == bad ? 1 : 2 });
        Assert.Throws<ArgumentException>(() => engine.DepthwiseConv2DBatchNorm(input, kernel,
            new[] { 1, 1 }, new[] { 1, 1 }, parameters[0], parameters[1], parameters[2], parameters[3],
            1e-5, FusedActivationType.ReLU));
    }

    [Fact]
    public void Rejects_short_stride_before_indexing_it()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 1, 2, 3, 3 });
        var kernel = new Tensor<float>(new[] { 2, 1, 3, 3 });
        var p = new Tensor<float>(new[] { 2 });
        Assert.Throws<ArgumentException>(() => engine.DepthwiseConv2DBatchNorm(input, kernel,
            new[] { 1 }, new[] { 1, 1 }, p, p, p, p, 1e-5, FusedActivationType.ReLU));
    }

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
#if NET5_0_OR_GREATER
        int previousMaxWidth = DepthwiseConv3x3.s_channelBlockMaxWidth;
        int previousMaxWidth1 = DepthwiseConv3x3.s_channelBlockMaxWidthStride1;
        DepthwiseConv3x3.s_channelBlockMaxWidth = channelBlocks ? int.MaxValue : 0;
        DepthwiseConv3x3.s_channelBlockMaxWidthStride1 = channelBlocks ? int.MaxValue : 0;
#else
        // The channel-blocked AVX2 kernels exist only on .NET 5+; here both cases run the scalar path.
        _ = channelBlocks;
#endif
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
                    Assert.True(Bits(expected[i]) == Bits(actual[i])
                        || (float.IsNaN(expected[i]) && float.IsNaN(actual[i])),
                        $"{activation} [{i}]: unfused {expected[i]}, fused {actual[i]}");
            }
        }
        finally
        {
#if NET5_0_OR_GREATER
            DepthwiseConv3x3.s_channelBlockMaxWidth = previousMaxWidth;
            DepthwiseConv3x3.s_channelBlockMaxWidthStride1 = previousMaxWidth1;
#endif
        }
    }

    private static int Bits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

#if NET5_0_OR_GREATER
    /// <summary>
    /// The epilogue where there is no FMA (Arm64, x86 without FMA), forced here: the product and the sum round
    /// separately and the clamp keeps a NaN, bit for bit what the scalar expression gives, and it stays in vector
    /// registers. It used to copy every vector of outputs to a new array and build a vector from it: two heap
    /// allocations per vector, 115 KB for the 64 KB result of the stride-1 case below.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void WithoutFma_TheEpilogueRoundsSeparately_AndAllocatesNothingPerVector(int stride)
    {
        bool previousFma = DepthwiseConv3x3.s_epilogueFma;
        int previousMaxWidth = DepthwiseConv3x3.s_channelBlockMaxWidth;
        int previousMaxWidth1 = DepthwiseConv3x3.s_channelBlockMaxWidthStride1;
        DepthwiseConv3x3.s_epilogueFma = false;
        DepthwiseConv3x3.s_channelBlockMaxWidth = 0;          // the channel-blocked kernels need FMA
        DepthwiseConv3x3.s_channelBlockMaxWidthStride1 = 0;
        try
        {
            var engine = new CpuEngine();
            const int size = 128;
            // One plane: a single task, so the work and its allocations are on this thread.
            var x = Random(new[] { 1, 1, size, size }, 1, -2f, 2f);
            x[5] = float.NaN;
            var w = Random(new[] { 1, 1, 3, 3 }, 2, -1f, 1f);
            var gamma = Random(new[] { 1 }, 3, 0.5f, 1.5f);
            var beta = Random(new[] { 1 }, 4, -0.5f, 0.5f);
            var mean = Random(new[] { 1 }, 5, -0.3f, 0.3f);
            var variance = Random(new[] { 1 }, 6, 0.2f, 2f);
            int[] s = { stride, stride }, p = { 1, 1 };
            var plain = engine.DepthwiseConv2D(x, w, s, p).ToArray();
            // NchwcBatchNorm.PackScaleBias.
            const float epsilon = 1e-5f;
            double scaleD = gamma[0] / Math.Sqrt((double)variance[0] + epsilon);
            float scale = (float)scaleD, shift = beta[0] - (float)(scaleD * mean[0]);

            foreach (var (activation, high) in new[] { (FusedActivationType.ReLU6, 6f), (FusedActivationType.ReLU, float.PositiveInfinity) })
            {
                engine.DepthwiseConv2DBatchNorm(x, w, s, p, gamma, beta, mean, variance, 1e-5, activation);   // scratch
                long before = GC.GetAllocatedBytesForCurrentThread();
                var fused = engine.DepthwiseConv2DBatchNorm(x, w, s, p, gamma, beta, mean, variance, 1e-5, activation);
                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                var actual = fused.ToArray();

                Assert.Equal(plain.Length, actual.Length);
                bool sawNaN = false;
                for (int i = 0; i < plain.Length; i++)
                {
                    float product = plain[i] * scale;
                    float expected = product + shift;
                    if (expected < 0f) expected = 0f;
                    if (expected > high) expected = high;
                    sawNaN |= float.IsNaN(expected);
                    Assert.True(Bits(expected) == Bits(actual[i]) || (float.IsNaN(expected) && float.IsNaN(actual[i])),
                        $"{activation} [{i}]: scalar {expected}, fused {actual[i]}");
                }
                Assert.True(sawNaN, "the NaN input reached no output");
                long resultBytes = (long)actual.Length * sizeof(float);
                Assert.True(allocated <= resultBytes + 8 * 1024,
                    $"{activation}: {allocated} bytes allocated for a {resultBytes}-byte result");
            }
        }
        finally
        {
            DepthwiseConv3x3.s_epilogueFma = previousFma;
            DepthwiseConv3x3.s_channelBlockMaxWidth = previousMaxWidth;
            DepthwiseConv3x3.s_channelBlockMaxWidthStride1 = previousMaxWidth1;
        }
    }
#endif

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

    [SkippableTheory]
    [InlineData(1, FusedActivationType.ReLU6)]
    [InlineData(2, FusedActivationType.ReLU)]
    [InlineData(1, FusedActivationType.None)]
    public void Gpu_MatchesTheCpuComposition(int stride, FusedActivationType activation)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "No DirectGpu backend.");
        using var _ = gpu;
        var cpu = new CpuEngine();
        var x = Random(new[] { 2, 24, 15, 15 }, 21, -2f, 2f);
        var w = Random(new[] { 24, 1, 3, 3 }, 22, -1f, 1f);
        var gamma = Random(new[] { 24 }, 23, 0.5f, 1.5f);
        var beta = Random(new[] { 24 }, 24, -0.5f, 0.5f);
        var mean = Random(new[] { 24 }, 25, -0.3f, 0.3f);
        var variance = Random(new[] { 24 }, 26, 0.2f, 2f);
        int[] s = { stride, stride }, p = { 1, 1 };
        var expected = cpu.BatchNormInference(cpu.DepthwiseConv2D(x, w, s, p), gamma, beta, mean, variance, 1e-5,
            activation).ToArray();
        var actual = gpu!.DepthwiseConv2DBatchNorm(x, w, s, p, gamma, beta, mean, variance, 1e-5, activation).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * Math.Max(1f, Math.Abs(expected[i])),
                $"[{i}] GPU {actual[i]} vs CPU {expected[i]}");
    }
}
