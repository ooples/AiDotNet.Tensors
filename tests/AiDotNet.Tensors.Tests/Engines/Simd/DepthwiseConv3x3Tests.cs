// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// The vectorized 3x3 depthwise kernels (<see cref="DepthwiseConv3x3"/>) against the generic loops they bypass, for
/// forward, input gradient and weight gradient at strides 1 and 2: odd and even sizes, rows narrower than a vector,
/// exactly one vector and a vector plus a tail, and batches. The kernels sum in a different order (vector lanes, nine
/// taps per register), so the comparison is to float rounding of the sums. Every shape runs through the width kernels
/// and through the channel-blocked ones (taken for multiples of eight channels on AVX2 hosts).
/// </summary>
[Collection("ConvPerfSerial")]
public class DepthwiseConv3x3Tests
{
    public static TheoryData<int, int, int, int> Shapes => new()
    {
        // batch, channels, size, stride
        { 1, 1, 1, 1 },
        { 2, 3, 5, 1 },
        { 2, 3, 5, 2 },
        { 3, 4, 7, 1 },
        { 3, 4, 7, 2 },
        { 2, 5, 8, 1 },
        { 2, 5, 8, 2 },
        { 2, 6, 16, 2 },
        { 2, 3, 17, 1 },
        { 2, 3, 17, 2 },
        { 1, 2, 34, 2 },
        { 1, 2, 47, 2 },
        { 2, 3, 64, 2 },
        { 4, 8, 14, 1 },
        { 2, 3, 13, 2 },
        // Multiples of eight channels: the channel-blocked kernels where the host has AVX2.
        { 2, 8, 7, 1 },
        { 2, 8, 7, 2 },
        { 3, 16, 5, 1 },
        { 3, 16, 14, 2 },
        { 1, 8, 1, 1 },
        { 2, 24, 9, 2 },
        { 2, 8, 17, 1 },
        { 1, 8, 28, 2 },
    };

    public static TheoryData<int, int, int, int, bool> ShapesAndKernels()
    {
        var data = new TheoryData<int, int, int, int, bool>();
        foreach (var row in Shapes)
        {
            data.Add((int)row[0], (int)row[1], (int)row[2], (int)row[3], false);
            data.Add((int)row[0], (int)row[1], (int)row[2], (int)row[3], true);
        }
        return data;
    }

#if NET9_0_OR_GREATER
    [SkippableTheory]
    [InlineData(2, 16, 7, 1)]
    [InlineData(2, 16, 7, 2)]
    [InlineData(1, 32, 14, 1)]
    [InlineData(3, 16, 14, 2)]
    [InlineData(2, 48, 9, 2)]
    [InlineData(1, 16, 28, 1)]
    [InlineData(2, 32, 1, 1)]
    public void TheSixteenChannelKernelsMatchTheGenericLoops(int batch, int channels, int size, int stride)
    {
        // The sixteen-channel kernels sit behind the eight-channel gate (AVX2 + FMA) and the operator opt-out; without
        // either, Compare would check another route and pass without running them.
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported,
            "The channel-blocked kernels need AVX2 + FMA.");
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_DW_AVX512") == "0", "AIDOTNET_DW_AVX512=0 opts them out.");
        // Forced on hosts without AVX-512: Vector512 is emulated there, so the arithmetic is the same.
        bool previousForce = DepthwiseConv3x3.s_forceAvx512ForTest;
        int previousMaxWidth = DepthwiseConv3x3.s_channelBlockMaxWidth;
        DepthwiseConv3x3.s_forceAvx512ForTest = true;
        DepthwiseConv3x3.s_channelBlockMaxWidth = int.MaxValue;
        try
        {
            Assert.True(DepthwiseConv3x3.UseAvx512Blocks(channels, size), "the sixteen-channel kernels would not run");
            Compare(batch, channels, size, stride);
        }
        finally
        {
            DepthwiseConv3x3.s_forceAvx512ForTest = previousForce;
            DepthwiseConv3x3.s_channelBlockMaxWidth = previousMaxWidth;
        }
    }
#endif

    [Theory]
    [MemberData(nameof(ShapesAndKernels))]
    public void TheVectorKernelsMatchTheGenericLoops(int batch, int channels, int size, int stride, bool channelBlocks)
    {
#if NET5_0_OR_GREATER
        int previousMaxWidth = DepthwiseConv3x3.s_channelBlockMaxWidth;
        int previousMaxWidth1 = DepthwiseConv3x3.s_channelBlockMaxWidthStride1;
        DepthwiseConv3x3.s_channelBlockMaxWidth = channelBlocks ? int.MaxValue : 0;
        DepthwiseConv3x3.s_channelBlockMaxWidthStride1 = channelBlocks ? int.MaxValue : 0;
        try
        {
            Compare(batch, channels, size, stride);
        }
        finally
        {
            DepthwiseConv3x3.s_channelBlockMaxWidth = previousMaxWidth;
            DepthwiseConv3x3.s_channelBlockMaxWidthStride1 = previousMaxWidth1;
        }
#else
        // The channel-blocked kernels are .NET 5+ (AVX2 intrinsics): only the width kernels exist here.
        Compare(batch, channels, size, stride);
#endif
    }

    private static void Compare(int batch, int channels, int size, int stride)
    {
        var engine = new CpuEngine();
        var rng = new Random(batch * 97 + channels * 13 + size * 3 + stride);
        int o = (size - 1) / stride + 1;
        var x = Random(rng, batch, channels, size, size);
        var w = Random(rng, channels, 1, 3, 3);
        var dy = Random(rng, batch, channels, o, o);
        int[] s = { stride, stride }, p = { 1, 1 };
        int[] xShape = { batch, channels, size, size }, wShape = { channels, 1, 3, 3 };

        Assert.True(DepthwiseConv3x3.Applies(1, 3, 3, stride, stride, 1, 1));
        var y = engine.DepthwiseConv2D(x, w, s, p).ToArray();
        var dx = engine.DepthwiseConv2DBackwardInput(dy, w, xShape, s, p).ToArray();
        var dw = engine.DepthwiseConv2DBackwardKernel(dy, x, wShape, s, p).ToArray();

        float[] yRef, dxRef, dwRef;
        DepthwiseConv3x3.s_disable = true;
        try
        {
            yRef = engine.DepthwiseConv2D(x, w, s, p).ToArray();
            dxRef = engine.DepthwiseConv2DBackwardInput(dy, w, xShape, s, p).ToArray();
            dwRef = engine.DepthwiseConv2DBackwardKernel(dy, x, wShape, s, p).ToArray();
        }
        finally
        {
            DepthwiseConv3x3.s_disable = false;
        }
        AssertClose("forward", yRef, y, 9);
        AssertClose("input gradient", dxRef, dx, 9);
        AssertClose("weight gradient", dwRef, dw, batch * o * o);
    }

    [Fact]
    public void OtherShapesKeepTheGenericLoops()
    {
        Assert.False(DepthwiseConv3x3.Applies(2, 3, 3, 1, 1, 1, 1));   // channel multiplier
        Assert.False(DepthwiseConv3x3.Applies(1, 5, 5, 1, 1, 2, 2));   // 5x5
        Assert.False(DepthwiseConv3x3.Applies(1, 3, 3, 1, 1, 0, 0));   // no padding
        Assert.False(DepthwiseConv3x3.Applies(1, 3, 3, 1, 2, 1, 1));   // unequal strides
        Assert.False(DepthwiseConv3x3.Applies(1, 3, 3, 3, 3, 1, 1));   // stride 3
    }

    private static void AssertClose(string what, float[] expected, float[] actual, int terms)
    {
        Assert.Equal(expected.Length, actual.Length);
        // Each value sums `terms` products of magnitude < 1: reordering moves it by a few ulps of the term count.
        double tolerance = 1e-6 * terms + 1e-6;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"{what}[{i}]: expected {expected[i]}, got {actual[i]} (tolerance {tolerance}).");
    }

    private static Tensor<float> Random(Random rng, params int[] shape)
    {
        var t = new Tensor<float>(shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}
