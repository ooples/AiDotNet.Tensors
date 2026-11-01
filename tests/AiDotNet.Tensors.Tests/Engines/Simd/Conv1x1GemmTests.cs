// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET5_0_OR_GREATER
using System;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// The unpadded 1x1 GEMM route (<see cref="Conv1x1Gemm"/>) against a scalar convolution, through the engine entry points
/// training uses (Conv2DInto, Conv2DBackwardInputInto, Conv2DBackwardKernelInto), at strides 1-3, odd channel counts and
/// odd sizes, overwriting and accumulating.
/// </summary>
public class Conv1x1GemmTests
{
    public static TheoryData<int, int, int, int, int, int> Shapes => new()
    {
        // batch, inChannels, outChannels, height, width, stride
        { 2, 16, 32, 16, 16, 2 },
        { 3, 13, 21, 9, 11, 2 },
        { 2, 8, 24, 7, 7, 3 },
        { 4, 24, 16, 6, 10, 1 },
        { 2, 64, 128, 32, 32, 2 },
        { 8, 5, 3, 3, 3, 2 },
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void TheThreePassesMatchAScalarConvolution(int batch, int inChannels, int outChannels, int height, int width, int stride)
    {
        Skip.IfNot(Avx2.IsSupported && Fma.IsSupported, "The 1x1 GEMM route needs AVX2 and FMA.");
        Assert.True(Conv1x1Gemm.Applies(batch, 1, 1, 0, 0, 1, 1));
        var engine = new CpuEngine();
        var rng = new Random(batch * 1000 + inChannels * 10 + stride);
        int oh = (height - 1) / stride + 1, ow = (width - 1) / stride + 1;
        var x = Random(rng, batch, inChannels, height, width);
        var w = Random(rng, outChannels, inChannels, 1, 1);
        var dy = Random(rng, batch, outChannels, oh, ow);
        int[] s = { stride, stride }, p = { 0, 0 }, d = { 1, 1 };

        var (yRef, dxRef, dwRef) = Reference(x, w, dy, batch, inChannels, outChannels, height, width, stride, oh, ow);

        var y = new Tensor<float>(new[] { batch, outChannels, oh, ow });
        int runs = Conv1x1Gemm.t_runs;
        engine.Conv2DInto(y, x, w, s, p, d);
        AssertClose(yRef, y.ToArray(), "forward");
        // The engine sends strided 1x1 forwards here; a stride-1 forward is a plain GEMM elsewhere.
        Assert.Equal(stride > 1 ? runs + 1 : runs, Conv1x1Gemm.t_runs);

        foreach (bool accumulate in new[] { false, true })
        {
            var seedDx = Random(rng, batch, inChannels, height, width);
            var seedDw = Random(rng, outChannels, inChannels, 1, 1);
            var dx = new Tensor<float>(seedDx.ToArray(), seedDx.Shape.ToArray());
            var dw = new Tensor<float>(seedDw.ToArray(), seedDw.Shape.ToArray());
            runs = Conv1x1Gemm.t_runs;
            engine.Conv2DBackwardInputInto(dx, dy, w, x.Shape.ToArray(), s, p, d, accumulate);
            Assert.Equal(runs + 1, Conv1x1Gemm.t_runs);
            engine.Conv2DBackwardKernelInto(dw, dy, x, w.Shape.ToArray(), s, p, d, accumulate);
            Assert.Equal(runs + 2, Conv1x1Gemm.t_runs);
            var expectDx = accumulate ? dxRef.Zip(seedDx.ToArray(), (a, b) => a + b).ToArray() : dxRef;
            var expectDw = accumulate ? dwRef.Zip(seedDw.ToArray(), (a, b) => a + b).ToArray() : dwRef;
            AssertClose(expectDx, dx.ToArray(), $"input gradient (accumulate {accumulate})");
            AssertClose(expectDw, dw.ToArray(), $"weight gradient (accumulate {accumulate})");
        }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void PendingDeviceDownloadsRunBeforeTheRouteTouchesTheBuffers(bool accumulate)
    {
        Skip.IfNot(Avx2.IsSupported && Fma.IsSupported, "The 1x1 GEMM route needs AVX2 and FMA.");
        const int batch = 2, inChannels = 16, outChannels = 32, height = 16, width = 16, stride = 2;
        int oh = (height - 1) / stride + 1, ow = (width - 1) / stride + 1;
        var rng = new Random(11);
        var x = Random(rng, batch, inChannels, height, width);
        var w = Random(rng, outChannels, inChannels, 1, 1);
        var source = Random(rng, batch, outChannels, oh, ow);
        var seed = Random(rng, batch, inChannels, height, width);
        var (_, dxRef, _) = Reference(x, w, source, batch, inChannels, outChannels, height, width, stride, oh, ow);

        // dY's host array is stale (zeros) until its pending download copies the device data in. dX's pending download
        // holds its current device value: the accumulating pass adds to it, the overwriting pass replaces it (a download
        // left pending would later run and put it back over the result).
        var dy = new Tensor<float>(new[] { batch, outChannels, oh, ow });
        float[] dyHost = dy.GetCpuBackingForStridedRead(out _)!;
        float[] sourceData = source.ToArray();
        AiDotNet.Tensors.Helpers.HostSync.Register(dyHost, _ => sourceData.CopyTo(dyHost, 0));
        var dx = new Tensor<float>(new[] { batch, inChannels, height, width });
        float[] dxHost = dx.GetCpuBackingForStridedRead(out _)!;
        float[] seedData = seed.ToArray();
        AiDotNet.Tensors.Helpers.HostSync.Register(dxHost, _ => seedData.CopyTo(dxHost, 0));

        int runs = Conv1x1Gemm.t_runs;
        new CpuEngine().Conv2DBackwardInputInto(dx, dy, w, x.Shape.ToArray(), new[] { stride, stride }, new[] { 0, 0 },
            new[] { 1, 1 }, accumulate);
        Assert.Equal(runs + 1, Conv1x1Gemm.t_runs);
        var expect = accumulate ? dxRef.Zip(seedData, (p, q) => p + q).ToArray() : dxRef;
        AssertClose(expect, dx.ToArray(), $"input gradient over pending downloads (accumulate {accumulate})");
    }

    private static (float[] Y, float[] Dx, float[] Dw) Reference(Tensor<float> xt, Tensor<float> wt, Tensor<float> dyt,
        int batch, int inChannels, int outChannels, int height, int width, int stride, int oh, int ow)
    {
        float[] x = xt.ToArray(), w = wt.ToArray(), dy = dyt.ToArray();
        var y = new float[batch * outChannels * oh * ow];
        var dx = new float[batch * inChannels * height * width];
        var dw = new float[outChannels * inChannels];
        for (int n = 0; n < batch; n++)
            for (int o = 0; o < outChannels; o++)
                for (int r = 0; r < oh; r++)
                    for (int c = 0; c < ow; c++)
                    {
                        int yi = ((n * outChannels + o) * oh + r) * ow + c;
                        double sum = 0;
                        for (int i = 0; i < inChannels; i++)
                        {
                            int xi = ((n * inChannels + i) * height + r * stride) * width + c * stride;
                            sum += (double)w[o * inChannels + i] * x[xi];
                            dx[xi] += w[o * inChannels + i] * dy[yi];
                            dw[o * inChannels + i] += x[xi] * dy[yi];
                        }
                        y[yi] = (float)sum;
                    }
        return (y, dx, dw);
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float tolerance = 1e-4f * Math.Max(1f, Math.Abs(expected[i]));
            if (Math.Abs(expected[i] - actual[i]) > tolerance)
                Assert.Fail($"{what}: element {i} expected {expected[i]}, got {actual[i]}");
        }
    }

    private static Tensor<float> Random(Random rng, params int[] shape)
    {
        var data = new float[shape.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }
}
#endif
