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
        engine.Conv2DInto(y, x, w, s, p, d);
        AssertClose(yRef, y.ToArray(), "forward");

        foreach (bool accumulate in new[] { false, true })
        {
            var seedDx = Random(rng, batch, inChannels, height, width);
            var seedDw = Random(rng, outChannels, inChannels, 1, 1);
            var dx = new Tensor<float>(seedDx.ToArray(), seedDx.Shape.ToArray());
            var dw = new Tensor<float>(seedDw.ToArray(), seedDw.Shape.ToArray());
            engine.Conv2DBackwardInputInto(dx, dy, w, x.Shape.ToArray(), s, p, d, accumulate);
            engine.Conv2DBackwardKernelInto(dw, dy, x, w.Shape.ToArray(), s, p, d, accumulate);
            var expectDx = accumulate ? dxRef.Zip(seedDx.ToArray(), (a, b) => a + b).ToArray() : dxRef;
            var expectDw = accumulate ? dwRef.Zip(seedDw.ToArray(), (a, b) => a + b).ToArray() : dwRef;
            AssertClose(expectDx, dx.ToArray(), $"input gradient (accumulate {accumulate})");
            AssertClose(expectDw, dw.ToArray(), $"weight gradient (accumulate {accumulate})");
        }
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
