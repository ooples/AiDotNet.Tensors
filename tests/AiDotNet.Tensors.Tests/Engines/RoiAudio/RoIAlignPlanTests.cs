// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.RoiAudio;

/// <summary>
/// RoIAlign forward and backward run from precomputed bilinear taps, in parallel. These tests compare them with a
/// direct per-sample rendering of torchvision's roi_align on several images, channels and overlapping RoIs, for both
/// <c>aligned</c> settings and both sampling-ratio rules.
/// </summary>
public class RoIAlignPlanTests
{
    private const int N = 2, C = 3, H = 7, W = 9, OutH = 3, OutW = 2;

    // [batch, x1, y1, x2, y2]: overlapping boxes on image 0, one on image 1, one past the right and bottom edges, and
    // one smaller than a pixel.
    private static readonly double[] Boxes =
    {
        0, 0.5, 0.25, 6.0, 5.5,
        0, 2.0, 1.0, 7.5, 6.0,
        1, 1.25, 0.75, 4.0, 3.5,
        1, 5.0, 4.0, 11.0, 9.5,
        0, 3.1, 2.2, 3.4, 2.6,
    };

    private static double[] Image() =>
        Enumerable.Range(0, N * C * H * W).Select(i => Math.Sin(i * 0.37) + (i % 11) * 0.05).ToArray();

    private static double Sample(double[] x, int plane, double y, double xx)
    {
        if (y < -1.0 || y > H || xx < -1.0 || xx > W) return 0;
        if (y <= 0) y = 0;
        if (xx <= 0) xx = 0;
        int y0 = (int)y, x0 = (int)xx, y1, x1;
        if (y0 >= H - 1) { y0 = y1 = H - 1; y = y0; } else y1 = y0 + 1;
        if (x0 >= W - 1) { x0 = x1 = W - 1; xx = x0; } else x1 = x0 + 1;
        double ly = y - y0, lx = xx - x0;
        return (1 - ly) * (1 - lx) * x[plane + (y0 * W) + x0] + (1 - ly) * lx * x[plane + (y0 * W) + x1]
            + ly * (1 - lx) * x[plane + (y1 * W) + x0] + ly * lx * x[plane + (y1 * W) + x1];
    }

    // torchvision's roi_align, one output at a time. The weights of the samples are what the backward scatters, so the
    // reference gradient of sum(output * g) is read off the same rule by linearity (see ReferenceGradient).
    private static double[] Reference(double[] x, float scale, int ratio, bool aligned)
    {
        int k = Boxes.Length / 5;
        var output = new double[k * C * OutH * OutW];
        double shift = aligned ? 0.5 : 0;
        for (int r = 0; r < k; r++)
        {
            int n = (int)Boxes[r * 5];
            double x1 = Boxes[(r * 5) + 1] * scale - shift, y1 = Boxes[(r * 5) + 2] * scale - shift;
            double x2 = Boxes[(r * 5) + 3] * scale - shift, y2 = Boxes[(r * 5) + 4] * scale - shift;
            double rw = aligned ? x2 - x1 : Math.Max(x2 - x1, 1), rh = aligned ? y2 - y1 : Math.Max(y2 - y1, 1);
            double bh = rh / OutH, bw = rw / OutW;
            int gy = Math.Max(1, ratio > 0 ? ratio : (int)Math.Ceiling(rh / OutH));
            int gx = Math.Max(1, ratio > 0 ? ratio : (int)Math.Ceiling(rw / OutW));
            for (int c = 0; c < C; c++)
            for (int ph = 0; ph < OutH; ph++)
            for (int pw = 0; pw < OutW; pw++)
            {
                double sum = 0;
                for (int iy = 0; iy < gy; iy++)
                for (int ix = 0; ix < gx; ix++)
                    sum += Sample(x, (n * C + c) * H * W, y1 + ph * bh + (iy + 0.5) * bh / gy, x1 + pw * bw + (ix + 0.5) * bw / gx);
                output[(((r * C) + c) * OutH + ph) * OutW + pw] = sum / (gy * gx);
            }
        }

        return output;
    }

    // RoIAlign is linear in its input, so d/dx_i sum(output * g) = sum(Reference(e_i) * g).
    private static double[] ReferenceGradient(double[] g, float scale, int ratio, bool aligned)
    {
        var gradient = new double[N * C * H * W];
        var basis = new double[gradient.Length];
        for (int i = 0; i < gradient.Length; i++)
        {
            basis[i] = 1;
            var output = Reference(basis, scale, ratio, aligned);
            double sum = 0;
            for (int j = 0; j < output.Length; j++) sum += output[j] * g[j];
            gradient[i] = sum;
            basis[i] = 0;
        }

        return gradient;
    }

    [Theory]
    [InlineData(false, 2, 1.0f)]
    [InlineData(false, 0, 1.0f)]
    [InlineData(true, 2, 1.0f)]
    [InlineData(true, 0, 0.75f)]
    public void Forward_matches_the_per_sample_rule(bool aligned, int ratio, float scale)
    {
        var image = Image();
        var input = new Tensor<double>(image, new[] { N, C, H, W });
        var boxes = new Tensor<double>(Boxes, new[] { Boxes.Length / 5, 5 });
        var actual = new CpuEngine().RoIAlign(input, boxes, OutH, OutW, scale, ratio, aligned).ToArray();
        var expected = Reference(image, scale, ratio, aligned);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12, $"[{i}] reference {expected[i]} vs {actual[i]}");
    }

    [Fact]
    public void Float_forward_matches_the_per_sample_rule()
    {
        var image = Image();
        var input = new Tensor<float>(image.Select(v => (float)v).ToArray(), new[] { N, C, H, W });
        var boxes = new Tensor<float>(Boxes.Select(v => (float)v).ToArray(), new[] { Boxes.Length / 5, 5 });
        var actual = new CpuEngine().RoIAlign(input, boxes, OutH, OutW, 1.0f, 2, false).ToArray();
        var expected = Reference(image.Select(v => (double)(float)v).ToArray(), 1.0f, 2, false);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5 * (1 + Math.Abs(expected[i])), $"[{i}] reference {expected[i]} vs {actual[i]}");
    }

    [Theory]
    [InlineData(false, 2, 1.0f)]
    [InlineData(true, 0, 0.75f)]
    public void Backward_matches_the_adjoint_of_the_per_sample_rule(bool aligned, int ratio, float scale)
    {
        var engine = new CpuEngine();
        var input = new Tensor<double>(Image(), new[] { N, C, H, W });
        var boxes = new Tensor<double>(Boxes, new[] { Boxes.Length / 5, 5 });
        int outputs = Boxes.Length / 5 * C * OutH * OutW;
        var g = Enumerable.Range(0, outputs).Select(i => Math.Cos(i * 0.21)).ToArray();
        var weights = new Tensor<double>(g, new[] { Boxes.Length / 5, C, OutH, OutW });

        double[] actual;
        using (var tape = new GradientTape<double>())
        {
            var output = engine.RoIAlign(input, boxes, OutH, OutW, scale, ratio, aligned);
            var loss = engine.ReduceSum(engine.TensorMultiply(output, weights), new[] { 0, 1, 2, 3 }, keepDims: false);
            var grads = tape.ComputeGradients(loss, new[] { input });
            actual = grads[input].ToArray();
        }

        var expected = ReferenceGradient(g, scale, ratio, aligned);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12, $"[{i}] reference {expected[i]} vs {actual[i]}");
    }
}
