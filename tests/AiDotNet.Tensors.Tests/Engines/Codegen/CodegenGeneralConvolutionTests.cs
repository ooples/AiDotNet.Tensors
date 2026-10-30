// Copyright (c) CrowdTrainer LLC. All rights reserved.
// General convolution (#890 phase 4b): stride, dilation and groups as index maps, and their
// gradients DERIVED by the adjoint. Forward is held to a hand-written convolution; each
// derived gradient to the adjoint identity <dy, F x> = <F^T dy, x> (data) and to a direct
// sum (weights), so a dilation or group term the adjoint mis-inverts cannot agree with it.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenGeneralConvolutionTests
{
    private static double[] Fill(long count, int salt)
    {
        var v = new double[count];
        for (long i = 0; i < count; i++) v[i] = (((i * 37 + salt * 101) % 97) - 48) / 48.0;
        return v;
    }

    /// <summary>A direct NCHW convolution: y[n,k,oh,ow] = sum_{c in group, kh, kw} x * w.</summary>
    private static double[] Reference(double[] x, double[] w, int n, int c, int k, int h, int wd,
        int kernel, int stride, int pad, int dilation, int groups, out int oh, out int ow)
    {
        oh = (h + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
        ow = (wd + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
        int cg = c / groups, kg = k / groups;
        var y = new double[n * k * oh * ow];
        for (int b = 0; b < n; b++)
        for (int o = 0; o < k; o++)
        for (int i = 0; i < oh; i++)
        for (int j = 0; j < ow; j++)
        {
            int g = o / kg;
            double acc = 0;
            for (int ci = 0; ci < cg; ci++)
            for (int a = 0; a < kernel; a++)
            for (int e = 0; e < kernel; e++)
            {
                int ih = i * stride + a * dilation - pad, iw = j * stride + e * dilation - pad;
                if (ih < 0 || ih >= h || iw < 0 || iw >= wd) continue;
                acc += x[((b * c + g * cg + ci) * h + ih) * wd + iw] * w[((o * cg + ci) * kernel + a) * kernel + e];
            }
            y[((b * k + o) * oh + i) * ow + j] = acc;
        }
        return y;
    }

    [Theory]
    [InlineData(2, 6, 4, 9, 8, 3, 2, 1, 1, 1)]   // stride 2, odd extent
    [InlineData(2, 4, 4, 9, 9, 3, 1, 2, 2, 1)]   // dilation 2
    [InlineData(2, 8, 8, 7, 7, 3, 1, 1, 1, 4)]   // 4 groups
    [InlineData(1, 6, 9, 10, 9, 3, 2, 2, 2, 3)]  // all three together
    public void ForwardMatchesADirectConvolution(int n, int c, int k, int h, int w, int kernel,
        int stride, int pad, int dilation, int groups)
    {
        var spec = CodegenKernelCatalog.Conv2D(n, c, k, h, w, kernel, stride, pad, dilation, groups, epilogue: false);
        double[] x = Fill((long)n * c * h * w, 1), wt = Fill((long)k * (c / groups) * kernel * kernel, 2);
        double[] want = Reference(x, wt, n, c, k, h, w, kernel, stride, pad, dilation, groups, out _, out _);
        double[] got = spec.Interpret(new[] { x, wt });
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++) Assert.Equal(want[i], got[i], 10);
    }

    [Theory]
    [InlineData(2, 6, 4, 9, 8, 3, 2, 1, 1, 1)]
    [InlineData(2, 4, 4, 9, 9, 3, 1, 2, 2, 1)]
    [InlineData(2, 8, 8, 7, 7, 3, 1, 1, 1, 4)]
    [InlineData(1, 6, 9, 10, 9, 3, 2, 2, 2, 3)]
    public void DerivedDataGradientIsTheAdjoint(int n, int c, int k, int h, int w, int kernel,
        int stride, int pad, int dilation, int groups)
    {
        var forward = CodegenKernelCatalog.Conv2D(n, c, k, h, w, kernel, stride, pad, dilation, groups, epilogue: false);
        var backward = CodegenAdjoint.BackwardData(forward, 0);
        double[] x = Fill(forward.Inputs[0].ElementCount, 3), wt = Fill(forward.Inputs[1].ElementCount, 4);
        double[] dy = Fill(forward.Output.ElementCount, 5);

        double[] y = forward.Interpret(new[] { x, wt });
        double[] dx = backward.Interpret(new[] { dy, wt });
        Assert.Equal(x.Length, dx.Length);
        Assert.Equal(Dot(dy, y), Dot(dx, x), 8);
    }

    [Theory]
    [InlineData(2, 6, 4, 9, 8, 3, 2, 1, 1, 1)]
    [InlineData(2, 4, 4, 9, 9, 3, 1, 2, 2, 1)]
    [InlineData(2, 8, 8, 7, 7, 3, 1, 1, 1, 4)]
    public void DerivedWeightGradientIsTheAdjoint(int n, int c, int k, int h, int w, int kernel,
        int stride, int pad, int dilation, int groups)
    {
        var forward = CodegenKernelCatalog.Conv2D(n, c, k, h, w, kernel, stride, pad, dilation, groups, epilogue: false);
        var backward = CodegenAdjoint.BackwardWeights(forward, 1);
        double[] x = Fill(forward.Inputs[0].ElementCount, 6), wt = Fill(forward.Inputs[1].ElementCount, 7);
        double[] dy = Fill(forward.Output.ElementCount, 8);

        double[] y = forward.Interpret(new[] { x, wt });
        double[] dw = backward.Interpret(new[] { dy, x });   // BackwardData over the weights: (dOut, input)
        Assert.Equal(wt.Length, dw.Length);
        Assert.Equal(Dot(dy, y), Dot(dw, wt), 8);
    }

    [Fact]
    public void GroupsMustDivideBothChannelCounts() =>
        Assert.Throws<ArgumentException>(() => CodegenKernelCatalog.Conv2D(1, 6, 4, 8, 8, 3, 1, 1, 1, 4, epilogue: false));

    private static double Dot(double[] a, double[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }
}
