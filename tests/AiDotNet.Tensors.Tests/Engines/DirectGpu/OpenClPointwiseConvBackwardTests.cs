using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// OpenCL computes the gradients of a 1 x 1, stride-1, unpadded convolution as one GEMM over the batch with layout
/// passes: dX = W^T . dY and dW = dY . X^T per image, summed over the batch for dW. Both must equal the float64 direct
/// sums to float summation's bound on ragged shapes (channels and spatial sizes off any tile), and the candidate must
/// admit exactly that shape class.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class OpenClPointwiseConvBackwardTests
{
    private static float[] Rand(int seed, int n)
    {
        var rng = new Random(seed);
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() * 2 - 1);
        return d;
    }

    [SkippableTheory]
    [InlineData(3, 5, 7, 4, 6)]
    [InlineData(2, 16, 8, 8, 8)]
    [InlineData(1, 8, 33, 3, 3)]
    public void Gradients_EqualTheDirectSums(int batch, int ci, int co, int h, int w)
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        int hw = h * w;
        var x = Rand(1, batch * ci * hw);
        var weights = Rand(2, co * ci);
        var dy = Rand(3, batch * co * hw);

        using var xBuf = backend.AllocateBuffer(x);
        using var wBuf = backend.AllocateBuffer(weights);
        using var dyBuf = backend.AllocateBuffer(dy);
        using var dxBuf = backend.AllocateBuffer(batch * ci * hw);
        using var dwBuf = backend.AllocateBuffer(co * ci);
        var dataArgs = new TunedConv2DBackwardArgs(dyBuf, wBuf, dxBuf, false, batch, ci, h, w, co, h, w, 1, 1, 1, 1, 0, 0, 1, 1);
        var filterArgs = new TunedConv2DBackwardArgs(xBuf, dyBuf, dwBuf, true, batch, ci, h, w, co, h, w, 1, 1, 1, 1, 0, 0, 1, 1);
        Assert.True(OpenClBackend.AdmitsPointwiseGemm(dataArgs.Shape()));
        backend.Conv2DBackwardInputPointwiseGemm(dataArgs);
        backend.Conv2DBackwardFilterPointwiseGemm(filterArgs);
        var dx = backend.DownloadBuffer(dxBuf);
        var dw = backend.DownloadBuffer(dwBuf);

        for (int b = 0; b < batch; b++)
            for (int c = 0; c < ci; c++)
                for (int p = 0; p < hw; p++)
                {
                    double sum = 0, mag = 0;
                    for (int o = 0; o < co; o++)
                    {
                        double t = (double)weights[o * ci + c] * dy[(b * co + o) * hw + p];
                        sum += t; mag += Math.Abs(t);
                    }
                    float got = dx[(b * ci + c) * hw + p];
                    Assert.True(Math.Abs(sum - got) <= (co + 1) * Math.Pow(2, -23) * mag + 1e-7, $"dX[{b},{c},{p}]: {got} vs {sum}");
                }
        for (int o = 0; o < co; o++)
            for (int c = 0; c < ci; c++)
            {
                double sum = 0, mag = 0;
                for (int b = 0; b < batch; b++)
                    for (int p = 0; p < hw; p++)
                    {
                        double t = (double)dy[(b * co + o) * hw + p] * x[(b * ci + c) * hw + p];
                        sum += t; mag += Math.Abs(t);
                    }
                float got = dw[o * ci + c];
                Assert.True(Math.Abs(sum - got) <= (batch * hw + 1) * Math.Pow(2, -23) * mag + 1e-7, $"dW[{o},{c}]: {got} vs {sum}");
            }
    }

    // Any convolution through the column matrix (im2col / col2im around one GEMM per batch chunk). The last case has
    // 576 x 1024 columns per image, so its 32 images run in two chunks.
    [SkippableTheory]
    [InlineData(3, 4, 6, 7, 5, 3, 1, 1, 1)]     // 3x3 stride 1, ragged sizes
    [InlineData(2, 5, 4, 9, 9, 3, 2, 1, 1)]     // 3x3 stride 2
    [InlineData(2, 3, 8, 12, 12, 7, 2, 3, 1)]   // a 7x7 stride-2 stem
    [InlineData(2, 4, 4, 9, 9, 3, 1, 2, 2)]     // dilation 2
    [InlineData(32, 64, 4, 32, 32, 3, 1, 1, 1)] // chunked
    public void ColumnGemm_EqualsTheDirectSums(int batch, int ci, int co, int h, int w, int k, int stride, int pad, int dil)
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        int oh = (h + 2 * pad - dil * (k - 1) - 1) / stride + 1, ow = (w + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        var x = Rand(4, batch * ci * h * w);
        var weights = Rand(5, co * ci * k * k);
        var dy = Rand(6, batch * co * oh * ow);
        using var xBuf = backend.AllocateBuffer(x);
        using var wBuf = backend.AllocateBuffer(weights);
        using var dyBuf = backend.AllocateBuffer(dy);
        using var dxBuf = backend.AllocateBuffer(batch * ci * h * w);
        using var dwBuf = backend.AllocateBuffer(co * ci * k * k);
        var dataArgs = new TunedConv2DBackwardArgs(dyBuf, wBuf, dxBuf, false, batch, ci, h, w, co, oh, ow, k, k, stride, stride, pad, pad, dil, dil);
        var filterArgs = new TunedConv2DBackwardArgs(xBuf, dyBuf, dwBuf, true, batch, ci, h, w, co, oh, ow, k, k, stride, stride, pad, pad, dil, dil);
        Assert.True(OpenClBackend.AdmitsColumnGemm(dataArgs.Shape()));
        backend.Conv2DBackwardInputColumnGemm(dataArgs);
        backend.Conv2DBackwardFilterColumnGemm(filterArgs);
        var dx = backend.DownloadBuffer(dxBuf);
        var dw = backend.DownloadBuffer(dwBuf);

        var refDx = new double[dx.Length];
        var magDx = new double[dx.Length];
        var refDw = new double[dw.Length];
        var magDw = new double[dw.Length];
        for (int b = 0; b < batch; b++)
            for (int o = 0; o < co; o++)
                for (int y = 0; y < oh; y++)
                    for (int xo = 0; xo < ow; xo++)
                    {
                        double g = dy[((b * co + o) * oh + y) * ow + xo];
                        for (int c = 0; c < ci; c++)
                            for (int kh = 0; kh < k; kh++)
                                for (int kw = 0; kw < k; kw++)
                                {
                                    int ih = y * stride - pad + kh * dil, iw = xo * stride - pad + kw * dil;
                                    if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                    int xi = ((b * ci + c) * h + ih) * w + iw, wi = ((o * ci + c) * k + kh) * k + kw;
                                    refDx[xi] += g * weights[wi]; magDx[xi] += Math.Abs(g * weights[wi]);
                                    refDw[wi] += g * x[xi]; magDw[wi] += Math.Abs(g * x[xi]);
                                }
                    }
        double ulp = Math.Pow(2, -23);
        for (int i = 0; i < dx.Length; i++)
            Assert.True(Math.Abs(refDx[i] - dx[i]) <= (co * k * k + 2) * ulp * magDx[i] + 1e-7, $"dX[{i}]: {dx[i]} vs {refDx[i]}");
        for (int i = 0; i < dw.Length; i++)
            Assert.True(Math.Abs(refDw[i] - dw[i]) <= (batch * oh * ow + 2) * ulp * magDw[i] + 1e-6, $"dW[{i}]: {dw[i]} vs {refDw[i]}");
    }
    [Fact]
    public void Admits_OnlyPointwiseStrideOneUnpaddedShapes()
    {
        // TunedConv2DBackwardArgs.Shape()'s layout: batch, Ci, H, W, Co, kH, kW, sH, sW, pH, pW, dH, dW, outH, outW.
        TunedShape Shape(int k, int s, int p, int dil)
        {
            int o = (8 + 2 * p - dil * (k - 1) - 1) / s + 1;
            return TunedShape.Create(TunedKernelDType.Float32, new[] { 2, 4, 8, 8, 4, k, k, s, s, p, p, dil, dil, o, o });
        }
        Assert.True(OpenClBackend.AdmitsPointwiseGemm(Shape(1, 1, 0, 1)));
        Assert.False(OpenClBackend.AdmitsPointwiseGemm(Shape(3, 1, 1, 1)));
        Assert.False(OpenClBackend.AdmitsPointwiseGemm(Shape(1, 2, 0, 1)));
        Assert.False(OpenClBackend.AdmitsPointwiseGemm(Shape(1, 1, 1, 1)));
    }
}
