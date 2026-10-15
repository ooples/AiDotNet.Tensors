using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// conv2d_implicit_gemm (CudaBackend.TryConv2DImplicitGemm): a narrow convolution as a GEMM whose im2col matrix is
/// gathered from the input inside the kernel. Every case asserts the kernel engaged -- a declined shape would fall
/// back to the other kernels and prove nothing -- and compares every output with a double-precision reference.
/// </summary>
public class CudaImplicitConvTests
{
    [SkippableTheory]
    [InlineData(64, 16, 14, 14, 32, 3, 3, 1, 1, 1, 1, 1, 1)]   // the parity CNN's second layer
    [InlineData(20, 5, 17, 13, 20, 3, 3, 2, 2, 1, 1, 1, 1)]    // stride 2; pixels and channels past tile edges
    [InlineData(3, 8, 20, 20, 64, 3, 3, 1, 1, 2, 2, 2, 2)]     // dilation 2, padding 2, two channel tiles
    [InlineData(8, 4, 16, 24, 7, 5, 3, 1, 2, 0, 1, 1, 1)]      // rectangular kernel and stride, no vertical padding
    [InlineData(17, 64, 8, 8, 33, 1, 1, 1, 1, 0, 0, 1, 1)]     // 1x1 kernel, reduction = channels
    public void ImplicitGemmConv_EqualsTheReference(int batch, int inC, int inH, int inW, int outC,
        int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        int outH = (inH + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        int outW = (inW + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
        var random = new Random(inC * 31 + outC);
        float[] x = Enumerable.Range(0, batch * inC * inH * inW).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        float[] w = Enumerable.Range(0, outC * inC * kH * kW).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        using var xBuf = backend.AllocateBuffer(x);
        using var wBuf = backend.AllocateBuffer(w);
        using var yBuf = backend.AllocateBuffer(Enumerable.Repeat(float.NaN, batch * outC * outH * outW).ToArray());

        Assert.True(backend.TryConv2DImplicitGemm(xBuf, wBuf, yBuf, batch, inC, inH, inW, outC, outH, outW,
            kH, kW, sH, sW, pH, pW, dH, dW), "conv2d_implicit_gemm declined the shape");

        var y = backend.DownloadBuffer(yBuf);
        for (int n = 0; n < batch; n++)
            for (int oc = 0; oc < outC; oc++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        double s = 0, bound = 0;
                        for (int ic = 0; ic < inC; ic++)
                            for (int r = 0; r < kH; r++)
                                for (int c = 0; c < kW; c++)
                                {
                                    int ih = oh * sH - pH + r * dH, iw = ow * sW - pW + c * dW;
                                    if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) continue;
                                    double p = (double)x[((n * inC + ic) * inH + ih) * inW + iw] * w[((oc * inC + ic) * kH + r) * kW + c];
                                    s += p; bound += Math.Abs(p);
                                }
                        float got = y[((n * outC + oc) * outH + oh) * outW + ow];
                        if (!(Math.Abs(got - s) <= 1e-5 * bound + 1e-6))
                            Assert.Fail($"y[{n},{oc},{oh},{ow}] = {got}, expected {s}");
                    }
    }

    /// <summary>conv2d_weight_grad_implicit against the double reference dW[oc,ic,r,s] = sum over (n, oh, ow) of
    /// dY[n,oc,oh,ow] * X[n,ic,oh*sH-pH+r*dH, ow*sW-pW+s*dW]. Each shape runs twice with different dY: the kernel's
    /// last block per tile resets the tile's counter, so a second launch must be complete too.</summary>
    [SkippableTheory]
    [InlineData(64, 16, 14, 14, 32, 3, 3, 1, 1, 1, 1, 1, 1)]   // the parity CNN's second layer: 3 tiles
    [InlineData(70, 5, 17, 13, 20, 3, 3, 2, 2, 1, 1, 1, 1)]    // stride 2; a partial channel tile
    [InlineData(12, 8, 20, 20, 64, 3, 3, 1, 1, 2, 2, 2, 2)]    // dilation 2, padding 2; two channel tiles
    [InlineData(32, 4, 16, 24, 7, 5, 3, 1, 2, 0, 1, 1, 1)]     // rectangular kernel and stride
    public void ImplicitGemmConvWeightGradient_EqualsTheReference(int batch, int inC, int inH, int inW, int outC,
        int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        int outH = (inH + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        int outW = (inW + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
        var random = new Random(inC * 13 + outC);
        float[] x = Enumerable.Range(0, batch * inC * inH * inW).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        using var xBuf = backend.AllocateBuffer(x);
        using var dwBuf = backend.AllocateBuffer(outC * inC * kH * kW);
        for (int launch = 0; launch < 2; launch++)
        {
            float[] dy = Enumerable.Range(0, batch * outC * outH * outW).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            using var dyBuf = backend.AllocateBuffer(dy);
            backend.UploadBufferInPlace(Enumerable.Repeat(float.NaN, outC * inC * kH * kW).ToArray(), dwBuf);
            Assert.True(backend.TryConv2DBackwardKernelImplicitGemm(xBuf, dyBuf, dwBuf, batch, inC, inH, inW, outC, outH, outW,
                kH, kW, sH, sW, pH, pW, dH, dW), "conv2d_weight_grad_implicit declined the shape");
            var dwGot = backend.DownloadBuffer(dwBuf);
            for (int oc = 0; oc < outC; oc++)
                for (int ic = 0; ic < inC; ic++)
                    for (int r = 0; r < kH; r++)
                        for (int q = 0; q < kW; q++)
                        {
                            double s = 0, bound = 0;
                            for (int n = 0; n < batch; n++)
                                for (int oh = 0; oh < outH; oh++)
                                    for (int ow = 0; ow < outW; ow++)
                                    {
                                        int ih = oh * sH - pH + r * dH, iw = ow * sW - pW + q * dW;
                                        if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) continue;
                                        double p = (double)dy[((n * outC + oc) * outH + oh) * outW + ow] * x[((n * inC + ic) * inH + ih) * inW + iw];
                                        s += p; bound += Math.Abs(p);
                                    }
                            float got = dwGot[((oc * inC + ic) * kH + r) * kW + q];
                            if (!(Math.Abs(got - s) <= 1e-5 * bound + 1e-6))
                                Assert.Fail($"launch {launch}: dW[{oc},{ic},{r},{q}] = {got}, expected {s}");
                        }
        }
    }
    [SkippableTheory]
    [InlineData(64, 1, 28, 28, 16)]    // reduction 9: Winograd serves it better
    [InlineData(64, 16, 14, 14, 128)]  // more output channels than the kernel takes
    [InlineData(1, 16, 8, 8, 32)]      // too few output pixels
    public void ImplicitGemmConv_DeclinesShapesItDoesNotServe(int batch, int inC, int inH, int inW, int outC)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        using var xBuf = backend.AllocateBuffer(batch * inC * inH * inW);
        using var wBuf = backend.AllocateBuffer(outC * inC * 9);
        using var yBuf = backend.AllocateBuffer(batch * outC * inH * inW);
        Assert.False(backend.TryConv2DImplicitGemm(xBuf, wBuf, yBuf, batch, inC, inH, inW, outC, inH, inW,
            3, 3, 1, 1, 1, 1, 1, 1));
    }
}
