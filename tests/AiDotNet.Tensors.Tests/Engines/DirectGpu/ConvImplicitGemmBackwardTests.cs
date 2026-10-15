using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1313: the CUDA Conv2D weight and input gradients through the implicit-GEMM kernels (no im2col / col buffer,
/// deterministic split-K) against a float64 reference, on every ResNet-18 conv shape at batch 8 and on shapes that exercise the edges:
/// non-square input, a non-square kernel, unequal strides, dilation, and K / P / Cout that do not tile 16 or 64.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class ConvImplicitGemmBackwardTests
{
    private readonly ITestOutputHelper _out;
    public ConvImplicitGemmBackwardTests(ITestOutputHelper output) => _out = output;

    // batch, Cin, H, W, Cout, kH, kW, sH, sW, pH, pW, dH, dW
    public static System.Collections.Generic.IEnumerable<object[]> Shapes() => new[]
    {
        new object[] { 8, 3, 32, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1 },      // ResNet-18 stem
        new object[] { 8, 64, 32, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1 },     // stage 1
        new object[] { 8, 64, 32, 32, 128, 3, 3, 2, 2, 1, 1, 1, 1 },    // stage 2 entry
        new object[] { 8, 64, 32, 32, 128, 1, 1, 2, 2, 0, 0, 1, 1 },    // stage 2 shortcut
        new object[] { 8, 128, 16, 16, 128, 3, 3, 1, 1, 1, 1, 1, 1 },
        new object[] { 8, 256, 8, 8, 512, 3, 3, 2, 2, 1, 1, 1, 1 },     // stage 4 entry
        new object[] { 8, 512, 4, 4, 512, 3, 3, 1, 1, 1, 1, 1, 1 },
        new object[] { 8, 256, 8, 8, 512, 1, 1, 2, 2, 0, 0, 1, 1 },
        new object[] { 3, 5, 7, 9, 3, 3, 2, 2, 1, 1, 0, 2, 1 },         // odd: non-square, dilation, unequal strides
        new object[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1 },         // one element
        new object[] { 2, 17, 13, 11, 70, 5, 3, 1, 2, 2, 1, 1, 1 },     // P = 255, Cout = 70: partial tiles in both dims
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void WeightGradient_MatchesFloat64ReferenceAndIsBitReproducible(
        int batch, int inC, int h, int w, int outC, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        int outH = (h + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        int outW = (w + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
        Assert.True(outH > 0 && outW > 0, "invalid test shape");

        var rng = new Random(1313 + batch * 7 + inC * 3 + outC);
        var x = new float[batch * inC * h * w];
        var dy = new float[batch * outC * outH * outW];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < dy.Length; i++) dy[i] = (float)(rng.NextDouble() * 2 - 1);
        int p = inC * kH * kW;

        using var dX = backend.AllocateBuffer(x);
        using var dDy = backend.AllocateBuffer(dy);
        using var dW1 = backend.AllocateBuffer(outC * p);
        using var dW2 = backend.AllocateBuffer(outC * p);
        backend.Conv2DBackwardKernel(dX, dDy, dW1, batch, inC, h, w, outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW);
        backend.Conv2DBackwardKernel(dX, dDy, dW2, batch, inC, h, w, outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW);
        backend.Synchronize();
        float[] got = backend.DownloadBuffer(dW1), again = backend.DownloadBuffer(dW2);

        double maxRel = 0;
        for (int co = 0; co < outC; co++)
            for (int ci = 0; ci < inC; ci++)
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++)
                    {
                        double sum = 0, magnitude = 0;
                        for (int b = 0; b < batch; b++)
                            for (int oh = 0; oh < outH; oh++)
                            {
                                int ih = oh * sH - pH + kh * dH;
                                if (ih < 0 || ih >= h) continue;
                                for (int ow = 0; ow < outW; ow++)
                                {
                                    int iw = ow * sW - pW + kw * dW;
                                    if (iw < 0 || iw >= w) continue;
                                    double term = (double)dy[((b * outC + co) * outH + oh) * outW + ow]
                                                  * x[((b * inC + ci) * h + ih) * w + iw];
                                    sum += term;
                                    magnitude += Math.Abs(term);
                                }
                            }
                        int idx = ((co * inC + ci) * kH + kh) * kW + kw;
                        // fp32 accumulation over K terms: error bounded relative to the sum of magnitudes.
                        double error = Math.Abs(got[idx] - sum);
                        double bound = 1e-5 * Math.Max(1.0, magnitude);
                        maxRel = Math.Max(maxRel, error / Math.Max(1.0, magnitude));
                        Assert.True(error <= bound, $"dW[{co},{ci},{kh},{kw}] = {got[idx]} vs {sum} (|terms| {magnitude})");
                        Assert.True(BitConverter.ToInt32(BitConverter.GetBytes(got[idx]), 0) ==
                                    BitConverter.ToInt32(BitConverter.GetBytes(again[idx]), 0),
                            $"dW[{co},{ci},{kh},{kw}] differs between two identical runs: {got[idx]:R} vs {again[idx]:R}");
                    }
        _out.WriteLine($"max error / sum|terms| = {maxRel:E2}");
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void InputGradient_MatchesFloat64ReferenceAndIsBitReproducible(
        int batch, int inC, int h, int w, int outC, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        int outH = (h + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        int outW = (w + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
        Assert.True(outH > 0 && outW > 0, "invalid test shape");

        var rng = new Random(7919 + batch * 7 + inC * 3 + outC);
        var weights = new float[outC * inC * kH * kW];
        var dy = new float[batch * outC * outH * outW];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < dy.Length; i++) dy[i] = (float)(rng.NextDouble() * 2 - 1);

        // Reference by scattering each dY element through the kernel, the forward conv's own index map.
        var sum = new double[batch * inC * h * w];
        var magnitude = new double[sum.Length];
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outC; co++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        double g = dy[((b * outC + co) * outH + oh) * outW + ow];
                        for (int ci = 0; ci < inC; ci++)
                            for (int kh = 0; kh < kH; kh++)
                            {
                                int ih = oh * sH - pH + kh * dH;
                                if (ih < 0 || ih >= h) continue;
                                for (int kw = 0; kw < kW; kw++)
                                {
                                    int iw = ow * sW - pW + kw * dW;
                                    if (iw < 0 || iw >= w) continue;
                                    double term = g * weights[((co * inC + ci) * kH + kh) * kW + kw];
                                    int idx = ((b * inC + ci) * h + ih) * w + iw;
                                    sum[idx] += term;
                                    magnitude[idx] += Math.Abs(term);
                                }
                            }
                    }

        using var dDy = backend.AllocateBuffer(dy);
        using var dWt = backend.AllocateBuffer(weights);
        // Pre-filled with NaN: every element, including ones no output position reaches, must be written.
        var poison = new float[sum.Length];
        for (int i = 0; i < poison.Length; i++) poison[i] = float.NaN;
        using var dX1 = backend.AllocateBuffer(poison);
        using var dX2 = backend.AllocateBuffer(poison);
        backend.Conv2DBackwardInput(dDy, dWt, dX1, batch, inC, h, w, outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW);
        backend.Conv2DBackwardInput(dDy, dWt, dX2, batch, inC, h, w, outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW);
        backend.Synchronize();
        float[] got = backend.DownloadBuffer(dX1), again = backend.DownloadBuffer(dX2);

        double maxRel = 0;
        for (int i = 0; i < sum.Length; i++)
        {
            double error = Math.Abs(got[i] - sum[i]);
            maxRel = Math.Max(maxRel, error / Math.Max(1.0, magnitude[i]));
            Assert.True(error <= 1e-5 * Math.Max(1.0, magnitude[i]), $"dX[{i}] = {got[i]} vs {sum[i]} (|terms| {magnitude[i]})");
            Assert.True(BitConverter.ToInt32(BitConverter.GetBytes(got[i]), 0) ==
                        BitConverter.ToInt32(BitConverter.GetBytes(again[i]), 0),
                $"dX[{i}] differs between two identical runs: {got[i]:R} vs {again[i]:R}");
        }
        _out.WriteLine($"max error / sum|terms| = {maxRel:E2}");
    }
}
