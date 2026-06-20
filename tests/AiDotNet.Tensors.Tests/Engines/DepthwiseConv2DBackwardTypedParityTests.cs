using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

// #639: the typed float/double fast paths added to DepthwiseConv2DBackwardInput/Kernel
// must produce exactly what the depthwise-conv backward math defines. We validate the
// engine output against an independent naive reference (same formula, scalar) across a
// spread of shapes incl. multiplier>1, 5x5, strided, and asymmetric H!=W.
public class DepthwiseConv2DBackwardTypedParityTests
{
    private readonly CpuEngine _engine = new CpuEngine();

    private static Tensor<float> Rnd(int[] shape, int seed)
    {
        int n = 1; foreach (var d in shape) n *= d;
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)System.Math.Sin(i * 0.7 + seed) * 0.5f;
        return new Tensor<float>(a, shape);
    }

    // Naive reference: dInput[b,ic,ih,iw] += gradOut[b,oc,oh,ow] * kernel[oc,kh,kw]
    private static float[] RefInput(float[] go, float[] k, int batch, int inC, int mult,
        int H, int W, int kH, int kW, int sH, int sW, int pH, int pW, int oH, int oW)
    {
        int outC = inC * mult;
        var gi = new float[batch * inC * H * W];
        for (int b = 0; b < batch; b++)
            for (int oc = 0; oc < outC; oc++)
            {
                int ic = oc / mult, m = oc % mult;
                for (int oh = 0; oh < oH; oh++)
                    for (int ow = 0; ow < oW; ow++)
                    {
                        float g = go[((b * outC + oc) * oH + oh) * oW + ow];
                        for (int kh = 0; kh < kH; kh++)
                            for (int kw = 0; kw < kW; kw++)
                            {
                                int ih = oh * sH + kh - pH, iw = ow * sW + kw - pW;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                gi[((b * inC + ic) * H + ih) * W + iw] +=
                                    g * k[((ic * mult + m) * kH + kh) * kW + kw];
                            }
                    }
            }
        return gi;
    }

    // Naive reference: dKernel[oc,kh,kw] = sum_b,oh,ow gradOut[b,oc,oh,ow]*input[b,ic,ih,iw]
    private static float[] RefKernel(float[] go, float[] inp, int batch, int inC, int mult,
        int H, int W, int kH, int kW, int sH, int sW, int pH, int pW, int oH, int oW)
    {
        int outC = inC * mult;
        var gk = new float[inC * mult * kH * kW];
        for (int ic = 0; ic < inC; ic++)
            for (int m = 0; m < mult; m++)
            {
                int oc = ic * mult + m;
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++)
                    {
                        float sum = 0f;
                        for (int b = 0; b < batch; b++)
                            for (int oh = 0; oh < oH; oh++)
                                for (int ow = 0; ow < oW; ow++)
                                {
                                    int ih = oh * sH + kh - pH, iw = ow * sW + kw - pW;
                                    if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                    sum += go[((b * outC + oc) * oH + oh) * oW + ow]
                                         * inp[((b * inC + ic) * H + ih) * W + iw];
                                }
                        gk[((ic * mult + m) * kH + kh) * kW + kw] = sum;
                    }
            }
        return gk;
    }

    [Theory]
    // batch, inC, mult, H, W, kH, kW, sH, sW, pH, pW
    [InlineData(1, 8, 1, 8, 8, 3, 3, 1, 1, 1, 1)]
    [InlineData(1, 16, 1, 8, 8, 5, 5, 1, 1, 2, 2)]
    [InlineData(2, 6, 2, 7, 9, 3, 3, 1, 1, 1, 1)]   // multiplier>1, H!=W, batch>1
    [InlineData(1, 12, 1, 8, 8, 3, 3, 2, 2, 1, 1)]  // stride 2
    [InlineData(1, 4, 3, 6, 6, 3, 3, 1, 1, 0, 0)]   // multiplier 3, valid pad
    public void TypedDepthwiseBackward_MatchesReference(
        int batch, int inC, int mult, int H, int W, int kH, int kW, int sH, int sW, int pH, int pW)
    {
        int oH = (H + 2 * pH - kH) / sH + 1;
        int oW = (W + 2 * pW - kW) / sW + 1;
        int outC = inC * mult;

        var gradOut = Rnd(new[] { batch, outC, oH, oW }, 1);
        var kernel = Rnd(new[] { inC, mult, kH, kW }, 2);
        var input = Rnd(new[] { batch, inC, H, W }, 3);
        var stride = new[] { sH, sW };
        var padding = new[] { pH, pW };

        var go = gradOut.GetFlattenedData();
        var k = kernel.GetFlattenedData();
        var inp = input.GetFlattenedData();

        var gi = _engine.DepthwiseConv2DBackwardInput(gradOut, kernel, new[] { batch, inC, H, W }, stride, padding);
        var gk = _engine.DepthwiseConv2DBackwardKernel(gradOut, input, new[] { inC, mult, kH, kW }, stride, padding);

        var refGi = RefInput(go, k, batch, inC, mult, H, W, kH, kW, sH, sW, pH, pW, oH, oW);
        var refGk = RefKernel(go, inp, batch, inC, mult, H, W, kH, kW, sH, sW, pH, pW, oH, oW);

        var giData = gi.GetFlattenedData();
        var gkData = gk.GetFlattenedData();
        Assert.Equal(refGi.Length, giData.Length);
        Assert.Equal(refGk.Length, gkData.Length);
        for (int i = 0; i < refGi.Length; i++)
            Assert.Equal(refGi[i], giData[i], 4);
        for (int i = 0; i < refGk.Length; i++)
            Assert.Equal(refGk[i], gkData[i], 4);
    }
}
