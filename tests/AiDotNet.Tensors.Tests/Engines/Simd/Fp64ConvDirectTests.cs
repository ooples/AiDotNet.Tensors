using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Stage 5 (#415) parity tests for the new FP64 Conv2D direct kernels:
/// Conv1x1Stride1Double, Conv3x3Stride2Double, Conv7x7Stride2Double.
/// Each test compares the SIMD direct path against the engine's
/// im2col+Dgemm reference path. Tolerance 1e-10 absolute — these are
/// pure FMA chains; reordering across IC/kw could perturb the last
/// few ulps but nothing beyond.
/// </summary>
public class Fp64ConvDirectTests
{
    private static Tensor<double> RandomTensor(int[] shape, int seed)
    {
        var rng = new System.Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() - 0.5;
        return t;
    }

    [Theory]
    [InlineData(1, 64, 56, 256)]     // ResNet bottleneck pointwise: [1,64,56,56] × [256,64,1,1]
    [InlineData(1, 256, 28, 64)]     // bottleneck reduce
    [InlineData(2, 128, 14, 128)]    // batch=2 mid layer
    [InlineData(1, 2048, 7, 1000)]   // ResNet head FC-as-conv would not normally hit this, but exercises wide N
    public void Conv1x1Stride1Double_MatchesIm2ColReference(int batch, int inC, int hw, int outC)
    {
        var engine = new CpuEngine();
        var input = RandomTensor(new[] { batch, inC, hw, hw }, 42);
        var kernel = RandomTensor(new[] { outC, inC, 1, 1 }, 7);

        // Direct path (Conv1x1Stride1Double routed via Conv2DDirectDouble).
        var direct = engine.Conv2D(input, kernel, stride: 1, padding: 0, dilation: 1);

        // Reference: pad=0 & stride=1 means im2col(1×1) is identity, so we
        // can compare against a hand-rolled per-pixel matmul.
        var expected = new Tensor<double>(new[] { batch, outC, hw, hw });
        int spatial = hw * hw;
        for (int b = 0; b < batch; b++)
            for (int oc = 0; oc < outC; oc++)
                for (int s = 0; s < spatial; s++)
                {
                    double sum = 0;
                    for (int ic = 0; ic < inC; ic++)
                        sum += input[b * inC * spatial + ic * spatial + s] * kernel[oc * inC + ic];
                    expected[b * outC * spatial + oc * spatial + s] = sum;
                }

        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - direct[i]) < 1e-10,
                $"[{i}] expected={expected[i]:F12} actual={direct[i]:F12}");
    }

    [Theory]
    [InlineData(1, 8, 32, 16, 1)]    // small stride=2 with padding
    [InlineData(1, 16, 28, 32, 1)]
    [InlineData(1, 32, 14, 64, 1)]
    [InlineData(2, 8, 16, 16, 0)]    // no padding
    public void Conv3x3Stride2Double_MatchesScalarReference(int batch, int inC, int hw, int outC, int padding)
    {
        var engine = new CpuEngine();
        var input = RandomTensor(new[] { batch, inC, hw, hw }, 11);
        var kernel = RandomTensor(new[] { outC, inC, 3, 3 }, 22);

        var direct = engine.Conv2D(input, kernel, stride: 2, padding: padding, dilation: 1);

        int outHW = (hw + 2 * padding - 3) / 2 + 1;
        var expected = new Tensor<double>(new[] { batch, outC, outHW, outHW });
        for (int b = 0; b < batch; b++)
            for (int oc = 0; oc < outC; oc++)
                for (int oh = 0; oh < outHW; oh++)
                    for (int ow = 0; ow < outHW; ow++)
                    {
                        double sum = 0;
                        for (int ic = 0; ic < inC; ic++)
                            for (int kh = 0; kh < 3; kh++)
                                for (int kw = 0; kw < 3; kw++)
                                {
                                    int ih = oh * 2 + kh - padding;
                                    int iw = ow * 2 + kw - padding;
                                    if (ih < 0 || ih >= hw || iw < 0 || iw >= hw) continue;
                                    sum += input[((b * inC + ic) * hw + ih) * hw + iw]
                                         * kernel[((oc * inC + ic) * 3 + kh) * 3 + kw];
                                }
                        expected[((b * outC + oc) * outHW + oh) * outHW + ow] = sum;
                    }

        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - direct[i]) < 1e-10,
                $"[{i}] expected={expected[i]:F12} actual={direct[i]:F12}");
    }

    [Theory]
    [InlineData(1, 3, 28, 8, 3)]      // small ResNet50-stem-like
    [InlineData(1, 3, 32, 16, 3)]
    [InlineData(2, 3, 24, 8, 2)]      // batch=2, padding=2
    public void Conv7x7Stride2Double_MatchesScalarReference(int batch, int inC, int hw, int outC, int padding)
    {
        var engine = new CpuEngine();
        var input = RandomTensor(new[] { batch, inC, hw, hw }, 33);
        var kernel = RandomTensor(new[] { outC, inC, 7, 7 }, 44);

        var direct = engine.Conv2D(input, kernel, stride: 2, padding: padding, dilation: 1);

        int outHW = (hw + 2 * padding - 7) / 2 + 1;
        var expected = new Tensor<double>(new[] { batch, outC, outHW, outHW });
        for (int b = 0; b < batch; b++)
            for (int oc = 0; oc < outC; oc++)
                for (int oh = 0; oh < outHW; oh++)
                    for (int ow = 0; ow < outHW; ow++)
                    {
                        double sum = 0;
                        for (int ic = 0; ic < inC; ic++)
                            for (int kh = 0; kh < 7; kh++)
                                for (int kw = 0; kw < 7; kw++)
                                {
                                    int ih = oh * 2 + kh - padding;
                                    int iw = ow * 2 + kw - padding;
                                    if (ih < 0 || ih >= hw || iw < 0 || iw >= hw) continue;
                                    sum += input[((b * inC + ic) * hw + ih) * hw + iw]
                                         * kernel[((oc * inC + ic) * 7 + kh) * 7 + kw];
                                }
                        expected[((b * outC + oc) * outHW + oh) * outHW + ow] = sum;
                    }

        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - direct[i]) < 1e-9,
                $"[{i}] expected={expected[i]:F12} actual={direct[i]:F12}");
    }
}
