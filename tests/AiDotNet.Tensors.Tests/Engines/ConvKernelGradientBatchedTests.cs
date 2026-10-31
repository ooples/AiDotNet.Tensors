using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The batched float kernel gradient (one GEMM over the whole batch) computes dW = G . cols^T. With fewer output
/// channels than column rows it computes dW^T = cols . G^T instead, packing the gradient straight into G^T, so no
/// column-matrix-sized transpose is made. Both forms must equal the direct sum to float summation's bound, overwriting
/// and accumulating: 1x1 and strided 3x3 convolutions (the 3x3 stride-1 kernel has its own direct route), each with
/// fewer and with more output channels than column rows.
/// </summary>
public sealed class ConvKernelGradientBatchedTests
{
    [Theory]
    [InlineData(4, 16, 6, 6, 8, 1, 1, 0)]    // 1x1, outC 8 < colH 16: the transposed product
    [InlineData(4, 4, 6, 6, 16, 1, 1, 0)]    // 1x1, outC 16 >= colH 4: G . cols^T
    [InlineData(3, 5, 9, 7, 6, 3, 2, 1)]     // 3x3 stride 2, outC 6 < colH 45, ragged spatial size
    [InlineData(3, 2, 8, 8, 20, 3, 2, 1)]    // 3x3 stride 2, outC 20 >= colH 18
    public void EqualsTheDirectSum_OverwritingAndAccumulating(
        int batch, int inC, int h, int w, int outC, int k, int stride, int pad)
    {
        var engine = new CpuEngine();
        int oh = (h + 2 * pad - k) / stride + 1, ow = (w + 2 * pad - k) / stride + 1;
        var input = Filled(new[] { batch, inC, h, w }, 1);
        var gradOut = Filled(new[] { batch, outC, oh, ow }, 2);
        int[] kernelShape = { outC, inC, k, k }, strides = { stride, stride }, pads = { pad, pad }, dil = { 1, 1 };

        var overwrite = new Tensor<float>(kernelShape);
        engine.Conv2DBackwardKernelInto(overwrite, gradOut, input, kernelShape, strides, pads, dil, accumulate: false);
        var prior = Filled(kernelShape, 3);
        var accumulated = Filled(kernelShape, 3);
        engine.Conv2DBackwardKernelInto(accumulated, gradOut, input, kernelShape, strides, pads, dil, accumulate: true);

        for (int o = 0; o < outC; o++)
            for (int c = 0; c < inC; c++)
                for (int kh = 0; kh < k; kh++)
                    for (int kw = 0; kw < k; kw++)
                    {
                        double sum = 0, magnitude = 0;
                        for (int b = 0; b < batch; b++)
                            for (int y = 0; y < oh; y++)
                                for (int x = 0; x < ow; x++)
                                {
                                    int ih = y * stride + kh - pad, iw = x * stride + kw - pad;
                                    if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                    double term = (double)gradOut[((b * outC + o) * oh + y) * ow + x] * input[((b * inC + c) * h + ih) * w + iw];
                                    sum += term;
                                    magnitude += Math.Abs(term);
                                }
                        int i = ((o * inC + c) * k + kh) * k + kw;
                        double bound = (batch * oh * ow + 2) * Math.Pow(2, -24) * (magnitude + Math.Abs(prior[i]));
                        Assert.True(Math.Abs(sum - overwrite[i]) <= bound, $"overwrite [{i}]: {overwrite[i]} vs {sum}");
                        Assert.True(Math.Abs(prior[i] + sum - accumulated[i]) <= bound,
                            $"accumulate [{i}]: {accumulated[i]} vs {prior[i] + sum}");
                    }
    }

    private static Tensor<float> Filled(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}
