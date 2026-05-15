using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

// Validates that ConvTranspose2DBackwardKernel — now implemented as a swap of
// input/gradOutput into Conv2DBackwardKernel — produces the same result as the
// previous 6-nested naive loop. ConvTranspose2D is the adjoint of Conv2D so
// the backward kernel reduces to the same im2col+GEMM machinery with the
// SMALL ↔ BIG spatial roles swapped. Shapes cover DCGAN's generator stack
// (4×4 stride 2 padding 1) and edge cases.
public class ConvTranspose2DBackwardKernelGemmCorrectnessTests
{
    private static Tensor<double> MakeRandomTensor(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var rng = new Random(seed);
        var span = t.Data.Span;
        for (int i = 0; i < span.Length; i++) span[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    // Inline reference replicating the old naive 6-nested-loop ConvTranspose2DBackwardKernel.
    // gradKernel[ic, oc, kh, kw] = Σ_b Σ_ih Σ_iw input[b,ic,ih,iw] * gradOutput[b,oc,ih*sH-pH+kh,iw*sW-pW+kw]
    private static double[] NaiveConvTranspose2DBackwardKernel(
        double[] gradOutput, double[] input,
        int batch, int inChannels, int height, int width,
        int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW,
        int outputHeight, int outputWidth)
    {
        var gradKernel = new double[inChannels * outChannels * kernelHeight * kernelWidth];
        for (int ic = 0; ic < inChannels; ic++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int kh = 0; kh < kernelHeight; kh++)
                {
                    for (int kw = 0; kw < kernelWidth; kw++)
                    {
                        double sum = 0.0;
                        for (int b = 0; b < batch; b++)
                        {
                            for (int ih = 0; ih < height; ih++)
                            {
                                int oh = ih * strideH - padH + kh;
                                if (oh < 0 || oh >= outputHeight) continue;
                                for (int iw = 0; iw < width; iw++)
                                {
                                    int ow = iw * strideW - padW + kw;
                                    if (ow < 0 || ow >= outputWidth) continue;
                                    int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                                    int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                    sum += gradOutput[gradOutIdx] * input[inputIdx];
                                }
                            }
                        }
                        int kernelIdx = ((ic * outChannels + oc) * kernelHeight + kh) * kernelWidth + kw;
                        gradKernel[kernelIdx] = sum;
                    }
                }
            }
        }
        return gradKernel;
    }

    [Theory]
    // DCGAN generator stack (input has small spatial, gradOutput has big spatial):
    [InlineData(1, 512, 4, 4, 256, 4, 4, 2, 1)]
    [InlineData(1, 256, 8, 8, 128, 4, 4, 2, 1)]
    [InlineData(1, 128, 16, 16, 64, 4, 4, 2, 1)]
    [InlineData(1, 64, 32, 32, 3, 4, 4, 2, 1)]
    // Edge: stride 1
    [InlineData(2, 8, 4, 4, 16, 3, 3, 1, 1)]
    // Edge: stride 2 padding 0
    [InlineData(1, 4, 3, 3, 8, 3, 3, 2, 0)]
    // Edge: kernel == stride (no overlap)
    [InlineData(1, 8, 4, 4, 8, 2, 2, 2, 0)]
    // Edge: batch=2
    [InlineData(2, 4, 4, 4, 8, 4, 4, 2, 1)]
    public void GemmPath_MatchesNaiveReference_Double(
        int batch, int inChannels, int inH, int inW,
        int outChannels, int kH, int kW, int stride, int padding)
    {
        int outH = (inH - 1) * stride - 2 * padding + kH;
        int outW = (inW - 1) * stride - 2 * padding + kW;

        var inputT = MakeRandomTensor(new[] { batch, inChannels, inH, inW }, seed: 42);
        var gradOutT = MakeRandomTensor(new[] { batch, outChannels, outH, outW }, seed: 43);

        // Naive reference
        var inputArr = inputT.Data.ToArray();
        var gradOutArr = gradOutT.Data.ToArray();
        var expected = NaiveConvTranspose2DBackwardKernel(
            gradOutArr, inputArr,
            batch, inChannels, inH, inW,
            outChannels, kH, kW,
            stride, stride, padding, padding,
            outH, outW);

        // New BLAS path via CpuEngine
        var engine = new CpuEngine();
        int[] kernelShape = new[] { inChannels, outChannels, kH, kW };
        var actualT = engine.ConvTranspose2DBackwardKernel(
            gradOutT, inputT, kernelShape,
            new[] { stride, stride }, new[] { padding, padding });

        Assert.Equal(kernelShape, actualT.Shape);

        var actualSpan = actualT.Data.Span;
        double maxDiff = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = Math.Abs(actualSpan[i] - expected[i]);
            if (d > maxDiff) maxDiff = d;
        }
        // L∞ tolerance: BLAS accumulation order differs from naive; bound by
        // n_reductions × machine_epsilon × max(magnitude).
        Assert.True(maxDiff < 1e-9,
            $"GEMM vs naive ConvTranspose2DBackwardKernel drift: maxDiff={maxDiff:E3} for shape "
            + $"[B={batch},Ci={inChannels},H={inH},W={inW}] → [Co={outChannels}] k={kH}x{kW} s={stride} p={padding}");
    }
}
