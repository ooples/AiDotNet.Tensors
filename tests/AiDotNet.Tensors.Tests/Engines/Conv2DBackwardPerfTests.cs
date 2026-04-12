using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Performance tests proving Conv2DBackwardInput/BackwardKernel im2col+GEMM
/// is significantly faster than naive loops. Gated by Category=Performance
/// trait on timing methods only — correctness tests always run.
/// </summary>
public class Conv2DBackwardPerfTests
{
    private readonly ITestOutputHelper _output;
    private readonly CpuEngine _engine = new();

    public Conv2DBackwardPerfTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Naive float Conv2DBackwardInput using nested loops — same algorithm as the
    /// generic T fallback but operating on float[] directly. This isolates the
    /// algorithm speedup (im2col+GEMM vs loops) from any data-type differences.
    /// </summary>
    private static float[] NaiveConv2DBackwardInputFloat(
        float[] gradOutputData, int[] gradOutputShape,
        float[] kernelData, int[] kernelShape,
        int[] inputShape, int[] stride, int[] padding, int[] dilation)
    {
        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];
        int outChannels = kernelShape[0];
        int kernelHeight = kernelShape[2];
        int kernelWidth = kernelShape[3];
        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];
        int outputHeight = gradOutputShape[2];
        int outputWidth = gradOutputShape[3];

        var gradInput = new float[batch * inChannels * height * width];

        for (int b = 0; b < batch; b++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                            float gradVal = gradOutputData[gradOutIdx];

                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int ih = oh * strideH + kh * dilationH - padH;
                                    int iw = ow * strideW + kw * dilationW - padW;

                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                    {
                                        int gradInputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                        int kernelIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;
                                        gradInput[gradInputIdx] += gradVal * kernelData[kernelIdx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return gradInput;
    }

    /// <summary>
    /// Naive float Conv2DBackwardKernel using nested loops — same algorithm as the
    /// generic T fallback but operating on float[] directly.
    /// </summary>
    private static float[] NaiveConv2DBackwardKernelFloat(
        float[] gradOutputData, int[] gradOutputShape,
        float[] inputData, int[] inputShape,
        int[] kernelShape, int[] stride, int[] padding, int[] dilation)
    {
        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];
        int outChannels = kernelShape[0];
        int kernelHeight = kernelShape[2];
        int kernelWidth = kernelShape[3];
        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];
        int outputHeight = gradOutputShape[2];
        int outputWidth = gradOutputShape[3];

        var gradKernel = new float[outChannels * inChannels * kernelHeight * kernelWidth];

        for (int oc = 0; oc < outChannels; oc++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                for (int kh = 0; kh < kernelHeight; kh++)
                {
                    for (int kw = 0; kw < kernelWidth; kw++)
                    {
                        float sum = 0f;

                        for (int b = 0; b < batch; b++)
                        {
                            for (int oh = 0; oh < outputHeight; oh++)
                            {
                                for (int ow = 0; ow < outputWidth; ow++)
                                {
                                    int ih = oh * strideH + kh * dilationH - padH;
                                    int iw = ow * strideW + kw * dilationW - padW;

                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                    {
                                        int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                                        int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                        sum += gradOutputData[gradOutIdx] * inputData[inputIdx];
                                    }
                                }
                            }
                        }

                        int kernelIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;
                        gradKernel[kernelIdx] = sum;
                    }
                }
            }
        }

        return gradKernel;
    }

    [Theory]
    [Trait("Category", "Performance")]
    [InlineData(4, 8, 16, 32, 32, 3, 3)]   // Small shape
    [InlineData(4, 64, 64, 56, 56, 3, 3)]  // ResNet50-like shape (from issue #148)
    public void Conv2DBackwardInput_FloatFastPath_IsFasterThanNaiveFloatPath(
        int batch, int inC, int outC, int h, int w, int kH, int kW)
    {
        int outH = h - kH + 1;
        int outW = w - kW + 1;

        var rng = new Random(42);
        var gradOutput = new Tensor<float>([batch, outC, outH, outW]);
        var kernel = new Tensor<float>([outC, inC, kH, kW]);
        for (int i = 0; i < gradOutput.Length; i++) gradOutput[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kernel.Length; i++) kernel[i] = (float)(rng.NextDouble() * 2 - 1);

        var inputShape = new[] { batch, inC, h, w };
        var stride = new[] { 1, 1 };
        var padding = new[] { 0, 0 };
        var dilation = new[] { 1, 1 };

        // Extract raw float arrays for the naive baseline
        var gradOutputF = new float[gradOutput.Length];
        for (int i = 0; i < gradOutput.Length; i++) gradOutputF[i] = gradOutput[i];
        var kernelF = new float[kernel.Length];
        for (int i = 0; i < kernel.Length; i++) kernelF[i] = kernel[i];
        var gradOutputShape = new[] { batch, outC, outH, outW };
        var kernelShape = new[] { outC, inC, kH, kW };

        // Warmup both paths
        _engine.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);
        NaiveConv2DBackwardInputFloat(gradOutputF, gradOutputShape, kernelF, kernelShape, inputShape, stride, padding, dilation);

        // Time the float im2col+GEMM fast path
        var sw = Stopwatch.StartNew();
        int iters = 10;
        for (int i = 0; i < iters; i++)
            _engine.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);
        sw.Stop();
        double gemmMs = sw.Elapsed.TotalMilliseconds / iters;

        // Time the naive float loop baseline (same data type — apples to apples)
        sw.Restart();
        for (int i = 0; i < iters; i++)
            NaiveConv2DBackwardInputFloat(gradOutputF, gradOutputShape, kernelF, kernelShape, inputShape, stride, padding, dilation);
        sw.Stop();
        double naiveMs = sw.Elapsed.TotalMilliseconds / iters;

        double speedup = naiveMs / gemmMs;
        _output.WriteLine($"Conv2DBackwardInput [{batch},{inC},{h},{w}] outC={outC} kernel {kH}x{kW}:");
        _output.WriteLine($"  Float im2col+GEMM: {gemmMs:F2}ms");
        _output.WriteLine($"  Float naive loops: {naiveMs:F2}ms");
        _output.WriteLine($"  Speedup: {speedup:F1}x");

        Assert.True(speedup > 10.0,
            $"Expected at least 10x speedup from im2col+GEMM but got {speedup:F1}x " +
            $"(gemm={gemmMs:F2}ms, naive={naiveMs:F2}ms)");
    }

    [Theory]
    [Trait("Category", "Performance")]
    [InlineData(4, 8, 16, 32, 32, 3, 3)]
    [InlineData(4, 64, 64, 56, 56, 3, 3)]
    public void Conv2DBackwardKernel_FloatFastPath_IsFasterThanNaiveFloatPath(
        int batch, int inC, int outC, int h, int w, int kH, int kW)
    {
        int outH = h - kH + 1;
        int outW = w - kW + 1;

        var rng = new Random(42);
        var gradOutput = new Tensor<float>([batch, outC, outH, outW]);
        var input = new Tensor<float>([batch, inC, h, w]);
        for (int i = 0; i < gradOutput.Length; i++) gradOutput[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var kernelShape = new[] { outC, inC, kH, kW };
        var stride = new[] { 1, 1 };
        var padding = new[] { 0, 0 };
        var dilation = new[] { 1, 1 };

        // Extract raw float arrays for the naive baseline
        var gradOutputF = new float[gradOutput.Length];
        for (int i = 0; i < gradOutput.Length; i++) gradOutputF[i] = gradOutput[i];
        var inputF = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputF[i] = input[i];
        var gradOutputShape = new[] { batch, outC, outH, outW };
        var inputShape = new[] { batch, inC, h, w };

        // Warmup both paths
        _engine.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);
        NaiveConv2DBackwardKernelFloat(gradOutputF, gradOutputShape, inputF, inputShape, kernelShape, stride, padding, dilation);

        // Time the float im2col+GEMM fast path
        var sw = Stopwatch.StartNew();
        int iters = 10;
        for (int i = 0; i < iters; i++)
            _engine.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);
        sw.Stop();
        double gemmMs = sw.Elapsed.TotalMilliseconds / iters;

        // Time the naive float loop baseline (same data type — apples to apples)
        sw.Restart();
        for (int i = 0; i < iters; i++)
            NaiveConv2DBackwardKernelFloat(gradOutputF, gradOutputShape, inputF, inputShape, kernelShape, stride, padding, dilation);
        sw.Stop();
        double naiveMs = sw.Elapsed.TotalMilliseconds / iters;

        double speedup = naiveMs / gemmMs;
        _output.WriteLine($"Conv2DBackwardKernel [{batch},{inC},{h},{w}] outC={outC} kernel {kH}x{kW}:");
        _output.WriteLine($"  Float im2col+GEMM: {gemmMs:F2}ms");
        _output.WriteLine($"  Float naive loops: {naiveMs:F2}ms");
        _output.WriteLine($"  Speedup: {speedup:F1}x");

        Assert.True(speedup > 10.0,
            $"Expected at least 10x speedup from im2col+GEMM but got {speedup:F1}x " +
            $"(gemm={gemmMs:F2}ms, naive={naiveMs:F2}ms)");
    }

    [Fact]
    public void Conv2DBackwardInput_Correctness_MatchesNaivePath()
    {
        // Verify the GEMM path produces the same results as the naive float path
        int batch = 2, inC = 3, outC = 4, h = 8, w = 8, kH = 3, kW = 3;
        int outH = h - kH + 1;
        int outW = w - kW + 1;

        var rng = new Random(123);
        var gradOutputT = new Tensor<float>([batch, outC, outH, outW]);
        var kernelT = new Tensor<float>([outC, inC, kH, kW]);
        for (int i = 0; i < gradOutputT.Length; i++) gradOutputT[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kernelT.Length; i++) kernelT[i] = (float)(rng.NextDouble() * 2 - 1);

        var inputShape = new[] { batch, inC, h, w };
        var stride = new[] { 1, 1 };
        var padding = new[] { 0, 0 };
        var dilation = new[] { 1, 1 };

        // im2col+GEMM fast path via CpuEngine
        var resultGemm = _engine.Conv2DBackwardInput(gradOutputT, kernelT, inputShape, stride, padding, dilation);

        // Naive float loops baseline
        var gradOutputF = new float[gradOutputT.Length];
        for (int i = 0; i < gradOutputT.Length; i++) gradOutputF[i] = gradOutputT[i];
        var kernelF = new float[kernelT.Length];
        for (int i = 0; i < kernelT.Length; i++) kernelF[i] = kernelT[i];
        var gradOutputShape = new[] { batch, outC, outH, outW };
        var kernelShape = new[] { outC, inC, kH, kW };

        var resultNaive = NaiveConv2DBackwardInputFloat(gradOutputF, gradOutputShape, kernelF, kernelShape, inputShape, stride, padding, dilation);

        for (int i = 0; i < resultGemm.Length; i++)
        {
            double diff = Math.Abs(resultGemm[i] - resultNaive[i]);
            Assert.True(diff < 1e-3,
                $"Mismatch at [{i}]: gemm={resultGemm[i]:F6}, naive={resultNaive[i]:F6}, diff={diff:E2}");
        }
    }

    [Fact]
    public void Conv2DBackwardKernel_Correctness_MatchesNaivePath()
    {
        int batch = 2, inC = 3, outC = 4, h = 8, w = 8, kH = 3, kW = 3;
        int outH = h - kH + 1;
        int outW = w - kW + 1;

        var rng = new Random(456);
        var gradOutputT = new Tensor<float>([batch, outC, outH, outW]);
        var inputT = new Tensor<float>([batch, inC, h, w]);
        for (int i = 0; i < gradOutputT.Length; i++) gradOutputT[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < inputT.Length; i++) inputT[i] = (float)(rng.NextDouble() * 2 - 1);

        var kernelShape = new[] { outC, inC, kH, kW };
        var stride = new[] { 1, 1 };
        var padding = new[] { 0, 0 };
        var dilation = new[] { 1, 1 };

        // im2col+GEMM fast path via CpuEngine
        var resultGemm = _engine.Conv2DBackwardKernel(gradOutputT, inputT, kernelShape, stride, padding, dilation);

        // Naive float loops baseline
        var gradOutputF = new float[gradOutputT.Length];
        for (int i = 0; i < gradOutputT.Length; i++) gradOutputF[i] = gradOutputT[i];
        var inputF = new float[inputT.Length];
        for (int i = 0; i < inputT.Length; i++) inputF[i] = inputT[i];
        var gradOutputShape = new[] { batch, outC, outH, outW };
        var inputShape = new[] { batch, inC, h, w };

        var resultNaive = NaiveConv2DBackwardKernelFloat(gradOutputF, gradOutputShape, inputF, inputShape, kernelShape, stride, padding, dilation);

        for (int i = 0; i < resultGemm.Length; i++)
        {
            double diff = Math.Abs(resultGemm[i] - resultNaive[i]);
            Assert.True(diff < 1e-2,
                $"Mismatch at [{i}]: gemm={resultGemm[i]:F6}, naive={resultNaive[i]:F6}, diff={diff:E2}");
        }
    }

    [Theory]
    [InlineData(2, 1, 1, 1)]  // stride=2, pad=1, dilation=1
    [InlineData(1, 1, 2, 1)]  // stride=1, pad=1, dilation=2 (dilated)
    [InlineData(2, 2, 1, 1)]  // stride=2, pad=2 (larger padding)
    public void Conv2DBackwardInput_Correctness_NonTrivialStridePaddingDilation(
        int strideVal, int padVal, int dilationVal, int _unused)
    {
        int batch = 2, inC = 3, outC = 4, h = 16, w = 16, kH = 3, kW = 3;
        int effKH = dilationVal * (kH - 1) + 1;
        int effKW = dilationVal * (kW - 1) + 1;
        int outH = (h + 2 * padVal - effKH) / strideVal + 1;
        int outW = (w + 2 * padVal - effKW) / strideVal + 1;

        var rng = new Random(789);
        var gradOutputT = new Tensor<float>([batch, outC, outH, outW]);
        var kernelT = new Tensor<float>([outC, inC, kH, kW]);
        for (int i = 0; i < gradOutputT.Length; i++) gradOutputT[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kernelT.Length; i++) kernelT[i] = (float)(rng.NextDouble() * 2 - 1);

        var inputShape = new[] { batch, inC, h, w };
        var stride = new[] { strideVal, strideVal };
        var padding = new[] { padVal, padVal };
        var dilation = new[] { dilationVal, dilationVal };

        // im2col+GEMM fast path
        var resultGemm = _engine.Conv2DBackwardInput(gradOutputT, kernelT, inputShape, stride, padding, dilation);

        // Naive float baseline
        var gradOutputF = new float[gradOutputT.Length];
        for (int i = 0; i < gradOutputT.Length; i++) gradOutputF[i] = gradOutputT[i];
        var kernelF = new float[kernelT.Length];
        for (int i = 0; i < kernelT.Length; i++) kernelF[i] = kernelT[i];
        var gradOutputShape = new[] { batch, outC, outH, outW };
        var kernelShape = new[] { outC, inC, kH, kW };

        var resultNaive = NaiveConv2DBackwardInputFloat(gradOutputF, gradOutputShape, kernelF, kernelShape, inputShape, stride, padding, dilation);

        for (int i = 0; i < resultGemm.Length; i++)
        {
            double diff = Math.Abs(resultGemm[i] - resultNaive[i]);
            Assert.True(diff < 1e-2,
                $"BackwardInput stride={strideVal} pad={padVal} dil={dilationVal} mismatch at [{i}]: " +
                $"gemm={resultGemm[i]:F6}, naive={resultNaive[i]:F6}, diff={diff:E2}");
        }
    }

    [Theory]
    [InlineData(2, 1, 1)]
    [InlineData(1, 1, 2)]
    [InlineData(2, 2, 1)]
    public void Conv2DBackwardKernel_Correctness_NonTrivialStridePaddingDilation(
        int strideVal, int padVal, int dilationVal)
    {
        int batch = 2, inC = 3, outC = 4, h = 16, w = 16, kH = 3, kW = 3;
        int effKH = dilationVal * (kH - 1) + 1;
        int effKW = dilationVal * (kW - 1) + 1;
        int outH = (h + 2 * padVal - effKH) / strideVal + 1;
        int outW = (w + 2 * padVal - effKW) / strideVal + 1;

        var rng = new Random(101);
        var gradOutputT = new Tensor<float>([batch, outC, outH, outW]);
        var inputT = new Tensor<float>([batch, inC, h, w]);
        for (int i = 0; i < gradOutputT.Length; i++) gradOutputT[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < inputT.Length; i++) inputT[i] = (float)(rng.NextDouble() * 2 - 1);

        var kernelShape = new[] { outC, inC, kH, kW };
        var stride = new[] { strideVal, strideVal };
        var padding = new[] { padVal, padVal };
        var dilation = new[] { dilationVal, dilationVal };

        // im2col+GEMM fast path
        var resultGemm = _engine.Conv2DBackwardKernel(gradOutputT, inputT, kernelShape, stride, padding, dilation);

        // Naive float baseline
        var gradOutputF = new float[gradOutputT.Length];
        for (int i = 0; i < gradOutputT.Length; i++) gradOutputF[i] = gradOutputT[i];
        var inputF = new float[inputT.Length];
        for (int i = 0; i < inputT.Length; i++) inputF[i] = inputT[i];
        var gradOutputShape = new[] { batch, outC, outH, outW };
        var inputShape = new[] { batch, inC, h, w };

        var resultNaive = NaiveConv2DBackwardKernelFloat(gradOutputF, gradOutputShape, inputF, inputShape, kernelShape, stride, padding, dilation);

        for (int i = 0; i < resultGemm.Length; i++)
        {
            double diff = Math.Abs(resultGemm[i] - resultNaive[i]);
            Assert.True(diff < 1e-1,
                $"BackwardKernel stride={strideVal} pad={padVal} dil={dilationVal} mismatch at [{i}]: " +
                $"gemm={resultGemm[i]:F6}, naive={resultNaive[i]:F6}, diff={diff:E2}");
        }
    }
}
