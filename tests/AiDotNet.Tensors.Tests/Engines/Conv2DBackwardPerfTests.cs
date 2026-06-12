using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>Runs the conv perf tests serially and NOT in parallel with any other collection, so the
/// wall-clock im2col+GEMM-vs-naive timings aren't corrupted by oversubscription from concurrent tests
/// (a Category=Performance timing test under heavy pool contention measures the scheduler, not the
/// kernel). Mirrors the repo's BlasManaged-Perf-Serial collection.</summary>
[CollectionDefinition("ConvPerfSerial", DisableParallelization = true)]
public class ConvPerfSerialCollection { }

/// <summary>
/// Performance tests proving Conv2DBackwardInput/BackwardKernel im2col+GEMM
/// is significantly faster than naive loops. Gated by Category=Performance
/// trait on timing methods only — correctness tests always run.
/// </summary>
[Collection("ConvPerfSerial")]
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

        // The 10× threshold relies on AVX2 intrinsics under
        // System.Runtime.Intrinsics.X86, available only on .NET Core 3.0+.
        // On net471 the GEMM path falls back to Vector<T> / scalar, where
        // 5–8× over the naive nested-loop baseline is the realistic ceiling.
#if NET5_0_OR_GREATER
        const double minSpeedup = 10.0;
#else
        const double minSpeedup = 5.0;
#endif
        Assert.True(speedup >= minSpeedup,
            $"Expected at least {minSpeedup}x speedup from im2col+GEMM but got {speedup:F1}x " +
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

        // Measure each path as the MIN over iterations, not the average. This is a
        // [Category=Performance] test (excluded from the parallel CI run) and is sometimes executed
        // alongside other parallel tests; under that oversubscription the OS preempts iterations and
        // the *average* reflects scheduler noise rather than kernel speed, while the min recovers the
        // least-preempted (true-kernel-speed) run. Mirrors PrePackSpeedupTest's min-of-repeats
        // methodology — the gate is unchanged, only the estimator is the standard noisy-box one.
        const int iters = 15;
        var sw = new Stopwatch();
        double gemmMs = double.MaxValue;
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            _engine.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);
            sw.Stop();
            gemmMs = Math.Min(gemmMs, sw.Elapsed.TotalMilliseconds);
        }

        double naiveMs = double.MaxValue;
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            NaiveConv2DBackwardKernelFloat(gradOutputF, gradOutputShape, inputF, inputShape, kernelShape, stride, padding, dilation);
            sw.Stop();
            naiveMs = Math.Min(naiveMs, sw.Elapsed.TotalMilliseconds);
        }

        double speedup = naiveMs / gemmMs;
        _output.WriteLine($"Conv2DBackwardKernel [{batch},{inC},{h},{w}] outC={outC} kernel {kH}x{kW}:");
        _output.WriteLine($"  Float im2col+GEMM: {gemmMs:F2}ms");
        _output.WriteLine($"  Float naive loops: {naiveMs:F2}ms");
        _output.WriteLine($"  Speedup: {speedup:F1}x");

        // The 10× threshold relies on AVX2 intrinsics under
        // System.Runtime.Intrinsics.X86, available only on .NET Core 3.0+.
        // On net471 the GEMM path falls back to Vector<T> / scalar, where
        // 5–8× over the naive nested-loop baseline is the realistic ceiling.
#if NET5_0_OR_GREATER
        const double minSpeedup = 10.0;
#else
        const double minSpeedup = 5.0;
#endif
        Assert.True(speedup >= minSpeedup,
            $"Expected at least {minSpeedup}x speedup from im2col+GEMM but got {speedup:F1}x " +
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

    // #403 Phase F: validate the channel-parallel K-concat im2col build at a
    // high-channel-count shape. The default correctness shape above has inC=3
    // (colW=36) — this one uses inC=32 (colH = 32·9 = 288, colW = 8·8 = 64 ≤ 256,
    // batch ≥ 2 → K-concat with the channel-parallel im2col build over 32 disjoint
    // channel tasks), so the channel-range row-offset math is exercised across many
    // blocks. Asserted for both element types against the naive reference.
    [Fact]
    public void Conv2DBackwardKernel_Correctness_ChannelParallelManyChannels_MatchesNaive()
    {
        int batch = 2, inC = 32, outC = 8, h = 10, w = 10, kH = 3, kW = 3;
        int outH = h - kH + 1, outW = w - kW + 1;
        int colW = outH * outW;                   // 64 ≤ 256 → channel-parallel build
        Assert.True(colW <= 256 && inC >= 16);    // guard the precondition

        var rng = new Random(789);
        var gradOutputF = new float[batch * outC * outH * outW];
        var inputF = new float[batch * inC * h * w];
        for (int i = 0; i < gradOutputF.Length; i++) gradOutputF[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < inputF.Length; i++) inputF[i] = (float)(rng.NextDouble() * 2 - 1);

        var kernelShape = new[] { outC, inC, kH, kW };
        var stride = new[] { 1, 1 };
        var padding = new[] { 0, 0 };
        var dilation = new[] { 1, 1 };
        var gradOutputShape = new[] { batch, outC, outH, outW };
        var inputShape = new[] { batch, inC, h, w };

        var resultNaive = NaiveConv2DBackwardKernelFloat(
            gradOutputF, gradOutputShape, inputF, inputShape, kernelShape, stride, padding, dilation);

        // FP32 swapped path
        var gradOutF = new Tensor<float>(gradOutputShape);
        var inF = new Tensor<float>(inputShape);
        for (int i = 0; i < gradOutputF.Length; i++) gradOutF[i] = gradOutputF[i];
        for (int i = 0; i < inputF.Length; i++) inF[i] = inputF[i];
        var resultF = _engine.Conv2DBackwardKernel(gradOutF, inF, kernelShape, stride, padding, dilation);

        // FP64 swapped path (same swap+transpose code, double element type)
        var gradOutD = new Tensor<double>(gradOutputShape);
        var inD = new Tensor<double>(inputShape);
        for (int i = 0; i < gradOutputF.Length; i++) gradOutD[i] = gradOutputF[i];
        for (int i = 0; i < inputF.Length; i++) inD[i] = inputF[i];
        var resultD = _engine.Conv2DBackwardKernel(gradOutD, inD, kernelShape, stride, padding, dilation);

        for (int i = 0; i < resultNaive.Length; i++)
        {
            double diffF = Math.Abs(resultF[i] - resultNaive[i]);
            Assert.True(diffF < 1e-2,
                $"FP32 swap mismatch at [{i}]: swap={resultF[i]:F6}, naive={resultNaive[i]:F6}, diff={diffF:E2}");
            double diffD = Math.Abs(resultD[i] - resultNaive[i]);
            Assert.True(diffD < 1e-2,
                $"FP64 swap mismatch at [{i}]: swap={resultD[i]:F6}, naive={resultNaive[i]:F6}, diff={diffD:E2}");
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
