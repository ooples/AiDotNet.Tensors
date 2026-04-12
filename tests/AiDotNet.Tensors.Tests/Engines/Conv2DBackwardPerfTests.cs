using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Performance tests proving Conv2DBackwardInput/BackwardKernel im2col+GEMM
/// is significantly faster than naive loops. Gated by Category=Performance
/// trait — exclude from CI with --filter "Category!=Performance" if needed.
/// </summary>
[Trait("Category", "Performance")]
public class Conv2DBackwardPerfTests
{
    private readonly ITestOutputHelper _output;
    private readonly CpuEngine _engine = new();

    public Conv2DBackwardPerfTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Theory]
    [InlineData(4, 8, 16, 32, 32, 3, 3)]   // Small shape
    [InlineData(4, 64, 64, 56, 56, 3, 3)]  // ResNet50-like shape (from issue #148)
    public void Conv2DBackwardInput_FloatFastPath_IsFasterThanGenericPath(
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

        // Warmup
        _engine.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);

        // Time the float fast path (im2col+GEMM)
        var sw = Stopwatch.StartNew();
        int iters = 10;
        for (int i = 0; i < iters; i++)
            _engine.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);
        sw.Stop();
        double floatMs = sw.Elapsed.TotalMilliseconds / iters;

        // Time the generic path (naive loops) using double to force the generic path
        var gradOutputD = new Tensor<double>([batch, outC, outH, outW]);
        var kernelD = new Tensor<double>([outC, inC, kH, kW]);
        for (int i = 0; i < gradOutput.Length; i++) gradOutputD[i] = gradOutput[i];
        for (int i = 0; i < kernel.Length; i++) kernelD[i] = kernel[i];

        _engine.Conv2DBackwardInput(gradOutputD, kernelD, inputShape, stride, padding, dilation);

        sw.Restart();
        for (int i = 0; i < iters; i++)
            _engine.Conv2DBackwardInput(gradOutputD, kernelD, inputShape, stride, padding, dilation);
        sw.Stop();
        double genericMs = sw.Elapsed.TotalMilliseconds / iters;

        double speedup = genericMs / floatMs;
        _output.WriteLine($"Conv2DBackwardInput [{batch},{inC},{h},{w}] outC={outC} kernel {kH}x{kW}:");
        _output.WriteLine($"  Float im2col+GEMM: {floatMs:F2}ms");
        _output.WriteLine($"  Double generic loops: {genericMs:F2}ms");
        _output.WriteLine($"  Speedup: {speedup:F1}x");

                Assert.True(speedup > 10.0,
            $"Expected at least 10x speedup from im2col+GEMM but got {speedup:F1}x " +
            $"(float={floatMs:F2}ms, generic={genericMs:F2}ms)");
    }

    [Theory]
    [InlineData(4, 8, 16, 32, 32, 3, 3)]
    [InlineData(4, 64, 64, 56, 56, 3, 3)]
    public void Conv2DBackwardKernel_FloatFastPath_IsFasterThanGenericPath(
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

        _engine.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);

        var sw = Stopwatch.StartNew();
        int iters = 10;
        for (int i = 0; i < iters; i++)
            _engine.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);
        sw.Stop();
        double floatMs = sw.Elapsed.TotalMilliseconds / iters;

        var gradOutputD = new Tensor<double>([batch, outC, outH, outW]);
        var inputD = new Tensor<double>([batch, inC, h, w]);
        for (int i = 0; i < gradOutput.Length; i++) gradOutputD[i] = gradOutput[i];
        for (int i = 0; i < input.Length; i++) inputD[i] = input[i];

        _engine.Conv2DBackwardKernel(gradOutputD, inputD, kernelShape, stride, padding, dilation);

        sw.Restart();
        for (int i = 0; i < iters; i++)
            _engine.Conv2DBackwardKernel(gradOutputD, inputD, kernelShape, stride, padding, dilation);
        sw.Stop();
        double genericMs = sw.Elapsed.TotalMilliseconds / iters;

        double speedup = genericMs / floatMs;
        _output.WriteLine($"Conv2DBackwardKernel [{batch},{inC},{h},{w}] outC={outC} kernel {kH}x{kW}:");
        _output.WriteLine($"  Float im2col+GEMM: {floatMs:F2}ms");
        _output.WriteLine($"  Double generic loops: {genericMs:F2}ms");
        _output.WriteLine($"  Speedup: {speedup:F1}x");

                Assert.True(speedup > 10.0,
            $"Expected at least 10x speedup from im2col+GEMM but got {speedup:F1}x " +
            $"(float={floatMs:F2}ms, generic={genericMs:F2}ms)");
    }

    [Fact]
    public void Conv2DBackwardInput_Correctness_MatchesGenericPath()
    {
        // Verify the GEMM path produces the same results as the generic path
        int batch = 2, inC = 3, outC = 4, h = 8, w = 8, kH = 3, kW = 3;
        int outH = h - kH + 1;
        int outW = w - kW + 1;

        var rng = new Random(123);
        var gradOutputF = new Tensor<float>([batch, outC, outH, outW]);
        var kernelF = new Tensor<float>([outC, inC, kH, kW]);
        for (int i = 0; i < gradOutputF.Length; i++) gradOutputF[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kernelF.Length; i++) kernelF[i] = (float)(rng.NextDouble() * 2 - 1);

        // Double path uses generic loops
        var gradOutputD = new Tensor<double>([batch, outC, outH, outW]);
        var kernelD = new Tensor<double>([outC, inC, kH, kW]);
        for (int i = 0; i < gradOutputF.Length; i++) gradOutputD[i] = gradOutputF[i];
        for (int i = 0; i < kernelF.Length; i++) kernelD[i] = kernelF[i];

        var inputShape = new[] { batch, inC, h, w };
        var stride = new[] { 1, 1 };
        var padding = new[] { 0, 0 };
        var dilation = new[] { 1, 1 };

        var resultFloat = _engine.Conv2DBackwardInput(gradOutputF, kernelF, inputShape, stride, padding, dilation);
        var resultDouble = _engine.Conv2DBackwardInput(gradOutputD, kernelD, inputShape, stride, padding, dilation);

        for (int i = 0; i < resultFloat.Length; i++)
        {
            double diff = Math.Abs(resultFloat[i] - (float)resultDouble[i]);
            Assert.True(diff < 1e-3,
                $"Mismatch at [{i}]: float={resultFloat[i]:F6}, double={(float)resultDouble[i]:F6}, diff={diff:E2}");
        }
    }

    [Fact]
    public void Conv2DBackwardKernel_Correctness_MatchesGenericPath()
    {
        int batch = 2, inC = 3, outC = 4, h = 8, w = 8, kH = 3, kW = 3;
        int outH = h - kH + 1;
        int outW = w - kW + 1;

        var rng = new Random(456);
        var gradOutputF = new Tensor<float>([batch, outC, outH, outW]);
        var inputF = new Tensor<float>([batch, inC, h, w]);
        for (int i = 0; i < gradOutputF.Length; i++) gradOutputF[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < inputF.Length; i++) inputF[i] = (float)(rng.NextDouble() * 2 - 1);

        var gradOutputD = new Tensor<double>([batch, outC, outH, outW]);
        var inputD = new Tensor<double>([batch, inC, h, w]);
        for (int i = 0; i < gradOutputF.Length; i++) gradOutputD[i] = gradOutputF[i];
        for (int i = 0; i < inputF.Length; i++) inputD[i] = inputF[i];

        var kernelShape = new[] { outC, inC, kH, kW };
        var stride = new[] { 1, 1 };
        var padding = new[] { 0, 0 };
        var dilation = new[] { 1, 1 };

        var resultFloat = _engine.Conv2DBackwardKernel(gradOutputF, inputF, kernelShape, stride, padding, dilation);
        var resultDouble = _engine.Conv2DBackwardKernel(gradOutputD, inputD, kernelShape, stride, padding, dilation);

        for (int i = 0; i < resultFloat.Length; i++)
        {
            double diff = Math.Abs(resultFloat[i] - (float)resultDouble[i]);
            Assert.True(diff < 1e-2,
                $"Mismatch at [{i}]: float={resultFloat[i]:F6}, double={(float)resultDouble[i]:F6}, diff={diff:E2}");
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
        var gradOutputF = new Tensor<float>([batch, outC, outH, outW]);
        var kernelF = new Tensor<float>([outC, inC, kH, kW]);
        for (int i = 0; i < gradOutputF.Length; i++) gradOutputF[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kernelF.Length; i++) kernelF[i] = (float)(rng.NextDouble() * 2 - 1);

        var gradOutputD = new Tensor<double>([batch, outC, outH, outW]);
        var kernelD = new Tensor<double>([outC, inC, kH, kW]);
        for (int i = 0; i < gradOutputF.Length; i++) gradOutputD[i] = gradOutputF[i];
        for (int i = 0; i < kernelF.Length; i++) kernelD[i] = kernelF[i];

        var inputShape = new[] { batch, inC, h, w };
        var stride = new[] { strideVal, strideVal };
        var padding = new[] { padVal, padVal };
        var dilation = new[] { dilationVal, dilationVal };

        var resultF = _engine.Conv2DBackwardInput(gradOutputF, kernelF, inputShape, stride, padding, dilation);
        var resultD = _engine.Conv2DBackwardInput(gradOutputD, kernelD, inputShape, stride, padding, dilation);

        for (int i = 0; i < resultF.Length; i++)
        {
            double diff = Math.Abs(resultF[i] - (float)resultD[i]);
            Assert.True(diff < 1e-2,
                $"BackwardInput stride={strideVal} pad={padVal} dil={dilationVal} mismatch at [{i}]: " +
                $"float={resultF[i]:F6}, double={(float)resultD[i]:F6}, diff={diff:E2}");
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
        var gradOutputF = new Tensor<float>([batch, outC, outH, outW]);
        var inputF = new Tensor<float>([batch, inC, h, w]);
        for (int i = 0; i < gradOutputF.Length; i++) gradOutputF[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < inputF.Length; i++) inputF[i] = (float)(rng.NextDouble() * 2 - 1);

        var gradOutputD = new Tensor<double>([batch, outC, outH, outW]);
        var inputD = new Tensor<double>([batch, inC, h, w]);
        for (int i = 0; i < gradOutputF.Length; i++) gradOutputD[i] = gradOutputF[i];
        for (int i = 0; i < inputF.Length; i++) inputD[i] = inputF[i];

        var kernelShape = new[] { outC, inC, kH, kW };
        var stride = new[] { strideVal, strideVal };
        var padding = new[] { padVal, padVal };
        var dilation = new[] { dilationVal, dilationVal };

        var resultF = _engine.Conv2DBackwardKernel(gradOutputF, inputF, kernelShape, stride, padding, dilation);
        var resultD = _engine.Conv2DBackwardKernel(gradOutputD, inputD, kernelShape, stride, padding, dilation);

        for (int i = 0; i < resultF.Length; i++)
        {
            double diff = Math.Abs(resultF[i] - (float)resultD[i]);
            Assert.True(diff < 1e-1,
                $"BackwardKernel stride={strideVal} pad={padVal} dil={dilationVal} mismatch at [{i}]: " +
                $"float={resultF[i]:F6}, double={(float)resultD[i]:F6}, diff={diff:E2}");
        }
    }
}
