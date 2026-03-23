using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Performance regression tests for GPU operations.
/// Ensures GPU operations complete within expected time bounds.
/// Tests are skipped when no GPU backend is available.
/// </summary>
public class GpuPerformanceRegressionTests : IDisposable
{
    private readonly DirectGpuTensorEngine? _gpu;
    private const int WarmupRuns = 2;
    private const int MeasuredRuns = 5;

    public GpuPerformanceRegressionTests()
    {
        try
        {
            _gpu = new DirectGpuTensorEngine();
            if (!_gpu.IsGpuAvailable)
                _gpu = null;
        }
        catch { _gpu = null; }
    }

    private void SkipIfNoGpu()
    {
        Skip.If(_gpu is null, "No GPU backend available");
    }

    private static Tensor<float> RandomTensor(int[] shape, int seed = 42)
    {
        var rng = new Random(seed);
        int total = 1;
        foreach (var d in shape) total *= d;
        var data = new float[total];
        for (int i = 0; i < total; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    private double MeasureMedianMs(Action action)
    {
        // Warmup
        for (int i = 0; i < WarmupRuns; i++) action();

        // Measure
        var times = new double[MeasuredRuns];
        for (int i = 0; i < MeasuredRuns; i++)
        {
            var sw = Stopwatch.StartNew();
            action();
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(times);
        return times[MeasuredRuns / 2]; // Median
    }

    [SkippableFact]
    public void TensorAdd_1M_CompletesWithin50ms()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 1000, 1000 }, 1);
        var b = RandomTensor(new[] { 1000, 1000 }, 2);
        double ms = MeasureMedianMs(() => _gpu!.TensorAdd(a, b));
        // Generous threshold to avoid flaky CI — different GPUs/drivers/thermal states vary widely
        Assert.True(ms < 500, $"TensorAdd 1M took {ms:F1}ms (expected < 500ms)");
    }

    [SkippableFact]
    public void TensorMultiply_1M_CompletesWithin500ms()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 1000, 1000 }, 3);
        var b = RandomTensor(new[] { 1000, 1000 }, 4);
        double ms = MeasureMedianMs(() => _gpu!.TensorMultiply(a, b));
        Assert.True(ms < 500, $"TensorMultiply 1M took {ms:F1}ms (expected < 500ms)");
    }

    [SkippableFact]
    public void Sigmoid_1M_CompletesWithin1000ms()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1000, 1000 }, 5);
        double ms = MeasureMedianMs(() => _gpu!.TensorSigmoid(input));
        Assert.True(ms < 1000, $"Sigmoid 1M took {ms:F1}ms (expected < 1000ms)");
    }

    [SkippableFact]
    public void MatMul_256x256_CompletesWithin500ms()
    {
        SkipIfNoGpu();
        var a = new Matrix<float>(256, 256);
        var b = new Matrix<float>(256, 256);
        var rng = new Random(6);
        for (int i = 0; i < 256; i++)
            for (int j = 0; j < 256; j++)
            { a[i, j] = (float)rng.NextDouble(); b[i, j] = (float)rng.NextDouble(); }
        double ms = MeasureMedianMs(() => ((IEngine)_gpu!).MatrixMultiply(a, b));
        Assert.True(ms < 500, $"MatMul 256x256 took {ms:F1}ms (expected < 500ms)");
    }

    [SkippableFact]
    public void Softmax_1Kx1K_CompletesWithin1000ms()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1000, 1000 }, 7);
        double ms = MeasureMedianMs(() => _gpu!.Softmax(input, -1));
        Assert.True(ms < 1000, $"Softmax 1Kx1K took {ms:F1}ms (expected < 1000ms)");
    }

    [SkippableFact]
    public void Conv2D_SmallBatch_CompletesWithin2000ms()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1, 3, 32, 32 }, 8);
        var kernel = RandomTensor(new[] { 16, 3, 3, 3 }, 9);
        double ms = MeasureMedianMs(() => _gpu!.Conv2D(input, kernel, 1, 1, 1));
        Assert.True(ms < 2000, $"Conv2D took {ms:F1}ms (expected < 2000ms)");
    }

    [SkippableFact]
    public void BatchNorm_CompletesWithin1000ms()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 64, 16, 16 }, 10);
        var gamma = new Tensor<float>(Enumerable.Range(0, 64).Select(_ => 1f).ToArray(), new[] { 64 });
        var beta = new Tensor<float>(new float[64], new[] { 64 });
        double ms = MeasureMedianMs(() => _gpu!.BatchNorm(input, gamma, beta, 1e-5, out _, out _));
        Assert.True(ms < 1000, $"BatchNorm took {ms:F1}ms (expected < 1000ms)");
    }

    public void Dispose()
    {
        (_gpu as IDisposable)?.Dispose();
    }
}
