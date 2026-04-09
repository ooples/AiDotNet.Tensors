using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// End-to-end CNN inference benchmark: compiled vs eager.
/// Conv2D → ReLU → MaxPool2D pipeline at realistic sizes.
/// </summary>
[Trait("Category", "Benchmark")]
public class CnnInferenceBenchmark
{
    private readonly ITestOutputHelper _output;
    public CnnInferenceBenchmark(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(1, 3, 32, 32, 16, 3, 10)]   // Small: CIFAR-like
    [InlineData(1, 3, 64, 64, 32, 3, 5)]    // Medium
    [InlineData(4, 3, 32, 32, 16, 3, 10)]   // Batched
    public void CnnInference_CompiledVsEager(int batch, int inC, int h, int w, int outC, int kSize, int iters)
    {
        var engine = new CpuEngine();
        var input = CreateRandom4D(batch, inC, h, w, 42);
        var kernel = CreateRandom4D(outC, inC, kSize, kSize, 43);
        int warmup = 3;

        // Eager: Conv2D → ReLU → MaxPool2D
        double eagerMs = Measure(() =>
        {
            var conv = engine.Conv2D(input, kernel, stride: 1, padding: 1);
            var relu = engine.ReLU(conv);
            engine.MaxPool2D(relu, poolSize: 2, stride: 2);
        }, warmup, iters);

        // Compiled inference
        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var conv = engine.Conv2D(input, kernel, stride: 1, padding: 1);
            var relu = engine.ReLU(conv);
            engine.MaxPool2D(relu, poolSize: 2, stride: 2);
            plan = scope.CompileInference<float>();
        }

        double compiledMs;
        try
        {
            compiledMs = Measure(() => plan.Execute(), warmup, iters);
        }
        finally { plan.Dispose(); }

        double speedup = eagerMs / compiledMs;
        _output.WriteLine($"CNN [{batch}x{inC}x{h}x{w}] Conv{kSize}→ReLU→Pool  " +
            $"Eager: {eagerMs:F3}ms  Compiled: {compiledMs:F3}ms  Speedup: {speedup:F2}x");
    }

    private static double Measure(Action action, int warmup, int iters)
    {
        for (int i = 0; i < warmup; i++) action();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) action();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }

    private static Tensor<float> CreateRandom4D(int n, int c, int h, int w, int seed)
    {
        var rng = new Random(seed);
        var data = new float[n * c * h * w];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, new[] { n, c, h, w });
    }
}
