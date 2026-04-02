using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Benchmarks fused vs unfused Linear+Activation forward+backward performance.
/// Measures wall-clock time for multiple iterations to verify the fused optimization
/// provides measurable speedup.
/// </summary>
public class FusedLinearBenchmark
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public FusedLinearBenchmark(ITestOutputHelper output)
    {
        _output = output;
    }

    private static Tensor<float> CreateRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<float>(shape);
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return tensor;
    }

    [Fact]
    [Trait("Category", "Benchmark")]
    public void Benchmark_FusedVsUnfused_ReLU()
    {
        int batchSize = 32;
        int inFeatures = 256;
        int outFeatures = 128;
        int warmupIterations = 50;
        int benchmarkIterations = 500;

        var input = CreateRandom([batchSize, inFeatures], 42);
        var weight = CreateRandom([inFeatures, outFeatures], 43);
        var bias = CreateRandom([outFeatures], 44);

        // Warmup
        for (int i = 0; i < warmupIterations; i++)
        {
            using var tape = new GradientTape<float>();
            var output = _engine.FusedLinearReLU(input, weight, bias);
            var loss = _engine.ReduceSum(output, [0, 1], keepDims: false);
            tape.ComputeGradients(loss, [input, weight, bias]);
        }

        // Benchmark unfused
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < benchmarkIterations; i++)
        {
            using var tape = new GradientTape<float>();
            var linear = _engine.TensorMatMul(input, weight);
            var biased = _engine.TensorBroadcastAdd(linear, bias);
            var activated = _engine.ReLU(biased);
            var loss = _engine.ReduceSum(activated, [0, 1], keepDims: false);
            tape.ComputeGradients(loss, [input, weight, bias]);
        }
        sw.Stop();
        double unfusedMs = sw.Elapsed.TotalMilliseconds;

        // Benchmark fused
        sw.Restart();
        for (int i = 0; i < benchmarkIterations; i++)
        {
            using var tape = new GradientTape<float>();
            var output = _engine.FusedLinearReLU(input, weight, bias);
            var loss = _engine.ReduceSum(output, [0, 1], keepDims: false);
            tape.ComputeGradients(loss, [input, weight, bias]);
        }
        sw.Stop();
        double fusedMs = sw.Elapsed.TotalMilliseconds;

        double speedup = unfusedMs / fusedMs;
        _output.WriteLine($"Unfused: {unfusedMs:F1}ms ({benchmarkIterations} iterations)");
        _output.WriteLine($"Fused:   {fusedMs:F1}ms ({benchmarkIterations} iterations)");
        _output.WriteLine($"Speedup: {speedup:F2}x");
        _output.WriteLine($"Per-iteration: unfused={unfusedMs / benchmarkIterations:F3}ms, fused={fusedMs / benchmarkIterations:F3}ms");

        // Fused should be at least as fast as unfused (accounting for measurement noise)
        Assert.True(fusedMs <= unfusedMs * 1.1,
            $"Fused ({fusedMs:F1}ms) should not be significantly slower than unfused ({unfusedMs:F1}ms)");
    }
}
