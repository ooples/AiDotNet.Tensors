using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A/B test: GroupNorm [32x64x8x8]. PyTorch = 357us, our compiled = 1,282us (3.6x slower).
/// Same pattern as BatchNorm — check if specialization is missing or mismatched.
/// </summary>
public class GroupNormABTest
{
    private readonly ITestOutputHelper _output;
    public GroupNormABTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public void GroupNorm_Compiled_vs_Eager()
    {
        var engine = new CpuEngine();
        int batch = 32, channels = 64, h = 8, w = 8, numGroups = 8;
        var input = CreateRandom(new[] { batch, channels, h, w }, 42);
        var gamma = CreateRandom(new[] { channels }, 43);
        var beta = CreateRandom(new[] { channels }, 44);
        float eps = 1e-5f;
        int warmup = 5, iters = 50;

        // Path A: Eager
        double eagerMs = Measure(() =>
        {
            engine.GroupNorm(input, numGroups, gamma, beta, eps, out _, out _);
        }, warmup, iters);

        // Path B: Compiled
        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.GroupNorm(input, numGroups, gamma, beta, eps, out _, out _);
            plan = scope.CompileInference<float>();
        }
        double compiledMs = Measure(() => plan.Execute(), warmup, iters);
        plan.Dispose();

        _output.WriteLine($"GroupNorm [32x64x8x8] (groups={numGroups}):");
        _output.WriteLine($"  Path A (Eager):    {eagerMs:F3}ms");
        _output.WriteLine($"  Path B (Compiled): {compiledMs:F3}ms");
        _output.WriteLine($"  Compiled vs Eager: {eagerMs / compiledMs:F2}x");
        _output.WriteLine($"");
        _output.WriteLine($"  PyTorch BDN: ~0.357ms");
        _output.WriteLine($"  Our compiled: {compiledMs:F3}ms = {compiledMs / 0.357:F1}x vs PyTorch");
    }

    private static double Measure(Action action, int warmup, int iters)
    {
        for (int i = 0; i < warmup; i++) action();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) action();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }

    private static Tensor<float> CreateRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int length = 1;
        for (int i = 0; i < shape.Length; i++) length *= shape[i];
        var data = new float[length];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }
}
