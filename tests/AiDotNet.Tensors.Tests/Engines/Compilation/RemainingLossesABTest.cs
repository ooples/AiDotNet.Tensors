using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A/B tests for remaining operations where PyTorch beat us in BDN.
/// Each test measures eager vs compiled vs PyTorch reference.
/// </summary>
public class RemainingLossesABTest
{
    private readonly ITestOutputHelper _output;
    public RemainingLossesABTest(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData("Max", 100_000, 0.144)]
    [InlineData("LogSoftmax", 65536, 0.099)]       // 256x256
    [InlineData("Round", 100_000, 0.084)]
    [InlineData("HardSwish", 100_000, 0.156)]
    [InlineData("Tanh", 100_000, 0.266)]
    [InlineData("Mean", 1_000_000, 0.089)]
    [InlineData("SELU", 100_000, 0.109)]
    [InlineData("Exp", 100_000, 0.069)]
    [InlineData("Sigmoid", 1_000_000, 0.488)]
    [InlineData("Softplus", 100_000, 0.343)]
    public void Op_CompiledVsEager(string opName, int size, double pytorchMs)
    {
        var engine = new CpuEngine();
        int warmup = 5, iters = 50;

        // Create appropriately shaped inputs
        var input = opName == "LogSoftmax"
            ? CreateRandom(new[] { 256, 256 }, 42)
            : CreateRandom(new[] { size }, 42);

        // Eager
        double eagerMs = Measure(() => RunOp(engine, opName, input), warmup, iters);

        // Compiled
        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            RunOp(engine, opName, input);
            plan = scope.CompileInference<float>();
        }
        double compiledMs = Measure(() => plan.Execute(), warmup, iters);
        plan.Dispose();

        double ratio = compiledMs / pytorchMs;
        string status = ratio < 1.0 ? "WIN" : ratio < 1.1 ? "~TIE" : "LOSE";

        _output.WriteLine($"{opName}[{size}]: Eager={eagerMs:F3}ms  Compiled={compiledMs:F3}ms  " +
            $"PyTorch={pytorchMs:F3}ms  Ratio={ratio:F2}x  [{status}]");
    }

    private static void RunOp(CpuEngine engine, string opName, Tensor<float> input)
    {
        switch (opName)
        {
            case "Max": engine.ReduceMax(input, new[] { 0 }, false, out _); break;
            case "LogSoftmax": engine.TensorLogSoftmax(input, -1); break;
            case "Round": engine.TensorRound(input); break;
            case "HardSwish": engine.HardSwish(input); break;
            case "Tanh": engine.Tanh(input); break;
            case "Mean": engine.ReduceMean(input, new[] { 0 }, false); break;
            case "SELU": engine.TensorSELU(input); break;
            case "Exp": engine.TensorExp(input); break;
            case "Sigmoid": engine.Sigmoid(input); break;
            case "Softplus": engine.Softplus(input); break;
        }
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
