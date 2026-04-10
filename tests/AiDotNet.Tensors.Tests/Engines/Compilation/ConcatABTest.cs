using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A/B test: Concat [2x100K]. PyTorch = 176us, our compiled = 863us (4.9x slower).
/// Hypothesis: compiled Concat runs the eager allocating path.
/// Measure: eager vs compiled vs raw Buffer.BlockCopy baseline.
/// </summary>
[Trait("Category", "Benchmark")]
public class ConcatABTest
{
    private readonly ITestOutputHelper _output;
    public ConcatABTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Concat_Compiled_vs_Eager_vs_RawCopy()
    {
        var engine = new CpuEngine();
        int size = 100_000;
        var a = CreateRandom(new[] { size }, 42);
        var b = CreateRandom(new[] { size }, 43);
        int warmup = 5, iters = 50;

        // Path A: Eager engine Concat
        double eagerMs = Measure(() =>
        {
            engine.TensorConcatenate(new[] { a, b }, 0);
        }, warmup, iters);

        // Path B: Compiled Concat
        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.TensorConcatenate(new[] { a, b }, 0);
            plan = scope.CompileInference<float>();
        }
        double compiledMs = Measure(() => plan.Execute(), warmup, iters);
        plan.Dispose();

        // Path C: Raw Buffer.BlockCopy baseline (theoretical minimum)
        var rawOutput = new float[size * 2];
        var aArr = a.GetDataArray();
        var bArr = b.GetDataArray();
        double rawMs = Measure(() =>
        {
            Buffer.BlockCopy(aArr, 0, rawOutput, 0, size * 4);
            Buffer.BlockCopy(bArr, 0, rawOutput, size * 4, size * 4);
        }, warmup, iters);

        _output.WriteLine($"Concat [2x{size}] ({size * 2} elements output):");
        _output.WriteLine($"  Path A (Eager engine):        {eagerMs:F3}ms");
        _output.WriteLine($"  Path B (Compiled plan):       {compiledMs:F3}ms");
        _output.WriteLine($"  Path C (Raw BlockCopy):       {rawMs:F3}ms");
        _output.WriteLine($"");
        _output.WriteLine($"  Eager overhead vs raw: {eagerMs / rawMs:F1}x");
        _output.WriteLine($"  Compiled overhead vs raw: {compiledMs / rawMs:F1}x");
        _output.WriteLine($"  Compiled vs Eager: {eagerMs / compiledMs:F2}x");
        _output.WriteLine($"");
        _output.WriteLine($"  PyTorch BDN: ~0.176ms");
        _output.WriteLine($"  Our best (raw): {rawMs:F3}ms = {rawMs / 0.176:F1}x vs PyTorch");
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
