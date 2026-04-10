using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A/B regression investigation: disable one optimization pass at a time
/// to find which pass causes the compiled path to be SLOWER than eager.
///
/// Known regressions:
/// 1. ReLU MLP: compiled 0.02x (50x slower than eager)
/// 2. Large [128x512->256->64]: compiled 0.03x (33x slower)
/// </summary>
[Trait("Category", "Benchmark")]
public class PassRegressionABTest
{
    private readonly ITestOutputHelper _output;
    public PassRegressionABTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ReLU_MLP_DisableOnePassAtATime()
    {
        var engine = new CpuEngine();
        int m = 32, k = 128, h = 64, n = 10;
        var input = CreateRandom(new[] { m, k }, 42);
        var w1 = CreateRandom(new[] { k, h }, 43);
        var w2 = CreateRandom(new[] { h, n }, 44);
        int warmup = 5, iters = 50;

        // Eager baseline
        double eagerMs = Measure(() =>
        {
            var h1 = engine.ReLU(engine.TensorMatMul(input, w1));
            engine.TensorMatMul(h1, w2);
        }, warmup, iters);

        // Compiled with ALL passes (the regressing case)
        double allPassesMs = CompileAndMeasure(engine, input, w1, w2, null, warmup, iters);

        _output.WriteLine($"Eager:       {eagerMs:F4}ms");
        _output.WriteLine($"All passes:  {allPassesMs:F4}ms  ({eagerMs / allPassesMs:F2}x vs eager)");
        _output.WriteLine("");

        // Disable one pass at a time
        string[] passNames = { "DataflowFusion", "PointwiseFusion", "ConstantFolding",
            "ForwardCSE", "BlasBatch", "ConvBnFusion", "AttentionFusion" };

        foreach (var passName in passNames)
        {
            var opts = TensorCodecOptions.Default;
            switch (passName)
            {
                case "DataflowFusion": opts.EnableDataflowFusion = false; break;
                case "PointwiseFusion": opts.EnablePointwiseFusion = false; break;
                case "ConstantFolding": opts.EnableConstantFolding = false; break;
                case "ForwardCSE": opts.EnableForwardCSE = false; break;
                case "BlasBatch": opts.EnableBlasBatch = false; break;
                case "ConvBnFusion": opts.EnableConvBnFusion = false; break;
                case "AttentionFusion": opts.EnableAttentionFusion = false; break;
            }

            double ms = CompileAndMeasure(engine, input, w1, w2, opts, warmup, iters);
            string delta = ms < allPassesMs ? "FASTER" : "slower";
            _output.WriteLine($"  Without {passName,-20}: {ms:F4}ms  ({eagerMs / ms:F2}x vs eager)  [{delta} than all-passes]");
        }

        // No passes at all
        var noPasses = TensorCodecOptions.Default;
        noPasses.EnableDataflowFusion = false;
        noPasses.EnablePointwiseFusion = false;
        noPasses.EnableConstantFolding = false;
        noPasses.EnableForwardCSE = false;
        noPasses.EnableBlasBatch = false;
        noPasses.EnableConvBnFusion = false;
        noPasses.EnableAttentionFusion = false;
        double noPassMs = CompileAndMeasure(engine, input, w1, w2, noPasses, warmup, iters);
        _output.WriteLine($"\n  NO passes:          {noPassMs:F4}ms  ({eagerMs / noPassMs:F2}x vs eager)");
    }

    [Fact]
    public void LargeModel_DisableOnePassAtATime()
    {
        var engine = new CpuEngine();
        int m = 128, k = 512, h = 256, n = 64;
        var input = CreateRandom(new[] { m, k }, 42);
        var w1 = CreateRandom(new[] { k, h }, 43);
        var w2 = CreateRandom(new[] { h, n }, 44);
        int warmup = 2, iters = 10;

        double eagerMs = Measure(() =>
        {
            var h1 = engine.ReLU(engine.TensorMatMul(input, w1));
            engine.TensorMatMul(h1, w2);
        }, warmup, iters);

        double allPassesMs = CompileAndMeasure(engine, input, w1, w2, null, warmup, iters);

        _output.WriteLine($"Large [128x512->256->64]:");
        _output.WriteLine($"Eager:       {eagerMs:F4}ms");
        _output.WriteLine($"All passes:  {allPassesMs:F4}ms  ({eagerMs / allPassesMs:F2}x vs eager)");
        _output.WriteLine("");

        string[] passNames = { "DataflowFusion", "PointwiseFusion", "ConstantFolding",
            "ForwardCSE", "BlasBatch", "SpectralDecomposition" };

        foreach (var passName in passNames)
        {
            var opts = TensorCodecOptions.Default;
            switch (passName)
            {
                case "DataflowFusion": opts.EnableDataflowFusion = false; break;
                case "PointwiseFusion": opts.EnablePointwiseFusion = false; break;
                case "ConstantFolding": opts.EnableConstantFolding = false; break;
                case "ForwardCSE": opts.EnableForwardCSE = false; break;
                case "BlasBatch": opts.EnableBlasBatch = false; break;
                case "SpectralDecomposition": opts.EnableSpectralDecomposition = false; break;
            }

            double ms = CompileAndMeasure(engine, input, w1, w2, opts, warmup, iters);
            string delta = ms < allPassesMs ? "FASTER" : "slower";
            _output.WriteLine($"  Without {passName,-25}: {ms:F4}ms  ({eagerMs / ms:F2}x vs eager)  [{delta}]");
        }

        var noPasses = TensorCodecOptions.Default;
        noPasses.EnableDataflowFusion = false;
        noPasses.EnablePointwiseFusion = false;
        noPasses.EnableConstantFolding = false;
        noPasses.EnableForwardCSE = false;
        noPasses.EnableBlasBatch = false;
        double noPassMs = CompileAndMeasure(engine, input, w1, w2, noPasses, warmup, iters);
        _output.WriteLine($"\n  NO passes:                    {noPassMs:F4}ms  ({eagerMs / noPassMs:F2}x vs eager)");
    }

    private double CompileAndMeasure(CpuEngine engine, Tensor<float> input,
        Tensor<float> w1, Tensor<float> w2, TensorCodecOptions? opts, int warmup, int iters)
    {
        var prev = TensorCodecOptions.Current;
        if (opts is not null) TensorCodecOptions.SetCurrent(opts);
        try
        {
            CompiledInferencePlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var h1 = engine.ReLU(engine.TensorMatMul(input, w1));
                engine.TensorMatMul(h1, w2);
                plan = scope.CompileInference<float>();
            }

            double ms = Measure(() => plan.Execute(), warmup, iters);
            plan.Dispose();
            return ms;
        }
        finally
        {
            TensorCodecOptions.SetCurrent(prev);
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
