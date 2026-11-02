// SCRATCH (probe branch only): the h2h CNN's layers one at a time (forward, and forward + backward through the tape with
// a given output gradient) against torch.autograd.grad on the same host.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Parity;

[Collection("ZzSerial")]
public class ZzCnnOpsBench
{
    private readonly ITestOutputHelper _o;
    public ZzCnnOpsBench(ITestOutputHelper o) => _o = o;

    private static Tensor<float> Rand(Random r, params int[] s)
    {
        var d = new float[s.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < d.Length; i++) d[i] = (float)(r.NextDouble() * 2 - 1);
        return new Tensor<float>(d, s);
    }

    private void Line(string s) { _o.WriteLine(s); Console.WriteLine(s); }

    private static (double Min, double Med) Measure(Action run)
    {
        var sw0 = Stopwatch.StartNew(); while (sw0.ElapsedMilliseconds < 300) run();
        var t = new List<double>();
        for (int round = 0; round < 15; round++)
        {
            double best = double.MaxValue;
            for (int i = 0; i < 3; i++) { var sw = Stopwatch.StartNew(); run(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
            t.Add(best);
        }
        t.Sort();
        return (t[0], t[t.Count / 2]);
    }

    [Fact]
    public void Run()
    {
        if (Environment.GetEnvironmentVariable("GITHUB_ACTIONS") != "true" && Environment.GetEnvironmentVariable("ZZ_CONV") != "1") return;
        int threads = int.TryParse(Environment.GetEnvironmentVariable("ZZ_THREADS"), out int t) ? t : Environment.ProcessorCount;
        CpuParallelSettings.MaxDegreeOfParallelism = threads;
        ZzDeepConvBenchTorch.Run(_o, "tools/parity/zz_cnn_torch.py", threads);
        var engine = new CpuEngine();
        var rng = new Random(9);
        const int b = 64;
        var x1 = Rand(rng, b, 1, 28, 28); var w1 = Rand(rng, 32, 1, 3, 3); var b1 = Rand(rng, 32);
        var x2 = Rand(rng, b, 32, 28, 28);
        var x3 = Rand(rng, b, 32, 14, 14); var w3 = Rand(rng, 64, 32, 3, 3); var b3 = Rand(rng, 64);
        var x4 = Rand(rng, b, 64, 14, 14);
        var x5 = Rand(rng, b, 3136); var w5 = Rand(rng, 3136, 128); var b5 = Rand(rng, 128);
        var x6 = Rand(rng, b, 128); var w6 = Rand(rng, 128, 10); var b6 = Rand(rng, 10);
        var layers = new (string Name, Func<Tensor<float>> Forward, Tensor<float>[] Sources)[]
        {
            ("conv1 1->32 28", () => engine.FusedConv2D(x1, w1, b1, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU), new[] { w1, b1 }),
            ("pool1 32x28", () => engine.MaxPool2D(x2, 2), new[] { x2 }),
            ("conv2 32->64 14", () => engine.FusedConv2D(x3, w3, b3, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU), new[] { x3, w3, b3 }),
            ("pool2 64x14", () => engine.MaxPool2D(x4, 2), new[] { x4 }),
            ("fc1 3136->128", () => engine.FusedLinear(x5, w5, b5, FusedActivationType.ReLU), new[] { x5, w5, b5 }),
            ("fc2 128->10", () => engine.FusedLinear(x6, w6, b6, FusedActivationType.None), new[] { x6, w6, b6 }),
        };
        foreach (var (name, forward, sources) in layers)
        {
            var y0 = forward();
            var dy = Rand(rng, y0.Shape.ToArray());
            var f = Measure(() => forward());
            var fb = Measure(() =>
            {
                using var tape = new GradientTape<float>();
                var y = forward();
                tape.ComputeGradients(y, sources, false, new[] { new KeyValuePair<Tensor<float>, Tensor<float>>(y, dy) });
            });
            Line($"ZC {name,-16} fwd {f.Min:F3}/{f.Med:F3}  fwd+bwd {fb.Min:F3}/{fb.Med:F3}  bwd~ {fb.Min - f.Min:F3}");
        }
        ZzDeepConvBenchTorch.Run(_o, "tools/parity/zz_cnn_torch.py", threads);
    }
}
