// SCRATCH (probe branch only): the h2h MLP's eager step split into phases and backward pieces, in process.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Parity;

public class ZzMlpStepBench
{
    private readonly ITestOutputHelper _o;
    public ZzMlpStepBench(ITestOutputHelper o) => _o = o;

    private static Tensor<float> Rand(Random r, float scale, params int[] s)
    {
        var d = new float[s.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < d.Length; i++) d[i] = (float)(r.NextDouble() * 2 - 1) * scale;
        return new Tensor<float>(d, s);
    }

    private void Line(string s) { _o.WriteLine(s); Console.WriteLine(s); }

    private static double Med(List<double> v) { v.Sort(); return v[v.Count / 2]; }

    private static (double Min, double Med) Measure(Action run)
    {
        var sw0 = Stopwatch.StartNew(); while (sw0.ElapsedMilliseconds < 250) run();
        var t = new List<double>();
        for (int round = 0; round < 25; round++)
        {
            double best = double.MaxValue;
            for (int i = 0; i < 5; i++) { var sw = Stopwatch.StartNew(); run(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
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
        ZzDeepConvBenchTorch.Run(_o, "tools/parity/zz_mlpstep_torch.py", threads);
        var engine = new CpuEngine();
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var rng = new Random(3);
            var x = Rand(rng, 1f, 128, 784); var y = Rand(rng, 1f, 128, 10);
            var w1 = Rand(rng, 0.03f, 784, 512); var b1 = Rand(rng, 0.03f, 512);
            var w2 = Rand(rng, 0.04f, 512, 256); var b2 = Rand(rng, 0.04f, 256);
            var w3 = Rand(rng, 0.06f, 256, 10); var b3 = Rand(rng, 0.06f, 10);
            var ps = new[] { w1, b1, w2, b2, w3, b3 };
            var axes = new[] { 0, 1 };
            var opt = new SgdOptimizer();
            var group = opt.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.01 });
            foreach (var p in ps) group.AddParameter(p);
            using var arena = TensorArena.Create();
            var f = new List<double>(); var b = new List<double>(); var o = new List<double>(); var s = new List<double>();
            var sw = new Stopwatch();
            for (int step = 0; step < 1200; step++)
            {
                sw.Restart();
                using var tape = new GradientTape<float>();
                var h = engine.FusedLinear(x, w1, b1, FusedActivationType.ReLU);
                h = engine.FusedLinear(h, w2, b2, FusedActivationType.ReLU);
                h = engine.FusedLinear(h, w3, b3, FusedActivationType.None);
                var loss = engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(h, y)), axes, keepDims: false);
                double t1 = sw.Elapsed.TotalMilliseconds;
                var grads = tape.ComputeGradients(loss, ps);
                double t2 = sw.Elapsed.TotalMilliseconds;
                opt.Step(grads);
                double t3 = sw.Elapsed.TotalMilliseconds;
                if (step >= 200) { f.Add(t1); b.Add(t2 - t1); o.Add(t3 - t2); s.Add(t3); }
            }
            Line($"ZS step {Med(s):F4} fwd {Med(f):F4} bwd {Med(b):F4} opt {Med(o):F4}");

            // Backward pieces in isolation.
            var a512 = engine.FusedLinear(x, w1, b1, FusedActivationType.ReLU);
            var a256 = engine.FusedLinear(a512, w2, b2, FusedActivationType.ReLU);
            var g512 = Rand(rng, 1f, 128, 512); var g256 = Rand(rng, 1f, 128, 256);
            var r512 = Measure(() => { using var tp = new GradientTape<float>(); engine.ApplyFusedActivationBackward(g512, a512, FusedActivationType.ReLU, null); });
            var r256 = Measure(() => { using var tp = new GradientTape<float>(); engine.ApplyFusedActivationBackward(g256, a256, FusedActivationType.ReLU, null); });
            var n512 = Measure(() => { using var tp = new GradientTape<float>(); });
            Line($"ZS empty-tape {n512.Min:F4}/{n512.Med:F4}");
            var out10 = engine.FusedLinear(a256, w3, b3, FusedActivationType.None);
            var mse = Measure(() =>
            {
                var leaf = out10;
                using var tp = new GradientTape<float>();
                var l = engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(leaf, y)), axes, keepDims: false);
                tp.ComputeGradients(l, new[] { leaf });
            });
            Line($"ZS pieces relu512 {r512.Min:F4}/{r512.Med:F4} relu256 {r256.Min:F4}/{r256.Med:F4} mse-fwd+bwd {mse.Min:F4}/{mse.Med:F4}");
            foreach (var (name, act, inp, w, bb, srcs) in new (string, FusedActivationType, Tensor<float>, Tensor<float>, Tensor<float>, Tensor<float>[])[]
            {
                ("L1", FusedActivationType.ReLU, x, w1, b1, new[] { w1, b1 }),
                ("L2", FusedActivationType.ReLU, a512, w2, b2, new[] { a512, w2, b2 }),
                ("L3", FusedActivationType.None, a256, w3, b3, new[] { a256, w3, b3 }),
            })
            {
                var y0 = engine.FusedLinear(inp, w, bb, act);
                var dy = Rand(rng, 1f, y0.Shape.ToArray());
                var fw = Measure(() => engine.FusedLinear(inp, w, bb, act));
                var fb = Measure(() =>
                {
                    using var tp = new GradientTape<float>();
                    var yy = engine.FusedLinear(inp, w, bb, act);
                    tp.ComputeGradients(yy, srcs, false, new[] { new KeyValuePair<Tensor<float>, Tensor<float>>(yy, dy) });
                });
                Line($"ZS {name} fwd {fw.Min:F4}/{fw.Med:F4} fwd+bwd {fb.Min:F4}/{fb.Med:F4} bwd~ {fb.Min - fw.Min:F4}");
            }
        }
        finally { AiDotNetEngine.Current = previous; }
        ZzDeepConvBenchTorch.Run(_o, "tools/parity/zz_mlpstep_torch.py", threads);
    }
}
