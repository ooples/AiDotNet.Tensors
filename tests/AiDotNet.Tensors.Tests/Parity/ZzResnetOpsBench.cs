// SCRATCH (probe branch only): every ResNet-18 (CIFAR, batch 8) conv pass timed in one process, min of short runs.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Parity;

public class ZzResnetOpsBench
{
    private readonly ITestOutputHelper _o;
    public ZzResnetOpsBench(ITestOutputHelper o) => _o = o;

    private static Tensor<float> Rand(Random r, params int[] s)
    {
        var d = new float[s.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < d.Length; i++) d[i] = (float)(r.NextDouble() * 2 - 1);
        return new Tensor<float>(d, s);
    }

    [Fact]
    public void Time()
    {
        if (Environment.GetEnvironmentVariable("GITHUB_ACTIONS") != "true" && Environment.GetEnvironmentVariable("ZZ_CONV") != "1") return;
        CpuParallelSettings.MaxDegreeOfParallelism = 4;
        var engine = new CpuEngine();
        var rng = new Random(3);
        // (cin, cout, hw, k, s, p, count in the network)
        var convs = new (int Cin, int Cout, int Hw, int K, int S, int P, int N)[]
        {
            (3, 64, 32, 3, 1, 1, 1), (64, 64, 32, 3, 1, 1, 4), (64, 128, 32, 3, 2, 1, 1), (64, 128, 32, 1, 2, 0, 1),
            (128, 128, 16, 3, 1, 1, 3), (128, 256, 16, 3, 2, 1, 1), (128, 256, 16, 1, 2, 0, 1),
            (256, 256, 8, 3, 1, 1, 3), (256, 512, 8, 3, 2, 1, 1), (256, 512, 8, 1, 2, 0, 1), (512, 512, 4, 3, 1, 1, 3),
        };
        const int b = 8;
        foreach (var c in convs)
        {
            int o = (c.Hw + 2 * c.P - c.K) / c.S + 1;
            var x = Rand(rng, b, c.Cin, c.Hw, c.Hw); var w = Rand(rng, c.Cout, c.Cin, c.K, c.K); var dy = Rand(rng, b, c.Cout, o, o);
            var y = new Tensor<float>(new[] { b, c.Cout, o, o }); var dx = new Tensor<float>(x.Shape.ToArray());
            var dw = new Tensor<float>(w.Shape.ToArray());
            int[] s = { c.S, c.S }, p = { c.P, c.P }, d = { 1, 1 };
            var ops = new (string Name, Action Run)[]
            {
                ("conv.fwd", () => engine.Conv2DInto(y, x, w, s, p, d)),
                ("conv.dgrad", () => engine.Conv2DBackwardInputInto(dx, dy, w, x.Shape.ToArray(), s, p, d, accumulate: false)),
                ("conv.wgrad", () => engine.Conv2DBackwardKernelInto(dw, dy, x, w.Shape.ToArray(), s, p, d, accumulate: false)),
                ("conv.fwd2", () => engine.Conv2DInto(y, x, w, s, p, d)),
            };
            foreach (var op in ops)
            {
                var sw0 = Stopwatch.StartNew();
                while (sw0.ElapsedMilliseconds < 300) op.Run();
                var t = new List<double>();
                for (int round = 0; round < 25; round++)
                {
                    double best = double.MaxValue;
                    for (int i = 0; i < 3; i++) { var sw = Stopwatch.StartNew(); op.Run(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
                    t.Add(best);
                }
                t.Sort();
                string line = $"ZZ {op.Name,-11} resnet18 b8 {c.Cin}->{c.Cout} {c.Hw}x{c.Hw} k{c.K} s{c.S} x{c.N}  {t[0]:F3}/{t[t.Count / 2]:F3}";
                _o.WriteLine(line); Console.WriteLine(line);
            }
        }
    }
}
