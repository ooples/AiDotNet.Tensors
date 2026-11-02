// SCRATCH (probe branch only): Winograd vs direct vs blocked on ResNet-18 3x3 convs, interleaved in one process.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Parity;

public class ZzConvScratchBench
{
    private readonly ITestOutputHelper _o;
    public ZzConvScratchBench(ITestOutputHelper o) => _o = o;

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
        var variants = new (string Name, Action Set)[]
        {
            ("wino", () => { WinogradF43.s_disable = false; WinogradF43.s_forceBlockedForTest = false; }),
            ("direct", () => { WinogradF43.s_disable = true; WinogradF43.s_forceBlockedForTest = false; }),
            ("blocked", () => { WinogradF43.s_disable = false; WinogradF43.s_forceBlockedForTest = true; }),
        };
        foreach (var (c, hw) in new[] { (64, 32), (128, 16), (256, 8) })
        {
            int b = 8;
            var x = Rand(rng, b, c, hw, hw); var w = Rand(rng, c, c, 3, 3); var dy = Rand(rng, b, c, hw, hw);
            var y = new Tensor<float>(new[] { b, c, hw, hw }); var dx = new Tensor<float>(new[] { b, c, hw, hw });
            var dw = new Tensor<float>(new[] { c, c, 3, 3 });
            int[] s = { 1, 1 }, p = { 1, 1 }, d = { 1, 1 };
            var ops = new (string Name, Action Run)[]
            {
                ("fwd", () => engine.Conv2DInto(y, x, w, s, p, d)),
                ("dgrad", () => engine.Conv2DBackwardInputInto(dx, dy, w, x.Shape.ToArray(), s, p, d, accumulate: false)),
                ("wgrad", () => engine.Conv2DBackwardKernelInto(dw, dy, x, w.Shape.ToArray(), s, p, d, accumulate: false)),
            };
            foreach (var op in ops)
            {
                var times = variants.ToDictionary(v => v.Name, _ => new List<double>());
                foreach (var v in variants) { v.Set(); var sw0 = Stopwatch.StartNew(); while (sw0.ElapsedMilliseconds < 300) op.Run(); }
                for (int round = 0; round < 25; round++)
                    foreach (var v in variants)
                    {
                        v.Set(); op.Run();
                        double best = double.MaxValue;
                        for (int i = 0; i < 5; i++) { var sw = Stopwatch.StartNew(); op.Run(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
                        times[v.Name].Add(best);
                    }
                string line = $"ZZ {c}->{c} {hw}x{hw} {op.Name,-5}";
                foreach (var v in variants) { var t = times[v.Name]; t.Sort(); line += $"  {v.Name} {t[0]:F3}/{t[t.Count / 2]:F3}"; }
                _o.WriteLine(line); Console.WriteLine(line);
            }
        }
        WinogradF43.s_disable = false; WinogradF43.s_forceBlockedForTest = false;
    }
}
