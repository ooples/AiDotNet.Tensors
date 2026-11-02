// SCRATCH (probe branch only): ResNet-18 deep-layer convs, direct vs GEMM routes vs Winograd, interleaved in one process.
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

public class ZzDeepConvBench
{
    private readonly ITestOutputHelper _o;
    public ZzDeepConvBench(ITestOutputHelper o) => _o = o;

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
        var rng = new Random(5);
        var all = DirectConvAvx2.EnabledPasses;
        int fewP = WinogradF43.s_fewMaxPatches;
        foreach (var (m, k, n) in new[] { (512, 4608, 128), (256, 2304, 512), (512, 2304, 128), (128, 1152, 2048) })
        {
            var a = Rand(rng, m, k); var bb = Rand(rng, k, n); var c = new Tensor<float>(new[] { m, n });
            var sw0 = Stopwatch.StartNew(); while (sw0.ElapsedMilliseconds < 250) engine.MatMulInto(c, a, bb);
            var t = new List<double>();
            for (int round = 0; round < 15; round++)
            {
                double best = double.MaxValue;
                for (int i = 0; i < 3; i++) { var sw = Stopwatch.StartNew(); engine.MatMulInto(c, a, bb); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
                t.Add(best);
            }
            t.Sort();
            var ml = $"ZM {m}x{k}x{n}  ours {t[0]:F3}/{t[t.Count / 2]:F3}"; _o.WriteLine(ml); Console.WriteLine(ml);
        }
        var variants = new (string Name, Action Set)[]
        {
            ("cur", () => { DirectConvAvx2.EnabledPasses = all; WinogradF43.s_disable = false; WinogradF43.s_fewMaxPatches = fewP; }),
            ("nofew", () => { DirectConvAvx2.EnabledPasses = all; WinogradF43.s_disable = false; WinogradF43.s_fewMaxPatches = 0; }),
            ("few32", () => { DirectConvAvx2.EnabledPasses = all; WinogradF43.s_disable = false; WinogradF43.s_fewMaxPatches = 32; }),
            ("nowino", () => { DirectConvAvx2.EnabledPasses = all; WinogradF43.s_disable = true; WinogradF43.s_fewMaxPatches = fewP; }),
            ("gemm", () => { DirectConvAvx2.EnabledPasses = DirectConvPasses.None; WinogradF43.s_disable = true; WinogradF43.s_fewMaxPatches = fewP; }),
        };
        var convs = new (int Cin, int Cout, int Hw, int S)[]
        {
            (512, 512, 4, 1), (256, 256, 8, 1), (256, 512, 8, 2), (128, 256, 16, 2), (64, 128, 32, 2), (128, 128, 16, 1),
            (3, 64, 224, 2), (64, 64, 56, 1), (64, 64, 32, 1),
        };
        const int b = 8;
        Torch();
        foreach (var c in convs)
        {
            int k = c.Cin == 3 ? 7 : 3, pad = c.Cin == 3 ? 3 : 1;
            int o = (c.Hw + 2 * pad - k) / c.S + 1;
            var x = Rand(rng, b, c.Cin, c.Hw, c.Hw); var w = Rand(rng, c.Cout, c.Cin, k, k); var dy = Rand(rng, b, c.Cout, o, o);
            var y = new Tensor<float>(new[] { b, c.Cout, o, o }); var dx = new Tensor<float>(x.Shape.ToArray());
            var dw = new Tensor<float>(w.Shape.ToArray());
            int[] s = { c.S, c.S }, p = { pad, pad }, d = { 1, 1 };
            var ops = new (string Name, Action Run)[]
            {
                ("fwd", () => engine.Conv2DInto(y, x, w, s, p, d)),
                ("dgrad", () => engine.Conv2DBackwardInputInto(dx, dy, w, x.Shape.ToArray(), s, p, d, accumulate: false)),
                ("wgrad", () => engine.Conv2DBackwardKernelInto(dw, dy, x, w.Shape.ToArray(), s, p, d, accumulate: false)),
            };
            foreach (var op in ops)
            {
                {
                    var outT = op.Name == "fwd" ? y : op.Name == "dgrad" ? dx : dw;
                    variants[0].Set(); op.Run(); var refv = outT.ToArray();
                    variants[1].Set(); op.Run(); var got = outT.ToArray();
                    double md = 0, mx = 0; for (int i = 0; i < refv.Length; i++) { md = Math.Max(md, Math.Abs(refv[i] - got[i])); mx = Math.Max(mx, Math.Abs(refv[i])); }
                    var dl = $"ZZ {c.Cin}->{c.Cout} {c.Hw}x{c.Hw} s{c.S} {op.Name,-5} nofew maxdiff {md:G3} of {mx:G3}"; _o.WriteLine(dl); Console.WriteLine(dl);
                }
                var times = variants.ToDictionary(v => v.Name, _ => new List<double>());
                foreach (var v in variants) { v.Set(); var sw0 = Stopwatch.StartNew(); while (sw0.ElapsedMilliseconds < 250) op.Run(); }
                for (int round = 0; round < 15; round++)
                    foreach (var v in variants)
                    {
                        v.Set(); op.Run();
                        double best = double.MaxValue;
                        for (int i = 0; i < 3; i++) { var sw = Stopwatch.StartNew(); op.Run(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
                        times[v.Name].Add(best);
                    }
                string line = $"ZZ {c.Cin}->{c.Cout} {c.Hw}x{c.Hw} s{c.S} {op.Name,-5}";
                foreach (var v in variants) { var t = times[v.Name]; t.Sort(); line += $"  {v.Name} {t[0]:F3}/{t[t.Count / 2]:F3}"; }
                _o.WriteLine(line); Console.WriteLine(line);
            }
        }
        DirectConvAvx2.EnabledPasses = all; WinogradF43.s_disable = false; WinogradF43.s_fewMaxPatches = fewP;
        Torch();
    }

    private void Torch()
    {
        var py = Environment.GetEnvironmentVariable("PARITY_PYTHON");
        if (string.IsNullOrEmpty(py)) return;
        var dir = AppContext.BaseDirectory;
        while (dir != null && !System.IO.File.Exists(System.IO.Path.Combine(dir, "tools/parity/zz_deep_torch.py"))) dir = System.IO.Path.GetDirectoryName(dir);
        if (dir == null) return;
        var psi = new ProcessStartInfo(py, System.IO.Path.Combine(dir, "tools/parity/zz_deep_torch.py")) { RedirectStandardOutput = true, UseShellExecute = false };
        using var proc = Process.Start(psi)!;
        string? l;
        while ((l = proc.StandardOutput.ReadLine()) != null) { _o.WriteLine(l); Console.WriteLine(l); }
        proc.WaitForExit();
    }
}
