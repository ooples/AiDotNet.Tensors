using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

// AiDotNet side of the #639 per-op head-to-head: forward, input-gradient and weight-gradient time of every MobileNetV3
// conv shape (mbv3_convs.json), median of `reps` timed calls after warmup, at a fixed DOP. Depthwise shapes (groups ==
// channels) run the engine's depthwise ops, the rest Conv2D, as the MobileNetV3 layers do.
// Usage: ConvH2H <dop> <reps> <shapes.json> <out.json>
internal static class Program
{
    private static double Median(Action fn, int reps)
    {
        for (int i = 0; i < 10; i++) fn();
        var ts = new double[reps];
        var sw = new Stopwatch();
        for (int i = 0; i < reps; i++)
        {
            sw.Restart();
            fn();
            ts[i] = sw.Elapsed.TotalMilliseconds * 1000.0;
        }
        Array.Sort(ts);
        return ts[reps / 2];
    }

    private static Tensor<float> Random(Random rng, params int[] shape)
    {
        int length = 1;
        foreach (int d in shape) length *= d;
        var data = new float[length];
        for (int i = 0; i < length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    private static int Main(string[] args)
    {
        int dop = int.Parse(args[0]), reps = int.Parse(args[1]);
        CpuParallelSettings.MaxDegreeOfParallelism = dop;
        var engine = new CpuEngine();
        var rng = new Random(0);
        var rows = new List<Dictionary<string, object>>();
        using var doc = JsonDocument.Parse(File.ReadAllText(args[2]));
        using var arena = TensorArena.Create();   // as the training loop: step buffers reused, not fresh LOH arrays
        foreach (var s in doc.RootElement.EnumerateArray())
        {
            int n = s.GetProperty("N").GetInt32(), c = s.GetProperty("C").GetInt32(), h = s.GetProperty("H").GetInt32();
            int w = s.GetProperty("W").GetInt32(), o = s.GetProperty("O").GetInt32(), k = s.GetProperty("K").GetInt32();
            int st = s.GetProperty("S").GetInt32(), p = s.GetProperty("P").GetInt32(), g = s.GetProperty("G").GetInt32();
            bool depthwise = g == c && g > 1;
            var x = Random(rng, n, c, h, w);
            var kernel = depthwise ? Random(rng, c, o / c, k, k) : Random(rng, o, c / g, k, k);
            var stride = new[] { st, st }; var pad = new[] { p, p }; var dil = new[] { 1, 1 };
            var y = depthwise ? engine.DepthwiseConv2D(x, kernel, stride, pad) : engine.Conv2D(x, kernel, stride, pad, dil);
            var gy = Random(rng, y.Shape.ToArray());
            int[] xShape = x.Shape.ToArray(), kShape = kernel.Shape.ToArray();
            double fwd, din, dw;
            if (depthwise)
            {
                fwd = Median(() => engine.DepthwiseConv2D(x, kernel, stride, pad), reps);
                din = Median(() => engine.DepthwiseConv2DBackwardInput(gy, kernel, xShape, stride, pad), reps);
                dw = Median(() => engine.DepthwiseConv2DBackwardKernel(gy, x, kShape, stride, pad), reps);
            }
            else
            {
                fwd = Median(() => engine.Conv2D(x, kernel, stride, pad, dil), reps);
                din = Median(() => engine.Conv2DBackwardInput(gy, kernel, xShape, stride, pad, dil), reps);
                dw = Median(() => engine.Conv2DBackwardKernel(gy, x, kShape, stride, pad, dil), reps);
            }
            rows.Add(new Dictionary<string, object>
            {
                ["N"] = n, ["C"] = c, ["H"] = h, ["W"] = w, ["O"] = o, ["K"] = k, ["S"] = st, ["P"] = p, ["G"] = g,
                ["count"] = s.GetProperty("count").GetInt32(), ["fwd"] = fwd, ["din"] = din, ["dw"] = dw,
            });
        }
        File.WriteAllText(args[3], JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"aidotnet dop {dop} shapes {rows.Count}");
        return 0;
    }
}
