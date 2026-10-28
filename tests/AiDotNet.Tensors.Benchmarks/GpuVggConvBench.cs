using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// VGG-16's thirteen 3x3 convolutions (batch 1, 224x224, bias + ReLU) on OpenCL (#1432): for each layer, the forward
/// slot's own choice against every applicable candidate, timed as paired AB/BA brackets
/// (<see cref="StableTimer.MeasurePairedRatioBound"/>) so a desktop GPU's noise widens the interval instead of deciding the
/// answer. Candidates run directly, not through the slot, whose per-shape decision cache would otherwise serve the first
/// decision to every later "forced" run.
/// Usage: --gpu-vgg-convs [pairs]   (default 21 pairs per comparison)
/// </summary>
internal static class GpuVggConvBench
{
    private static readonly (int C, int K, int Hw)[] Layers =
    {
        (3, 64, 224), (64, 64, 224),
        (64, 128, 112), (128, 128, 112),
        (128, 256, 56), (256, 256, 56), (256, 256, 56),
        (256, 512, 28), (512, 512, 28), (512, 512, 28),
        (512, 512, 14), (512, 512, 14), (512, 512, 14),
    };

    public static void Run(string[] args)
    {
        int pairs = args.Length > 0 && int.TryParse(args[0], out int p) ? p : 21;
        using var backend = new OpenClBackend();
        if (!backend.IsAvailable) { Console.WriteLine("OpenCL is not available."); return; }
        var slot = backend.Conv2DForwardSlot;
        var rng = new Random(16);
        float[] Rand(int n)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;
            return a;
        }

        double tunedTotal = 0, bestTotal = 0;
        foreach (var (c, k, hw) in Layers)
        {
            using var input = backend.AllocateBuffer(Rand(c * hw * hw));
            using var kernel = backend.AllocateBuffer(Rand(k * c * 9));
            using var bias = backend.AllocateBuffer(Rand(k));
            using var output = backend.AllocateBuffer(k * hw * hw);
            var a = new TunedConv2DArgs(input, kernel, output, 1, c, hw, hw, k, hw, hw, 3, 3, 1, 1, 1, 1, 1, 1,
                bias, FusedActivationType.ReLU);
            var shape = a.Shape();
            TunedKernelPolicy.SetOverride(TunedKernelOp.Conv2DForward, null);
            slot.InvalidateDecisions();
            var tuned = slot.Resolve(shape, a);
            if (tuned is null) { Console.WriteLine($"{c}->{k} @{hw}: no candidate applies"); continue; }
            double flops = 2.0 * k * c * 9 * hw * hw;
            double tunedMs = MedianMs(() => tuned.Execute(a), backend.Synchronize);
            tunedTotal += tunedMs;
            Console.WriteLine($"{c}->{k} @{hw}: slot picks {tuned.Id}  {tunedMs:F3} ms ({flops / tunedMs / 1e9:F1} TFLOP/s)");

            double bestMs = tunedMs;
            foreach (var id in slot.CandidateIds)
            {
                var candidate = slot.Candidate(id);
                if (candidate is null || id == tuned.Id || !candidate.IsApplicable(shape)) continue;
                StableTimer.RatioBound bound;
                try
                {
                    bound = StableTimer.MeasurePairedRatioBound(() => tuned.Execute(a), backend.Synchronize,
                        () => candidate.Execute(a), backend.Synchronize, pairs);
                }
                catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException)
                {
                    Console.WriteLine($"    {id,-36} failed: {ex.GetType().Name}");
                    continue;
                }
                string verdict = bound.Lower > 1.0 ? "FASTER than the pick" : bound.Upper < 1.0 ? "slower" : "no verdict";
                Console.WriteLine($"    {id,-36} speedup {bound.Median:F2}x [{bound.Lower:F2}, {bound.Upper:F2}]  {verdict}");
                if (bound.Lower > 1.0) bestMs = Math.Min(bestMs, tunedMs / bound.Median);
            }
            bestTotal += bestMs;
        }
        Console.WriteLine($"total: slot's picks {tunedTotal:F2} ms; with every clearly faster candidate {bestTotal:F2} ms");
        Console.WriteLine("speedup = pick time / candidate time, median of paired AB/BA ratios with its 95% interval");
    }

    private static double MedianMs(Action launch, Action synchronize)
    {
        for (int w = 0; w < 5; w++) launch();
        synchronize();
        var times = new double[21];
        for (int i = 0; i < times.Length; i++)
        {
            var sw = Stopwatch.StartNew();
            launch();
            synchronize();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(times);
        return times[times.Length / 2];
    }
}
