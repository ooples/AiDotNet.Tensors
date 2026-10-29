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
        int pairs = 21;
        if (args.Length > 0 && (!int.TryParse(args[0], out pairs) || pairs < 6))
        {
            // StableTimer.MeasurePairedRatioBound needs 6 pairs for a 95% interval.
            Console.WriteLine($"--gpu-vgg-convs [pairs]: pairs must be an integer of at least 6, got '{args[0]}'.");
            return;
        }
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
            // What FusedConv2D enters for host-authoritative weights outside a tape: routes that derive a form from the
            // weights (the F(4x4, 3x3) filter transform) cache it, as they do in real inference.
            var weightKey = new object();
            Action Run(ITunedKernelCandidate<TunedConv2DArgs> candidate) => () =>
            {
                using (ConvWeightIdentity.Enter(weightKey, 0)) candidate.Execute(a);
            };
            TunedKernelPolicy.SetOverride(TunedKernelOp.Conv2DForward, null);
            slot.InvalidateDecisions();
            ITunedKernelCandidate<TunedConv2DArgs>? tuned;
            using (ConvWeightIdentity.Enter(weightKey, 0)) tuned = slot.Resolve(shape, a);
            if (tuned is null) { Console.WriteLine($"{c}->{k} @{hw}: no candidate applies"); continue; }
            double flops = 2.0 * k * c * 9 * hw * hw;
            double tunedMs = MedianMs(Run(tuned), backend.Synchronize);
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
                    bound = StableTimer.MeasurePairedRatioBound(Run(tuned), backend.Synchronize,
                        Run(candidate), backend.Synchronize, pairs);
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

    /// <summary>
    /// The F(4x4, 3x3) route's three per-call stages on each VGG layer (the filter transform is cached by the real route):
    /// input transform, the 36 batched GEMMs, and the output transform with its epilogue. Usage: --gpu-vgg-f43-stages
    /// </summary>
    public static void RunF43Stages()
    {
        using var backend = new OpenClBackend();
        if (!backend.IsAvailable) { Console.WriteLine("OpenCL is not available."); return; }
        IWinogradF43Transforms t = backend;
        var rng = new Random(16);
        float[] Rand(int n)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;
            return a;
        }
        Console.WriteLine($"{"layer",-16}{"input ms",10}{"gemm ms",10}{"output ms",10}{"gemm TFLOP/s",14}");
        foreach (var (c, k, hw) in Layers)
        {
            int tiles = (hw + 3) / 4, p = tiles * tiles;
            using var x = backend.AllocateBuffer(Rand(c * hw * hw));
            using var w = backend.AllocateBuffer(Rand(k * c * 9));
            using var bias = backend.AllocateBuffer(Rand(k));
            using var y = backend.AllocateBuffer(k * hw * hw);
            using var u = backend.AllocateBuffer(36 * k * c);
            using var v = backend.AllocateBuffer(36 * c * p);
            using var m = backend.AllocateBuffer(36 * k * p);
            t.FilterF43(w, u, k, c);
            double input = MedianMs(() => t.InputF43(x, v, 1, c, hw, hw, 1, tiles, tiles), backend.Synchronize);
            double gemm = MedianMs(() => backend.BatchedGemm(u, v, m, k, p, c, 36), backend.Synchronize);
            double output = MedianMs(() => t.OutputF43(m, y, 1, k, hw, hw, tiles, tiles, bias, 3), backend.Synchronize);
            Console.WriteLine($"{$"{c}->{k} @{hw}",-16}{input,10:F3}{gemm,10:F3}{output,10:F3}{2.0 * 36 * k * p * c / gemm / 1e9,14:F2}");
        }
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
