using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// #1115: runs the tuned-kernel registry's evidence gate on the Vulkan device for a set of shape classes, several
/// independent rounds each, and writes a shippable profile holding only the shapes where the rounds agreed (see the
/// selection rule below). Usage: <c>--vulkan-tuned-profile [output.json] [rounds]</c>.
/// </summary>
internal static class VulkanTunedProfileExperiment
{
    private static readonly (int Rows, int N)[] RowShapes =
    {
        (8192, 32), (2048, 64), (4096, 64), (4096, 128), (1024, 256), (2364, 197), (512, 384), (512, 512),
        (256, 768), (128, 1024), (64, 4096),
    };

    // batch, inC, H, W, outC, k, stride, pad
    private static readonly (int B, int C, int H, int W, int O, int K, int S, int P)[] ConvShapes =
    {
        (64, 1, 28, 28, 16, 3, 1, 1), (64, 16, 14, 14, 32, 3, 1, 1), (32, 3, 32, 32, 16, 3, 1, 1),
        (16, 64, 16, 16, 64, 3, 1, 1), (8, 64, 32, 32, 128, 3, 2, 1), (8, 128, 16, 16, 128, 1, 1, 0),
        (4, 3, 224, 224, 64, 7, 2, 3),
    };

    public static void Run(string[] args)
    {
        string? output = args.Length > 0 ? args[0] : null;
        int rounds = args.Length > 1 ? int.Parse(args[1], CultureInfo.InvariantCulture) : 5;

        var backend = VulkanBackend.Instance;
        if (!backend.Initialize() || !backend.IsGlslCompilerAvailable)
        {
            Console.WriteLine("No Vulkan device with the runtime GLSL compiler.");
            return;
        }
        TunedKernelPolicy.ModeOverride = TunedKernelMode.Tune;
        TunedKernelPolicy.PersistDecisions = false;
        TunedKernelPolicy.BudgetMilliseconds = double.MaxValue;
        Console.WriteLine($"device {backend.TunedDeviceKey}; clock {(backend.TunedKernelTimingUsesDeviceClock ? "device timestamps" : "host")}; rounds {rounds}");

        // (op, shape) -> the decision of each round
        var history = new Dictionary<(TunedKernelOp, string), List<TunedKernelDecision>>();
        var rng = new Random(1115);
        float[] Values(int count)
        {
            var a = new float[count];
            for (int i = 0; i < count; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            return a;
        }

        for (int round = 0; round < rounds; round++)
        {
            foreach (var slot in VulkanSlots(backend)) slot.Invalidate();
            foreach (var (rows, n) in RowShapes)
            {
                using var x = backend.AllocateBuffer(Values(rows * n));
                using var dy = backend.AllocateBuffer(Values(rows * n));
                using var gamma = backend.AllocateBuffer(Values(n));
                using var beta = backend.AllocateBuffer(Values(n));
                using var y = backend.AllocateBuffer(rows * n);
                using var dx = backend.AllocateBuffer(rows * n);
                using var mean = backend.AllocateBuffer(rows);
                using var inv = backend.AllocateBuffer(rows);
                using var dGamma = backend.AllocateBuffer(n);
                using var dBeta = backend.AllocateBuffer(n);
                backend.Softmax(x, y, rows, n);
                backend.SoftmaxBackward(dy, y, dx, rows, n);
                backend.LayerNorm(x, y, gamma, beta, mean, inv, rows, n, 1e-5f);
                backend.LayerNormBackward(dy, x, gamma, mean, inv, dx, dGamma, dBeta, rows, n, 1e-5f);
                backend.RmsNorm(x, y, gamma, inv, rows, n, 1e-5f);
                backend.RmsNormBackward(dy, x, gamma, inv, dx, dGamma, rows, n, 1e-5f);
            }
            foreach (var c in ConvShapes)
            {
                int oh = (c.H + 2 * c.P - c.K) / c.S + 1, ow = (c.W + 2 * c.P - c.K) / c.S + 1;
                using var x = backend.AllocateBuffer(Values(c.B * c.C * c.H * c.W));
                using var w = backend.AllocateBuffer(Values(c.O * c.C * c.K * c.K));
                using var y = backend.AllocateBuffer(c.B * c.O * oh * ow);
                backend.Conv2D(x, w, y, c.B, c.C, c.H, c.W, c.O, oh, ow, c.K, c.K, c.S, c.S, c.P, c.P, 1, 1);
            }
            foreach (var slot in VulkanSlots(backend))
                foreach (var d in slot.Info.CachedDecisions)
                {
                    var key = (d.Op, d.Shape.ToString());
                    if (!history.TryGetValue(key, out var list)) history[key] = list = new List<TunedKernelDecision>();
                    list.Add(d);
                }
        }

        var stable = new List<TunedKernelDecision>();
        Console.WriteLine();
        Console.WriteLine("| op | shape | reference ms | chosen (rounds) | chosen ms | speedup per round | profile entry |");
        Console.WriteLine("|---|---|---|---|---|---|---|");
        foreach (var kv in history.OrderBy(k => k.Key.Item1).ThenBy(k => k.Key.Item2, StringComparer.Ordinal))
        {
            var list = kv.Value;
            var tuned = list.Where(d => d.Reason == TunedKernelDecisionReason.Tuned).ToList();
            bool allReference = list.Count == rounds && list.All(d => d.Reason == TunedKernelDecisionReason.ReferenceWon);
            bool allTuned = list.Count == rounds && tuned.Count == rounds;
            string speedups = string.Join(" ", list.Select(d =>
                d.Reason == TunedKernelDecisionReason.Tuned ? d.MedianSpeedup.ToString("F2", CultureInfo.InvariantCulture) + "x" : "ref"));
            string refMs = tuned.Count > 0 ? Median(tuned.Select(d => d.ReferenceMilliseconds)).ToString("F4", CultureInfo.InvariantCulture) : "-";
            string candMs = tuned.Count > 0 ? Median(tuned.Select(d => d.CandidateMilliseconds)).ToString("F4", CultureInfo.InvariantCulture) : "-";
            // A shape enters the profile when every round agreed the reference is best, or when every round selected a
            // generated variant; then the variant selected most often is taken (variants that all reach the dispatch
            // floor trade places between rounds), represented by its middle round. A shape where rounds disagreed about
            // the reference is left to online tuning.
            TunedKernelDecision? pick = null;
            if (allReference) pick = list[0];
            else if (allTuned)
            {
                var modal = tuned.GroupBy(d => d.CandidateId)
                    .OrderByDescending(g => g.Count()).ThenByDescending(g => Median(g.Select(d => d.MedianSpeedup))).First();
                pick = modal.OrderBy(d => d.MedianSpeedup).ElementAt(modal.Count() / 2);
            }
            string chosen = string.Join(" / ", list.GroupBy(d => d.CandidateId).OrderByDescending(g => g.Count())
                .Select(g => $"{g.Key} x{g.Count()}"));
            Console.WriteLine($"| {kv.Key.Item1} | {kv.Key.Item2} | {refMs} | {chosen} | {candMs} | {speedups} | {(pick is null ? "-" : pick.CandidateId)} |");
            if (pick is not null) stable.Add(pick);
        }

        string json = TunedKernelProfiles.ExportProfile(backend.TunedDeviceKey, stable, deterministic: true);
        if (output is not null)
        {
            File.WriteAllText(output, json + Environment.NewLine);
            Console.WriteLine();
            Console.WriteLine($"wrote {stable.Count} of {history.Count} shape classes to {output}");
        }
        else
        {
            Console.WriteLine();
            Console.WriteLine(json);
        }
    }

    private static double Median(IEnumerable<double> values)
    {
        double[] a = values.OrderBy(v => v).ToArray();
        return a[a.Length / 2];
    }

    private static IEnumerable<(ITunedKernelSlotInfo Info, Action Invalidate)> VulkanSlots(VulkanBackend b)
    {
        yield return (b.SoftmaxSlot, b.SoftmaxSlot.InvalidateDecisions);
        yield return (b.SoftmaxBackwardSlot, b.SoftmaxBackwardSlot.InvalidateDecisions);
        yield return (b.LayerNormSlot, b.LayerNormSlot.InvalidateDecisions);
        yield return (b.LayerNormBackwardSlot, b.LayerNormBackwardSlot.InvalidateDecisions);
        yield return (b.LayerNormGradParametersSlot, b.LayerNormGradParametersSlot.InvalidateDecisions);
        yield return (b.RmsNormGradGammaSlot, b.RmsNormGradGammaSlot.InvalidateDecisions);
        yield return (b.Conv2DForwardSlot, b.Conv2DForwardSlot.InvalidateDecisions);
    }
}
