using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Focused A/B benchmark for the Conv3x3Stride1 variants (per-channel,
/// 4-oc-blocked, 2-oc-blocked). Same process, same warmup, same data —
/// answers "which variant is fastest at the BDN benchmark shape?" without
/// the BDN-subprocess overhead that obscured the prior comparison.
///
/// Approach:
///  1. Build deterministic input/kernel tensors once
///  2. Reflectively poke the static <c>SimdConvHelper.ActiveConv3x3Variant</c>
///     field to switch variants without env vars
///  3. Warmup each variant for fixed iters, then time fixed iters
///  4. Report mean + min + stddev side-by-side
///
/// Run with: <c>dotnet run -c Release -- --ab-conv2d</c>
/// </summary>
internal static class Conv2DAbBench
{
    public static void Run()
    {
        Console.WriteLine("=== Conv3x3Stride1 A/B variant benchmark ===");
        Console.WriteLine($"OS: {System.Runtime.InteropServices.RuntimeInformation.OSDescription}");
        Console.WriteLine($"Cores: {Environment.ProcessorCount}");
        Console.WriteLine();

        // Reflective access to internal SimdConvHelper.ActiveConv3x3Variant
        var helperType = typeof(CpuEngine).Assembly
            .GetType("AiDotNet.Tensors.Helpers.SimdConvHelper", throwOnError: true);
        var variantField = helperType!.GetField("ActiveConv3x3Variant",
            BindingFlags.Static | BindingFlags.NonPublic);
        var variantEnum = helperType.GetNestedType("Conv3x3Variant",
            BindingFlags.NonPublic);
        if (variantField is null || variantEnum is null)
            throw new InvalidOperationException("Could not locate SimdConvHelper.ActiveConv3x3Variant via reflection.");

        // Shapes: cover the BDN benchmark + a few representative ResNet/transformer shapes
        var shapes = new (int b, int ic, int h, int w, int oc)[]
        {
            (1, 16, 64, 64, 32),    // BDN benchmark
            (1, 32, 32, 32, 64),    // medium ResNet
            (1, 64, 32, 32, 64),    // wider ResNet
            (1, 64, 16, 16, 128),   // deeper ResNet
            (4, 16, 64, 64, 32),    // batched
        };

        foreach (var (b, ic, h, w, oc) in shapes)
        {
            Console.WriteLine($"--- shape: input=[{b},{ic},{h},{w}] kernel=[{oc},{ic},3,3] ---");
            var (input, kernel) = MakeTensors(b, ic, h, w, oc, seed: 42);
            var engine = new CpuEngine();

            var results = new (string name, double meanUs, double minUs, double stddevUs)[3];
            string[] variants = { "PerChannel", "Block2", "Block4" };
            for (int v = 0; v < variants.Length; v++)
            {
                object enumValue = Enum.Parse(variantEnum, variants[v]);
                variantField.SetValue(null, enumValue);
                results[v] = TimeIt(engine, input, kernel, label: variants[v]);
            }

            Console.WriteLine($"  {"Variant",-12} {"Mean (µs)",10} {"Min (µs)",10} {"StdDev (µs)",12}");
            foreach (var r in results)
                Console.WriteLine($"  {r.name,-12} {r.meanUs,10:F1} {r.minUs,10:F1} {r.stddevUs,12:F1}");
            var fastest = results.OrderBy(r => r.minUs).First();
            Console.WriteLine($"  -> fastest by min: {fastest.name} @ {fastest.minUs:F1} µs");
            Console.WriteLine();
        }

        // Restore default
        variantField.SetValue(null, Enum.Parse(variantEnum, "Auto"));
    }

    private static (Tensor<float> input, Tensor<float> kernel) MakeTensors(
        int b, int ic, int h, int w, int oc, int seed)
    {
        var rng = new Random(seed);
        var inputData = new float[b * ic * h * w];
        var kernelData = new float[oc * ic * 9];
        for (int i = 0; i < inputData.Length; i++)  inputData[i]  = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kernelData.Length; i++) kernelData[i] = (float)(rng.NextDouble() * 2 - 1);
        return (new Tensor<float>(inputData, new[] { b, ic, h, w }),
                new Tensor<float>(kernelData, new[] { oc, ic, 3, 3 }));
    }

    private static (string name, double meanUs, double minUs, double stddevUs)
        TimeIt(CpuEngine engine, Tensor<float> input, Tensor<float> kernel, string label)
    {
        const int warmupIters = 20;
        const int measureIters = 100;

        // Warmup — JIT, populate caches, etc.
        for (int i = 0; i < warmupIters; i++)
        {
            var r = engine.Conv2D(input, kernel, 1, 1, 1);
            // Drop reference so AutoTensorCache can recycle. No explicit return
            // since we want the same alloc behavior the public API has.
            _ = r;
        }

        // Measure
        var samples = new double[measureIters];
        var sw = new Stopwatch();
        for (int i = 0; i < measureIters; i++)
        {
            sw.Restart();
            var r = engine.Conv2D(input, kernel, 1, 1, 1);
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMicroseconds;
            _ = r;
        }

        double mean = samples.Average();
        double min = samples.Min();
        double sumSq = 0;
        foreach (var s in samples) sumSq += (s - mean) * (s - mean);
        double stddev = Math.Sqrt(sumSq / samples.Length);
        return (label, mean, min, stddev);
    }
}
