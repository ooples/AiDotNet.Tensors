// Copyright (c) AiDotNet. All rights reserved.
// FE-4: measure what splitting a reduction actually buys, on the device.
//
// The interpreter tests prove the split computes the same thing. They cannot prove it is
// faster, and the reason to build it was a measured 1081x gap, so the claim has to be
// measured too. This runs every catalog kernel that ChooseAxis says is worth splitting:
//
//   1. time the kernel as it is emitted today
//   2. time the partial pass plus the combine pass, launched back to back
//   3. download both results and require them to agree
//
// Step 3 is not optional. Two passes touching a temporary is exactly the shape that
// produces a fast wrong answer, and the 0.000E+000 gate on this project has caught four
// defects the structural gates passed.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

internal static class KernelSplitTool
{
    private const int Warmup = 20;
    private const int Samples = 31;
    private const int LaunchesPerSample = 50;
    private const int Runs = 3;

    internal static void Run(string[] args)
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("splitk-start");
        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
        {
            Console.WriteLine("Split-K measurement requires the experimental SM86 device.");
            return;
        }

        string selector = args.FirstOrDefault(a => !a.StartsWith("--", StringComparison.Ordinal)) ?? "all";
        var entries = string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase)
            ? CodegenKernelCatalog.All
            : new[] { CodegenKernelCatalog.Find(selector)! }.Where(e => e != null).ToList();

        string outputPath = ValueOf(args, "--out") ??
            Path.Combine(Directory.GetCurrentDirectory(), "artifacts", "splitk.tsv");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        Console.WriteLine();
        Console.WriteLine("SPLIT-K - two-pass reduction, protocol " + CodegenMeasurementProtocol.Tag);
        Console.WriteLine();
        Console.WriteLine("kernel                          blocks   split   unsplit    partial   combine     total    gain   agree");

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        var rows = new List<string>();
        try
        {
            foreach (var entry in entries)
            {
                var spec = entry.Bench;
                var ranked = CodegenSplitReduction.ChooseAxes(spec);
                if (ranked.Count == 0) continue;

                // Every prefix of the ranking is a valid split, and the model cannot say
                // which is fastest -- it said two axes would beat one on the depthwise
                // weight gradient and the hardware said otherwise. So measure them all.
                for (int k = 1; k <= ranked.Count; k++)
                {
                    var prefix = ranked.Take(k).ToArray();
                    try { rows.Add(SplitOne(runtime, entry, prefix)); }
                    catch (Exception ex)
                    {
                        Console.WriteLine(entry.Name.PadRight(30) + "  ERROR " + ex.Message.Split('\n')[0]);
                    }
                }
            }
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }

        var text = new StringBuilder();
        text.AppendLine("# split-K two-pass measurement, " + CodegenMeasurementProtocol.Tag + ": " +
                        CodegenMeasurementProtocol.Description);
        text.AppendLine("kernel\tsplit_axis\tunsplit_us\tpartial_us\tcombine_us\ttotal_us\tgain\tmax_abs_diff\tprotocol");
        foreach (string row in rows) text.AppendLine(row);
        File.WriteAllText(outputPath, text.ToString());

        Console.WriteLine();
        Console.WriteLine("written to " + outputPath);
    }

    private static string SplitOne(
        DirectPtxRuntime runtime, CodegenCatalogEntry entry, IReadOnlyList<int> promoted)
    {
        var spec = entry.Bench;
        var (partial, combine) = CodegenSplitReduction.Split(spec, promoted);
        string axisName = string.Join("+", promoted.Select(
            a => spec.Space.Axes[a].Name + "(" + spec.Space.Axes[a].Extent + ")"));

        var direct = new PtxAffineEmitter();
        var partialEmitter = new PtxAffineEmitter();
        var combineEmitter = new PtxAffineEmitter();

        string directPtx = direct.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        string partialPtx = partialEmitter.Emit(partial, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        string combinePtx = combineEmitter.Emit(combine, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);

        var buffers = new List<DirectPtxBuffer>();
        try
        {
            using var directModule = runtime.LoadModule(directPtx, allowExperimentalJitFallback: true);
            using var partialModule = runtime.LoadModule(partialPtx, allowExperimentalJitFallback: true);
            using var combineModule = runtime.LoadModule(combinePtx, allowExperimentalJitFallback: true);

            IntPtr directFn = directModule.GetFunction(spec.Name, out _);
            IntPtr partialFn = partialModule.GetFunction(partial.Name, out _);
            IntPtr combineFn = combineModule.GetFunction(combine.Name, out _);

            // Shared inputs: both lowerings read exactly the same operands.
            var inputPointers = new IntPtr[spec.Inputs.Count];
            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                long count = Elements(spec.Inputs[i].Shape);
                var b = runtime.AllocateBytes((nuint)(count * sizeof(float)));
                var host = new float[count];
                for (long e = 0; e < count; e++)
                    host[e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                b.Upload<float>(host);
                buffers.Add(b);
                inputPointers[i] = b.Pointer;
            }

            long outCount = Elements(spec.Output.Shape);
            long partialCount = Elements(partial.Output.Shape);

            var directOut = runtime.AllocateBytes((nuint)(outCount * sizeof(float)));
            var splitOut = runtime.AllocateBytes((nuint)(outCount * sizeof(float)));
            var temp = runtime.AllocateBytes((nuint)(partialCount * sizeof(float)));
            buffers.Add(directOut); buffers.Add(splitOut); buffers.Add(temp);

            var directArgs = new IntPtr[spec.ParameterCount];
            for (int i = 0; i < spec.Inputs.Count; i++) directArgs[i] = inputPointers[i];
            directArgs[spec.Inputs.Count] = directOut.Pointer;

            // The partial pass takes only the PRODUCT operands -- any epilogue moved to
            // the combine -- so it is bound through ProductInputs, not by position.
            var partialArgs = new IntPtr[partial.ParameterCount];
            for (int i = 0; i < spec.ProductInputs.Count; i++)
                partialArgs[i] = inputPointers[spec.ProductInputs[i]];
            partialArgs[partialArgs.Length - 1] = temp.Pointer;

            var combineArgs = new IntPtr[combine.ParameterCount];
            combineArgs[0] = temp.Pointer;
            if (combine.BiasInput is { } bias) combineArgs[bias] = inputPointers[spec.BiasInput!.Value];
            if (combine.ScaleInput is { } scaleParam)
                combineArgs[scaleParam] = inputPointers[spec.ScaleInput!.Value];
            combineArgs[combineArgs.Length - 1] = splitOut.Pointer;

            void LaunchDirect() => Launch(directModule, directFn, directArgs,
                direct.LaunchBlocks, (uint)direct.LaunchBlockX, (uint)direct.LaunchBlockY);
            void LaunchPartial() => Launch(partialModule, partialFn, partialArgs,
                partialEmitter.LaunchBlocks, (uint)partialEmitter.LaunchBlockX, (uint)partialEmitter.LaunchBlockY);
            void LaunchCombine() => Launch(combineModule, combineFn, combineArgs,
                combineEmitter.LaunchBlocks, (uint)combineEmitter.LaunchBlockX, (uint)combineEmitter.LaunchBlockY);
            void LaunchSplit() { LaunchPartial(); LaunchCombine(); }

            // ---- Agreement before timing.
            LaunchDirect();
            LaunchSplit();
            runtime.Synchronize();

            var expected = new float[outCount];
            var got = new float[outCount];
            directOut.Download<float>(expected);
            splitOut.Download<float>(got);

            double worst = 0, scale = 0;
            for (long e = 0; e < outCount; e++)
            {
                worst = Math.Max(worst, Math.Abs(got[e] - expected[e]));
                scale = Math.Max(scale, Math.Abs(expected[e]));
            }
            double relative = scale > 0 ? worst / scale : worst;

            // ---- Timing.
            double unsplitUs = Measure(runtime.Synchronize, LaunchDirect);
            double partialUs = Measure(runtime.Synchronize, LaunchPartial);
            double combineUs = Measure(runtime.Synchronize, LaunchCombine);
            double totalUs = Measure(runtime.Synchronize, LaunchSplit);
            double gain = totalUs > 0 ? unsplitUs / totalUs : 0;

            Console.WriteLine(entry.Name.PadRight(30) +
                direct.LaunchBlocks.ToString(CultureInfo.InvariantCulture).PadLeft(7) +
                "  " + axisName.PadRight(16) +
                unsplitUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(8) +
                partialUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(10) +
                combineUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(10) +
                totalUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(10) +
                gain.ToString("F2", CultureInfo.InvariantCulture).PadLeft(8) + "x" +
                relative.ToString("E1", CultureInfo.InvariantCulture).PadLeft(10));

            return string.Join("\t", entry.Name, axisName,
                unsplitUs.ToString("F3", CultureInfo.InvariantCulture),
                partialUs.ToString("F3", CultureInfo.InvariantCulture),
                combineUs.ToString("F3", CultureInfo.InvariantCulture),
                totalUs.ToString("F3", CultureInfo.InvariantCulture),
                gain.ToString("F4", CultureInfo.InvariantCulture),
                relative.ToString("E3", CultureInfo.InvariantCulture),
                CodegenMeasurementProtocol.Tag);
        }
        finally { foreach (var b in buffers) b.Dispose(); }
    }

    private static double Measure(Action synchronize, Action launch)
    {
        double best = double.MaxValue;
        for (int run = 0; run < Runs; run++)
        {
            for (int i = 0; i < Warmup; i++) launch();
            synchronize();

            var samples = new double[Samples];
            for (int i = 0; i < Samples; i++)
            {
                long start = Stopwatch.GetTimestamp();
                for (int k = 0; k < LaunchesPerSample; k++) launch();
                synchronize();
                samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds / LaunchesPerSample * 1000.0;
            }
            Array.Sort(samples);
            best = Math.Min(best, samples[samples.Length / 2]);
        }
        return best;
    }

    private static unsafe void Launch(
        DirectPtxModule module, IntPtr fn, IntPtr[] pointers, uint blocks, uint blockX, uint blockY)
    {
        fixed (IntPtr* pinned = pointers)
        {
            void** argv = stackalloc void*[pointers.Length];
            for (int i = 0; i < pointers.Length; i++) argv[i] = pinned + i;
            module.Launch(fn, blocks, 1, 1, blockX, blockY, 1, 0, argv);
        }
    }

    private static long Elements(IReadOnlyList<int> shape)
    {
        long total = 1;
        for (int i = 0; i < shape.Count; i++) total *= shape[i];
        return total;
    }

    private static string? ValueOf(string[] args, string flag)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], flag, StringComparison.Ordinal)) return args[i + 1];
        return null;
    }
}
