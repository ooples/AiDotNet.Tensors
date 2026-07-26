// Copyright (c) AiDotNet. All rights reserved.
// FE-3: measure candidate lowerings instead of modelling them.
//
// Every lowering decision came from a static cost model with one fitted constant, and
// that model was wrong every time it was checked:
//
//   occupancy      predicted a 2.78x penalty where 1.46x was measured
//   tile search    picked a 4x8 tile that ran SLOWER than the 4x4 it replaced
//   staging        under a 2D block returned 5.277 instead of zero
//   transposed conv BOTH post-emission measures called the chosen tile worse
//                  (32.4 us vs 28.5 predicted, 1.250 vs 1.111 loads/MAC) and the
//                  hardware disagreed: 99.4 us vs 111.2, the pick was 1.12x FASTER
//
// The last one is decisive: a model cannot arbitrate lowering quality, because the
// models do not contain whatever made that kernel faster. Measurement does. This emits
// several lowerings of the SAME spec, checks they agree numerically, times them under
// the p4 protocol, and keeps the winner.

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

internal static class KernelAutotuneTool
{
    private const int Warmup = 20;
    private const int Samples = 31;
    private const int LaunchesPerSample = 50;
    private const int Runs = 3;

    /// <summary>One candidate lowering: a name and the knobs that produce it.</summary>
    private sealed record Candidate(string Name, Action<PtxAffineEmitter> Configure);

    private static readonly Candidate[] Candidates =
    {
        // The modelled choice, first so it is the reference every other is compared to.
        new("modelled", _ => { }),
        new("no-tile", e => e.Coarsening = 1),
        new("tile2", e => { e.Coarsening = 2; }),
        new("lanes4", e => { e.MaxTileLanes = 4; }),
        new("no-staging", e => e.EnableSharedStaging = false),
        new("no-vector", e => e.EnableVectorLoads = false),
    };

    internal static void Run(string[] args)
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("autotune-start");
        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
        {
            Console.WriteLine("Autotuning requires the experimental SM86 device.");
            return;
        }

        string selector = KernelToolArgs.Selector(args);
        var entries = string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase)
            ? CodegenKernelCatalog.All
            : new[] { CodegenKernelCatalog.Find(selector)! }.Where(e => e != null).ToList();
        KernelToolArgs.RequireNonEmptySelection(selector, entries.Count, "kernel-autotune");

        string outputPath = ValueOf(args, "--out") ??
            Path.Combine(Directory.GetCurrentDirectory(), "artifacts", "autotune.tsv");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        Console.WriteLine();
        Console.WriteLine("AUTOTUNE - measured candidate lowerings, protocol " + CodegenMeasurementProtocol.Tag);
        Console.WriteLine("candidates: " + string.Join(", ", Candidates.Select(c => c.Name)));
        Console.WriteLine();
        Console.WriteLine("kernel                          modelled   best      winner        gain");

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        var rows = new List<string>();
        int improved = 0, regressed = 0;
        try
        {
            foreach (var entry in entries)
            {
                try
                {
                    var (bestName, bestUs, modelledUs) = TuneOne(runtime, entry);
                    if (bestUs <= 0) continue;

                    double gain = modelledUs / bestUs;
                    if (gain > 1.03) improved++;
                    if (gain < 0.97) regressed++;

                    Console.WriteLine(entry.Name.PadRight(30) +
                        modelledUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(9) +
                        bestUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(9) +
                        "   " + bestName.PadRight(12) +
                        gain.ToString("F3", CultureInfo.InvariantCulture).PadLeft(7) + "x");

                    rows.Add(string.Join("\t", entry.Name, bestName,
                        bestUs.ToString("F3", CultureInfo.InvariantCulture),
                        modelledUs.ToString("F3", CultureInfo.InvariantCulture),
                        gain.ToString("F4", CultureInfo.InvariantCulture),
                        CodegenMeasurementProtocol.Tag));
                }
                catch (Exception ex)
                {
                    Console.WriteLine(entry.Name.PadRight(30) + "  ERROR " + ex.Message.Split('\n')[0]);
                }
            }
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }

        var text = new StringBuilder();
        text.AppendLine("# autotune winners, " + CodegenMeasurementProtocol.Tag + ": " +
                        CodegenMeasurementProtocol.Description);
        text.AppendLine("kernel\twinner\tbest_us\tmodelled_us\tgain\tprotocol");
        foreach (string row in rows) text.AppendLine(row);
        File.WriteAllText(outputPath, text.ToString());

        Console.WriteLine();
        Console.WriteLine(improved + " kernels improved past the 1.05% noise floor, " +
                          regressed + " regressed");
        Console.WriteLine("winners written to " + outputPath);
    }

    private static (string Name, double BestUs, double ModelledUs) TuneOne(
        DirectPtxRuntime runtime, CodegenCatalogEntry entry)
    {
        var spec = entry.Bench;
        string bestName = "modelled";
        double bestUs = double.MaxValue, modelledUs = 0;
        float[]? reference = null;

        foreach (var candidate in Candidates)
        {
            var emitter = new PtxAffineEmitter();
            candidate.Configure(emitter);

            string ptx;
            try { ptx = emitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor); }
            catch (NotSupportedException) { continue; }

            var buffers = new List<DirectPtxBuffer>();
            try
            {
                using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
                IntPtr fn = module.GetFunction(spec.Name, out _);

                var pointers = new IntPtr[spec.ParameterCount];
                for (int i = 0; i < spec.Inputs.Count; i++)
                {
                    long count = Elements(spec.Inputs[i].Shape);
                    var b = runtime.AllocateBytes((nuint)(count * sizeof(float)));
                    var host = new float[count];
                    for (long e = 0; e < count; e++)
                        host[e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                    b.Upload<float>(host);
                    buffers.Add(b);
                    pointers[i] = b.Pointer;
                }
                long outCount = Elements(spec.Output.Shape);
                var outBuffer = runtime.AllocateBytes((nuint)(outCount * sizeof(float)));
                buffers.Add(outBuffer);
                pointers[spec.Inputs.Count] = outBuffer.Pointer;

                void Launch() => LaunchOne(module, fn, pointers,
                    emitter.LaunchBlocks, (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY);

                // CORRECTNESS BEFORE SPEED. Every candidate lowers the SAME spec, so they
                // must agree; a faster candidate that computes something else is not a
                // faster candidate. The first one measured becomes the reference.
                Launch();
                runtime.Synchronize();
                var got = new float[outCount];
                outBuffer.Download<float>(got);

                if (reference is null) reference = got;
                else if (!Agrees(got, reference, out double deviation))
                {
                    Console.WriteLine("    candidate '" + candidate.Name + "' disagrees by " +
                                      deviation.ToString("E3", CultureInfo.InvariantCulture) +
                                      " relative; rejected");
                    continue;
                }

                double us = Measure(runtime.Synchronize, Launch);
                if (candidate.Name == "modelled") modelledUs = us;
                if (us < bestUs) { bestUs = us; bestName = candidate.Name; }
            }
            finally { foreach (var b in buffers) b.Dispose(); }
        }

        if (modelledUs <= 0) modelledUs = bestUs;

        // The split is a candidate like any other lowering, and for the same reason:
        // whether it wins cannot be predicted. Block count says the opposite of the truth
        // (a 64-block kernel won 2.05x while a 16-block one lost 1.89x) and arithmetic
        // volume does not separate the cases -- the reduction that won and the linear
        // layer that lost do the SAME 131,072 multiply-accumulates per block. See
        // docs/SPLIT_K_REDUCTION.md.
        //
        // Run LAST, deliberately: `reference` now holds a single-kernel result, so the
        // split is checked against the established answer. Running it first would let a
        // wrong split become the reference and reject the correct kernels.
        if (reference is not null &&
            TrySplitCandidate(runtime, spec, reference) is { } split && split.Us < bestUs)
        {
            bestUs = split.Us;
            bestName = split.Name;
        }

        return (bestName, bestUs, modelledUs);
    }

    /// <summary>
    /// Times the two-kernel split route, or null when there is no split or it disagrees.
    /// </summary>
    /// <remarks>
    /// <paramref name="reference"/> is an established single-kernel result, so the split
    /// is held to exactly the same numerical bar as every other lowering. A two-kernel
    /// path through a temporary is the shape that produces a fast wrong answer, and a
    /// tuner that preferred it on speed alone would install one.
    /// </remarks>
    private static (string Name, double Us)? TrySplitCandidate(
        DirectPtxRuntime runtime, CodegenKernelSpec spec, float[] reference)
    {
        CodegenSplitPlan? plan;
        try { plan = CodegenSplitReduction.TryPlan(spec); }
        catch (NotSupportedException) { return null; }
        if (plan is null) return null;

        var buffers = new List<DirectPtxBuffer>();
        try
        {
            var partialEmitter = new PtxAffineEmitter();
            var combineEmitter = new PtxAffineEmitter();
            string partialPtx, combinePtx;
            try
            {
                partialPtx = partialEmitter.Emit(plan.Partial,
                    runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                combinePtx = combineEmitter.Emit(plan.Combine,
                    runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
            }
            catch (NotSupportedException) { return null; }

            using var partialModule = runtime.LoadModule(partialPtx, allowExperimentalJitFallback: true);
            using var combineModule = runtime.LoadModule(combinePtx, allowExperimentalJitFallback: true);
            IntPtr partialFn = partialModule.GetFunction(plan.Partial.Name, out _);
            IntPtr combineFn = combineModule.GetFunction(plan.Combine.Name, out _);

            var uploaded = new IntPtr[spec.Inputs.Count];
            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                long count = Elements(spec.Inputs[i].Shape);
                var b = runtime.AllocateBytes((nuint)(count * sizeof(float)));
                var host = new float[count];
                for (long e = 0; e < count; e++)
                    host[e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                b.Upload<float>(host);
                buffers.Add(b);
                uploaded[i] = b.Pointer;
            }

            var temp = runtime.AllocateBytes((nuint)(plan.TempElements * sizeof(float)));
            buffers.Add(temp);
            long outCount = Elements(spec.Output.Shape);
            var outBuffer = runtime.AllocateBytes((nuint)(outCount * sizeof(float)));
            buffers.Add(outBuffer);

            // The partial pass reads only the product operands; the epilogue moved to the
            // combine, so binding by position would hand it the bias.
            var partialArgs = new IntPtr[plan.Partial.ParameterCount];
            for (int i = 0; i < spec.ProductInputs.Count; i++)
                partialArgs[i] = uploaded[spec.ProductInputs[i]];
            partialArgs[partialArgs.Length - 1] = temp.Pointer;

            var combineArgs = new IntPtr[plan.Combine.ParameterCount];
            combineArgs[0] = temp.Pointer;
            if (plan.Combine.BiasInput is { } bias) combineArgs[bias] = uploaded[spec.BiasInput!.Value];
            if (plan.Combine.ScaleInput is { } scaleAt) combineArgs[scaleAt] = uploaded[spec.ScaleInput!.Value];
            combineArgs[combineArgs.Length - 1] = outBuffer.Pointer;

            void Launch()
            {
                LaunchOne(partialModule, partialFn, partialArgs, partialEmitter.LaunchBlocks,
                    (uint)partialEmitter.LaunchBlockX, (uint)partialEmitter.LaunchBlockY);
                LaunchOne(combineModule, combineFn, combineArgs, combineEmitter.LaunchBlocks,
                    (uint)combineEmitter.LaunchBlockX, (uint)combineEmitter.LaunchBlockY);
            }

            Launch();
            runtime.Synchronize();
            var got = new float[outCount];
            outBuffer.Download<float>(got);

            if (!Agrees(got, reference, out double deviation))
            {
                Console.WriteLine("    candidate 'split' disagrees by " +
                                  deviation.ToString("E3", CultureInfo.InvariantCulture) +
                                  " relative; rejected");
                return null;
            }

            string axes = string.Join("+", plan.PromotedAxes);
            return ("split:" + axes, Measure(runtime.Synchronize, Launch));
        }
        finally { foreach (var b in buffers) b.Dispose(); }
    }

    /// <summary>
    /// Whether a candidate reproduces the reference, judged RELATIVE to the reference's
    /// own magnitude.
    /// </summary>
    /// <remarks>
    /// An absolute tolerance is a fp32-epsilon test, not an agreement test, and it
    /// silently scales with the reduction length. The absolute form rejected a CORRECT
    /// split of depthwise_conv2d_3x3_bwd_weights over a deviation of 8.575 -- which is
    /// 5.6E-004 relative, the ordinary fp32 accumulation-order difference across 100,352
    /// summed terms, and the same figure that kernel already shows on the conveyor. That
    /// false negative cost a measured 17x.
    /// </remarks>
    private static bool Agrees(float[] candidate, float[] reference, out double deviation)
    {
        double worst = 0, scale = 0;
        for (long e = 0; e < candidate.Length; e++)
        {
            worst = Math.Max(worst, Math.Abs(candidate[e] - reference[e]));
            scale = Math.Max(scale, Math.Abs((double)reference[e]));
        }
        deviation = scale > 0 ? worst / scale : worst;
        return deviation <= 2e-3;
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

    private static unsafe void LaunchOne(
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
