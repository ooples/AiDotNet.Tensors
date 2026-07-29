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
// the current protocol, and keeps the winner.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using static AiDotNet.Tensors.Benchmarks.KernelToolArgs;

namespace AiDotNet.Tensors.Benchmarks;

internal static class KernelAutotuneTool
{
    /// <summary>One candidate lowering: a name and the knobs that produce it.</summary>
    private sealed record Candidate(
        string Name, Action<PtxAffineEmitter>? Configure,
        bool TiledContraction = false, bool TiledConv2D = false,
        bool DepthwiseWeightGradient = false,
        bool ParityTransposedConv2D = false);

    private sealed record TuneResult(
        string Name, double BestUs, double ModelledUs, double Gain);

    /// <summary>A loaded candidate kept alive while it is paired against the baseline.</summary>
    private sealed class CandidateProgram : IDisposable
    {
        private readonly List<IDisposable> _resources;
        private readonly DirectPtxBuffer _output;
        private readonly int _outputElements;

        internal CandidateProgram(
            string name, Action launch, DirectPtxBuffer output, int outputElements,
            List<IDisposable> resources)
        {
            Name = name;
            Launch = launch;
            _output = output;
            _outputElements = outputElements;
            _resources = resources;
        }

        internal string Name { get; }
        internal Action Launch { get; }

        internal float[] ReadOutput()
        {
            var values = new float[_outputElements];
            _output.Download<float>(values);
            return values;
        }

        public void Dispose()
        {
            for (int i = _resources.Count - 1; i >= 0; i--)
                _resources[i].Dispose();
        }
    }

    private static readonly Candidate[] Candidates =
    {
        // The modelled choice, first so it is the reference every other is compared to.
        new("modelled", _ => { }),

        // A different execution model rather than another scalar-emitter knob: one CTA
        // cooperatively stages both operands and owns an FP32 output tile. Keep it adjacent
        // to the baseline so a targeted schedule investigation gets its paired window before
        // the legacy knob sweep; it remains subject to the same numerical and noise gates.
        new("tiled-contraction", null, TiledContraction: true),

        // A dense 3x3 row tile: stage three activation rows and all nine weights for a
        // channel slice, then reuse them across output channels and adjacent columns.
        new("tiled-conv2d", null, TiledConv2D: true),

        // A cooperative reduction over contiguous NCHW positions. One block owns a
        // (channel,kh) row, so all three kw accumulators share each dOut load instead
        // of independently replaying the same reduction.
        new("depthwise-weight-gradient", null, DepthwiseWeightGradient: true),

        // One input coordinate owns the complete stride-2 output parity tile. This
        // removes the affine lowering's exact-division and remainder predicates while
        // preserving deterministic assignment: no output is shared between threads.
        new("parity-transposed", null, ParityTransposedConv2D: true),

        new("no-tile", e => e.Coarsening = 1),
        new("tile2", e => { e.Coarsening = 2; }),
        new("lanes4", e => { e.MaxTileLanes = 4; }),
        new("no-staging", e => e.EnableSharedStaging = false),
        new("no-vector", e => e.EnableVectorLoads = false),

        // PER-DIMENSION STAGING, the lever docs/PATH_TO_WINS.md aims at all five
        // competitor losses: each is L1-bound and none stages its activation operand.
        // Measured as a candidate rather than switched on, because staging is not free --
        // it adds two barriers per strip-mine step, and staging the wrong operand cost
        // conv_transpose 104 -> 131.4 us.
        new("input-staging", e => e.EnableInputStaging = true),
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
        Console.WriteLine("candidates: " + string.Join(", ", Candidates.Select(c => c.Name)) +
                          ", split, tiled-split");
        Console.WriteLine();
        Console.WriteLine("kernel                          modelled   best      winner        gain");

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        var rows = new List<string>();
        int improved = 0;
        try
        {
            foreach (var entry in entries)
            {
                try
                {
                    TuneResult? result = TuneOne(runtime, entry);
                    if (result is null || !double.IsFinite(result.BestUs) ||
                        result.BestUs == double.MaxValue ||
                        !double.IsFinite(result.ModelledUs) || result.ModelledUs == double.MaxValue)
                        continue;

                    double gain = result.Gain;
                    if (gain > CodegenMeasurementProtocol.AutotuneGainNoiseFloor) improved++;

                    Console.WriteLine(entry.Name.PadRight(30) +
                        result.ModelledUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(9) +
                        result.BestUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(9) +
                        "   " + result.Name.PadRight(12) +
                        gain.ToString("F3", CultureInfo.InvariantCulture).PadLeft(7) + "x");

                    var identity = CodegenAutotuneIdentity.Create(
                        entry.Bench, runtime.DeviceFingerprint,
                        runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                    rows.Add(string.Join("\t", entry.Name, result.Name,
                        result.BestUs.ToString("F3", CultureInfo.InvariantCulture),
                        result.ModelledUs.ToString("F3", CultureInfo.InvariantCulture),
                        gain.ToString("F4", CultureInfo.InvariantCulture),
                        CodegenMeasurementProtocol.Tag,
                        identity.DeviceFingerprint,
                        identity.Target,
                        identity.SpecFingerprint,
                        identity.EmitterFingerprint));
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
        text.AppendLine(
            "kernel\twinner\tbest_us\tmodelled_us\tgain\tprotocol\tdevice\ttarget\tspec\temitter");
        foreach (string row in rows) text.AppendLine(row);
        File.WriteAllText(outputPath, text.ToString());

        Console.WriteLine();
        double noiseFloorPercent =
            (CodegenMeasurementProtocol.AutotuneGainNoiseFloor - 1.0) * 100.0;
        Console.WriteLine(improved + " kernels improved past the " +
                          noiseFloorPercent.ToString("0.##", CultureInfo.InvariantCulture) +
                          "% noise floor");
        Console.WriteLine("winners written to " + outputPath);
        CodegenAutotuneCache.Invalidate();
    }

    private static TuneResult? TuneOne(
        DirectPtxRuntime runtime, CodegenCatalogEntry entry)
    {
        CodegenKernelSpec spec = entry.Bench;
        long workUnits = WorkUnits(spec);

        using CandidateProgram modelled = CreateSingle(runtime, spec, Candidates[0]);
        modelled.Launch();
        runtime.Synchronize();
        float[] reference = modelled.ReadOutput();

        StableTimer.Result baseline = StableTimer.Measure(runtime, modelled.Launch, workUnits);
        bool hasStableTiming = baseline.Stable;
        if (!baseline.Stable)
        {
            Console.WriteLine("    modelled lowering " + baseline.Describe() +
                              "; trying independently gated paired windows");
        }

        string bestName = "modelled";
        double bestUs = baseline.Stable ? baseline.Microseconds : double.MaxValue;
        double bestModelledUs = bestUs;
        double bestGain = 1.0;

        for (int i = 1; i < Candidates.Length; i++)
        {
            CandidateProgram? program = null;
            try { program = CreateSingle(runtime, spec, Candidates[i]); }
            catch (NotSupportedException) { continue; }
            using (program)
            {
                program.Launch();
                runtime.Synchronize();
                if (!Agrees(program.ReadOutput(), reference, out double deviation))
                {
                    Console.WriteLine("    candidate '" + program.Name + "' disagrees by " +
                                      deviation.ToString("E3", CultureInfo.InvariantCulture) +
                                      " relative; rejected");
                    continue;
                }

                Consider(runtime, modelled, program, workUnits,
                    ref hasStableTiming, ref bestName, ref bestUs,
                    ref bestModelledUs, ref bestGain);
            }
        }

        // Both split forms preserve the deterministic affine combine. The second changes
        // only the expensive partial pass to a cooperative outer-product tile. Each stays
        // paired against the live modelled program, with both launches in the timed region.
        for (int splitKind = 0; splitKind < 2; splitKind++)
        {
            using CandidateProgram? split = TryCreateSplit(
                runtime, spec, tiledPartial: splitKind == 1);
            if (split is null) continue;

            split.Launch();
            runtime.Synchronize();
            if (!Agrees(split.ReadOutput(), reference, out double deviation))
            {
                Console.WriteLine("    candidate '" + split.Name + "' disagrees by " +
                                  deviation.ToString("E3", CultureInfo.InvariantCulture) +
                                  " relative; rejected");
            }
            else
            {
                Consider(runtime, modelled, split, workUnits,
                    ref hasStableTiming, ref bestName, ref bestUs,
                    ref bestModelledUs, ref bestGain);
            }
        }

        if (!hasStableTiming)
        {
            Console.WriteLine("    no standalone or paired timing window stabilized; " +
                              "no winner recorded");
            return null;
        }
        return new TuneResult(bestName, bestUs, bestModelledUs, bestGain);
    }

    private static void Consider(
        DirectPtxRuntime runtime,
        CandidateProgram modelled, CandidateProgram candidate, long workUnits,
        ref bool hasStableTiming,
        ref string bestName, ref double bestUs, ref double bestModelledUs, ref double bestGain)
    {
        StableTimer.PairResult timing = StableTimer.MeasurePair(
            runtime, modelled.Launch, candidate.Launch, workUnits, workUnits);
        if (!timing.Stable)
        {
            Console.WriteLine("    candidate '" + candidate.Name +
                              "' has unstable paired timing (modelled " +
                              timing.A.Describe() + ", candidate " + timing.B.Describe() +
                              ", ratio +-" +
                              (timing.RelativeSpread * 100).ToString(
                                  "0.0", CultureInfo.InvariantCulture) + "%); rejected");
            return;
        }

        Console.WriteLine("    candidate '" + candidate.Name + "': modelled " +
                          timing.A.Describe() + ", candidate " + timing.B.Describe() +
                          ", paired " + timing.DescribeRatio() + " +-" +
                          (timing.RelativeSpread * 100).ToString("0.0", CultureInfo.InvariantCulture) +
                          "%");

        // A stable pair is also valid baseline evidence. This lets an interrupted standalone
        // window recover without accepting a candidate whose own time or ratio was unstable.
        if (!hasStableTiming)
        {
            hasStableTiming = true;
            bestUs = timing.A.Microseconds;
            bestModelledUs = timing.A.Microseconds;
        }

        // A winner must clear both the observed paired spread and the protocol noise floor.
        // Merely having the smallest median is not a promotion criterion.
        double required = Math.Max(
            CodegenMeasurementProtocol.AutotuneGainNoiseFloor,
            1.0 + timing.RelativeSpread);
        if (timing.Ratio <= required || timing.Ratio <= bestGain) return;

        bestName = candidate.Name;
        bestUs = timing.B.Microseconds;
        bestModelledUs = timing.A.Microseconds;
        bestGain = timing.Ratio;
    }

    private static CandidateProgram CreateSingle(
        DirectPtxRuntime runtime, CodegenKernelSpec spec, Candidate candidate)
    {
        RequireFloat32(spec);
        var resources = new List<IDisposable>();
        try
        {
            string ptx;
            uint blocks, blockX, blockY;
            if (candidate.TiledContraction)
            {
                var tiled = new PtxTiledContractionEmitter();
                ptx = tiled.Emit(
                    spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                blocks = tiled.LaunchBlocks;
                blockX = checked((uint)tiled.LaunchBlockThreads);
                blockY = 1;
            }
            else if (candidate.TiledConv2D)
            {
                var tiled = new PtxTiledConv2DEmitter();
                ptx = tiled.Emit(
                    spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                blocks = tiled.LaunchBlocks;
                blockX = checked((uint)tiled.LaunchBlockThreads);
                blockY = 1;
            }
            else if (candidate.DepthwiseWeightGradient)
            {
                var cooperative = new PtxDepthwiseConv2DWeightGradientEmitter();
                ptx = cooperative.Emit(
                    spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                blocks = cooperative.LaunchBlocks;
                blockX = checked((uint)cooperative.LaunchBlockThreads);
                blockY = 1;
            }
            else if (candidate.ParityTransposedConv2D)
            {
                var parity = new PtxParityTransposedConv2DEmitter();
                ptx = parity.Emit(
                    spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                blocks = parity.LaunchBlocks;
                blockX = checked((uint)parity.LaunchBlockThreads);
                blockY = 1;
            }
            else
            {
                var emitter = new PtxAffineEmitter();
                candidate.Configure!(emitter);
                ptx = emitter.Emit(
                    spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                blocks = emitter.LaunchBlocks;
                blockX = checked((uint)emitter.LaunchBlockX);
                blockY = checked((uint)emitter.LaunchBlockY);
            }
            var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
            resources.Add(module);
            IntPtr fn = module.GetFunction(spec.Name, out _);

            var pointers = new IntPtr[spec.ParameterCount];
            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                var binding = spec.Inputs[i];
                var buffer = runtime.AllocateBytes(
                    (nuint)(binding.ElementCount * binding.ElementBytes));
                resources.Add(buffer);
                var host = new float[binding.ElementCount];
                for (long e = 0; e < host.LongLength; e++)
                    host[e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                buffer.Upload<float>(host);
                pointers[binding.ParameterIndex] = buffer.Pointer;
            }

            var output = runtime.AllocateBytes(
                (nuint)(spec.Output.ElementCount * spec.Output.ElementBytes));
            resources.Add(output);
            pointers[spec.Output.ParameterIndex] = output.Pointer;
            foreach (var extra in spec.ExtraOutputs)
            {
                var buffer = runtime.AllocateBytes(
                    (nuint)(extra.Binding.ElementCount * extra.Binding.ElementBytes));
                resources.Add(buffer);
                pointers[extra.Binding.ParameterIndex] = buffer.Pointer;
            }

            void Launch() => LaunchOne(module, fn, pointers, blocks, blockX, blockY);
            return new CandidateProgram(
                candidate.Name, Launch, output, checked((int)spec.Output.ElementCount), resources);
        }
        catch
        {
            for (int i = resources.Count - 1; i >= 0; i--) resources[i].Dispose();
            throw;
        }
    }

    private static CandidateProgram? TryCreateSplit(
        DirectPtxRuntime runtime, CodegenKernelSpec spec, bool tiledPartial = false)
    {
        CodegenSplitPlan? plan;
        try { plan = CodegenSplitReduction.TryPlan(spec); }
        catch (NotSupportedException) { return null; }
        if (plan is null) return null;
        RequireFloat32(spec);

        var resources = new List<IDisposable>();
        try
        {
            var combineEmitter = new PtxAffineEmitter();
            string partialPtx;
            uint partialBlocks, partialBlockX, partialBlockY;
            if (tiledPartial)
            {
                PtxTiledOuterProductProgram partial =
                    PtxTiledOuterProductDispatcher.Emit(
                        plan.Partial, runtime.ComputeCapabilityMajor,
                        runtime.ComputeCapabilityMinor);
                partialPtx = partial.Text;
                partialBlocks = partial.LaunchBlocks;
                partialBlockX = checked((uint)partial.BlockThreads);
                partialBlockY = 1;
            }
            else
            {
                var partialEmitter = new PtxAffineEmitter();
                partialPtx = partialEmitter.Emit(
                    plan.Partial, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                partialBlocks = partialEmitter.LaunchBlocks;
                partialBlockX = checked((uint)partialEmitter.LaunchBlockX);
                partialBlockY = checked((uint)partialEmitter.LaunchBlockY);
            }
            string combinePtx = combineEmitter.Emit(
                plan.Combine, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
            var partialModule = runtime.LoadModule(partialPtx, allowExperimentalJitFallback: true);
            var combineModule = runtime.LoadModule(combinePtx, allowExperimentalJitFallback: true);
            resources.Add(partialModule);
            resources.Add(combineModule);
            IntPtr partialFn = partialModule.GetFunction(plan.Partial.Name, out _);
            IntPtr combineFn = combineModule.GetFunction(plan.Combine.Name, out _);

            var uploaded = new IntPtr[spec.Inputs.Count];
            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                var binding = spec.Inputs[i];
                var buffer = runtime.AllocateBytes(
                    (nuint)(binding.ElementCount * binding.ElementBytes));
                resources.Add(buffer);
                var host = new float[binding.ElementCount];
                for (long e = 0; e < host.LongLength; e++)
                    host[e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                buffer.Upload<float>(host);
                uploaded[i] = buffer.Pointer;
            }

            var temporary = runtime.AllocateBytes((nuint)(plan.TempElements * sizeof(float)));
            var output = runtime.AllocateBytes((nuint)(spec.Output.ElementCount * sizeof(float)));
            resources.Add(temporary);
            resources.Add(output);

            var partialArgs = new IntPtr[plan.Partial.ParameterCount];
            for (int i = 0; i < spec.ProductInputs.Count; i++)
                partialArgs[i] = uploaded[spec.ProductInputs[i]];
            partialArgs[partialArgs.Length - 1] = temporary.Pointer;

            var combineArgs = new IntPtr[plan.Combine.ParameterCount];
            combineArgs[0] = temporary.Pointer;
            if (plan.Combine.BiasInput is { } bias)
                combineArgs[bias] = uploaded[spec.BiasInput!.Value];
            if (plan.Combine.ScaleInput is { } scale)
                combineArgs[scale] = uploaded[spec.ScaleInput!.Value];
            combineArgs[combineArgs.Length - 1] = output.Pointer;

            void Launch()
            {
                LaunchOne(partialModule, partialFn, partialArgs,
                    partialBlocks, partialBlockX, partialBlockY);
                LaunchOne(combineModule, combineFn, combineArgs, combineEmitter.LaunchBlocks,
                    (uint)combineEmitter.LaunchBlockX, (uint)combineEmitter.LaunchBlockY);
            }

            string name = (tiledPartial ? "tiled-split:" : "split:") +
                string.Join("+", plan.PromotedAxes);
            return new CandidateProgram(
                name, Launch, output, checked((int)spec.Output.ElementCount), resources);
        }
        catch (NotSupportedException)
        {
            for (int i = resources.Count - 1; i >= 0; i--) resources[i].Dispose();
            return null;
        }
        catch
        {
            for (int i = resources.Count - 1; i >= 0; i--) resources[i].Dispose();
            throw;
        }
    }

    private static void RequireFloat32(CodegenKernelSpec spec)
    {
        if (spec.Output.ElementType != CodegenElementType.Float32)
            throw new NotSupportedException("Autotune correctness reads require an fp32 output.");
        foreach (var input in spec.Inputs)
            if (input.ElementType != CodegenElementType.Float32)
                throw new NotSupportedException("Autotune input generation currently requires fp32.");
        foreach (var extra in spec.ExtraOutputs)
            if (extra.Binding.ElementType != CodegenElementType.Float32)
                throw new NotSupportedException("Autotune extra outputs currently require fp32.");
    }

    private static long WorkUnits(CodegenKernelSpec spec)
    {
        long bytes = spec.Output.ElementCount * spec.Output.ElementBytes;
        foreach (var input in spec.Inputs)
            bytes = checked(bytes + input.ElementCount * input.ElementBytes);
        foreach (var extra in spec.ExtraOutputs)
            bytes = checked(bytes + extra.Binding.ElementCount * extra.Binding.ElementBytes);
        long operations = checked(spec.Output.ElementCount * Math.Max(1, spec.Space.ReductionTripCount));
        return Math.Max(bytes, operations);
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

}
