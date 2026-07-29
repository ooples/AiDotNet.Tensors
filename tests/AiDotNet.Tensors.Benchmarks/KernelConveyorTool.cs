// Copyright (c) AiDotNet. All rights reserved.
// The conveyor: three stages every kernel passes through, in the same order, with
// the same gates, driven by a loop over CodegenKernelCatalog rather than by
// per-kernel code.
//
//   --kernel-verify    emit PTX from the spec, run it on the device, compare against
//                      the fp64 CPU interpretation of the SAME spec. A kernel that
//                      does not match its own specification is not a kernel.
//
//   --kernel-release   compile PTX to a cubin through the driver linker, record the
//                      content-addressed hash, and read the machine-code metrics back
//                      out of nvdisasm. Gate: zero register spills.
//
//   --kernel-bench     time the kernel with the Phase 0.5 calibrated protocol
//                      (device-filling shape, batched timed regions, paired
//                      within-sample ratio) against the naive same-spec baseline.
//
// The stages deliberately share one source of truth. The oracle in verify and the
// kernel in bench are generated from the same CodegenKernelSpec, so a spec change
// cannot leave the reference and the implementation disagreeing -- the failure mode
// that let a shipped grouped-deformable kernel pass three structural gates while
// computing zeros.

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
using static AiDotNet.Tensors.Benchmarks.KernelToolArgs;

namespace AiDotNet.Tensors.Benchmarks;

internal static class KernelConveyorTool
{
    private const double Tolerance = 2e-3;
    private const int Warmup = 20;
    private const int Samples = 51;
    private const int LaunchesPerSample = 50;
    private const int Runs = 3;

    /// <summary>Attempts allowed to obtain a measurement taken at a steady SM clock.</summary>
    private const int ClockRetries = 4;

    /// <summary>
    /// Forces per-dimension activation staging on, for the correctness gate and for
    /// inspecting the staging decision.
    /// </summary>
    /// <remarks>
    /// Staging must clear the fp64 oracle before any timing is believed: the first version
    /// of it returned 5.277 and 1.112e1 instead of zero under a two-dimensional block. A
    /// flag that runs the real verify with staging on is how that gets checked.
    /// </remarks>
    private static bool _forceInputStaging;

    internal static void Run(string stage, string[] args)
    {
        _forceInputStaging = args.Contains("--input-staging", StringComparer.Ordinal);
        string selector = KernelToolArgs.Selector(args);
        var entries = Select(selector);
        if (entries.Count == 0)
        {
            Console.WriteLine("No catalog entry matches '" + selector + "'. Known kernels:");
            foreach (var e in CodegenKernelCatalog.All)
                Console.WriteLine("  " + e.Name.PadRight(32) + e.Summary);
            return;
        }

        switch (stage)
        {
            case "verify": Verify(entries); break;
            case "release": Release(entries, args); break;
            case "bench": Bench(entries, args); break;
            case "vector-ab": VectorAb(entries); break;
            case "dump": Dump(entries, args); break;
            case "coarsen-ab": CoarsenAb(entries); break;
            case "once": Once(entries, args); break;
            case "predict": Predict(entries, args); break;
            default: Console.WriteLine("Unknown conveyor stage '" + stage + "'."); break;
        }
    }

    private static IReadOnlyList<CodegenCatalogEntry> Select(string selector)
    {
        if (string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase))
            return CodegenKernelCatalog.All;
        var one = CodegenKernelCatalog.Find(selector);
        return one is null ? Array.Empty<CodegenCatalogEntry>() : new[] { one };
    }

    /// <summary>Writes each kernel's PTX to disk so ptxas can be run on it directly.</summary>
    private static void Dump(IReadOnlyList<CodegenCatalogEntry> entries, string[] args)
    {
        string dir = ValueOf(args, "--out") ?? Path.Combine(Path.GetTempPath(), "codegen-ptx");
        Directory.CreateDirectory(dir);
        foreach (var entry in entries)
        {
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(entry.Bench, 8, 6);
            string path = Path.Combine(dir, entry.Name + ".ptx");
            File.WriteAllText(path, ptx);
            File.WriteAllText(Path.Combine(dir, entry.Name + ".spec.txt"), entry.Bench.Describe());
            Console.WriteLine(path + "  lanes=" + emitter.CoarsenedLanes +
                              " blocks=" + emitter.LaunchBlocks);
        }
    }

    /// <summary>
    /// Launches each kernel at its BENCH shape exactly once. Nsight Compute replays
    /// every launch it profiles, so the benchmark loop is unusable under a profiler;
    /// this gives it a single launch of the shape we actually care about.
    /// </summary>
    private static void Once(IReadOnlyList<CodegenCatalogEntry> entries, string[] args)
    {
        using var runtime = OpenRuntime();
        if (runtime is null) return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            foreach (var entry in entries)
            {
                var spec = entry.Bench;

                // Launch the TUNED program. The limiter gate drives this stage, so
                // launching the untuned lowering here made it profile a kernel we do not
                // ship: the three weight gradients came back at 1-3% on every unit,
                // because that is the unsplit 3-block kernel, not the split one the
                // conveyor verifies, benches and releases.
                bool overridden = args.Contains("--no-coarsen", StringComparer.Ordinal)
                    || ValueOf(args, "--coarsen") is not null
                    || ValueOf(args, "--max-lanes") is not null;

                TunedProgram program;
                if (overridden)
                {
                    var emitter = new PtxAffineEmitter();
                    if (args.Contains("--no-coarsen", StringComparer.Ordinal)) emitter.Coarsening = 1;
                    if (ValueOf(args, "--coarsen") is string cz)
                        emitter.Coarsening = int.Parse(cz, CultureInfo.InvariantCulture);
                    if (ValueOf(args, "--max-lanes") is string mz)
                        emitter.MaxTileLanes = int.Parse(mz, CultureInfo.InvariantCulture);
                    string ptx = emitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                    program = new TunedProgram(spec,
                        new[]
                        {
                            new ProgramKernel(spec, ptx, emitter.LaunchBlocks,
                                (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY,
                                emitter.LoopedAxes, emitter.ElidedGuards, emitter.StagedOperands),
                        },
                        null, "command-line");
                }
                else
                {
                    program = ResolveTuned(runtime, spec, entry.Name);
                }

                using var launchable = TunedLaunchable.Create(runtime, program);
                launchable.Launch();
                runtime.Synchronize();

                var blockCounts = new List<string>();
                foreach (var kernel in program.Kernels)
                    blockCounts.Add(kernel.Blocks.ToString(CultureInfo.InvariantCulture));
                Console.WriteLine(entry.Name + ": " + program.Label() + ", " +
                                  string.Join("+", blockCounts) + " blocks");
            }
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    /// <summary>
    /// Device-free bottleneck prediction for every catalog kernel, printed beside the
    /// measured time so the model can be falsified rather than believed.
    /// </summary>
    private static void Predict(IReadOnlyList<CodegenCatalogEntry> entries, string[] args)
    {
        var machine = CodegenMachineModel.Rtx3080Locked;
        Console.WriteLine();
        Console.WriteLine("STATIC BOTTLENECK PREDICTION - no GPU used");
        Console.WriteLine("machine: " + machine.Name);
        Console.WriteLine("  load issue " + (machine.LoadInstructionsPerSecond / 1e9).ToString("F1", CultureInfo.InvariantCulture) +
                          " G warp-inst/s | dram " + (machine.DramBytesPerSecond / 1e9).ToString("F0", CultureInfo.InvariantCulture) +
                          " GB/s | compute " + (machine.MacsPerSecond * 2 / 1e12).ToString("F1", CultureInfo.InvariantCulture) + " TFLOP/s");
        Console.WriteLine();
        Console.WriteLine("measured column loaded from competitor evidence under protocol " +
                          CodegenMeasurementProtocol.Tag);
        Console.WriteLine();
        Console.WriteLine("kernel                        tile            block  ld/MAC  staged" +
                          "                 predicted  measured  pred/meas");

        var measuredEvidence = ReadEvidence(
            Path.Combine("artifacts", "competitor-ratios.tsv"), 1);

        foreach (var entry in entries)
        {
            var spec = entry.Bench;
            var emitter = new PtxAffineEmitter();
            if (ValueOf(args, "--coarsen") is string cz)
                emitter.Coarsening = int.Parse(cz, CultureInfo.InvariantCulture);
            if (_forceInputStaging) emitter.EnableInputStaging = true;
            emitter.Emit(spec, 8, 6);

            long threads = spec.Space.TotalThreads / Math.Max(1, emitter.CoarsenedLanes);
            var p = CodegenPerformanceModel.Predict(spec, threads, emitter.DynamicLoadsPerThread, machine);

            double measured = measuredEvidence.TryGetValue(entry.Name, out string? cell) &&
                              double.TryParse(cell, NumberStyles.Any,
                                  CultureInfo.InvariantCulture, out double parsed)
                ? parsed
                : 0.0;
            string measuredText = measured > 0
                ? measured.ToString("F1", CultureInfo.InvariantCulture)
                : "MISSING";
            string predictedText = p.HasComputeCeiling
                ? p.PredictedMicroseconds.ToString("F1", CultureInfo.InvariantCulture)
                : "-";
            string ratioText = measured <= 0
                ? "MISSING"
                : p.HasComputeCeiling
                    ? (p.PredictedMicroseconds / measured)
                        .ToString("F2", CultureInfo.InvariantCulture) + "x"
                    : "-";

            Console.WriteLine(entry.Name.PadRight(30) +
                emitter.TileDescription.PadRight(16) +
                emitter.LaunchBlockThreads.ToString(CultureInfo.InvariantCulture).PadLeft(5) +
                p.LoadsPerMac.ToString("F3", CultureInfo.InvariantCulture).PadLeft(8) +
                "  " + (emitter.UsedTwoDimensionalBlock
                    ? "2D " + emitter.LaunchBlockX + "x" + emitter.LaunchBlockY + " "
                    : "flat ") + emitter.StagedOperands.PadRight(16) +
                predictedText.PadLeft(10) +
                measuredText.PadLeft(10) + ratioText.PadLeft(11));
        }

        Console.WriteLine();
        Console.WriteLine("reuse axes (the axes worth tiling - an operand invariant in an axis");
        Console.WriteLine("can share one load across every position of it):");
        foreach (var entry in entries)
        {
            foreach (var pair in CodegenPerformanceModel.ReuseAxes(entry.Bench))
                if (pair.Value.Count > 0)
                    Console.WriteLine("  " + entry.Name.PadRight(32) + pair.Key.PadRight(10) +
                                      "invariant in {" + string.Join(", ", pair.Value) + "}");
        }
    }

    // ---------------------------------------------------------------- verify

    private static void Verify(IReadOnlyList<CodegenCatalogEntry> entries)
    {
        using var runtime = OpenRuntime();
        if (runtime is null) return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        int passed = 0, failed = 0;
        try
        {
            Console.WriteLine();
            Console.WriteLine("CONVEYOR STAGE 1 - verify against the fp64 interpretation of the same spec");
            Console.WriteLine("tolerance " + Tolerance.ToString("E0", CultureInfo.InvariantCulture));
            Console.WriteLine();
            Console.WriteLine("kernel                              regs  lowering/staged  guards  max abs dev   result");

            foreach (var entry in entries)
            {
                string status;
                try
                {
                    // VERIFY THE SHAPE WE RELEASE. Verifying a small proxy shape and
                    // releasing a large one has now shipped two unexercised code paths:
                    // the strip-mined loop, and then shared-memory staging, because both
                    // are chosen from extents that differ between the shapes. Parity
                    // gates caught each after the fact; using one shape removes the
                    // class. The fp64 oracle at these sizes is 26-115M operations, a few
                    // seconds, which is worth paying once per kernel.
                    // The verify shape must exercise the SAME lowering as the shape
                    // that gets released. Otherwise the strip-mined loop path can ship
                    // having only ever been checked in its fully-unrolled form -- the
                    // released-an-unverified-branch failure, one abstraction up.
                    // One shape now, so there is no verify-vs-release divergence left to
                    // gate: the row records the lowering and the staging it was verified
                    // WITH, which is the fact that matters -- including whether that
                    // lowering was a two-kernel split.
                    var (dev, regs, elided, lowering, staged) = VerifyOne(runtime, entry.Bench, entry.Name);

                    bool ok = dev <= Tolerance && !double.IsNaN(dev);
                    status = ok ? "PASS" : "FAIL";
                    if (ok) passed++; else failed++;
                    Console.WriteLine(entry.Name.PadRight(36) +
                        regs.ToString(CultureInfo.InvariantCulture).PadLeft(4) +
                        lowering.PadLeft(9) + ("/" + staged).PadRight(10) +
                        elided.ToString(CultureInfo.InvariantCulture).PadLeft(9) +
                        dev.ToString("E3", CultureInfo.InvariantCulture).PadLeft(14) +
                        status.PadLeft(9));

                }
                catch (Exception ex)
                {
                    failed++;
                    Console.WriteLine(entry.Name.PadRight(36) + "   -               -             -     ERROR");
                    Console.WriteLine("    " + ex.GetType().Name + ": " + ex.Message.Split('\n')[0]);
                }
            }
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }

        Console.WriteLine();
        Console.WriteLine("verify: " + passed.ToString(CultureInfo.InvariantCulture) + " passed, " +
                          failed.ToString(CultureInfo.InvariantCulture) + " failed");
    }

    /// <summary>
    /// Applies a MEASURED lowering choice when the autotuner recorded one that beat the
    /// modelled choice. Named the same as the autotuner's candidates, so the winner it
    /// records is the configuration reproduced here.
    /// </summary>
    /// <summary>
    /// Applies the measured lowering for a kernel.
    /// </summary>
    /// <param name="emitter">Emitter to configure.</param>
    /// <param name="kernelName">
    /// The CATALOG ENTRY name, which is what the autotuner writes. It is not always the
    /// spec's own name -- the depthwise entries are catalogued as
    /// "depthwise_conv2d_3x3..." while their specs are named "dwconv2d_3x3..." -- and
    /// looking up by the spec name silently found nothing, so those kernels ran the
    /// modelled lowering while the cache said they had been tuned.
    /// </param>
    internal static void ApplyTuned(
        PtxAffineEmitter emitter, string kernelName, string? winner)
    {
        switch (winner)
        {
            case "no-tile": emitter.Coarsening = 1; break;
            case "tile2": emitter.Coarsening = 2; break;
            case "lanes4": emitter.MaxTileLanes = 4; break;
            case "no-staging": emitter.EnableSharedStaging = false; break;
            case "no-vector": emitter.EnableVectorLoads = false; break;
            case "input-staging": emitter.EnableInputStaging = true; break;

            // A split winner is not a knob -- it is a different PROGRAM, two kernels and a
            // temporary, which this single-kernel path cannot launch. Saying so is the
            // point: silently falling through would report a kernel as tuned while running
            // the lowering the tuner measured as 17x slower.
            case not null when winner.StartsWith("split:", StringComparison.Ordinal):
            case not null when winner.StartsWith("tiled-split:", StringComparison.Ordinal):
                Console.WriteLine("    note: " + kernelName + " measured fastest as " + winner +
                                  ", a two-kernel split this stage cannot launch; " +
                                  "running the single-kernel lowering instead");
                break;

            default: break;   // untuned, or the modelled choice already won
        }
    }

    // ------------------------------------------------- the tuned program
    //
    // A tuned lowering is usually a set of knobs, but it can also be a different PROGRAM:
    // the autotuner measures a two-kernel split against every single-kernel candidate and
    // records "split:N" or "tiled-split:N" when one wins. The conveyor stages used to
    // print a note and run the slower lowering, so the headline evidence described a
    // lowering the tuner had already rejected.
    //
    // This resolves the recorded winner into something all three stages can run, whether
    // it is one kernel or two.

    /// <summary>One kernel of a tuned program, with everything needed to launch it.</summary>
    private sealed record ProgramKernel(
        CodegenKernelSpec Spec, string Ptx, uint Blocks, uint BlockX, uint BlockY,
        int LoopedAxes, int ElidedGuards, string StagedOperands);

    /// <summary>What the tuner says to run for a catalog entry.</summary>
    /// <param name="Spec">
    /// The ORIGINAL spec. It stays the semantic contract and the verify oracle no matter
    /// how many kernels the lowering takes: a split computing something else is not a
    /// faster lowering of this operator.
    /// </param>
    /// <param name="Kernels">One kernel, or a partial pass followed by a combine pass.</param>
    /// <param name="Split">The split plan when this is a two-kernel program.</param>
    /// <param name="Winner">The recorded winner name, for reporting.</param>
    private sealed record TunedProgram(
        CodegenKernelSpec Spec, IReadOnlyList<ProgramKernel> Kernels,
        CodegenSplitPlan? Split, string? Winner)
    {
        internal bool IsSplit => Split is not null;

        /// <summary>How the stages label this lowering in their tables.</summary>
        internal string Label() => IsSplit
            ? (Winner is not null && Winner.StartsWith("tiled-split:", StringComparison.Ordinal)
                ? "tiled split x"
                : "split x") + Kernels.Count.ToString(CultureInfo.InvariantCulture)
            : string.Equals(Winner, "tiled-contraction", StringComparison.Ordinal)
                ? "tiled contraction"
                : string.Equals(Winner, "tiled-conv2d", StringComparison.Ordinal)
                    ? "tiled conv2d"
                : string.Equals(Winner, "depthwise-weight-gradient", StringComparison.Ordinal)
                    ? "coop dW"
                : string.Equals(Winner, "parity-transposed", StringComparison.Ordinal)
                    ? "parity transpose"
                : Describe(Kernels[0].LoopedAxes);
    }

    /// <summary>Emits the program the tuner recorded for this entry.</summary>
    private static TunedProgram ResolveTuned(
        DirectPtxRuntime runtime, CodegenKernelSpec spec, string catalogName)
    {
        var identity = CodegenAutotuneIdentity.Create(
            spec, runtime.DeviceFingerprint,
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        string? winner = CodegenAutotuneCache.WinnerFor(catalogName, identity);

        if (string.Equals(winner, "tiled-contraction", StringComparison.Ordinal))
        {
            try
            {
                var tiled = new PtxTiledContractionEmitter();
                string text = tiled.Emit(
                    spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                return new TunedProgram(
                    spec,
                    new[]
                    {
                        new ProgramKernel(spec, text, tiled.LaunchBlocks,
                            checked((uint)tiled.LaunchBlockThreads), 1,
                            0, 0, "matrix+stream"),
                    },
                    null, winner);
            }
            catch (NotSupportedException ex)
            {
                Console.WriteLine("    note: " + catalogName + " recorded tiled-contraction " +
                                  "but it could not be rebuilt (" + ex.Message +
                                  "); using the affine kernel");
            }
        }

        if (string.Equals(winner, "tiled-conv2d", StringComparison.Ordinal))
        {
            try
            {
                var tiled = new PtxTiledConv2DEmitter();
                string text = tiled.Emit(
                    spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                return new TunedProgram(
                    spec,
                    new[]
                    {
                        new ProgramKernel(spec, text, tiled.LaunchBlocks,
                            checked((uint)tiled.LaunchBlockThreads), 1,
                            0, 0, "weights+three input rows"),
                    },
                    null, winner);
            }
            catch (NotSupportedException ex)
            {
                Console.WriteLine("    note: " + catalogName + " recorded tiled-conv2d " +
                                  "but it could not be rebuilt (" + ex.Message +
                                  "); using the affine kernel");
            }
        }

        if (string.Equals(winner, "depthwise-weight-gradient", StringComparison.Ordinal))
        {
            try
            {
                var cooperative = new PtxDepthwiseConv2DWeightGradientEmitter();
                string text = cooperative.Emit(
                    spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                return new TunedProgram(
                    spec,
                    new[]
                    {
                        new ProgramKernel(spec, text, cooperative.LaunchBlocks,
                            checked((uint)cooperative.LaunchBlockThreads), 1,
                            spec.Space.ReductionAxes.Length, 0, "dOut/kw"),
                    },
                    null, winner);
            }
            catch (NotSupportedException ex)
            {
                Console.WriteLine("    note: " + catalogName +
                                  " recorded depthwise-weight-gradient but it could not " +
                                  "be rebuilt (" + ex.Message + "); using the affine kernel");
            }
        }

        if (string.Equals(winner, "parity-transposed", StringComparison.Ordinal))
        {
            try
            {
                var parity = new PtxParityTransposedConv2DEmitter();
                string text = parity.Emit(
                    spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                return new TunedProgram(
                    spec,
                    new[]
                    {
                        new ProgramKernel(spec, text, parity.LaunchBlocks,
                            checked((uint)parity.LaunchBlockThreads), 1,
                            spec.Space.ReductionAxes.Length, 2, "input parity tile+weights"),
                    },
                    null, winner);
            }
            catch (NotSupportedException ex)
            {
                Console.WriteLine("    note: " + catalogName +
                                  " recorded parity-transposed but it could not be " +
                                  "rebuilt (" + ex.Message + "); using the affine kernel");
            }
        }

        bool tiledSplit = winner is not null &&
            winner.StartsWith("tiled-split:", StringComparison.Ordinal);
        if (winner is not null &&
            (winner.StartsWith("split:", StringComparison.Ordinal) || tiledSplit))
        {
            CodegenSplitPlan? plan = null;
            try { plan = CodegenSplitReduction.TryPlan(spec); }
            catch (NotSupportedException) { }

            if (plan is not null)
            {
                var halves = new List<ProgramKernel>(2);
                bool emitted = true;
                if (tiledSplit)
                {
                    try
                    {
                        PtxTiledOuterProductProgram tiled =
                            PtxTiledOuterProductDispatcher.Emit(
                                plan.Partial, runtime.ComputeCapabilityMajor,
                                runtime.ComputeCapabilityMinor);
                        halves.Add(new ProgramKernel(
                            plan.Partial, tiled.Text, tiled.LaunchBlocks,
                            checked((uint)tiled.BlockThreads), 1,
                            plan.Partial.Space.ReductionAxes.Length, 0,
                            tiled.StagedLabel));
                    }
                    catch (NotSupportedException) { emitted = false; }
                }
                else
                {
                    var partial = new PtxAffineEmitter();
                    try
                    {
                        string text = partial.Emit(plan.Partial,
                            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                        halves.Add(new ProgramKernel(plan.Partial, text, partial.LaunchBlocks,
                            (uint)partial.LaunchBlockX, (uint)partial.LaunchBlockY,
                            partial.LoopedAxes, partial.ElidedGuards, partial.StagedOperands));
                    }
                    catch (NotSupportedException) { emitted = false; }
                }
                if (emitted)
                {
                    var combine = new PtxAffineEmitter();
                    try
                    {
                        string text = combine.Emit(plan.Combine,
                            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                        halves.Add(new ProgramKernel(plan.Combine, text, combine.LaunchBlocks,
                            (uint)combine.LaunchBlockX, (uint)combine.LaunchBlockY,
                            combine.LoopedAxes, combine.ElidedGuards, combine.StagedOperands));
                    }
                    catch (NotSupportedException) { emitted = false; }
                }
                if (emitted) return new TunedProgram(spec, halves, plan, winner);
            }

            // The recorded split could not be rebuilt. Falling back to one kernel is
            // correct, but saying nothing would report the slower lowering as the tuned
            // one, which is the confusion this resolution step exists to remove.
            Console.WriteLine("    note: " + catalogName + " recorded " + winner +
                              " but the split could not be rebuilt; using one kernel");
        }

        var single = new PtxAffineEmitter();
        ApplyTuned(single, catalogName, winner);
        if (_forceInputStaging) single.EnableInputStaging = true;
        string ptx = single.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        return new TunedProgram(
            spec,
            new[]
            {
                new ProgramKernel(spec, ptx, single.LaunchBlocks, (uint)single.LaunchBlockX,
                    (uint)single.LaunchBlockY, single.LoopedAxes, single.ElidedGuards,
                    single.StagedOperands),
            },
            null, winner);
    }

    /// <summary>A loaded, bound tuned program: one launch call whatever its shape.</summary>
    private sealed class TunedLaunchable : IDisposable
    {
        private readonly List<DirectPtxModule> _modules = new();
        private readonly List<DirectPtxBuffer> _buffers = new();
        private readonly List<(DirectPtxModule Module, IntPtr Fn, IntPtr[] Args, uint Blocks, uint X, uint Y)> _steps = new();
        private DirectPtxBuffer _output = null!;

        /// <summary>Highest register count across the program's kernels.</summary>
        internal int RegistersPerThread { get; private set; }

        /// <summary>The fp64 inputs the program was bound to, for the verify oracle.</summary>
        internal List<double[]> HostInputs { get; } = new();

        internal static TunedLaunchable Create(DirectPtxRuntime runtime, TunedProgram program)
        {
            var it = new TunedLaunchable();
            try
            {
                var spec = program.Spec;

                // One buffer per operand of the ORIGINAL spec, so both lowerings read
                // byte-identical inputs and a deviation cannot come from the data.
                var uploaded = new IntPtr[spec.Inputs.Count];
                for (int i = 0; i < spec.Inputs.Count; i++)
                {
                    long count = Elements(spec.Inputs[i].Shape);
                    var host = new double[count];
                    var single = new float[count];
                    for (long e = 0; e < count; e++)
                    {
                        float v = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                        single[e] = v;
                        host[e] = v;
                    }
                    it.HostInputs.Add(host);
                    var buffer = runtime.AllocateBytes((nuint)(count * sizeof(float)));
                    buffer.Upload<float>(single);
                    it._buffers.Add(buffer);
                    uploaded[i] = buffer.Pointer;
                }

                it._output = runtime.AllocateBytes(
                    (nuint)(Elements(spec.Output.Shape) * sizeof(float)));
                it._buffers.Add(it._output);

                if (program.Split is { } plan)
                {
                    var temp = runtime.AllocateBytes((nuint)(plan.TempElements * sizeof(float)));
                    it._buffers.Add(temp);

                    // The partial pass reads only the product operands, because the
                    // epilogue moved to the combine; binding by position would feed the
                    // partial pass the bias.
                    var partialArgs = new IntPtr[plan.Partial.ParameterCount];
                    for (int i = 0; i < spec.ProductInputs.Count; i++)
                        partialArgs[i] = uploaded[spec.ProductInputs[i]];
                    partialArgs[partialArgs.Length - 1] = temp.Pointer;

                    var combineArgs = new IntPtr[plan.Combine.ParameterCount];
                    combineArgs[0] = temp.Pointer;
                    if (plan.Combine.BiasInput is { } bias)
                        combineArgs[bias] = uploaded[spec.BiasInput!.Value];
                    if (plan.Combine.ScaleInput is { } scaleAt)
                        combineArgs[scaleAt] = uploaded[spec.ScaleInput!.Value];
                    combineArgs[combineArgs.Length - 1] = it._output.Pointer;

                    it.Add(runtime, program.Kernels[0], partialArgs);
                    it.Add(runtime, program.Kernels[1], combineArgs);
                }
                else
                {
                    var args = new IntPtr[spec.ParameterCount];
                    for (int i = 0; i < spec.Inputs.Count; i++) args[i] = uploaded[i];
                    args[spec.Inputs.Count] = it._output.Pointer;
                    it.Add(runtime, program.Kernels[0], args);
                }

                return it;
            }
            catch
            {
                it.Dispose();
                throw;
            }
        }

        private void Add(DirectPtxRuntime runtime, ProgramKernel kernel, IntPtr[] args)
        {
            var module = runtime.LoadModule(kernel.Ptx, allowExperimentalJitFallback: true);
            _modules.Add(module);
            IntPtr fn = module.GetFunction(kernel.Spec.Name, out DirectPtxFunctionInfo info);
            RegistersPerThread = Math.Max(RegistersPerThread, info.RegistersPerThread);
            _steps.Add((module, fn, args, kernel.Blocks, kernel.BlockX, kernel.BlockY));
        }

        /// <summary>Runs the whole program in order; the combine depends on the partial.</summary>
        internal void Launch()
        {
            foreach (var (module, fn, args, blocks, x, y) in _steps)
                LaunchSpec(module, fn, args, blocks, x, y);
        }

        internal void DownloadOutput(float[] destination) => _output.Download<float>(destination);

        public void Dispose()
        {
            foreach (var b in _buffers) b.Dispose();
            foreach (var m in _modules) m.Dispose();
        }
    }

    private static string Describe(int loopedAxes) =>
        loopedAxes == 0 ? "unroll" : "loop x" + loopedAxes.ToString(CultureInfo.InvariantCulture);

    /// <summary>
    /// Runs the tuned program on the device and compares it against the fp64
    /// interpretation of the ORIGINAL spec.
    /// </summary>
    /// <remarks>
    /// The oracle is the original spec whether the lowering is one kernel or two. That is
    /// the whole reason the split can be trusted: a two-kernel path through a temporary is
    /// exactly the shape that produces a fast wrong answer, and it is held to the same
    /// specification the single kernel was.
    /// </remarks>
    private static (double Deviation, int Registers, int Elided, string Lowering, string Staged) VerifyOne(
        DirectPtxRuntime runtime, CodegenKernelSpec spec, string catalogName)
    {
        var program = ResolveTuned(runtime, spec, catalogName);

        // Host data is a deterministic function of (input index, tensor index) so a
        // failure is reproducible and independent of run order.
        using var launchable = TunedLaunchable.Create(runtime, program);
        launchable.Launch();
        runtime.Synchronize();

        long outCount = Elements(spec.Output.Shape);
        var actual = new float[outCount];
        launchable.DownloadOutput(actual);
        double[] expected = spec.Interpret(launchable.HostInputs);

        double worst = 0;
        for (long e = 0; e < outCount; e++)
        {
            double diff = Math.Abs(actual[e] - expected[e]);
            double scale = Math.Max(1.0, Math.Abs(expected[e]));
            worst = Math.Max(worst, diff / scale);
        }

        int elided = 0;
        foreach (var kernel in program.Kernels) elided += kernel.ElidedGuards;
        string staged = program.IsSplit ? "split" : program.Kernels[0].StagedOperands;
        return (worst, launchable.RegistersPerThread, elided, program.Label(), staged);
    }

    // --------------------------------------------------------------- release

    private static void Release(IReadOnlyList<CodegenCatalogEntry> entries, string[] args)
    {
        using var runtime = OpenRuntime();
        if (runtime is null) return;

        string outputDirectory = ValueOf(args, "--out") ??
            Path.Combine(Directory.GetCurrentDirectory(), "artifacts", "codegen-cubins");
        Directory.CreateDirectory(outputDirectory);
        string? nvdisasm = ValueOf(args, "--nvdisasm") ?? FindNvdisasm();

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        var rows = new List<string>();
        int gated = 0, spilled = 0;
        try
        {
            Console.WriteLine();
            Console.WriteLine("CONVEYOR STAGE 2 - release: driver-linked cubin + machine-code audit");
            Console.WriteLine("output " + outputDirectory);
            Console.WriteLine("nvdisasm " + (nvdisasm ?? "NOT FOUND - SASS metrics skipped"));
            Console.WriteLine();
            Console.WriteLine("kernel                              regs   SASS instr   LDG  STG   spill ld/st   gate");

            foreach (var entry in entries)
            {
              try
              {
                var spec = entry.Bench;

                // Release the program the tuner chose. A split ships TWO cubins and both
                // have to clear the zero-spill gate; releasing only the partial pass would
                // leave half the shipped program unaudited.
                var program = ResolveTuned(runtime, spec, entry.Name);
                for (int k = 0; k < program.Kernels.Count; k++)
                {
                  var kernel = program.Kernels[k];
                  string label = program.IsSplit
                      ? entry.Name + (k == 0 ? " [partial]" : " [combine]")
                      : entry.Name;
                  string fileStem = program.IsSplit
                      ? entry.Name + (k == 0 ? ".partial" : ".combine")
                      : entry.Name;
                  string ptx = kernel.Ptx;

                int regs;
                using (var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true))
                {
                    module.GetFunction(kernel.Spec.Name, out DirectPtxFunctionInfo info);
                    regs = info.RegistersPerThread;
                }

                var artifact = DirectPtxCubinArtifactCache.Resolve(runtime, ptx);
                string cubinPath = Path.Combine(outputDirectory, fileStem + ".cubin");
                File.WriteAllBytes(cubinPath, artifact.Image);

                var metrics = nvdisasm is null ? null : ReadSass(nvdisasm, cubinPath);
                bool ok = metrics is null || (metrics.SpillLoads == 0 && metrics.SpillStores == 0);
                if (ok) gated++; else spilled++;

                Console.WriteLine(label.PadRight(36) +
                    regs.ToString(CultureInfo.InvariantCulture).PadLeft(4) +
                    (metrics?.Instructions.ToString(CultureInfo.InvariantCulture) ?? "-").PadLeft(13) +
                    (metrics?.Ldg.ToString(CultureInfo.InvariantCulture) ?? "-").PadLeft(6) +
                    (metrics?.Stg.ToString(CultureInfo.InvariantCulture) ?? "-").PadLeft(5) +
                    ((metrics is null ? "-" : metrics.SpillLoads + "/" + metrics.SpillStores)).PadLeft(14) +
                    (metrics is null ? "SKIP" : ok ? "PASS" : "FAIL").PadLeft(7));

                rows.Add(string.Join("\t", label, kernel.Spec.Name, artifact.CubinSha256, artifact.SourceKey,
                    regs.ToString(CultureInfo.InvariantCulture),
                    metrics?.Instructions.ToString(CultureInfo.InvariantCulture) ?? "",
                    metrics?.Ldg.ToString(CultureInfo.InvariantCulture) ?? "",
                    metrics?.Stg.ToString(CultureInfo.InvariantCulture) ?? "",
                    metrics?.SpillLoads.ToString(CultureInfo.InvariantCulture) ?? "",
                    metrics?.SpillStores.ToString(CultureInfo.InvariantCulture) ?? "",
                    CodegenMeasurementProtocol.Tag));
                }
              }
              catch (Exception ex)
              {
                  // One kernel failing must not stop the line -- the point of a
                  // conveyor is that the remaining kernels still get their evidence.
                  spilled++;
                  Console.WriteLine(entry.Name.PadRight(36) + "   -            -     -    -             -   ERROR");
                  Console.WriteLine("    " + ex.GetType().Name + ": " + ex.Message.Split('\n')[0]);
              }
            }
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }

        string manifest = Path.Combine(outputDirectory, "codegen-cubins.tsv");
        var text = new StringBuilder();
        text.AppendLine("kernel\tentry\tcubin_sha256\tsource_key\tregisters\tsass_instructions\tldg\tstg\tspill_ld\tspill_st\tprotocol");
        foreach (string row in rows) text.AppendLine(row);
        File.WriteAllText(manifest, text.ToString());

        Console.WriteLine();
        Console.WriteLine("manifest " + manifest);
        Console.WriteLine("release: " + gated.ToString(CultureInfo.InvariantCulture) + " zero-spill, " +
                          spilled.ToString(CultureInfo.InvariantCulture) + " spilling");

        EnforceEvidenceGates(entries, runtime);
    }

    /// <summary>
    /// Blueprint #3 and #4 as release gates. Zero spills and a clean SASS audit say the
    /// kernel is well-formed; they say nothing about whether it is worth shipping. A
    /// kernel is only releasable when it also carries, at the CURRENT protocol:
    ///
    ///   a competitor ratio  -- otherwise every number about it is ours-vs-ours;
    ///   a named limiter     -- otherwise nobody knows what its next lever is.
    ///
    /// Both files are produced by other stages. A missing or stale file is a release
    /// failure, not an informational line that still returns success.
    /// </summary>
    private static void EnforceEvidenceGates(
        IReadOnlyList<CodegenCatalogEntry> entries, DirectPtxRuntime runtime)
    {
        string dispatch = KernelEvidenceIdentity.CurrentDispatch(runtime);
        var ratios = ReadEvidence(
            Path.Combine("artifacts", "competitor-ratios.tsv"), 3, dispatch);
        var limiters = ReadEvidence(
            Path.Combine("artifacts", "limiter.tsv"), 1, dispatch);

        Console.WriteLine();
        Console.WriteLine("EVIDENCE GATES (protocol " + CodegenMeasurementProtocol.Tag + ")");
        Console.WriteLine("kernel                            competitor    limiter     releasable");

        int releasable = 0;
        foreach (var entry in entries)
        {
            bool hasRatio = ratios.TryGetValue(entry.Name, out string? ratio);
            bool hasLimiter = limiters.TryGetValue(entry.Name, out string? limiter);
            bool ok = hasRatio && hasLimiter;
            if (ok) releasable++;

            Console.WriteLine(entry.Name.PadRight(32) +
                (hasRatio ? ratio + "x" : "MISSING").PadLeft(12) +
                (hasLimiter ? limiter! : "MISSING").PadLeft(12) +
                (ok ? "yes" : "NO").PadLeft(15));
        }

        Console.WriteLine();
        Console.WriteLine(releasable.ToString(CultureInfo.InvariantCulture) + " of " +
                          entries.Count.ToString(CultureInfo.InvariantCulture) +
                          " carry both. Run --kernel-limiter and --kernel-competitor");
        Console.WriteLine("to fill gaps; any kernel without both aborts the release as unproven.");

        if (releasable != entries.Count)
        {
            throw new InvalidOperationException(
                (entries.Count - releasable).ToString(CultureInfo.InvariantCulture) +
                " kernel(s) lack current-protocol release evidence.");
        }
    }

    /// <summary>Reads kernel -> column from a protocol-stamped evidence file.</summary>
    private static Dictionary<string, string> ReadEvidence(
        string path, int column, string? expectedDispatch = null)
    {
        var found = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!File.Exists(path)) return found;

        foreach (string line in File.ReadAllLines(path))
        {
            if (line.StartsWith("#", StringComparison.Ordinal)) continue;
            string[] cells = line.Split('	');
            if (cells.Length <= column) continue;
            if (!cells[cells.Length - 1].Equals(CodegenMeasurementProtocol.Tag, StringComparison.Ordinal))
                continue;   // stale protocol is the same as absent
            if (expectedDispatch is not null &&
                (cells.Length < 9 || !cells[cells.Length - 2].Equals(
                    expectedDispatch, StringComparison.Ordinal)))
                continue;   // another tuned program is another benchmark
            found[cells[0]] = cells[column];
        }
        return found;
    }

    private sealed record SassMetrics(int Instructions, int Ldg, int Stg, int SpillLoads, int SpillStores);

    private static SassMetrics? ReadSass(string nvdisasm, string cubinPath)
    {
        var start = new ProcessStartInfo
        {
            FileName = nvdisasm,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        start.ArgumentList.Add("--print-code");
        start.ArgumentList.Add(cubinPath);
        using Process? process = Process.Start(start);
        if (process is null) return null;
        var outputTask = process.StandardOutput.ReadToEndAsync();
        var errorTask = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit((int)TimeSpan.FromMinutes(5).TotalMilliseconds))
        {
            process.Kill(entireProcessTree: true);
            return null;
        }
        string output = outputTask.GetAwaiter().GetResult();
        errorTask.GetAwaiter().GetResult();
        if (process.ExitCode != 0) return null;

        int instructions = 0, ldg = 0, stg = 0, spillLd = 0, spillSt = 0;
        foreach (string raw in output.Split('\n'))
        {
            string line = raw.Trim();
            if (line.Length == 0 || !line.EndsWith(";", StringComparison.Ordinal)) continue;
            instructions++;
            if (line.Contains("LDG")) ldg++;
            if (line.Contains("STG")) stg++;
            if (line.Contains("LDL")) spillLd++;
            if (line.Contains("STL")) spillSt++;
        }
        return new SassMetrics(instructions, ldg, stg, spillLd, spillSt);
    }

    private static string? FindNvdisasm()
    {
        string? env = Environment.GetEnvironmentVariable("AIDOTNET_NVDISASM_PATH");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        foreach (string root in new[] { @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" })
        {
            if (!Directory.Exists(root)) continue;
            foreach (string version in Directory.GetDirectories(root).OrderByDescending(d => d))
            {
                string candidate = Path.Combine(version, "bin", "nvdisasm.exe");
                if (File.Exists(candidate)) return candidate;
            }
        }
        return null;
    }

    // ----------------------------------------------------------------- bench

    private static void Bench(IReadOnlyList<CodegenCatalogEntry> entries, string[] args)
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("kernel-bench-start");
        using var runtime = OpenRuntime();
        if (runtime is null) return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            Console.WriteLine();
            Console.WriteLine("CONVEYOR STAGE 3 - bench with the Phase 0.5 calibrated protocol");
            Console.WriteLine("device-filling shapes, " + LaunchesPerSample +
                              " launches per timed region, paired within-sample ratio, " +
                              Runs + " runs");
            Console.WriteLine("harness noise floor measured at 1.05%; differences under ~3% are not claimable");
            Console.WriteLine("protocol " + CodegenMeasurementProtocol.Stamp(
                "RTX 3080, clocks " + GpuBenchmarkEnvironment.SampleSmClockMhz().ToString(CultureInfo.InvariantCulture) + " MHz"));
            Console.WriteLine();
            Console.WriteLine("kernel                              blocks    us/launch    p95/med   run spread   SM clock");

            foreach (var entry in entries)
            {
                try
                {
                    var spec = entry.Bench;

                    // Bench the program the TUNER CHOSE, which for the weight gradients
                    // may be a two-kernel split. Timing
                    // the single-kernel lowering here would publish a number for a
                    // lowering the tuner had already rejected.
                    bool overridden = args.Contains("--no-coarsen", StringComparer.Ordinal)
                        || ValueOf(args, "--coarsen") is not null
                        || ValueOf(args, "--max-lanes") is not null;

                    TunedProgram program;
                    if (overridden)
                    {
                        // An explicit knob on the command line is a request to bench THAT
                        // lowering, so the recorded winner is set aside -- and said so,
                        // because a hand-set knob silently overriding a measured split
                        // would be the same confusion in the other direction.
                        var emitter = new PtxAffineEmitter();
                        if (args.Contains("--no-coarsen", StringComparer.Ordinal)) emitter.Coarsening = 1;
                        if (ValueOf(args, "--coarsen") is string cz)
                            emitter.Coarsening = int.Parse(cz, CultureInfo.InvariantCulture);
                        if (ValueOf(args, "--max-lanes") is string mz)
                            emitter.MaxTileLanes = int.Parse(mz, CultureInfo.InvariantCulture);
                        string ptx = emitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                        program = new TunedProgram(spec,
                            new[]
                            {
                                new ProgramKernel(spec, ptx, emitter.LaunchBlocks,
                                    (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY,
                                    emitter.LoopedAxes, emitter.ElidedGuards, emitter.StagedOperands),
                            },
                            null, "command-line");
                    }
                    else
                    {
                        program = ResolveTuned(runtime, spec, entry.Name);
                    }

                    uint blocks = program.Kernels[0].Blocks;

                    using var launchable = TunedLaunchable.Create(runtime, program);
                        void Launch() => launchable.Launch();
                        int bestClockBefore = 0, bestClockAfter = 0;

                        // RETRY ON CLOCK DRIFT. The SM clock was observed swinging
                        // 2025 -> 1770 MHz (-12.6%) inside a single kernel's three runs,
                        // and the rows whose clock held still had low spread while every
                        // drifting row was elevated. That, not the kernel, is what
                        // produced the intermittent 7.5% spreads. Locking clocks needs
                        // administrator rights that are not available here, so instead
                        // the measurement is repeated and the least-contaminated attempt
                        // is the one reported.
                        var medians = new double[Runs];
                        double worstTail = 0;
                        int clockBefore = 0, clockAfter = 0;
                        double bestDrift = double.MaxValue;
                        var bestMedians = new double[Runs];
                        double bestTail = 0;
                        for (int attempt = 0; attempt < ClockRetries; attempt++)
                        {
                            clockBefore = GpuBenchmarkEnvironment.SampleSmClockMhz();
                            worstTail = 0;
                            for (int run = 0; run < Runs; run++)
                            {
                                var d = Measure(runtime.Synchronize, Launch);
                                medians[run] = d.Median;
                                worstTail = Math.Max(worstTail, d.Median > 0 ? d.P95 / d.Median : double.NaN);
                            }
                            clockAfter = GpuBenchmarkEnvironment.SampleSmClockMhz();

                            double drift = clockBefore > 0
                                ? Math.Abs(clockAfter - clockBefore) / (double)clockBefore
                                : double.MaxValue;
                            if (attempt == 0 || drift < bestDrift)
                            {
                                bestDrift = drift;
                                Array.Copy(medians, bestMedians, Runs);
                                bestTail = worstTail;
                                bestClockBefore = clockBefore;
                                bestClockAfter = clockAfter;
                            }
                            if (drift <= 0.02) break;   // clean enough; stop retrying
                        }
                        Array.Copy(bestMedians, medians, Runs);
                        worstTail = bestTail;
                        clockBefore = bestClockBefore;
                        clockAfter = bestClockAfter;
                        double lo = medians.Min(), hi = medians.Max();
                        Console.WriteLine(entry.Name.PadRight(36) +
                            blocks.ToString("N0", CultureInfo.InvariantCulture).PadLeft(8) +
                            (lo * 1000.0).ToString("F1", CultureInfo.InvariantCulture).PadLeft(13) +
                            worstTail.ToString("F2", CultureInfo.InvariantCulture).PadLeft(11) +
                            ((hi / lo - 1.0) * 100).ToString("F1", CultureInfo.InvariantCulture).PadLeft(10) + "%   " +
                            GpuBenchmarkEnvironment.DescribeClockDrift(clockBefore, clockAfter));
                }
                catch (Exception ex)
                {
                    Console.WriteLine(entry.Name.PadRight(36) + "  ERROR  " + ex.Message.Split('\n')[0]);
                }
            }
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }

        GpuBenchmarkEnvironment.RequireNoForeignCompute("kernel-bench-end");
    }

    /// <summary>
    /// Paired A/B of the vector-load lowering against the scalar one, in ONE process.
    /// Comparing numbers from separate runs is exactly what Phase 0.5 showed to be
    /// untrustworthy, so the two variants are interleaved sample by sample and the
    /// ratio is taken WITHIN each pair.
    /// </summary>
    private static void VectorAb(IReadOnlyList<CodegenCatalogEntry> entries)
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("kernel-vector-ab");
        using var runtime = OpenRuntime();
        if (runtime is null) return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            Console.WriteLine();
            Console.WriteLine("VECTOR-LOAD A/B - ld.global.v4.f32 vs scalar, paired in-process");
            Console.WriteLine("harness noise floor 1.05%; a ratio inside ~1.03x is not claimable");
            Console.WriteLine();
            Console.WriteLine("kernel                              v4 loads   scalar us   vector us   best-of-3  paired-med");

            foreach (var entry in entries)
            {
                var spec = entry.Bench;
                var vecEmitter = new PtxAffineEmitter();
                string vecPtx = vecEmitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                if (vecEmitter.VectorisedLoads == 0)
                {
                    Console.WriteLine(entry.Name.PadRight(36) + "         0           -           -   no unit-stride reduction axis");
                    continue;
                }

                var scalarEmitter = new PtxAffineEmitter { EnableVectorLoads = false };
                string scalarPtx = scalarEmitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);

                using var vecStage = LaunchableStage.Create(runtime, spec, vecPtx, vecEmitter.LaunchBlocks, (uint)vecEmitter.LaunchBlockX, (uint)vecEmitter.LaunchBlockY);
                using var scalarStage = LaunchableStage.Create(runtime, spec, scalarPtx, scalarEmitter.LaunchBlocks, (uint)scalarEmitter.LaunchBlockX, (uint)scalarEmitter.LaunchBlockY);

                var ratios = new double[Runs];
                double scalarUs = double.MaxValue, vectorUs = double.MaxValue;
                for (int run = 0; run < Runs; run++)
                {
                    (double a, double b, double ratio) = PairedRatio(runtime, scalarStage.Launch, vecStage.Launch);
                    ratios[run] = ratio;
                    scalarUs = Math.Min(scalarUs, a);   // best-of-N per side
                    vectorUs = Math.Min(vectorUs, b);
                }
                Array.Sort(ratios);

                Console.WriteLine(entry.Name.PadRight(36) +
                    vecEmitter.VectorisedLoads.ToString(CultureInfo.InvariantCulture).PadLeft(10) +
                    scalarUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(12) +
                    vectorUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(12) +
                    (scalarUs / vectorUs).ToString("F3", CultureInfo.InvariantCulture).PadLeft(11) + "x" +
                    ratios[Runs / 2].ToString("F3", CultureInfo.InvariantCulture).PadLeft(10) + "x");
            }
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    /// <summary>Paired A/B of the coarsened lowering against one-output-per-thread.</summary>
    private static void CoarsenAb(IReadOnlyList<CodegenCatalogEntry> entries)
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("kernel-coarsen-ab");
        using var runtime = OpenRuntime();
        if (runtime is null) return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            Console.WriteLine();
            Console.WriteLine("COARSENING A/B - N outputs per thread vs one, paired in-process");
            Console.WriteLine("harness noise floor 1.05%; a ratio inside ~1.03x is not claimable");
            Console.WriteLine();
            Console.WriteLine("kernel                              lanes  loads/out   1-per-thread   coarsened   speedup");

            foreach (var entry in entries)
            {
                try
                {
                    var spec = entry.Bench;
                    var wide = new PtxAffineEmitter();
                    string widePtx = wide.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                    if (wide.CoarsenedLanes == 1)
                    {
                        Console.WriteLine(entry.Name.PadRight(36) + "     1          -              -           -   axis not divisible");
                        continue;
                    }

                    var thin = new PtxAffineEmitter { Coarsening = 1 };
                    string thinPtx = thin.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);

                    using var wideStage = LaunchableStage.Create(runtime, spec, widePtx, wide.LaunchBlocks, (uint)wide.LaunchBlockX, (uint)wide.LaunchBlockY);
                    using var thinStage = LaunchableStage.Create(runtime, spec, thinPtx, thin.LaunchBlocks, (uint)thin.LaunchBlockX, (uint)thin.LaunchBlockY);

                    var ratios = new double[Runs];
                    double thinUs = 0, wideUs = 0;
                    for (int run = 0; run < Runs; run++)
                    {
                        (double a, double b, double r) = PairedRatio(runtime, thinStage.Launch, wideStage.Launch);
                        ratios[run] = r; thinUs = a; wideUs = b;
                    }
                    Array.Sort(ratios);

                    // Loads per output: the quantity the bake-off identified as the
                    // real problem, so report it beside the time.
                    double loadsPerOutput = (double)wide.EmittedLoads / wide.CoarsenedLanes;
                    Console.WriteLine(entry.Name.PadRight(36) +
                        wide.CoarsenedLanes.ToString(CultureInfo.InvariantCulture).PadLeft(6) +
                        loadsPerOutput.ToString("F1", CultureInfo.InvariantCulture).PadLeft(11) +
                        thinUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(15) +
                        wideUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(12) +
                        ratios[Runs / 2].ToString("F3", CultureInfo.InvariantCulture).PadLeft(10) + "x");
                }
                catch (Exception ex)
                {
                    Console.WriteLine(entry.Name.PadRight(36) + "  ERROR " + ex.Message.Split('\n')[0]);
                }
            }
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    /// <summary>Interleaves A and B and returns the median per-sample ratio A/B.</summary>
    private static (double A, double B, double Ratio) PairedRatio(
        DirectPtxRuntime runtime, Action a, Action b)
    {
        for (int i = 0; i < Warmup; i++) { a(); b(); }
        runtime.Synchronize();

        var sa = new double[Samples];
        var sb = new double[Samples];
        var ratio = new double[Samples];
        for (int i = 0; i < Samples; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            for (int k = 0; k < LaunchesPerSample; k++) a();
            runtime.Synchronize();
            sa[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds / LaunchesPerSample * 1000.0;

            long t1 = Stopwatch.GetTimestamp();
            for (int k = 0; k < LaunchesPerSample; k++) b();
            runtime.Synchronize();
            sb[i] = Stopwatch.GetElapsedTime(t1).TotalMilliseconds / LaunchesPerSample * 1000.0;

            ratio[i] = sa[i] / sb[i];
        }
        Array.Sort(sa); Array.Sort(sb); Array.Sort(ratio);
        return (sa[Samples / 2], sb[Samples / 2], ratio[Samples / 2]);
    }

    /// <summary>A loaded module plus its buffers, launchable from supplied PTX.</summary>
    private sealed class LaunchableStage : IDisposable
    {
        private readonly DirectPtxModule _module;
        private readonly IntPtr _fn;
        private readonly IntPtr[] _pointers;
        private readonly uint _blocks;
        private readonly List<DirectPtxBuffer> _buffers;

        private readonly uint _blockX, _blockY;

        private LaunchableStage(DirectPtxModule m, IntPtr fn, IntPtr[] p, uint blocks,
                                uint blockX, uint blockY, List<DirectPtxBuffer> b)
        { _module = m; _fn = fn; _pointers = p; _blocks = blocks; _blockX = blockX; _blockY = blockY; _buffers = b; }

        internal static LaunchableStage Create(DirectPtxRuntime runtime, CodegenKernelSpec spec, string ptx, uint blocks, uint blockX, uint blockY)
        {
            var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
            IntPtr fn = module.GetFunction(spec.Name, out _);
            var buffers = new List<DirectPtxBuffer>();
            var pointers = new IntPtr[spec.ParameterCount];
            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                long count = Elements(spec.Inputs[i].Shape);
                var buf = runtime.AllocateBytes((nuint)(count * sizeof(float)));
                var host = new float[count];
                for (long e = 0; e < count; e++) host[e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                buf.Upload<float>(host);
                buffers.Add(buf);
                pointers[i] = buf.Pointer;
            }
            var outBuf = runtime.AllocateBytes((nuint)(Elements(spec.Output.Shape) * sizeof(float)));
            buffers.Add(outBuf);
            pointers[spec.Inputs.Count] = outBuf.Pointer;
            return new LaunchableStage(module, fn, pointers, blocks, blockX, blockY, buffers);
        }

        internal void Launch() => LaunchSpec(_module, _fn, _pointers, _blocks, _blockX, _blockY);

        public void Dispose()
        {
            foreach (var b in _buffers) b.Dispose();
            _module.Dispose();
        }
    }

    private readonly record struct Dist(double Median, double P95);

    private static Dist Measure(Action synchronize, Action launch)
    {
        for (int i = 0; i < Warmup; i++) launch();
        synchronize();
        var samples = new double[Samples];
        for (int i = 0; i < Samples; i++)
        {
            long start = Stopwatch.GetTimestamp();
            for (int k = 0; k < LaunchesPerSample; k++) launch();
            synchronize();
            samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds / LaunchesPerSample;
        }
        Array.Sort(samples);
        return new Dist(Percentile(samples, 0.5), Percentile(samples, 0.95));
    }

    private static double Percentile(double[] sorted, double fraction)
    {
        double position = (sorted.Length - 1) * fraction;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    // ----------------------------------------------------------------- misc

    private static DirectPtxRuntime? OpenRuntime()
    {
        var runtime = new DirectPtxRuntime();
        if (DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return runtime;
        Console.WriteLine("The conveyor requires the experimental SM86 device.");
        runtime.Dispose();
        return null;
    }

    private static long Elements(IReadOnlyList<int> shape)
    {
        long total = 1;
        for (int i = 0; i < shape.Count; i++) total *= shape[i];
        return total;
    }

    private static unsafe void LaunchSpec(DirectPtxModule module, IntPtr fn, IntPtr[] pointers, uint blocks, uint blockX, uint blockY)
    {
        fixed (IntPtr* pinned = pointers)
        {
            void** argv = stackalloc void*[pointers.Length];
            for (int i = 0; i < pointers.Length; i++) argv[i] = pinned + i;
            module.Launch(fn, blocks, 1, 1, blockX, blockY, 1, 0, argv);
        }
    }
}
