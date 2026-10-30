// Copyright (c) CrowdTrainer LLC. All rights reserved.
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
    private const double Tolerance = CodegenMeasurementProtocol.AccumulationTolerance;
    private const int Warmup = 20;
    private const int Samples = 51;
    private const int LaunchesPerSample = 50;
    private const int Runs = 3;
    private const int StabilityAttempts = 15;

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

    /// <summary>
    /// Runs row-shaped programs as their separate passes instead of one fused kernel, for
    /// a paired comparison of the two lowerings.
    /// </summary>
    private static bool _noRowFusion;

    /// <summary>
    /// Added to every value of a program's "x" source (--x-offset). Data far from zero is what
    /// exposes cancellation in a statistic, which data centred on zero never does.
    /// </summary>
    private static double _xOffset;

    internal static void Run(string stage, string[] args)
    {
        _forceInputStaging = args.Contains("--input-staging", StringComparer.Ordinal);
        _noRowFusion = args.Contains("--no-row-fusion", StringComparer.Ordinal);
        _xOffset = ValueOf(args, "--x-offset") is string xo
            ? double.Parse(xo, CultureInfo.InvariantCulture) : 0.0;
        string selector = KernelToolArgs.Selector(args);
        if (stage == "static-rows")
        {
            StaticRows();
            return;
        }
        var entries = Select(selector);
        var programs = SelectPrograms(selector);
        // Program entries are carried by verify and bench; the other stages are per-kernel.
        bool programStage = stage is "verify" or "bench" or "incumbent";
        if (entries.Count == 0 && (!programStage || programs.Count == 0))
        {
            Console.WriteLine("No catalog entry matches '" + selector + "'. Known kernels:");
            foreach (var e in CodegenKernelCatalog.All)
                Console.WriteLine("  " + e.Name.PadRight(32) + e.Summary);
            foreach (var e in CodegenKernelCatalog.Programs)
                Console.WriteLine("  " + e.Name.PadRight(32) + e.Summary + " (program)");
            KernelToolArgs.RequireNonEmptySelection(
                selector, entries.Count, "kernel-" + stage);
        }

        switch (stage)
        {
            case "verify":
                if (entries.Count != 0) Verify(entries);
                if (programs.Count != 0) VerifyPrograms(programs);
                break;
            case "release": Release(entries, args); break;
            case "bench":
                if (entries.Count != 0) Bench(entries, args);
                if (programs.Count != 0) BenchPrograms(programs);
                break;
            case "incumbent":
                Incumbent(programs);
                break;
            case "vector-ab": VectorAb(entries); break;
            case "dump": Dump(entries, args); break;
            case "coarsen-ab": CoarsenAb(entries); break;
            case "once": Once(entries, args); break;
            case "predict": Predict(entries, args); break;
            default: Console.WriteLine("Unknown conveyor stage '" + stage + "'."); break;
        }
    }

    private static IReadOnlyList<CodegenCatalogProgramEntry> SelectPrograms(string selector)
    {
        if (string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase))
            return CodegenKernelCatalog.Programs;
        var one = CodegenKernelCatalog.FindProgram(selector);
        return one is null ? Array.Empty<CodegenCatalogProgramEntry>() : new[] { one };
    }

    /// <summary>
    /// Prints the static-ratchet rows (codegen-static-baseline.tsv) for every ratcheted
    /// kernel, computed exactly as CodegenRatchetTests computes them, so a baseline change
    /// can be produced without running the test suite.
    /// </summary>
    private static void StaticRows()
    {
        Console.WriteLine("kernel\tptx_lines\tloads\tvector_loads\telided_guards\tlooped_axes");
        foreach (var (name, spec) in CodegenKernelCatalog.RatchetedKernels())
        {
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(spec, 8, 6);
            Console.WriteLine(string.Join("\t", name,
                ptx.Split('\n').Length.ToString(CultureInfo.InvariantCulture),
                emitter.EmittedLoads.ToString(CultureInfo.InvariantCulture),
                emitter.VectorisedLoads.ToString(CultureInfo.InvariantCulture),
                emitter.ElidedGuards.ToString(CultureInfo.InvariantCulture),
                emitter.LoopedAxes.ToString(CultureInfo.InvariantCulture)));
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
            // --winner applies a tuned configuration, so a candidate's PTX can be inspected
            // (and its emission failures reproduced) without a device; --arch 75 targets Turing.
            ApplyTuned(emitter, entry.Name, ValueOf(args, "--winner"));
            int major = ValueOf(args, "--arch") is string arch ? int.Parse(arch, CultureInfo.InvariantCulture) / 10 : 8;
            int minor = ValueOf(args, "--arch") is string arch2 ? int.Parse(arch2, CultureInfo.InvariantCulture) % 10 : 6;
            string ptx = emitter.Emit(entry.Bench, major, minor);
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
                    || ValueOf(args, "--max-lanes") is not null
                    || ValueOf(args, "--winner") is not null;

                TunedProgram program;
                if (overridden)
                {
                    var emitter = new PtxAffineEmitter();
                    if (args.Contains("--no-coarsen", StringComparer.Ordinal)) emitter.Coarsening = 1;
                    if (ValueOf(args, "--coarsen") is string cz)
                        emitter.Coarsening = int.Parse(cz, CultureInfo.InvariantCulture);
                    if (ValueOf(args, "--max-lanes") is string mz)
                        emitter.MaxTileLanes = int.Parse(mz, CultureInfo.InvariantCulture);
                    // --winner: profile a named candidate (implicit-gemm, ...) rather than the recorded winner.
                    ApplyTuned(emitter, entry.Name, ValueOf(args, "--winner"));
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
        if (failed != 0 || passed != entries.Count)
            throw new InvalidOperationException(
                "Kernel verification failed closed: " +
                passed.ToString(CultureInfo.InvariantCulture) + " of " +
                entries.Count.ToString(CultureInfo.InvariantCulture) +
                " selected operations passed.");
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
    /// <summary>Implicit-GEMM tile tokens of the split winners ("m128n64"): no 'x', which the chunk factor uses.</summary>
    internal static readonly string[] ImplicitSplitTiles = { "m64n64", "m96n64", "m128n64" };

    /// <summary>Configures an emitter for an implicit-GEMM tile token such as "m128n64".</summary>
    internal static void ApplyImplicitTile(PtxAffineEmitter emitter, string token)
    {
        int n = token.IndexOf('n');
        if (!token.StartsWith("m", StringComparison.Ordinal) || n < 2 ||
            !int.TryParse(token.Substring(1, n - 1), NumberStyles.None, CultureInfo.InvariantCulture, out int m) ||
            !int.TryParse(token.Substring(n + 1), NumberStyles.None, CultureInfo.InvariantCulture, out int nEdge))
            throw new ArgumentException("Not an implicit-GEMM tile token: '" + token + "'.", nameof(token));
        emitter.ImplicitGemm = true;
        emitter.ImplicitGemmTile = nEdge;
        emitter.ImplicitGemmTileM = m;
    }

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
            case "implicit-gemm": emitter.ImplicitGemm = true; break;
            case "implicit-gemm-32": emitter.ImplicitGemm = true; emitter.ImplicitGemmTile = 32; break;
            case "implicit-gemm-128x64": emitter.ImplicitGemm = true; emitter.ImplicitGemmTileM = 128; break;
            case "implicit-gemm-96x64": emitter.ImplicitGemm = true; emitter.ImplicitGemmTileM = 96; break;
            case "implicit-gemm-64x128": emitter.ImplicitGemm = true; emitter.ImplicitGemmTile = 128; emitter.ImplicitGemmTileM = 64; break;

            // A split winner is not a knob -- it is a different PROGRAM, two kernels and a
            // temporary, which this single-kernel path cannot launch. Saying so is the
            // point: silently falling through would report a kernel as tuned while running
            // the lowering the tuner measured as 17x slower.
            case not null when winner.StartsWith("split:", StringComparison.Ordinal):
            case not null when winner.StartsWith("tiled-split:", StringComparison.Ordinal):
            case not null when winner.StartsWith("tiled-chunked-split:", StringComparison.Ordinal):
            // ig-chunked-split is a split program too (TryBuildTunedProgram treats it as one at
            // "implicitSplit"), so it belongs with the other prefixes rather than in the default arm.
            case not null when winner.StartsWith("ig-chunked-split:", StringComparison.Ordinal):
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
        int LoopedAxes, int ElidedGuards, string StagedOperands,
        uint DynamicSharedMemoryBytes = 0);

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
            ? (Winner is not null &&
               Winner.StartsWith("tiled-chunked-split:", StringComparison.Ordinal)
                ? "tiled chunked split x"
                : Winner is not null && Winner.StartsWith("tiled-split:", StringComparison.Ordinal)
                    ? "tiled split x"
                : CodegenTiledConv2DSplitSchedule.Find(Winner) is not null
                    ? "tiled conv2d split x"
                    : "split x") + Kernels.Count.ToString(CultureInfo.InvariantCulture)
            : string.Equals(Winner, "tiled-contraction", StringComparison.Ordinal) ||
              CodegenTiledContractionSchedule.Find(Winner) is not null
                ? "tiled contraction"
                : string.Equals(Winner, "tiled-conv2d", StringComparison.Ordinal) ||
                  CodegenTiledConv2DSchedule.Find(Winner) is not null
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

        CodegenOuterProductWinogradSchedule? winogradSchedule =
            CodegenOuterProductWinogradSchedule.Find(winner);
        if (winogradSchedule is not null)
        {
            try
            {
                var winograd = new PtxOuterProductWinogradConv2DEmitter(winogradSchedule);
                string text = winograd.Emit(
                    spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                return new TunedProgram(
                    spec,
                    new[]
                    {
                        new ProgramKernel(spec, text, winograd.LaunchBlocks,
                            checked((uint)winograd.LaunchBlockThreads), 1,
                            0, 0, "weights+input tiles",
                            checked((uint)winograd.SharedMemoryBytes)),
                    },
                    null, winner);
            }
            catch (NotSupportedException ex)
            {
                Console.WriteLine("    note: " + catalogName +
                                  " recorded " + winner + " but it could not " +
                                  "be rebuilt (" + ex.Message + "); using the affine kernel");
            }
        }

        CodegenTiledContractionSchedule? contractionSchedule =
            CodegenTiledContractionSchedule.Find(winner);
        if (string.Equals(winner, "tiled-contraction", StringComparison.Ordinal) ||
            contractionSchedule is not null)
        {
            try
            {
                var tiled = contractionSchedule is null
                    ? new PtxTiledContractionEmitter()
                    : new PtxTiledContractionEmitter(contractionSchedule);
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
                Console.WriteLine("    note: " + catalogName + " recorded " + winner + " " +
                                  "but it could not be rebuilt (" + ex.Message +
                                  "); using the affine kernel");
            }
        }

        CodegenTiledConv2DSplitSchedule? conv2DSplitSchedule =
            CodegenTiledConv2DSplitSchedule.Find(winner);
        if (conv2DSplitSchedule is not null)
        {
            try
            {
                if (!CodegenTiledConv2DSplitPlan.TryCreate(
                        spec, conv2DSplitSchedule,
                        out CodegenTiledConv2DSplitPlan? exact, out string reason))
                    throw new NotSupportedException(reason);

                CodegenSplitPlan split = exact!.Split;
                var partial = new PtxTiledConv2DEmitter(conv2DSplitSchedule.Tile);
                string partialText = partial.Emit(
                    split.Partial, runtime.ComputeCapabilityMajor,
                    runtime.ComputeCapabilityMinor);
                var combine = new PtxAffineEmitter();
                string combineText = combine.Emit(
                    split.Combine, runtime.ComputeCapabilityMajor,
                    runtime.ComputeCapabilityMinor);
                return new TunedProgram(
                    spec,
                    new[]
                    {
                        new ProgramKernel(split.Partial, partialText,
                            partial.LaunchBlocks,
                            checked((uint)partial.LaunchBlockThreads), 1,
                            split.Partial.Space.ReductionAxes.Length, 0,
                            "weights+three input rows"),
                        new ProgramKernel(split.Combine, combineText,
                            combine.LaunchBlocks,
                            checked((uint)combine.LaunchBlockX),
                            checked((uint)combine.LaunchBlockY),
                            combine.LoopedAxes, combine.ElidedGuards,
                            combine.StagedOperands),
                    },
                    split, winner);
            }
            catch (NotSupportedException ex)
            {
                Console.WriteLine("    note: " + catalogName + " recorded " + winner +
                                  " but its tiled Conv2D split could not be rebuilt (" +
                                  ex.Message + "); using the single-kernel lowering");
            }
        }

        CodegenTiledConv2DSchedule? conv2DSchedule =
            CodegenTiledConv2DSchedule.Find(winner);
        if (string.Equals(winner, "tiled-conv2d", StringComparison.Ordinal) ||
            conv2DSchedule is not null)
        {
            try
            {
                var tiled = conv2DSchedule is null
                    ? new PtxTiledConv2DEmitter()
                    : new PtxTiledConv2DEmitter(conv2DSchedule);
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
                            spec.Space.ReductionAxes.Length, 0, "input parity tile+weights"),
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
        bool implicitSplit = winner is not null &&
            winner.StartsWith("ig-chunked-split:", StringComparison.Ordinal);
        bool tiledChunkedSplit = winner is not null &&
            winner.StartsWith("tiled-chunked-split:", StringComparison.Ordinal);
        if (winner is not null &&
            (winner.StartsWith("split:", StringComparison.Ordinal) || tiledSplit ||
             tiledChunkedSplit || implicitSplit))
        {
            CodegenSplitPlan? plan = null;
            try
            {
                if (implicitSplit)
                    plan = TryParseChunkFactor(winner, out int implicitFactor)
                        ? CodegenSplitReduction.TryPlanChunkedForTiles(spec, implicitFactor)
                        : null;
                else if (tiledChunkedSplit)
                    plan = TryParseChunkFactor(winner, out int chunkFactor)
                        ? CodegenSplitReduction.TryPlanChunked(spec, chunkFactor)
                        : null;
                else
                    plan = CodegenSplitReduction.TryPlan(spec);
            }
            catch (NotSupportedException) { }

            if (plan is not null)
            {
                var halves = new List<ProgramKernel>(2);
                bool emitted = true;
                if (tiledSplit || tiledChunkedSplit)
                {
                    try
                    {
                        PtxTiledOuterProductProgram tiled =
                            PtxTiledOuterProductDispatcher.Emit(
                                plan.Partial, runtime.ComputeCapabilityMajor,
                                runtime.ComputeCapabilityMinor,
                                CodegenTiledOuterProductSchedule.FindForWinner(winner));
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
                    // ig-chunked-split winners lower the partial pass as an implicit GEMM; the tile is the last field.
                    if (implicitSplit) ApplyImplicitTile(partial, winner.Substring(winner.LastIndexOf(':') + 1));
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

    private static bool TryParseChunkFactor(string winner, out int chunkFactor)
    {
        chunkFactor = 0;
        int marker = winner.LastIndexOf('x');
        int end = marker < 0 ? -1 : winner.IndexOf(':', marker + 1);
        if (end < 0) end = winner.Length;
        return marker >= 0 && marker + 1 < end &&
            int.TryParse(winner.Substring(marker + 1, end - marker - 1), NumberStyles.None,
                CultureInfo.InvariantCulture, out chunkFactor) &&
            CodegenAutotuneIdentity.IsChunkedSplitFactor(chunkFactor);
    }

    /// <summary>A loaded, bound tuned program: one launch call whatever its shape.</summary>
    private sealed class TunedLaunchable : IDisposable
    {
        private readonly List<DirectPtxModule> _modules = new();
        private readonly List<DirectPtxBuffer> _buffers = new();
        private readonly List<(DirectPtxModule Module, IntPtr Fn, IntPtr[] Args,
            uint Blocks, uint X, uint Y, uint DynamicSharedMemoryBytes)> _steps = new();
        private DirectPtxGraph? _graph;
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
                    int[] partialSources = CodegenSplitReduction.PartialOperandSources(spec);
                    for (int i = 0; i < partialSources.Length; i++)
                        partialArgs[i] = uploaded[partialSources[i]];
                    partialArgs[partialArgs.Length - 1] = temp.Pointer;

                    var combineArgs = new IntPtr[plan.Combine.ParameterCount];
                    combineArgs[0] = temp.Pointer;
                    int[] combineSources = CodegenSplitReduction.CombineOperandSources(spec);
                    for (int i = 0; i < combineSources.Length; i++)
                        combineArgs[1 + i] = uploaded[combineSources[i]];
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

                // The competitor's accepted lane is CUDA-graph replay. Capture the exact
                // tuned program too, so every catalog kernel gets the same launch-overhead
                // treatment and a one- versus two-kernel lowering remains comparable.
                it._graph = runtime.CaptureGraph(it.LaunchDirect);

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
            if (kernel.DynamicSharedMemoryBytes != 0)
                module.SetMaxDynamicSharedMemory(
                    fn, checked((int)kernel.DynamicSharedMemoryBytes));
            RegistersPerThread = Math.Max(RegistersPerThread, info.RegistersPerThread);
            _steps.Add((module, fn, args, kernel.Blocks, kernel.BlockX, kernel.BlockY,
                kernel.DynamicSharedMemoryBytes));
        }

        /// <summary>Runs the whole program in order; the combine depends on the partial.</summary>
        private void LaunchDirect()
        {
            foreach (var (module, fn, args, blocks, x, y, dynamicSharedMemoryBytes) in _steps)
                LaunchSpec(module, fn, args, blocks, x, y, dynamicSharedMemoryBytes);
        }

        internal void Launch()
        {
            if (_graph is not null) _graph.Launch();
            else LaunchDirect();
        }

        internal void DownloadOutput(float[] destination) => _output.Download<float>(destination);

        public void Dispose()
        {
            _graph?.Dispose();
            foreach (var b in _buffers) b.Dispose();
            foreach (var m in _modules) m.Dispose();
        }
    }

    /// <summary>
    /// A whole multi-pass program on the device: one buffer per NAME (sources, then every
    /// pass output), each pass lowered the way the tuner chose for it, and the chain captured
    /// as one CUDA graph so it is timed the way the competitor lane is.
    /// </summary>
    private sealed class ProgramLaunchable : IDisposable
    {
        private readonly List<DirectPtxModule> _modules = new();
        private readonly Dictionary<string, DirectPtxBuffer> _named = new(StringComparer.Ordinal);
        private readonly List<DirectPtxBuffer> _temporaries = new();
        // Every lowering buffer by name, temporaries ("#tempN") included.
        private readonly Dictionary<string, DirectPtxBuffer> _scratch = new(StringComparer.Ordinal);
        private readonly List<(DirectPtxModule Module, IntPtr Fn, IntPtr[] Args,
            uint Blocks, uint X, uint Y, uint DynamicSharedMemoryBytes)> _steps = new();
        private DirectPtxGraph? _graph;

        /// <summary>The fp64 value of every source, as uploaded.</summary>
        internal Dictionary<string, double[]> HostSources { get; } = new(StringComparer.Ordinal);

        /// <summary>Highest register count across every kernel of every pass.</summary>
        internal int RegistersPerThread { get; private set; }

        /// <summary>Kernels launched per program run, after each pass's lowering.</summary>
        internal int KernelCount => _steps.Count;

        /// <summary>Lowering label per pass, for the report.</summary>
        internal List<string> Lowerings { get; } = new();

        internal static ProgramLaunchable Create(
            DirectPtxRuntime runtime, CodegenProgram program, string catalogName)
        {
            var it = new ProgramLaunchable();
            try
            {
                int salt = 0;
                foreach (var source in program.Sources())
                {
                    long count = source.ElementCount;
                    var single = new float[count];
                    var wide = new double[count];
                    // Parameters (gamma, beta, bias) sit near one rather than near zero, so a
                    // dropped post-scale or post-bias moves the output instead of hiding in it.
                    double offset = source.Name == "x" ? _xOffset : 1.0;
                    for (long e = 0; e < count; e++)
                    {
                        float v = (float)((((e * 37 + salt * 101) % 97) - 48) / 64.0 + offset);
                        single[e] = v;
                        wide[e] = v;
                    }
                    salt++;
                    var buffer = runtime.AllocateBytes((nuint)(count * sizeof(float)));
                    buffer.Upload<float>(single);
                    it._named.Add(source.Name, buffer);
                    it.HostSources.Add(source.Name, wide);
                }

                // THE PRODUCTION LOWERING (PtxProgramLowering): what is verified and timed here is what a
                // backend launches -- row fusion, the planner's splits, scratch by name -- with the
                // tuner's recorded winner for each pass that is not fused.
                PtxPassLowering? Tuned(CodegenKernelSpec pass)
                {
                    var tuned = ResolveTuned(runtime, pass, catalogName + "/" + pass.Name);
                    // NO RECORDED WINNER means the modelled choice, and for a program that is what
                    // the backend does: it splits whenever the planner says one kernel would leave
                    // the device idle. A BatchNorm's per-channel statistics are 64 threads unsplit.
                    if (tuned.Winner is null && !tuned.IsSplit && CodegenSplitReduction.TryPlan(pass) is not null)
                        return null;
                    var kernels = new List<PtxPassKernel>();
                    foreach (var k in tuned.Kernels)
                        kernels.Add(new PtxPassKernel(k.Ptx, k.Spec.Name, k.Blocks, k.BlockX, k.BlockY,
                            k.DynamicSharedMemoryBytes));
                    return new PtxPassLowering(kernels, tuned.Split, tuned.Label());
                }

                var lowering = PtxProgramLowering.Create(program, catalogName, runtime.ComputeCapabilityMajor,
                    runtime.ComputeCapabilityMinor, Tuned, rowFusion: !_noRowFusion);
                it.Lowerings.AddRange(lowering.PassLowerings);
                foreach (var pair in lowering.Buffers)
                {
                    if (it._named.ContainsKey(pair.Key)) continue;
                    var buffer = runtime.AllocateBytes((nuint)(pair.Value * sizeof(float)));
                    if (pair.Key.StartsWith("#", StringComparison.Ordinal)) it._temporaries.Add(buffer);
                    else it._named.Add(pair.Key, buffer);
                    it._scratch.Add(pair.Key, buffer);
                }
                foreach (var step in lowering.Steps)
                {
                    var args = new IntPtr[step.Arguments.Count];
                    for (int i = 0; i < args.Length; i++)
                        args[i] = (it._named.TryGetValue(step.Arguments[i], out var named) ? named : it._scratch[step.Arguments[i]]).Pointer;
                    var module = runtime.LoadModule(step.Ptx, allowExperimentalJitFallback: true);
                    it._modules.Add(module);
                    IntPtr fn = module.GetFunction(step.EntryPoint, out DirectPtxFunctionInfo info);
                    if (step.DynamicSharedMemoryBytes != 0)
                        module.SetMaxDynamicSharedMemory(fn, checked((int)step.DynamicSharedMemoryBytes));
                    it.RegistersPerThread = Math.Max(it.RegistersPerThread, info.RegistersPerThread);
                    it._steps.Add((module, fn, args, step.Blocks, step.BlockX, step.BlockY, step.DynamicSharedMemoryBytes));
                }

                it._graph = runtime.CaptureGraph(it.LaunchDirect);
                return it;
            }
            catch
            {
                it.Dispose();
                throw;
            }
        }

        private void LaunchDirect()
        {
            foreach (var (module, fn, args, blocks, x, y, dynamicSharedMemoryBytes) in _steps)
                LaunchSpec(module, fn, args, blocks, x, y, dynamicSharedMemoryBytes);
        }

        internal void Launch()
        {
            if (_graph is not null) _graph.Launch();
            else LaunchDirect();
        }

        internal float[] Download(string name, long count)
        {
            var values = new float[count];
            _named[name].Download<float>(values);
            return values;
        }

        public void Dispose()
        {
            _graph?.Dispose();
            foreach (var b in _named.Values) b.Dispose();
            foreach (var b in _temporaries) b.Dispose();
            foreach (var m in _modules) m.Dispose();
        }
    }

    /// <summary>
    /// Verifies each program at its bench shape -- the shape it is released at, for the same
    /// reason the single kernels are -- against the fp64 chain. EVERY intermediate is
    /// compared, not just the final output: a wrong statistic that the affine pass happens to
    /// mask on this data would otherwise ship.
    /// </summary>
    private static void VerifyPrograms(IReadOnlyList<CodegenCatalogProgramEntry> entries)
    {
        using var runtime = OpenRuntime();
        if (runtime is null) return;

        int passed = 0, failed = 0;
        Console.WriteLine();
        Console.WriteLine("CONVEYOR STAGE 1 (programs) - every pass output against the fp64 chain");
        Console.WriteLine("tolerance " + Tolerance.ToString("E0", CultureInfo.InvariantCulture));
        Console.WriteLine();
        Console.WriteLine("program / output                    regs  kernels  lowering             max abs dev   result");
        // Every program runs TWICE: on data centred on zero, and on the same data offset by
        // 1000. A statistic that cancels catastrophically passes the first and fails the second
        // -- BatchNorm's running-mean shift did exactly that (NaN at an offset of 1000) while
        // the zero-centred check passed. An explicit --x-offset runs that offset alone.
        double[] offsets = _xOffset != 0.0 ? new[] { _xOffset } : new[] { 0.0, 1000.0 };
        double savedOffset = _xOffset;
        foreach (var entry in entries)
        foreach (double offset in offsets)
        {
            _xOffset = offset;
            try
            {
                var program = entry.Bench;
                using var launchable = ProgramLaunchable.Create(runtime, program, entry.Name);
                launchable.Launch();
                runtime.Synchronize();
                var expected = program.Interpret(launchable.HostSources);

                bool all = true;
                for (int p = 0; p < program.Passes.Count; p++)
                {
                    var pass = program.Passes[p];
                    long count = pass.Output.ElementCount;
                    float[] actual = launchable.Download(pass.Output.Name, count);
                    double[] want = expected[pass.Output.Name];
                    double worst = 0;
                    for (long e = 0; e < count; e++)
                    {
                        double diff = Math.Abs(actual[e] - want[e]);
                        worst = Math.Max(worst, diff / Math.Max(1.0, Math.Abs(want[e])));
                        if (double.IsNaN(actual[e])) worst = double.NaN;
                        if (double.IsNaN(worst)) break;
                    }
                    bool ok = worst <= Tolerance && !double.IsNaN(worst);
                    all &= ok;
                    string label = entry.Name + " / " + pass.Output.Name +
                        (offset == 0.0 ? "" : " @+" + offset.ToString("0", CultureInfo.InvariantCulture));
                    Console.WriteLine(label.PadRight(36) +
                        launchable.RegistersPerThread.ToString(CultureInfo.InvariantCulture).PadLeft(4) +
                        launchable.KernelCount.ToString(CultureInfo.InvariantCulture).PadLeft(9) + "  " +
                        launchable.Lowerings[p].PadRight(18) +
                        worst.ToString("E3", CultureInfo.InvariantCulture).PadLeft(14) +
                        (ok ? "PASS" : "FAIL").PadLeft(9));
                }
                if (all) passed++; else failed++;
            }
            catch (Exception ex)
            {
                failed++;
                Console.WriteLine(entry.Name.PadRight(36) + "   -        -  -                  -     ERROR");
                Console.WriteLine("    " + ex.GetType().Name + ": " + ex.Message.Split('\n')[0]);
            }
        }

        _xOffset = savedOffset;
        Console.WriteLine();
        Console.WriteLine("verify (programs): " + passed.ToString(CultureInfo.InvariantCulture) + " passed, " +
                          failed.ToString(CultureInfo.InvariantCulture) + " failed");
        if (failed != 0 || passed != entries.Count * offsets.Length)
            throw new InvalidOperationException(
                "Program verification failed closed: " +
                passed.ToString(CultureInfo.InvariantCulture) + " of " +
                (entries.Count * offsets.Length).ToString(CultureInfo.InvariantCulture) +
                " selected program runs passed.");
    }

    /// <summary>Times each whole program as one graph replay, under the stage-3 protocol.</summary>
    private static void BenchPrograms(IReadOnlyList<CodegenCatalogProgramEntry> entries)
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("kernel-bench-programs-start");
        using var runtime = OpenRuntime();
        if (runtime is null) return;
        var failures = new List<string>();

        Console.WriteLine();
        Console.WriteLine("CONVEYOR STAGE 3 (programs) - whole program per graph replay");
        Console.WriteLine();
        Console.WriteLine("program                           kernels    us/launch    p95/med   run spread   SM clock");
        foreach (var entry in entries)
        {
            try
            {
                using var launchable = ProgramLaunchable.Create(runtime, entry.Bench, entry.Name);
                var medians = new List<double>(Runs);
                var tails = new List<double>(Runs);
                int clockBefore = 0, clockAfter = 0;
                bool converged = false;
                for (int attempt = 0; attempt < StabilityAttempts; attempt++)
                {
                    clockBefore = GpuBenchmarkEnvironment.SampleSmClockMhz();
                    Dist d = Measure(runtime.Synchronize, launchable.Launch);
                    clockAfter = GpuBenchmarkEnvironment.SampleSmClockMhz();
                    medians.Add(d.Median);
                    tails.Add(d.Median > 0 ? d.P95 / d.Median : double.NaN);
                    if (medians.Count > Runs)
                    {
                        medians.RemoveAt(0);
                        tails.RemoveAt(0);
                    }
                    double spreadNow = medians.Min() > 0 ? medians.Max() / medians.Min() - 1.0 : double.MaxValue;
                    double drift = clockBefore > 0
                        ? Math.Abs(clockAfter - clockBefore) / (double)clockBefore
                        : double.MaxValue;
                    if (medians.Count == Runs && spreadNow <= StableTimer.StableSpread && drift <= 0.02)
                    {
                        converged = true;
                        break;
                    }
                }
                double reported = medians.OrderBy(value => value).ElementAt(medians.Count / 2);
                if (!converged)
                    failures.Add(entry.Name + ": timing did not converge within " +
                                 StabilityAttempts.ToString(CultureInfo.InvariantCulture) + " attempts");
                Console.WriteLine(entry.Name.PadRight(34) +
                    launchable.KernelCount.ToString(CultureInfo.InvariantCulture).PadLeft(7) +
                    (converged ? (reported * 1000.0).ToString("F1", CultureInfo.InvariantCulture) : "UNSTABLE").PadLeft(13) +
                    tails.Max().ToString("F2", CultureInfo.InvariantCulture).PadLeft(11) +
                    ((medians.Max() / medians.Min() - 1.0) * 100).ToString("F1", CultureInfo.InvariantCulture).PadLeft(10) + "%   " +
                    GpuBenchmarkEnvironment.DescribeClockDrift(clockBefore, clockAfter));
            }
            catch (Exception ex)
            {
                failures.Add(entry.Name + ": " + ex.Message.Split('\n')[0]);
                Console.WriteLine(entry.Name.PadRight(34) + "  ERROR  " + ex.Message.Split('\n')[0]);
            }
        }

        GpuBenchmarkEnvironment.RequireNoForeignCompute("kernel-bench-programs-end", afterSuite: true);
        if (failures.Count != 0)
            throw new InvalidOperationException(
                failures.Count.ToString(CultureInfo.InvariantCulture) +
                " selected program(s) produced no stable benchmark row. " + string.Join("; ", failures));
    }

    /// <summary>
    /// The phase-4 gate: each declared program against the HAND-WRITTEN backend kernel it
    /// replaces, in one process, at the catalog's bench shape. Rounds alternate the two
    /// sides so drift and preemption land on both, and the ratio is taken per round; the
    /// median of the rounds is reported (incumbent / ours, above one means the declared
    /// program is faster). The incumbent is launched eagerly, as the engine launches it;
    /// with 50 launches per timed region its launch cost is amortised, not removed.
    /// </summary>
    private static void Incumbent(IReadOnlyList<CodegenCatalogProgramEntry> entries)
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("kernel-incumbent-start");
        using var runtime = OpenRuntime();
        if (runtime is null) return;
        using var engine = new AiDotNet.Tensors.Engines.DirectGpu.DirectGpuEngine();
        var backend = engine.Backend ??
            throw new InvalidOperationException("No direct GPU backend is available for the incumbent lane.");        // The backend now routes these calls to the declared programs themselves; the incumbent is the
        // hand-written kernel, so the route is switched off for this lane.
        if (backend is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cuda) cuda.DeclaredKernelsSwitch = false;

        Console.WriteLine();
        Console.WriteLine("INCUMBENT - declared program vs the hand-written " + engine.BackendName + " kernel, paired rounds");
        Console.WriteLine();
        Console.WriteLine("program                       ours us   incumbent us   ratio (round spread)");

        // A gate that prints ERROR and exits 0 is not a gate: a program that fails to lower,
        // compile or launch has to fail the lane, as VerifyPrograms and BenchPrograms do.
        var failures = new List<string>();
        foreach (var entry in entries)
        {
            Action? incumbent = null;
            var buffers = new List<IDisposable>();
            try
            {
                var program = entry.Bench;
                var shapes = new Dictionary<string, IReadOnlyList<int>>(StringComparer.Ordinal);
                foreach (var source in program.Sources()) shapes[source.Name] = source.Shape;
                AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer Buffer(long count)
                {
                    var data = new float[count];
                    for (long i = 0; i < count; i++) data[i] = (float)((((i * 37) % 97) - 48) / 64.0 + 1.0);
                    var b = backend.AllocateBuffer(data);
                    buffers.Add(b);
                    return b;
                }

                switch (entry.Name)
                {
                    case "softmax_rows":
                    {
                        int r = shapes["x"][0], c = shapes["x"][1];
                        var a = Buffer((long)r * c); var o = Buffer((long)r * c);
                        incumbent = () => backend.Softmax(a, o, r, c);
                        break;
                    }
                    case "layernorm":
                    case "layernorm_d64":
                    {
                        int r = shapes["x"][0], c = shapes["x"][1];
                        var x = Buffer((long)r * c); var y = Buffer((long)r * c);
                        var g = Buffer(c); var b = Buffer(c); var m = Buffer(r); var v = Buffer(r);
                        incumbent = () => backend.LayerNorm(x, y, g, b, m, v, r, c, 1e-5f);
                        break;
                    }
                    case "rmsnorm":
                    {
                        int r = shapes["x"][0], c = shapes["x"][1];
                        var x = Buffer((long)r * c); var y = Buffer((long)r * c);
                        var g = Buffer(c); var s = Buffer(r);
                        incumbent = () => backend.RmsNorm(x, y, g, s, r, c, 1e-6f);
                        break;
                    }
                    case "batchnorm2d":
                    {
                        // Statistics bind x as [n, c, h*w].
                        int n = shapes["x"][0], c = shapes["x"][1], s = shapes["x"][2];
                        var x = Buffer((long)n * c * s); var y = Buffer((long)n * c * s);
                        var g = Buffer(c); var b = Buffer(c); var rm = Buffer(c); var rv = Buffer(c);
                        var sm = Buffer(c); var sv = Buffer(c);
                        incumbent = () => backend.BatchNorm(x, y, g, b, rm, rv, sm, sv, n, c, s, 1e-5f, 0.1f, true);
                        break;
                    }
                    case "groupnorm":
                    {
                        var xs = shapes["x"];   // [n, groups, channels/groups, spatial]
                        int n = xs[0], groups = xs[1], c = xs[1] * xs[2], s = xs[3];
                        var x = Buffer((long)n * c * s); var y = Buffer((long)n * c * s);
                        var g = Buffer(c); var b = Buffer(c); var m = Buffer((long)n * groups); var v = Buffer((long)n * groups);
                        incumbent = () => backend.GroupNorm(x, y, g, b, m, v, n, groups, c, s, 1e-5f);
                        break;
                    }
                    case "layernorm_backward":
                    {
                        int r = shapes["x"][0], c = shapes["x"][1];
                        var dy = Buffer((long)r * c); var x = Buffer((long)r * c); var dx = Buffer((long)r * c);
                        var g = Buffer(c); var m = Buffer(r); var v = Buffer(r); var dg = Buffer(c); var db = Buffer(c);
                        incumbent = () => backend.LayerNormBackward(dy, x, g, m, v, dx, dg, db, r, c, 1e-5f);
                        break;
                    }
                    case "rmsnorm_backward":
                    {
                        int r = shapes["x"][0], c = shapes["x"][1];
                        var dy = Buffer((long)r * c); var x = Buffer((long)r * c); var dx = Buffer((long)r * c);
                        var g = Buffer(c); var s = Buffer(r); var dg = Buffer(c);
                        incumbent = () => backend.RmsNormBackward(dy, x, g, s, dx, dg, r, c, 1e-6f);
                        break;
                    }
                    case "batchnorm2d_backward":
                    {
                        int n = shapes["dy"][0], c = shapes["dy"][1], s = shapes["dy"][2];
                        var dy = Buffer((long)n * c * s); var x = Buffer((long)n * c * s); var dx = Buffer((long)n * c * s);
                        var g = Buffer(c); var m = Buffer(c); var v = Buffer(c); var dg = Buffer(c); var db = Buffer(c);
                        incumbent = () => backend.BatchNormBackward(dy, x, g, m, v, dx, dg, db, n, c, s, 1e-5f);
                        break;
                    }
                    case "softmax_backward":
                    {
                        int r = shapes["y"][0], c = shapes["y"][1];
                        var dy = Buffer((long)r * c); var y = Buffer((long)r * c); var dx = Buffer((long)r * c);
                        incumbent = () => backend.SoftmaxBackward(dy, y, dx, r, c);
                        break;
                    }
                    default:
                        Console.WriteLine(entry.Name.PadRight(28) + "  no hand-written incumbent declared");
                        continue;
                }

                using var ours = ProgramLaunchable.Create(runtime, program, entry.Name);
                var oursUs = new List<double>();
                var theirsUs = new List<double>();
                var ratios = new List<double>();
                for (int round = 0; round < Runs; round++)
                {
                    double a = Measure(runtime.Synchronize, ours.Launch).Median * 1000.0;
                    double b = Measure(backend.Synchronize, incumbent).Median * 1000.0;
                    oursUs.Add(a);
                    theirsUs.Add(b);
                    ratios.Add(b / a);
                }
                ratios.Sort();
                double median = ratios[ratios.Count / 2];
                double spread = ratios[0] > 0 ? (ratios[ratios.Count - 1] / ratios[0] - 1) * 100 : double.NaN;
                Console.WriteLine(entry.Name.PadRight(28) +
                    oursUs.OrderBy(v => v).ElementAt(oursUs.Count / 2).ToString("F1", CultureInfo.InvariantCulture).PadLeft(9) +
                    theirsUs.OrderBy(v => v).ElementAt(theirsUs.Count / 2).ToString("F1", CultureInfo.InvariantCulture).PadLeft(15) +
                    (median.ToString("F2", CultureInfo.InvariantCulture) + "x").PadLeft(9) +
                    (" (" + spread.ToString("F1", CultureInfo.InvariantCulture) + "%)"));
            }
            catch (Exception ex)
            {
                failures.Add(entry.Name + ": " + ex.Message.Split('\n')[0]);
                Console.WriteLine(entry.Name.PadRight(28) + "  ERROR " + ex.Message.Split('\n')[0]);
            }
            finally
            {
                foreach (var b in buffers) b.Dispose();
            }
        }
        GpuBenchmarkEnvironment.RequireNoForeignCompute("kernel-incumbent-end", afterSuite: true);
        if (failures.Count != 0)
            throw new InvalidOperationException(
                failures.Count.ToString(CultureInfo.InvariantCulture) +
                " incumbent comparison(s) failed. " + string.Join("; ", failures));
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
        var failures = new List<string>();

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            Console.WriteLine();
            Console.WriteLine("CONVEYOR STAGE 3 - bench with the Phase 0.5 calibrated protocol");
            Console.WriteLine("device-filling shapes, " + LaunchesPerSample +
                              " launches per timed region, paired within-sample ratio, " +
                              "up to " + StabilityAttempts + " attempts with a latest-" +
                              Runs + " stability window");
            Console.WriteLine("harness noise floor measured at 1.05%; differences under ~3% are not claimable");
            Console.WriteLine("protocol " + CodegenMeasurementProtocol.Stamp(
                runtime.DeviceName + ", clocks " + GpuBenchmarkEnvironment.SampleSmClockMhz().ToString(CultureInfo.InvariantCulture) + " MHz"));
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
                    // p11: both cross-process lanes retain the latest three runs.
                    // A WDDM preemption ages out instead of poisoning a max/min over
                    // all history, and both sides publish the MEDIAN of that window.
                    var medians = new List<double>(Runs);
                    var tails = new List<double>(Runs);
                    int clockBefore = 0, clockAfter = 0;
                    bool converged = false;
                    for (int attempt = 0; attempt < StabilityAttempts; attempt++)
                    {
                        clockBefore = GpuBenchmarkEnvironment.SampleSmClockMhz();
                        Dist d = Measure(runtime.Synchronize, Launch);
                        clockAfter = GpuBenchmarkEnvironment.SampleSmClockMhz();
                        medians.Add(d.Median);
                        tails.Add(d.Median > 0 ? d.P95 / d.Median : double.NaN);
                        if (medians.Count > Runs)
                        {
                            medians.RemoveAt(0);
                            tails.RemoveAt(0);
                        }

                        double loNow = medians.Min(), hiNow = medians.Max();
                        double spreadNow = loNow > 0 ? hiNow / loNow - 1.0 : double.MaxValue;
                        double drift = clockBefore > 0
                            ? Math.Abs(clockAfter - clockBefore) / (double)clockBefore
                            : double.MaxValue;
                        if (medians.Count == Runs &&
                            spreadNow <= StableTimer.StableSpread && drift <= 0.02)
                        {
                            converged = true;
                            break;
                        }
                    }
                    double lo = medians.Min(), hi = medians.Max();
                    double reported = medians.OrderBy(value => value).ElementAt(medians.Count / 2);
                    double worstTail = tails.Max();
                    if (!converged)
                        failures.Add(entry.Name + ": timing did not converge within " +
                                     StabilityAttempts.ToString(CultureInfo.InvariantCulture) +
                                     " attempts");
                    Console.WriteLine(entry.Name.PadRight(36) +
                        blocks.ToString("N0", CultureInfo.InvariantCulture).PadLeft(8) +
                        (converged
                            ? (reported * 1000.0).ToString("F1", CultureInfo.InvariantCulture)
                            : "UNSTABLE").PadLeft(13) +
                        worstTail.ToString("F2", CultureInfo.InvariantCulture).PadLeft(11) +
                        ((hi / lo - 1.0) * 100).ToString("F1", CultureInfo.InvariantCulture).PadLeft(10) + "%   " +
                        GpuBenchmarkEnvironment.DescribeClockDrift(clockBefore, clockAfter));
                }
                catch (Exception ex)
                {
                    failures.Add(entry.Name + ": " + ex.Message.Split('\n')[0]);
                    Console.WriteLine(entry.Name.PadRight(36) + "  ERROR  " + ex.Message.Split('\n')[0]);
                }
            }
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }

        GpuBenchmarkEnvironment.RequireNoForeignCompute("kernel-bench-end", afterSuite: true);
        if (failures.Count != 0)
            throw new InvalidOperationException(
                failures.Count.ToString(CultureInfo.InvariantCulture) +
                " selected kernel(s) produced no stable benchmark row. " +
                string.Join("; ", failures));
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
        if (DirectPtxArchitecture.HasCodegenConveyor(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return runtime;
        Console.WriteLine("The conveyor requires an SM75 (Turing) or newer device.");
        runtime.Dispose();
        return null;
    }

    private static long Elements(IReadOnlyList<int> shape)
    {
        long total = 1;
        for (int i = 0; i < shape.Count; i++) total *= shape[i];
        return total;
    }

    private static unsafe void LaunchSpec(
        DirectPtxModule module, IntPtr fn, IntPtr[] pointers, uint blocks,
        uint blockX, uint blockY, uint dynamicSharedMemoryBytes = 0)
    {
        fixed (IntPtr* pinned = pointers)
        {
            void** argv = stackalloc void*[pointers.Length];
            for (int i = 0; i < pointers.Length; i++) argv[i] = pinned + i;
            module.Launch(
                fn, blocks, 1, 1, blockX, blockY, 1, dynamicSharedMemoryBytes, argv);
        }
    }
}
