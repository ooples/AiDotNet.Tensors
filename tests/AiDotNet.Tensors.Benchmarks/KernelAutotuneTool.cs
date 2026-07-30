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
        bool ParityTransposedConv2D = false,
        CodegenTiledContractionSchedule? TiledContractionSchedule = null,
        CodegenTiledConv2DSchedule? TiledConv2DSchedule = null);

    private sealed record TuneResult(
        string Name, double BestUs, double ModelledUs, double Gain);

    private sealed record CandidatePhase(string Name, Action Launch, long WorkUnits);

    /// <summary>A loaded candidate kept alive while it is paired against the baseline.</summary>
    private sealed class CandidateProgram : IDisposable
    {
        private readonly List<IDisposable> _resources;
        private readonly DirectPtxBuffer _output;
        private readonly int _outputElements;

        internal CandidateProgram(
            string name, Action launch, DirectPtxBuffer output, int outputElements,
            List<IDisposable> resources, IReadOnlyList<CandidatePhase>? phases = null,
            string? resourceSummary = null, bool promotable = true)
        {
            Name = name;
            Launch = launch;
            _output = output;
            _outputElements = outputElements;
            _resources = resources;
            Phases = phases ?? Array.Empty<CandidatePhase>();
            ResourceSummary = resourceSummary;
            Promotable = promotable;
        }

        internal string Name { get; }
        internal Action Launch { get; }
        internal IReadOnlyList<CandidatePhase> Phases { get; }
        internal string? ResourceSummary { get; }
        internal bool Promotable { get; }

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
        string? candidateSelector = ValueOf(args, "--candidate");
        ValidateCandidateSelector(candidateSelector);
        var entries = string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase)
            ? CodegenKernelCatalog.All
            : new[] { CodegenKernelCatalog.Find(selector)! }.Where(e => e != null).ToList();
        KernelToolArgs.RequireNonEmptySelection(selector, entries.Count, "kernel-autotune");

        string? profileCandidate = ValueOf(args, "--profile-candidate");
        if (!string.IsNullOrWhiteSpace(profileCandidate))
        {
            if (entries.Count != 1)
                throw new ArgumentException("Counter profiling requires one exact kernel selector.");
            ProfileOne(runtime, entries[0].Bench, profileCandidate);
            return;
        }

        string outputPath = ValueOf(args, "--out") ??
            Path.Combine(Directory.GetCurrentDirectory(), "artifacts", "autotune.tsv");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        Console.WriteLine();
        Console.WriteLine("AUTOTUNE - measured candidate lowerings, protocol " + CodegenMeasurementProtocol.Tag);
        Console.WriteLine("candidates: " +
            (string.IsNullOrWhiteSpace(candidateSelector)
                ? string.Join(", ", Candidates.Select(c => c.Name)) +
                    ", " + string.Join(", ", CodegenTiledContractionSchedule.SearchSpace
                        .Select(s => s.WinnerName)) +
                    ", " + string.Join(", ", CodegenTiledConv2DSchedule.SearchSpace
                        .Select(s => s.WinnerName)) +
                    ", " + string.Join(", ", CodegenTiledConv2DSplitSchedule.SearchSpace
                        .Select(s => s.WinnerName)) +
                    ", library-winograd-fp32-bn16, library-winograd-fp32-bn32, " +
                    "inline-outer-winograd-conv2d, " +
                    "library-bwd-input-direct, " +
                    "split, tiled-split, tiled-chunked-split"
                : candidateSelector));
        Console.WriteLine();
        Console.WriteLine("kernel                          modelled   best      winner        gain");

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        var rows = LoadUnselectedCurrentRows(outputPath, runtime, entries);
        if (File.Exists(outputPath)) File.Delete(outputPath);
        int improved = 0;
        var failures = new List<string>();
        try
        {
            foreach (var entry in entries)
            {
                try
                {
                    TuneResult? result = TuneOne(runtime, entry, candidateSelector);
                    if (result is null || !double.IsFinite(result.BestUs) ||
                        result.BestUs == double.MaxValue ||
                        !double.IsFinite(result.ModelledUs) || result.ModelledUs == double.MaxValue)
                        throw new InvalidOperationException(
                            "no numerically valid, stable candidate completed");

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
                    rows[entry.Name] = string.Join("\t", entry.Name, result.Name,
                        result.BestUs.ToString("F3", CultureInfo.InvariantCulture),
                        result.ModelledUs.ToString("F3", CultureInfo.InvariantCulture),
                        gain.ToString("F4", CultureInfo.InvariantCulture),
                        CodegenMeasurementProtocol.Tag,
                        identity.DeviceFingerprint,
                        identity.Target,
                        identity.SpecFingerprint,
                        identity.EmitterFingerprint);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(entry.Name.PadRight(30) + "  ERROR " + ex.Message.Split('\n')[0]);
                    failures.Add(entry.Name + ": " + ex.Message.Split('\n')[0]);
                }
            }
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }

        GpuBenchmarkEnvironment.RequireNoForeignCompute("autotune-end", afterSuite: true);
        if (failures.Count != 0)
            throw new InvalidOperationException(
                failures.Count.ToString(CultureInfo.InvariantCulture) +
                " selected kernel(s) failed autotuning; no winner artifact written. " +
                string.Join("; ", failures));

        var text = new StringBuilder();
        text.AppendLine("# autotune winners, " + CodegenMeasurementProtocol.Tag + ": " +
                        CodegenMeasurementProtocol.Description);
        text.AppendLine(
            "kernel\twinner\tbest_us\tmodelled_us\tgain\tprotocol\tdevice\ttarget\tspec\temitter");
        foreach (CodegenCatalogEntry entry in CodegenKernelCatalog.All)
            if (rows.TryGetValue(entry.Name, out string? row)) text.AppendLine(row);
        string temporaryOutput = outputPath + ".tmp-" +
            Environment.ProcessId.ToString(CultureInfo.InvariantCulture);
        File.WriteAllText(temporaryOutput, text.ToString());
        File.Move(temporaryOutput, outputPath, overwrite: true);

        Console.WriteLine();
        double noiseFloorPercent =
            (CodegenMeasurementProtocol.AutotuneGainNoiseFloor - 1.0) * 100.0;
        Console.WriteLine(improved + " kernels improved past the " +
                          noiseFloorPercent.ToString("0.##", CultureInfo.InvariantCulture) +
                          "% noise floor");
        Console.WriteLine("winners written to " + outputPath);
        CodegenAutotuneCache.Invalidate();
    }

    private static Dictionary<string, string> LoadUnselectedCurrentRows(
        string outputPath, DirectPtxRuntime runtime,
        IReadOnlyList<CodegenCatalogEntry> selectedEntries)
    {
        var rows = new Dictionary<string, string>(StringComparer.Ordinal);
        if (selectedEntries.Count == CodegenKernelCatalog.All.Count ||
            !File.Exists(outputPath))
            return rows;

        var selected = new HashSet<string>(
            selectedEntries.Select(entry => entry.Name), StringComparer.Ordinal);
        string[] lines = File.ReadAllLines(outputPath);
        if (lines.Length < 2 || !lines[1].StartsWith("kernel\twinner\t", StringComparison.Ordinal))
            return rows;

        for (int i = 2; i < lines.Length; i++)
        {
            string[] cells = lines[i].Split('\t');
            if (cells.Length != 10 || selected.Contains(cells[0]) ||
                !string.Equals(cells[5], CodegenMeasurementProtocol.Tag,
                    StringComparison.Ordinal))
                continue;

            CodegenCatalogEntry? entry = CodegenKernelCatalog.Find(cells[0]);
            if (entry is null) continue;
            CodegenAutotuneIdentity identity = CodegenAutotuneIdentity.Create(
                entry.Bench, runtime.DeviceFingerprint,
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
            if (!string.Equals(cells[6], identity.DeviceFingerprint, StringComparison.Ordinal) ||
                !string.Equals(cells[7], identity.Target, StringComparison.Ordinal) ||
                !string.Equals(cells[8], identity.SpecFingerprint, StringComparison.Ordinal) ||
                !string.Equals(cells[9], identity.EmitterFingerprint, StringComparison.Ordinal))
                continue;
            rows[entry.Name] = lines[i];
        }

        Console.WriteLine("preserving " + rows.Count.ToString(CultureInfo.InvariantCulture) +
                          " identity-current unselected autotune rows");
        return rows;
    }

    private static void ProfileOne(
        DirectPtxRuntime runtime, CodegenKernelSpec spec, string candidateName)
    {
        if (string.Equals(candidateName, "inline-outer-winograd-conv2d",
                StringComparison.Ordinal))
        {
            using CandidateProgram? winograd =
                TryCreateInlineOuterWinograd(runtime, spec);
            if (winograd is null)
                throw new ArgumentException(
                    "The inline outer-product Winograd candidate does not support this spec.");
            winograd.Launch();
            runtime.Synchronize();
            Console.WriteLine("profiled " + winograd.Name +
                              (winograd.ResourceSummary is null
                                  ? string.Empty
                                  : ": " + winograd.ResourceSummary));
            return;
        }

        Candidate? candidate = Candidates.FirstOrDefault(c =>
            string.Equals(c.Name, candidateName, StringComparison.Ordinal));
        if (candidate is null)
        {
            CodegenTiledContractionSchedule? contractionSchedule =
                CodegenTiledContractionSchedule.Find(candidateName);
            if (contractionSchedule is not null)
                candidate = new Candidate(
                    contractionSchedule.WinnerName, null,
                    TiledContractionSchedule: contractionSchedule);
        }
        if (candidate is null)
        {
            CodegenTiledConv2DSchedule? schedule =
                CodegenTiledConv2DSchedule.Find(candidateName);
            if (schedule is not null)
                candidate = new Candidate(
                    schedule.WinnerName, null, TiledConv2DSchedule: schedule);
        }
        if (candidate is null)
            throw new ArgumentException("Unknown profile candidate '" + candidateName + "'.");

        using CandidateProgram program = CreateSingle(runtime, spec, candidate);
        program.Launch();
        runtime.Synchronize();
        Console.WriteLine("profiled " + program.Name +
                          (program.ResourceSummary is null
                              ? string.Empty
                              : ": " + program.ResourceSummary));
    }

    private static TuneResult? TuneOne(
        DirectPtxRuntime runtime, CodegenCatalogEntry entry, string? candidateSelector)
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
            if (!CandidateEnabled(candidateSelector, Candidates[i].Name)) continue;
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

        foreach (CodegenTiledContractionSchedule schedule in
                 CodegenTiledContractionSchedule.SearchSpace)
        {
            if (!CandidateEnabled(candidateSelector, schedule.WinnerName)) continue;
            CandidateProgram? program = null;
            try
            {
                var candidate = new Candidate(
                    schedule.WinnerName, null, TiledContractionSchedule: schedule);
                program = CreateSingle(runtime, spec, candidate);
            }
            catch (NotSupportedException) { continue; }
            using (program)
            {
                program.Launch();
                runtime.Synchronize();
                if (!Agrees(program.ReadOutput(), reference, out double deviation))
                {
                    Console.WriteLine("    candidate '" + program.Name +
                                      "' disagrees by " +
                                      deviation.ToString("E3", CultureInfo.InvariantCulture) +
                                      " relative; rejected");
                    continue;
                }
                Consider(runtime, modelled, program, workUnits,
                    ref hasStableTiming, ref bestName, ref bestUs,
                    ref bestModelledUs, ref bestGain);
            }
        }

        foreach (CodegenTiledConv2DSchedule schedule in
                 CodegenTiledConv2DSchedule.SearchSpace)
        {
            if (!CandidateEnabled(candidateSelector, schedule.WinnerName)) continue;
            CandidateProgram? program = null;
            try
            {
                var candidate = new Candidate(
                    schedule.WinnerName, null, TiledConv2DSchedule: schedule);
                program = CreateSingle(runtime, spec, candidate);
            }
            catch (NotSupportedException) { continue; }
            using (program)
            {
                program.Launch();
                runtime.Synchronize();
                if (!Agrees(program.ReadOutput(), reference, out double deviation))
                {
                    Console.WriteLine("    candidate '" + program.Name +
                                      "' disagrees by " +
                                      deviation.ToString("E3", CultureInfo.InvariantCulture) +
                                      " relative; rejected");
                    continue;
                }
                Consider(runtime, modelled, program, workUnits,
                    ref hasStableTiming, ref bestName, ref bestUs,
                    ref bestModelledUs, ref bestGain);
            }
        }

        // A cooperative tile's block count can be far below the scalar iteration
        // space's apparent parallelism. Compose exact tiles with deterministic channel
        // chunks so that schedule-hidden underfill is measured and replayable too.
        foreach (CodegenTiledConv2DSplitSchedule schedule in
                 CodegenTiledConv2DSplitSchedule.SearchSpace)
        {
            if (!CandidateEnabled(candidateSelector, schedule.WinnerName)) continue;
            using CandidateProgram? split = TryCreateTiledConv2DSplit(
                runtime, spec, schedule);
            if (split is null) continue;

            split.Launch();
            runtime.Synchronize();
            if (!Agrees(split.ReadOutput(), reference, out double deviation))
            {
                Console.WriteLine("    candidate '" + split.Name + "' disagrees by " +
                                  deviation.ToString("E3", CultureInfo.InvariantCulture) +
                                  " relative; rejected");
                continue;
            }
            Consider(runtime, modelled, split, workUnits,
                ref hasStableTiming, ref bestName, ref bestUs,
                ref bestModelledUs, ref bestGain);
        }

        // Point the oracle at the independently developed true-FP32 hand-written
        // Winograd pipeline. All four launches stay in the timed region because the
        // general kernel contract permits weights and inputs to change on every call.
        int[] winogradBlockColumns = { 16, 32 };
        foreach (int blockColumns in winogradBlockColumns)
        {
            using CandidateProgram? library = TryCreateLibraryWinograd(
                runtime, spec, blockColumns);
            if (library is null || !CandidateEnabled(candidateSelector, library.Name)) continue;

            library.Launch();
            runtime.Synchronize();
            if (!Agrees(library.ReadOutput(), reference, out double deviation))
            {
                Console.WriteLine("    candidate '" + library.Name + "' disagrees by " +
                                  deviation.ToString("E3", CultureInfo.InvariantCulture) +
                                  " relative; rejected");
                continue;
            }

            Consider(runtime, modelled, library, workUnits,
                ref hasStableTiming, ref bestName, ref bestUs,
                ref bestModelledUs, ref bestGain);
        }

        using (CandidateProgram? winograd = TryCreateInlineOuterWinograd(
                   runtime, spec))
        {
            if (winograd is not null && CandidateEnabled(candidateSelector, winograd.Name))
            {
                winograd.Launch();
                runtime.Synchronize();
                if (!Agrees(winograd.ReadOutput(), reference, out double deviation,
                        out long worstIndex, out float actual, out float expected))
                {
                    Console.WriteLine("    candidate '" + winograd.Name +
                                      "' disagrees by " +
                                      deviation.ToString("E3", CultureInfo.InvariantCulture) +
                                      " relative at output[" +
                                      worstIndex.ToString(CultureInfo.InvariantCulture) +
                                      "]: actual " + actual.ToString("G9", CultureInfo.InvariantCulture) +
                                      ", expected " + expected.ToString("G9", CultureInfo.InvariantCulture) +
                                      ", nearest reference output[" +
                                      ClosestIndex(reference, actual).ToString(
                                          CultureInfo.InvariantCulture) + "]" +
                                      "; rejected");
                }
                else
                {
                    Consider(runtime, modelled, winograd, workUnits,
                        ref hasStableTiming, ref bestName, ref bestUs,
                        ref bestModelledUs, ref bestGain);
                }
            }
        }

        // Probe the same true-FP32 Winograd dataflow as a linear adjoint. This is
        // diagnostic until the oracle proves that it beats the generated lowering;
        // a library-only result is never eligible for conveyor promotion.
        using (CandidateProgram? library = TryCreateLibraryInlineWinogradBackward(
                   runtime, spec))
        {
            if (library is not null && CandidateEnabled(candidateSelector, library.Name))
            {
                library.Launch();
                runtime.Synchronize();
                if (!Agrees(library.ReadOutput(), reference, out double deviation))
                {
                    Console.WriteLine("    candidate '" + library.Name +
                                      "' disagrees by " +
                                      deviation.ToString("E3", CultureInfo.InvariantCulture) +
                                      " relative; rejected");
                }
                else
                {
                    Consider(runtime, modelled, library, workUnits,
                        ref hasStableTiming, ref bestName, ref bestUs,
                        ref bestModelledUs, ref bestGain);
                }
            }
        }

        // Measure the independently developed deterministic backward-input kernel too.
        // It has no conveyor representation, so it is evidence about the algorithmic
        // dataflow only and can never be recorded as a dispatch winner here.
        using (CandidateProgram? library = TryCreateLibraryBackwardInput(runtime, spec))
        {
            if (library is not null && CandidateEnabled(candidateSelector, library.Name))
            {
                library.Launch();
                runtime.Synchronize();
                if (!Agrees(library.ReadOutput(), reference, out double deviation))
                {
                    Console.WriteLine("    candidate '" + library.Name +
                                      "' disagrees by " +
                                      deviation.ToString("E3", CultureInfo.InvariantCulture) +
                                      " relative; rejected");
                }
                else
                {
                    Consider(runtime, modelled, library, workUnits,
                        ref hasStableTiming, ref bestName, ref bestUs,
                        ref bestModelledUs, ref bestGain);
                }
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
            if (!CandidateEnabled(candidateSelector, split.Name)) continue;

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

        foreach (int chunkFactor in CodegenAutotuneIdentity.ChunkedSplitFactors)
        {
            using CandidateProgram? split = TryCreateSplit(
                runtime, spec, tiledPartial: true, chunkFactor: chunkFactor);
            if (split is null || !CandidateEnabled(candidateSelector, split.Name)) continue;

            split.Launch();
            runtime.Synchronize();
            if (!Agrees(split.ReadOutput(), reference, out double deviation))
            {
                Console.WriteLine("    candidate '" + split.Name + "' disagrees by " +
                                  deviation.ToString("E3", CultureInfo.InvariantCulture) +
                                  " relative; rejected");
                continue;
            }

            Consider(runtime, modelled, split, workUnits,
                ref hasStableTiming, ref bestName, ref bestUs,
                ref bestModelledUs, ref bestGain);
        }

        if (!hasStableTiming)
        {
            Console.WriteLine("    no standalone or paired timing window stabilized; " +
                              "no winner recorded");
            return null;
        }
        return new TuneResult(bestName, bestUs, bestModelledUs, bestGain);
    }

    private static bool CandidateEnabled(string? selector, string name) =>
        string.IsNullOrWhiteSpace(selector) ||
        string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(selector, name, StringComparison.OrdinalIgnoreCase);

    private static void ValidateCandidateSelector(string? selector)
    {
        if (string.IsNullOrWhiteSpace(selector) ||
            string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase) ||
            Candidates.Any(candidate => string.Equals(
                selector, candidate.Name, StringComparison.OrdinalIgnoreCase)) ||
            CodegenTiledContractionSchedule.SearchSpace.Any(schedule => string.Equals(
                selector, schedule.WinnerName, StringComparison.OrdinalIgnoreCase)) ||
            CodegenTiledConv2DSchedule.SearchSpace.Any(schedule => string.Equals(
                selector, schedule.WinnerName, StringComparison.OrdinalIgnoreCase)) ||
            CodegenTiledConv2DSplitSchedule.SearchSpace.Any(schedule => string.Equals(
                selector, schedule.WinnerName, StringComparison.OrdinalIgnoreCase)) ||
            string.Equals(selector, "library-winograd-fp32-bn16", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(selector, "library-winograd-fp32-bn32", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(selector, "inline-outer-winograd-conv2d", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(selector, "library-winograd-inline-adjoint-fp32", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(selector, "library-bwd-input-direct", StringComparison.OrdinalIgnoreCase) ||
            selector.StartsWith("split:", StringComparison.OrdinalIgnoreCase) ||
            selector.StartsWith("tiled-split:", StringComparison.OrdinalIgnoreCase) ||
            selector.StartsWith("tiled-chunked-split:", StringComparison.OrdinalIgnoreCase))
            return;

        throw new ArgumentException("Unknown autotune candidate '" + selector + "'.");
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
            ReportPhases(runtime, candidate);
            return;
        }

        Console.WriteLine("    candidate '" + candidate.Name + "': modelled " +
                          timing.A.Describe() + ", candidate " + timing.B.Describe() +
                          ", paired " + timing.DescribeRatio() + " +-" +
                          (timing.RelativeSpread * 100).ToString("0.0", CultureInfo.InvariantCulture) +
                          "%");
        if (candidate.ResourceSummary is not null)
            Console.WriteLine("      resources: " + candidate.ResourceSummary);
        ReportPhases(runtime, candidate);

        // A stable pair is also valid baseline evidence. This lets an interrupted standalone
        // window recover without accepting a candidate whose own time or ratio was unstable.
        if (!hasStableTiming)
        {
            hasStableTiming = true;
            bestUs = timing.A.Microseconds;
            bestModelledUs = timing.A.Microseconds;
        }

        // Hand-written multi-kernel diagnostics do not yet have a conveyor replay
        // representation. They may explain an algorithmic gap, but recording one as a
        // dispatch winner would make later stages silently run a different program.
        if (!candidate.Promotable)
        {
            Console.WriteLine("      diagnostic only: no exact conveyor replay");
            return;
        }

        // A winner must clear both the observed paired spread and the protocol noise floor.
        // Merely having the smallest median is not a promotion criterion.
        double required = Math.Max(
            CodegenMeasurementProtocol.AutotuneGainNoiseFloor,
            1.0 + timing.RelativeSpread);
        if (timing.Ratio <= required) return;
        if (bestGain > 1.0 && timing.Ratio / bestGain <= required) return;

        bestName = candidate.Name;
        bestUs = timing.B.Microseconds;
        bestModelledUs = timing.A.Microseconds;
        bestGain = timing.Ratio;
    }

    private static void ReportPhases(DirectPtxRuntime runtime, CandidateProgram candidate)
    {
        if (candidate.Phases.Count <= 1) return;

        var descriptions = new List<string>(candidate.Phases.Count);
        foreach (CandidatePhase phase in candidate.Phases)
        {
            StableTimer.Result timing = StableTimer.Measure(
                runtime, phase.Launch, phase.WorkUnits);
            descriptions.Add(phase.Name + " " + timing.Describe());
        }
        Console.WriteLine("      phases: " + string.Join(", ", descriptions));
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
            if (candidate.TiledContraction || candidate.TiledContractionSchedule is not null)
            {
                var tiled = candidate.TiledContractionSchedule is null
                    ? new PtxTiledContractionEmitter()
                    : new PtxTiledContractionEmitter(candidate.TiledContractionSchedule);
                ptx = tiled.Emit(
                    spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                blocks = tiled.LaunchBlocks;
                blockX = checked((uint)tiled.LaunchBlockThreads);
                blockY = 1;
            }
            else if (candidate.TiledConv2D || candidate.TiledConv2DSchedule is not null)
            {
                var tiled = candidate.TiledConv2DSchedule is null
                    ? new PtxTiledConv2DEmitter()
                    : new PtxTiledConv2DEmitter(candidate.TiledConv2DSchedule);
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
                if (candidate.Configure is null)
                    throw new InvalidOperationException(
                        "Affine candidate '" + candidate.Name + "' has no configurator.");
                candidate.Configure(emitter);
                ptx = emitter.Emit(
                    spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                blocks = emitter.LaunchBlocks;
                blockX = checked((uint)emitter.LaunchBlockX);
                blockY = checked((uint)emitter.LaunchBlockY);
            }
            var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
            resources.Add(module);
            IntPtr fn = module.GetFunction(spec.Name, out DirectPtxFunctionInfo functionInfo);
            int activeBlocks = module.GetActiveBlocksPerMultiprocessor(
                fn, checked((int)(blockX * blockY)));

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
                candidate.Name, Launch, output, checked((int)spec.Output.ElementCount), resources,
                resourceSummary: functionInfo.RegistersPerThread.ToString(
                        CultureInfo.InvariantCulture) + " regs/thread, " +
                    functionInfo.StaticSharedBytes.ToString(CultureInfo.InvariantCulture) +
                        " B shared, " +
                    functionInfo.LocalBytesPerThread.ToString(CultureInfo.InvariantCulture) +
                        " B local/thread, " +
                    activeBlocks.ToString(CultureInfo.InvariantCulture) + " blocks/SM");
        }
        catch
        {
            for (int i = resources.Count - 1; i >= 0; i--) resources[i].Dispose();
            throw;
        }
    }

    private static CandidateProgram? TryCreateTiledConv2DSplit(
        DirectPtxRuntime runtime, CodegenKernelSpec spec,
        CodegenTiledConv2DSplitSchedule schedule)
    {
        if (!CodegenTiledConv2DSplitPlan.TryCreate(
                spec, schedule, out CodegenTiledConv2DSplitPlan? possible, out _))
            return null;
        CodegenSplitPlan plan = possible!.Split;
        RequireFloat32(spec);

        var resources = new List<IDisposable>();
        try
        {
            var partialEmitter = new PtxTiledConv2DEmitter(schedule.Tile);
            string partialPtx = partialEmitter.Emit(
                plan.Partial, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
            var combineEmitter = new PtxAffineEmitter();
            string combinePtx = combineEmitter.Emit(
                plan.Combine, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
            var partialModule = runtime.LoadModule(partialPtx, allowExperimentalJitFallback: true);
            var combineModule = runtime.LoadModule(combinePtx, allowExperimentalJitFallback: true);
            resources.Add(partialModule);
            resources.Add(combineModule);
            IntPtr partialFn = partialModule.GetFunction(plan.Partial.Name, out var partialInfo);
            IntPtr combineFn = combineModule.GetFunction(plan.Combine.Name, out var combineInfo);

            var uploaded = new IntPtr[spec.Inputs.Count];
            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                CodegenTensorBinding binding = spec.Inputs[i];
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
            partialArgs[^1] = temporary.Pointer;

            var combineArgs = new IntPtr[plan.Combine.ParameterCount];
            combineArgs[0] = temporary.Pointer;
            if (plan.Combine.BiasInput is { } bias)
                combineArgs[bias] = uploaded[spec.BiasInput!.Value];
            if (plan.Combine.ScaleInput is { } scale)
                combineArgs[scale] = uploaded[spec.ScaleInput!.Value];
            combineArgs[^1] = output.Pointer;

            void LaunchPartial() => LaunchOne(
                partialModule, partialFn, partialArgs,
                partialEmitter.LaunchBlocks,
                checked((uint)partialEmitter.LaunchBlockThreads), 1);
            void LaunchCombine() => LaunchOne(
                combineModule, combineFn, combineArgs,
                combineEmitter.LaunchBlocks,
                checked((uint)combineEmitter.LaunchBlockX),
                checked((uint)combineEmitter.LaunchBlockY));
            void Launch()
            {
                LaunchPartial();
                LaunchCombine();
            }

            string summary = "partial " +
                partialInfo.RegistersPerThread.ToString(CultureInfo.InvariantCulture) +
                " regs/thread, " +
                partialInfo.StaticSharedBytes.ToString(CultureInfo.InvariantCulture) +
                " B shared; combine " +
                combineInfo.RegistersPerThread.ToString(CultureInfo.InvariantCulture) +
                " regs/thread";
            return new CandidateProgram(
                schedule.WinnerName, Launch, output,
                checked((int)spec.Output.ElementCount), resources,
                new[]
                {
                    new CandidatePhase("partial", LaunchPartial, WorkUnits(plan.Partial)),
                    new CandidatePhase("combine", LaunchCombine, WorkUnits(plan.Combine)),
                },
                resourceSummary: summary);
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

    private static CandidateProgram? TryCreateSplit(
        DirectPtxRuntime runtime, CodegenKernelSpec spec,
        bool tiledPartial = false, int chunkFactor = 0)
    {
        CodegenSplitPlan? plan;
        try
        {
            plan = chunkFactor > 0
                ? CodegenSplitReduction.TryPlanChunked(spec, chunkFactor)
                : CodegenSplitReduction.TryPlan(spec);
        }
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

            void LaunchPartial()
            {
                LaunchOne(partialModule, partialFn, partialArgs,
                    partialBlocks, partialBlockX, partialBlockY);
            }

            void LaunchCombine()
            {
                LaunchOne(combineModule, combineFn, combineArgs, combineEmitter.LaunchBlocks,
                    (uint)combineEmitter.LaunchBlockX, (uint)combineEmitter.LaunchBlockY);
            }

            void Launch()
            {
                LaunchPartial();
                LaunchCombine();
            }

            string name = chunkFactor > 0
                ? "tiled-chunked-split:" + string.Join("+", plan.PromotedAxes) +
                    "x" + chunkFactor.ToString(CultureInfo.InvariantCulture)
                : (tiledPartial ? "tiled-split:" : "split:") +
                    string.Join("+", plan.PromotedAxes);
            return new CandidateProgram(
                name, Launch, output, checked((int)spec.Output.ElementCount), resources,
                new[]
                {
                    new CandidatePhase("partial", LaunchPartial, WorkUnits(plan.Partial)),
                    new CandidatePhase("combine", LaunchCombine, WorkUnits(plan.Combine)),
                });
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

    private static CandidateProgram? TryCreateLibraryWinograd(
        DirectPtxRuntime runtime, CodegenKernelSpec spec, int blockColumns)
    {
        if (!CodegenTiledConv2DPlan.TryCreate(spec, out var possible, out _)) return null;
        CodegenTiledConv2DPlan plan = possible!;
        if (plan.TapSign != 1 || !plan.BiasInput.HasValue ||
            spec.Activation != CodegenActivationKind.ReLU ||
            plan.M % 64 != 0 ||
            plan.ReductionChannels % 8 != 0 ||
            plan.OutputHeight % 2 != 0 || plan.OutputWidth % 2 != 0)
            return null;

        int tiles = plan.Batch * (plan.OutputHeight / 2) * (plan.OutputWidth / 2);
        if (tiles % blockColumns != 0) return null;

        var resources = new List<IDisposable>();
        try
        {
            var filter = new PtxWinogradF23FilterTransformKernel(
                runtime, plan.M, plan.ReductionChannels, positionMajor: true);
            var inputTransform = new PtxWinogradF23InputTransformKernel(
                runtime, plan.Batch, plan.ReductionChannels,
                plan.InputHeight, plan.InputWidth);
            var gemm = new PtxWinogradBatchedGemmKernel(
                runtime, plan.M, plan.ReductionChannels, tiles,
                blockM: 64, blockN: blockColumns, blockK: 8,
                threadM: 4, threadN: 4);
            var outputTransform = new PtxWinogradF23OutputTransformKernel(
                runtime, plan.Batch, plan.OutputHeight, plan.OutputWidth, plan.M);
            resources.Add(filter);
            resources.Add(inputTransform);
            resources.Add(gemm);
            resources.Add(outputTransform);

            var uploaded = new DirectPtxBuffer[spec.Inputs.Count];
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
                uploaded[i] = buffer;
            }

            var transformedWeights = runtime.AllocateBytes((nuint)filter.TransformedBytes);
            var transformedInput = runtime.AllocateBytes((nuint)inputTransform.TransformedBytes);
            var transformedOutput = runtime.AllocateBytes((nuint)gemm.MBytes);
            var output = runtime.AllocateBytes(
                (nuint)(spec.Output.ElementCount * spec.Output.ElementBytes));
            resources.Add(transformedWeights);
            resources.Add(transformedInput);
            resources.Add(transformedOutput);
            resources.Add(output);

            DirectPtxBuffer weights = uploaded[plan.MatrixInput];
            DirectPtxBuffer input = uploaded[plan.StreamInput];
            DirectPtxBuffer bias = uploaded[plan.BiasInput.Value];

            void LaunchFilter() => filter.Launch(
                DirectPtxTensorView.CreateOwned(weights, filter.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(
                    transformedWeights, filter.Blueprint.Tensors[1]));
            void LaunchInput() => inputTransform.Launch(
                DirectPtxTensorView.CreateOwned(input, inputTransform.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(
                    transformedInput, inputTransform.Blueprint.Tensors[1]));
            void LaunchGemm() => gemm.Launch(
                DirectPtxTensorView.CreateOwned(
                    transformedWeights, gemm.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(
                    transformedInput, gemm.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(
                    transformedOutput, gemm.Blueprint.Tensors[2]));
            void LaunchOutput() => outputTransform.Launch(
                DirectPtxTensorView.CreateOwned(
                    transformedOutput, outputTransform.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(bias, outputTransform.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(output, outputTransform.Blueprint.Tensors[2]));
            void Launch()
            {
                LaunchFilter();
                LaunchInput();
                LaunchGemm();
                LaunchOutput();
            }

            return new CandidateProgram(
                "library-winograd-fp32-bn" +
                    blockColumns.ToString(CultureInfo.InvariantCulture),
                Launch, output, checked((int)spec.Output.ElementCount), resources,
                new[]
                {
                    new CandidatePhase("filter-transform", LaunchFilter,
                        Math.Max(1L, plan.M * (long)plan.ReductionChannels * 9)),
                    new CandidatePhase("input-transform", LaunchInput,
                        Math.Max(1L, inputTransform.Tiles * (long)plan.ReductionChannels * 16)),
                    new CandidatePhase("winograd-gemm", LaunchGemm,
                        Math.Max(1L, 16L * plan.M * plan.ReductionChannels * tiles)),
                    new CandidatePhase("output-transform", LaunchOutput,
                        Math.Max(1L, plan.M * (long)tiles * 16)),
                }, promotable: false);
        }
        catch (NotSupportedException)
        {
            for (int i = resources.Count - 1; i >= 0; i--) resources[i].Dispose();
            return null;
        }
        catch (ArgumentException)
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

    private static CandidateProgram? TryCreateInlineOuterWinograd(
        DirectPtxRuntime runtime, CodegenKernelSpec spec)
    {
        if (!CodegenTiledConv2DPlan.TryCreate(spec, out var possible, out _)) return null;
        CodegenTiledConv2DPlan plan = possible!;
        if (plan.TapSign != -1 || !plan.MatrixReductionMajor ||
            plan.BiasInput.HasValue || spec.Activation != CodegenActivationKind.None)
            return null;

        var resources = new List<IDisposable>();
        try
        {
            var mainEmitter = new PtxOuterProductWinogradConv2DEmitter();
            string mainPtx = mainEmitter.Emit(
                spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
            var mainModule = runtime.LoadModule(mainPtx, allowExperimentalJitFallback: true);
            resources.Add(mainModule);
            IntPtr mainFn = mainModule.GetFunction(mainEmitter.EntryPoint!, out var mainInfo);
            mainModule.SetMaxDynamicSharedMemory(mainFn, mainEmitter.SharedMemoryBytes);

            var uploaded = new DirectPtxBuffer[spec.Inputs.Count];
            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                CodegenTensorBinding binding = spec.Inputs[i];
                var buffer = runtime.AllocateBytes(
                    (nuint)(binding.ElementCount * binding.ElementBytes));
                resources.Add(buffer);
                var host = new float[binding.ElementCount];
                for (long e = 0; e < host.LongLength; e++)
                    host[e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                buffer.Upload<float>(host);
                uploaded[i] = buffer;
            }

            var output = runtime.AllocateBytes(
                (nuint)(spec.Output.ElementCount * spec.Output.ElementBytes));
            resources.Add(output);

            var mainArgs = new IntPtr[spec.ParameterCount];
            mainArgs[spec.Inputs[plan.MatrixInput].ParameterIndex] =
                uploaded[plan.MatrixInput].Pointer;
            mainArgs[spec.Inputs[plan.StreamInput].ParameterIndex] =
                uploaded[plan.StreamInput].Pointer;
            mainArgs[spec.Output.ParameterIndex] = output.Pointer;

            void LaunchMain() => LaunchOne(
                mainModule, mainFn, mainArgs, mainEmitter.LaunchBlocks,
                checked((uint)mainEmitter.LaunchBlockThreads), 1,
                checked((uint)mainEmitter.SharedMemoryBytes));

            int activeBlocks = mainModule.GetActiveBlocksPerMultiprocessor(
                mainFn, mainEmitter.LaunchBlockThreads,
                checked((nuint)mainEmitter.SharedMemoryBytes));
            return new CandidateProgram(
                "inline-outer-winograd-conv2d", LaunchMain, output,
                checked((int)spec.Output.ElementCount), resources,
                new[]
                {
                    new CandidatePhase("inline-winograd", LaunchMain,
                        Math.Max(1L, 16L * plan.M * plan.ReductionChannels *
                            plan.Batch * (plan.OutputHeight / 2) * (plan.OutputWidth / 2))),
                },
                resourceSummary: "main " + mainInfo.RegistersPerThread.ToString(CultureInfo.InvariantCulture) +
                    " regs/thread, " + mainEmitter.SharedMemoryBytes.ToString(
                        CultureInfo.InvariantCulture) + " B dynamic shared, " +
                    mainInfo.LocalBytesPerThread.ToString(CultureInfo.InvariantCulture) +
                    " B local/thread, " + activeBlocks.ToString(
                        CultureInfo.InvariantCulture) + " blocks/SM",
                promotable: true);
        }
        catch (NotSupportedException)
        {
            for (int i = resources.Count - 1; i >= 0; i--) resources[i].Dispose();
            return null;
        }
        catch (ArgumentException)
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

    private static CandidateProgram? TryCreateLibraryInlineWinogradBackward(
        DirectPtxRuntime runtime, CodegenKernelSpec spec)
    {
        if (!CodegenTiledConv2DPlan.TryCreate(spec, out var possible, out _)) return null;
        CodegenTiledConv2DPlan plan = possible!;
        if (plan.TapSign != -1 || !plan.MatrixReductionMajor ||
            plan.BiasInput.HasValue || spec.Activation != CodegenActivationKind.None ||
            plan.InputHeight != plan.OutputHeight ||
            plan.InputWidth != plan.OutputWidth ||
            (plan.OutputHeight & 1) != 0 || (plan.OutputWidth & 1) != 0)
            return null;

        var resources = new List<IDisposable>();
        try
        {
            var shape = new Conv2DWinogradShape(
                plan.Batch, plan.ReductionChannels,
                plan.OutputHeight, plan.OutputWidth, plan.M,
                linear: true, reductionMajor: true, invertFilter: true);
            var kernel = new PtxConv2DNchw3x3WinogradF23Kernel(runtime, shape);
            resources.Add(kernel);

            var uploaded = new DirectPtxBuffer[spec.Inputs.Count];
            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                CodegenTensorBinding binding = spec.Inputs[i];
                var buffer = runtime.AllocateBytes(
                    (nuint)(binding.ElementCount * binding.ElementBytes));
                resources.Add(buffer);
                var host = new float[binding.ElementCount];
                for (long e = 0; e < host.LongLength; e++)
                    host[e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                buffer.Upload<float>(host);
                uploaded[i] = buffer;
            }

            var dummyBias = runtime.AllocateBytes((nuint)shape.BiasBytes);
            var output = runtime.AllocateBytes((nuint)shape.OutputBytes);
            resources.Add(dummyBias);
            resources.Add(output);

            void Launch() => kernel.Launch(
                DirectPtxTensorView.CreateOwned(
                    uploaded[plan.StreamInput], kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(
                    uploaded[plan.MatrixInput], kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(dummyBias, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));

            DirectPtxFunctionInfo info = kernel.FunctionInfo;
            return new CandidateProgram(
                "library-winograd-inline-adjoint-fp32", Launch, output,
                checked((int)spec.Output.ElementCount), resources,
                resourceSummary: info.RegistersPerThread.ToString(
                        CultureInfo.InvariantCulture) + " regs/thread, " +
                    info.StaticSharedBytes.ToString(CultureInfo.InvariantCulture) +
                        " B shared, " +
                    info.LocalBytesPerThread.ToString(CultureInfo.InvariantCulture) +
                        " B local/thread, " +
                    kernel.Audit.ActiveBlocksPerMultiprocessor.ToString(
                        CultureInfo.InvariantCulture) + " blocks/SM",
                promotable: false);
        }
        catch (NotSupportedException)
        {
            for (int i = resources.Count - 1; i >= 0; i--) resources[i].Dispose();
            return null;
        }
        catch (ArgumentException)
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

    private static CandidateProgram? TryCreateLibraryBackwardInput(
        DirectPtxRuntime runtime, CodegenKernelSpec spec)
    {
        if (!CodegenTiledConv2DPlan.TryCreate(spec, out var possible, out _)) return null;
        CodegenTiledConv2DPlan plan = possible!;
        if (plan.TapSign != -1 || plan.BiasInput.HasValue ||
            spec.Activation != CodegenActivationKind.None)
            return null;

        var resources = new List<IDisposable>();
        try
        {
            var kernel = new PtxConv2DBackwardInput3x3Kernel(
                runtime, plan.Batch, plan.ReductionChannels, plan.M,
                plan.InputHeight, plan.InputWidth);
            resources.Add(kernel);

            var uploaded = new DirectPtxBuffer[spec.Inputs.Count];
            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                CodegenTensorBinding binding = spec.Inputs[i];
                var buffer = runtime.AllocateBytes(
                    (nuint)(binding.ElementCount * binding.ElementBytes));
                resources.Add(buffer);
                var host = new float[binding.ElementCount];
                for (long e = 0; e < host.LongLength; e++)
                    host[e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                buffer.Upload<float>(host);
                uploaded[i] = buffer;
            }

            var output = runtime.AllocateBytes(
                (nuint)(spec.Output.ElementCount * spec.Output.ElementBytes));
            resources.Add(output);

            void Launch() => kernel.Launch(
                DirectPtxTensorView.CreateOwned(
                    uploaded[plan.StreamInput], kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(
                    uploaded[plan.MatrixInput], kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));

            DirectPtxFunctionInfo info = kernel.FunctionInfo;
            return new CandidateProgram(
                "library-bwd-input-direct", Launch, output,
                checked((int)spec.Output.ElementCount), resources,
                resourceSummary: info.RegistersPerThread.ToString(
                        CultureInfo.InvariantCulture) + " regs/thread, " +
                    info.StaticSharedBytes.ToString(CultureInfo.InvariantCulture) +
                        " B shared, " +
                    info.LocalBytesPerThread.ToString(CultureInfo.InvariantCulture) +
                        " B local/thread, " +
                    kernel.Audit.ActiveBlocksPerMultiprocessor.ToString(
                        CultureInfo.InvariantCulture) + " blocks/SM",
                promotable: false);
        }
        catch (NotSupportedException)
        {
            for (int i = resources.Count - 1; i >= 0; i--) resources[i].Dispose();
            return null;
        }
        catch (ArgumentException)
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
        return Agrees(candidate, reference, out deviation, out _, out _, out _);
    }

    private static bool Agrees(
        float[] candidate, float[] reference, out double deviation,
        out long worstIndex, out float actual, out float expected)
    {
        return CodegenOutputAgreement.Agrees(
            candidate, reference, CodegenMeasurementProtocol.AccumulationTolerance,
            out deviation, out worstIndex, out actual, out expected);
    }

    private static long ClosestIndex(float[] values, float target)
    {
        long closest = -1;
        double distance = double.MaxValue;
        for (long i = 0; i < values.LongLength; i++)
        {
            double candidate = Math.Abs(values[i] - target);
            if (candidate >= distance) continue;
            distance = candidate;
            closest = i;
        }
        return closest;
    }

    private static unsafe void LaunchOne(
        DirectPtxModule module, IntPtr fn, IntPtr[] pointers, uint blocks, uint blockX,
        uint blockY, uint dynamicSharedMemoryBytes = 0)
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
