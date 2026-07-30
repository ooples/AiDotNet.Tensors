// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Joins the semantic roofline, exact tuned lowering, competitor ratio, and hardware
/// counters for each catalog kernel that does not beat the standing competitor.
/// </summary>
/// <remarks>
/// The family oracle answers whether the emitter can cover each algebraic family. It does
/// not answer why a concrete tuned catalog row loses to cuDNN. This tool deliberately starts
/// from the release artifacts for the exact dispatch being judged, rejects stale or missing
/// rows, and turns the independent evidence into a falsifiable diagnosis and next action.
/// </remarks>
internal static class KernelCatalogOracleTool
{
    private sealed record EvidenceRow(IReadOnlyDictionary<string, string> Cells)
    {
        internal string this[string name] => Cells.TryGetValue(name, out string? value)
            ? value
            : throw new InvalidOperationException("Evidence row has no '" + name + "' column.");

        internal double Number(string name) => double.Parse(this[name],
            NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    private sealed record ScheduleEvidence(
        string Winner,
        string Shape,
        double MinimumTrafficAmplification,
        double WarpLoadsPerMac,
        double? PredictedMicroseconds,
        string Reuse,
        int SharedMemoryBytes,
        int BlockThreads,
        bool AsyncCopy,
        bool DoubleBuffered,
        bool VectorSharedLoads,
        bool SoftwarePrefetch,
        string Features);

    private sealed record Diagnosis(string Cause, string Action);

    private sealed record Result(
        string Kernel,
        string Outcome,
        double Ratio,
        double OursUs,
        double CompetitorUs,
        string CompetitorPlanStrategy,
        double CompetitorPlanSpread,
        double? CeilingUs,
        double? SemanticEfficiency,
        ScheduleEvidence Schedule,
        string Limiter,
        double LimiterPct,
        double LongScoreboard,
        double Wait,
        double Mio,
        double AluShare,
        double FmaShare,
        double LsuShare,
        double ControlShare,
        string ProfiledPhase,
        int PhaseCount,
        double PhaseShare,
        Diagnosis Diagnosis);

    internal static void Run(string[] args)
    {
        string competitorPath = Path.GetFullPath(KernelToolArgs.ValueOf(args, "--competitor") ??
            Path.Combine("artifacts", "competitor-ratios.tsv"));
        string limiterPath = Path.GetFullPath(KernelToolArgs.ValueOf(args, "--limiter") ??
            Path.Combine("artifacts", "limiter.tsv"));
        string outputPath = Path.GetFullPath(KernelToolArgs.ValueOf(args, "--out") ??
            Path.Combine("artifacts", "kernel-diagnosis.tsv"));
        bool includeWins = args.Contains("--all", StringComparer.Ordinal);

        var competitor = ReadEvidence(competitorPath);
        var limiter = ReadEvidence(limiterPath);
        RequireColumns(competitorPath, competitor,
            "ours_us", "competitor_us", "ratio", "competitor_plan_spread_pct",
            "competitor_plan_strategy", "dispatch");
        RequireColumns(limiterPath, limiter,
            "limiter", "pct_of_peak", "status", "stall_wait", "stall_long_sb",
            "stall_mio", "pipe_alu", "pipe_fma", "pipe_lsu", "pipe_cbu", "phase",
            "phase_count", "phase_share_pct", "dispatch");

        string selector = KernelToolArgs.Selector(args);
        var entries = string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase)
            ? CodegenKernelCatalog.All
            : new[] { CodegenKernelCatalog.Find(selector)! }.Where(e => e is not null).ToList();
        KernelToolArgs.RequireNonEmptySelection(selector, entries.Count, "kernel-oracle --catalog");

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        if (File.Exists(outputPath)) File.Delete(outputPath);
        GpuBenchmarkEnvironment.RequireIdleGpu("catalog-oracle-start");

        using var runtime = new DirectPtxRuntime();
        string dispatch = KernelEvidenceIdentity.CurrentDispatch(runtime);
        var rates = DeviceCalibration.Measure(
            runtime, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        var reference = CodegenMachineModel.Rtx3080Locked;
        var machine = DeviceCalibration.ToMachineModel(
            rates, reference.Multiprocessors, reference.ClockHz);
        var results = new List<Result>();

        foreach (CodegenCatalogEntry entry in entries)
        {
            if (!competitor.TryGetValue(entry.Name, out EvidenceRow? competitorRow))
                throw Missing(entry.Name, competitorPath, "competitor");
            if (!limiter.TryGetValue(entry.Name, out EvidenceRow? limiterRow))
                throw Missing(entry.Name, limiterPath, "limiter");
            if (!string.Equals(competitorRow["dispatch"], dispatch, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    entry.Name + " competitor row measured another generated dispatch; " +
                    "rerun --kernel-competitor before diagnosing it.");
            if (!string.Equals(limiterRow["dispatch"], dispatch, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    entry.Name + " limiter row profiled another generated dispatch; " +
                    "rerun --kernel-limiter before diagnosing it.");

            double ratio = competitorRow.Number("ratio");
            string outcome = ratio <= 0.91 ? "LOSS" : ratio < 1.10 ? "TIE" : "WIN";
            if (!includeWins && outcome == "WIN") continue;

            CodegenKernelSpec spec = entry.Bench;
            var semantic = CodegenPerformanceModel.Predict(
                spec, spec.Space.TotalThreads, dynamicLoadsPerThread: 0, machine);
            double? ceiling = Ceiling(semantic);
            double ours = competitorRow.Number("ours_us");

            var identity = CodegenAutotuneIdentity.Create(
                spec, runtime.DeviceFingerprint,
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
            string? winner = CodegenAutotuneCache.WinnerFor(entry.Name, identity);
            ScheduleEvidence schedule = AnalyzeSchedule(
                entry, winner, semantic, machine,
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);

            double pipeAlu = limiterRow.Number("pipe_alu");
            double pipeFma = limiterRow.Number("pipe_fma");
            double pipeLsu = limiterRow.Number("pipe_lsu");
            double pipeCbu = limiterRow.Number("pipe_cbu");
            double pipeTotal = pipeAlu + pipeFma + pipeLsu + pipeCbu;
            double Share(double value) => pipeTotal > 0 ? value / pipeTotal * 100.0 : 0.0;

            string limiterName = limiterRow["limiter"];
            double limiterPct = limiterRow.Number("pct_of_peak");
            double longSb = limiterRow.Number("stall_long_sb");
            double wait = limiterRow.Number("stall_wait");
            double mio = limiterRow.Number("stall_mio");
            int phaseCount = (int)limiterRow.Number("phase_count");
            double phaseShare = limiterRow.Number("phase_share_pct");
            double? efficiency = ceiling / ours * 100.0;

            Diagnosis diagnosis = Diagnose(
                spec, outcome, limiterName, limiterRow["status"], limiterPct,
                efficiency, schedule, longSb, wait, mio,
                Share(pipeAlu), Share(pipeFma), Share(pipeLsu), Share(pipeCbu),
                limiterRow["phase"], phaseCount, phaseShare);

            results.Add(new Result(
                entry.Name, outcome, ratio, ours, competitorRow.Number("competitor_us"),
                competitorRow["competitor_plan_strategy"],
                competitorRow.Number("competitor_plan_spread_pct"),
                ceiling, efficiency, schedule, limiterName, limiterPct,
                longSb, wait, mio,
                Share(pipeAlu), Share(pipeFma), Share(pipeLsu), Share(pipeCbu),
                limiterRow["phase"], phaseCount, phaseShare, diagnosis));
        }

        GpuBenchmarkEnvironment.RequireNoForeignCompute(
            "catalog-oracle-end", afterSuite: true);
        if (results.Count == 0 && !includeWins)
        {
            Console.WriteLine("All selected catalog kernels are wins; nothing to diagnose.");
            return;
        }
        if (results.Count == 0)
            throw new InvalidOperationException("No catalog evidence could be joined.");

        Print(results, rates);
        Write(outputPath, results);
        Console.WriteLine();
        Console.WriteLine("diagnosis artifact: " + outputPath);
    }

    private static ScheduleEvidence AnalyzeSchedule(
        CodegenCatalogEntry entry,
        string? winner,
        CodegenPerformancePrediction semantic,
        CodegenMachineModel machine,
        int major,
        int minor)
    {
        long uniqueBytes = 0;
        long warpLoads = 0;
        double? predictedUs = 0.0;
        string shape;
        int sharedMemoryBytes = 0;
        int blockThreads = 0;
        bool asyncCopy = false;
        bool doubleBuffered = false;
        bool vectorSharedLoads = false;
        bool softwarePrefetch = false;
        var features = new List<string>();

        bool tiledChunkedSplit = winner is not null &&
            winner.StartsWith("tiled-chunked-split:", StringComparison.Ordinal);
        if (winner is not null &&
            (winner.StartsWith("tiled-split:", StringComparison.Ordinal) || tiledChunkedSplit))
        {
            int chunkFactor = 0;
            CodegenSplitPlan? split = tiledChunkedSplit
                ? TryParseChunkFactor(winner, out chunkFactor)
                    ? CodegenSplitReduction.TryPlanChunked(entry.Bench, chunkFactor)
                    : null
                : CodegenSplitReduction.TryPlan(entry.Bench);
            if (split is null)
                throw new InvalidOperationException(
                    entry.Name + " records " + winner + " but no split can be rebuilt.");

            PtxTiledOuterProductProgram tiled = PtxTiledOuterProductDispatcher.Emit(
                split.Partial, major, minor);
            int tiledBlocks = tiled.Blocks;
            int tiledSteps = tiled.Steps;
            int tiledInnerReduction = tiled.InnerReduction;
            int tiledM = tiled.TileM;
            int tiledN = tiled.TileN;
            int tiledBlockThreads = tiled.BlockThreads;
            sharedMemoryBytes = tiled.SharedMemoryBytes;
            blockThreads = tiledBlockThreads;
            asyncCopy = true;
            doubleBuffered = true;
            features.Add("split-reduction");
            features.Add("register-tiled");
            long partialThreads = split.Partial.Space.TotalThreads;
            var partialSemantic = CodegenPerformanceModel.Predict(
                split.Partial, partialThreads, 0, machine, tiledBlockThreads);
            long scalarLoads = checked((long)tiledBlocks * tiledSteps *
                tiledInnerReduction * (tiledM + tiledN));

            var combineEmitter = new PtxAffineEmitter();
            _ = combineEmitter.Emit(split.Combine, major, minor);
            long combineThreads = split.Combine.Space.TotalThreads /
                Math.Max(1, combineEmitter.CoarsenedLanes);
            var combinePrediction = CodegenPerformanceModel.Predict(
                split.Combine, combineThreads, combineEmitter.DynamicLoadsPerThread,
                machine, combineEmitter.LaunchBlockThreads);

            uniqueBytes = checked(partialSemantic.UniqueBytes + combinePrediction.UniqueBytes);
            warpLoads = checked((scalarLoads + 31) / 32 +
                combinePrediction.WarpLoadInstructions);
            double? partialCeiling = Ceiling(partialSemantic);
            predictedUs = partialCeiling is double partialUs &&
                combinePrediction.HasComputeCeiling
                    ? partialUs + combinePrediction.PredictedMicroseconds
                    : null;
            shape = (tiledChunkedSplit
                    ? "tiled chunked split " + chunkFactor.ToString(CultureInfo.InvariantCulture) +
                      "x2, "
                    : "tiled split x2, ") +
                tiledM + "x" + tiledN + " over " + tiledInnerReduction;
        }
        else if (CodegenTiledConv2DSplitSchedule.Find(winner) is { } splitSchedule)
        {
            if (!CodegenTiledConv2DSplitPlan.TryCreate(
                    entry.Bench, splitSchedule,
                    out CodegenTiledConv2DSplitPlan? exact, out string reason))
                throw new InvalidOperationException(
                    entry.Name + " records " + winner + " but cannot rebuild it: " + reason);

            CodegenSplitPlan split = exact!.Split;
            var partialEmitter = new PtxTiledConv2DEmitter(splitSchedule.Tile);
            _ = partialEmitter.Emit(split.Partial, major, minor);
            CodegenTiledConv2DPlan partialPlan = partialEmitter.Plan!;
            sharedMemoryBytes = partialEmitter.SharedMemoryBytes;
            blockThreads = partialEmitter.LaunchBlockThreads;
            asyncCopy = true;
            doubleBuffered = partialPlan.Stages > 1;
            vectorSharedLoads = true;
            features.Add("split-reduction");
            features.Add("register-tiled");
            if (partialPlan.DirectStream) features.Add("direct-stream");
            var partialSemantic = CodegenPerformanceModel.Predict(
                split.Partial, split.Partial.Space.TotalThreads, 0,
                machine, partialEmitter.LaunchBlockThreads);
            long scalarLoads = checked((long)partialPlan.Blocks * partialPlan.Steps *
                (partialPlan.MatrixStageElements + partialPlan.StreamStageElements));
            long partialWarpLoads = partialPlan.DirectStream
                ? partialSemantic.WarpLoadInstructions
                : (scalarLoads + 31) / 32;

            var combineEmitter = new PtxAffineEmitter();
            _ = combineEmitter.Emit(split.Combine, major, minor);
            long combineThreads = split.Combine.Space.TotalThreads /
                Math.Max(1, combineEmitter.CoarsenedLanes);
            var combinePrediction = CodegenPerformanceModel.Predict(
                split.Combine, combineThreads, combineEmitter.DynamicLoadsPerThread,
                machine, combineEmitter.LaunchBlockThreads);

            uniqueBytes = checked(partialSemantic.UniqueBytes + combinePrediction.UniqueBytes);
            warpLoads = checked(partialWarpLoads + combinePrediction.WarpLoadInstructions);
            double? partialCeiling = Ceiling(partialSemantic);
            predictedUs = partialCeiling is double partialUs &&
                combinePrediction.HasComputeCeiling
                    ? partialUs + combinePrediction.PredictedMicroseconds
                    : null;
            shape = "tiled conv2d split x2, " + partialPlan.TileM + "x" +
                partialPlan.TileRows + "x" + partialPlan.TileChannels;
        }
        else if (winner is not null && winner.StartsWith("split:", StringComparison.Ordinal))
        {
            CodegenSplitPlan? plan = CodegenSplitReduction.TryPlan(entry.Bench);
            if (plan is null)
                throw new InvalidOperationException(
                    entry.Name + " records " + winner + " but no split can be rebuilt.");

            foreach (CodegenKernelSpec half in new[] { plan.Partial, plan.Combine })
            {
                var emitter = new PtxAffineEmitter();
                _ = emitter.Emit(half, major, minor);
                long threads = half.Space.TotalThreads / Math.Max(1, emitter.CoarsenedLanes);
                var prediction = CodegenPerformanceModel.Predict(
                    half, threads, emitter.DynamicLoadsPerThread,
                    machine, emitter.LaunchBlockThreads);
                sharedMemoryBytes = Math.Max(sharedMemoryBytes, emitter.SharedMemoryBytes);
                blockThreads = Math.Max(blockThreads, emitter.LaunchBlockThreads);
                uniqueBytes = checked(uniqueBytes + prediction.UniqueBytes);
                warpLoads = checked(warpLoads + prediction.WarpLoadInstructions);
                predictedUs = predictedUs is double total && prediction.HasComputeCeiling
                    ? total + prediction.PredictedMicroseconds
                    : null;
            }
            features.Add("split-reduction");
            features.Add("affine");
            shape = "split x2";
        }
        else if (string.Equals(winner, "tiled-contraction", StringComparison.Ordinal) ||
                 CodegenTiledContractionSchedule.Find(winner) is { })
        {
            CodegenTiledContractionSchedule? contractionSchedule =
                CodegenTiledContractionSchedule.Find(winner);
            var emitter = contractionSchedule is null
                ? new PtxTiledContractionEmitter()
                : new PtxTiledContractionEmitter(contractionSchedule);
            _ = emitter.Emit(entry.Bench, major, minor);
            var plan = emitter.Plan!;

            // The plan stages every distinct operand tile once per K step. Count those
            // scalar values, convert to warp-level issue equivalents, and keep semantic
            // unique bytes as the traffic floor. This is schedule evidence, not a runtime
            // prediction: the hardware timing remains the promotion authority.
            long scalarLoads = checked((long)plan.Blocks * plan.Steps * plan.TileK *
                (plan.TileM + plan.TileN));
            uniqueBytes = semantic.UniqueBytes;
            warpLoads = (scalarLoads + 31) / 32;
            predictedUs = Ceiling(semantic);
            sharedMemoryBytes = emitter.SharedMemoryBytes;
            blockThreads = emitter.LaunchBlockThreads;
            asyncCopy = true;
            doubleBuffered = plan.Stages > 1;
            vectorSharedLoads = true;
            features.Add("register-tiled");
            if (plan.RegisterPrefetch) features.Add("register-prefetch");
            shape = plan.TileM + "x" + plan.TileN + "x" + plan.TileK +
                ", matrix+stream";
        }
        else if (string.Equals(
            winner, "depthwise-weight-gradient", StringComparison.Ordinal))
        {
            var emitter = new PtxDepthwiseConv2DWeightGradientEmitter();
            _ = emitter.Emit(entry.Bench, major, minor);
            var plan = emitter.Plan!;

            // Every warp executes one dOut load and three neighbouring input loads per
            // reduction step. The three outputs therefore share the dOut instruction;
            // the affine split issued it independently for each tap.
            long reductionWarps = (plan.ReductionElements + 31L) / 32L;
            warpLoads = checked((long)plan.Blocks * reductionWarps * 4L);
            uniqueBytes = semantic.UniqueBytes;
            predictedUs = Ceiling(semantic);
            sharedMemoryBytes = emitter.SharedMemoryBytes;
            blockThreads = emitter.LaunchBlockThreads;
            softwarePrefetch = true;
            features.Add("cooperative-warp-reduction");
            shape = "cooperative (channel,kh), three kw, 256-thread tree";
        }
        else if (string.Equals(winner, "parity-transposed", StringComparison.Ordinal))
        {
            var emitter = new PtxParityTransposedConv2DEmitter();
            _ = emitter.Emit(entry.Bench, major, minor);
            var plan = emitter.Plan!;

            // Each launched thread issues nine weight loads plus four predicated
            // activation-load instructions. Boundary predicates reduce traffic but not
            // the instruction stream, so retain all thirteen in schedule evidence.
            long warps = (plan.InputElements + 31L) / 32L;
            warpLoads = checked(warps * 13L);
            uniqueBytes = semantic.UniqueBytes;
            predictedUs = Ceiling(semantic);
            blockThreads = emitter.LaunchBlockThreads;
            features.Add("output-reuse-4x");
            shape = "one input per thread, deterministic 2x2 output parity tile";
        }
        else if (string.Equals(
            winner, "inline-outer-winograd-conv2d", StringComparison.Ordinal))
        {
            var emitter = new PtxOuterProductWinogradConv2DEmitter();
            _ = emitter.Emit(entry.Bench, major, minor);
            var plan = emitter.Plan!;

            // Each C8 stage issues, per warp, twelve input-patch loads (the two
            // always-valid centre columns are one v2 instruction) and nine raw
            // filter loads. All eight warps participate. Boundary predicates reduce
            // traffic but not the issued instruction stream represented here.
            const int channelsPerStage = 8;
            const int warpsPerBlock = 8;
            const int inputLoadsPerWarp = 12;
            const int filterLoadsPerWarp = 9;
            long stages = plan.ReductionChannels / channelsPerStage;
            warpLoads = checked((long)emitter.LaunchBlocks * stages *
                warpsPerBlock * (inputLoadsPerWarp + filterLoadsPerWarp));
            uniqueBytes = semantic.UniqueBytes;
            predictedUs = Ceiling(semantic);
            sharedMemoryBytes = emitter.SharedMemoryBytes;
            blockThreads = emitter.LaunchBlockThreads;
            asyncCopy = true;
            doubleBuffered = true;
            features.Add("winograd-f2x3");
            features.Add("register-tiled");
            shape = "Winograd F(2,3), M32 x 32 tiles, C8 staged 8x8 outer product";
        }
        else if (string.Equals(winner, "tiled-conv2d", StringComparison.Ordinal) ||
                 CodegenTiledConv2DSchedule.Find(winner) is { })
        {
            CodegenTiledConv2DSchedule? convSchedule =
                CodegenTiledConv2DSchedule.Find(winner);
            var emitter = convSchedule is null
                ? new PtxTiledConv2DEmitter()
                : new PtxTiledConv2DEmitter(convSchedule);
            _ = emitter.Emit(entry.Bench, major, minor);
            var plan = emitter.Plan!;

            long scalarLoads = checked((long)plan.Blocks * plan.Steps *
                (plan.MatrixStageElements + plan.StreamStageElements));
            uniqueBytes = semantic.UniqueBytes;
            warpLoads = (scalarLoads + 31) / 32;
            predictedUs = Ceiling(semantic);
            sharedMemoryBytes = emitter.SharedMemoryBytes;
            blockThreads = emitter.LaunchBlockThreads;
            asyncCopy = true;
            doubleBuffered = plan.Stages > 1;
            vectorSharedLoads = true;
            features.Add("register-tiled");
            if (plan.DirectStream) features.Add("direct-stream");
            shape = plan.TileM + "x" + plan.OutputWidth + "x" + plan.TileChannels +
                ", weights+three input rows";
        }
        else
        {
            var emitter = new PtxAffineEmitter();
            KernelConveyorTool.ApplyTuned(emitter, entry.Name, winner);
            _ = emitter.Emit(entry.Bench, major, minor);
            long threads = entry.Bench.Space.TotalThreads / Math.Max(1, emitter.CoarsenedLanes);
            var prediction = CodegenPerformanceModel.Predict(
                entry.Bench, threads, emitter.DynamicLoadsPerThread,
                machine, emitter.LaunchBlockThreads);
            uniqueBytes = prediction.UniqueBytes;
            warpLoads = prediction.WarpLoadInstructions;
            predictedUs = prediction.PredictedMicroseconds;
            sharedMemoryBytes = emitter.SharedMemoryBytes;
            blockThreads = emitter.LaunchBlockThreads;
            features.Add("affine");
            if (!string.Equals(emitter.StagedOperands, "none", StringComparison.Ordinal))
                features.Add("shared-staged");
            shape = emitter.TileDescription + ", " + emitter.StagedOperands;
        }

        double loadsPerMac = semantic.Macs > 0
            ? warpLoads * 32.0 / semantic.Macs
            : 0.0;
        double traffic = semantic.UniqueBytes > 0
            ? uniqueBytes / (double)semantic.UniqueBytes
            : 1.0;
        if (asyncCopy) features.Add("cp.async");
        if (doubleBuffered) features.Add("double-buffered");
        if (vectorSharedLoads) features.Add("vector-shared-loads");
        if (softwarePrefetch) features.Add("software-prefetch-2x");
        return new ScheduleEvidence(
            winner ?? "modelled", shape, traffic, loadsPerMac, predictedUs,
            ReuseText(entry.Bench), sharedMemoryBytes, blockThreads,
            asyncCopy, doubleBuffered, vectorSharedLoads, softwarePrefetch,
            string.Join(",", features.Distinct(StringComparer.Ordinal)));
    }

    private static bool TryParseChunkFactor(string winner, out int chunkFactor)
    {
        chunkFactor = 0;
        int marker = winner.LastIndexOf('x');
        return marker >= 0 && marker + 1 < winner.Length &&
            int.TryParse(winner.Substring(marker + 1), NumberStyles.None,
                CultureInfo.InvariantCulture, out chunkFactor) &&
            CodegenAutotuneIdentity.IsChunkedSplitFactor(chunkFactor);
    }

    private static Diagnosis Diagnose(
        CodegenKernelSpec spec,
        string outcome,
        string limiter,
        string limiterStatus,
        double limiterPct,
        double? semanticEfficiency,
        ScheduleEvidence schedule,
        double longSb,
        double wait,
        double mio,
        double aluShare,
        double fmaShare,
        double lsuShare,
        double controlShare,
        string profiledPhase,
        int phaseCount,
        double phaseShare)
    {
        bool atRoofline = string.Equals(
            limiterStatus, "at-roofline", StringComparison.OrdinalIgnoreCase);
        bool exactDivision = HasExactDivision(spec);
        bool tiledSplit = schedule.Winner.StartsWith(
            "tiled-split:", StringComparison.Ordinal);
        bool tiledChunkedSplit = schedule.Winner.StartsWith(
            "tiled-chunked-split:", StringComparison.Ordinal);
        string phaseContext = phaseCount > 1
            ? "The dominant phase " + profiledPhase + " accounts for " + F(phaseShare) +
              "% of profiled program time. "
            : string.Empty;

        if (outcome == "TIE" && atRoofline && limiter == "SM" && exactDivision)
        {
            return new Diagnosis(
                phaseContext + "SM issue is saturated at " + F(limiterPct) +
                "% by an exact-division/address-control schedule; the measured pipe mix is " +
                "ALU " + F(aluShare) + "%, control " + F(controlShare) +
                "%, FMA " + F(fmaShare) + "%, LSU " + F(lsuShare) + "%.",
                "Specialize output residue/parity classes so each kernel has affine integer " +
                "indices without per-tap div/rem guards; another load micro-optimization " +
                "cannot move an already saturated issue path.");
        }

        if (atRoofline && (limiter == "L1" || limiter == "L2") &&
            (tiledSplit || tiledChunkedSplit))
        {
            return new Diagnosis(
                phaseContext + limiter + " is saturated at " + F(limiterPct) +
                "% in the exact two-pass outer-product program while semantic efficiency " +
                "is " + F(semanticEfficiency) + "%. The partial pass owns the reusable " +
                "operands and dominates program time; the deterministic combine is not the gap. " +
                "Active lowering features: " + schedule.Features + ".",
                (tiledChunkedSplit
                    ? "The partial is already asynchronously double-buffered. Search CTA/thread " +
                      "fragments jointly with chunk count, vectorize shared fragments, and measure " +
                      "a longer prefetch distance; keep the combine unchanged."
                    : "The partial is already asynchronously double-buffered. Increase operand " +
                      "reuse with a wider measured fragment or warp-specialized producer/consumer " +
                      "roles; keep the combine unchanged."));
        }

        if (atRoofline && (limiter == "L1" || limiter == "L2"))
        {
            return new Diagnosis(
                phaseContext + limiter + " is saturated at " + F(limiterPct) +
                "% while the kernel reaches only " +
                F(semanticEfficiency) + "% of its semantic roofline. The chosen " +
                schedule.Winner + " program has " + F(schedule.MinimumTrafficAmplification) +
                "x minimum program traffic and " + F(schedule.WarpLoadsPerMac) +
                " thread-loads/MAC; long-scoreboard stalls are " + F(longSb) + "%.",
                "Increase cross-output operand reuse with a GEMM-style shared/register tile " +
                "or cp.async prefetch across " + schedule.Reuse +
                "; any remaining speedup must do less on-chip load work, not chase DRAM peak.");
        }

        if (!atRoofline && string.Equals(
                schedule.Winner, "inline-outer-winograd-conv2d", StringComparison.Ordinal))
        {
            return new Diagnosis(
                phaseContext + "The exact Winograd F(2,3) outer-product program is " +
                "under-filled: " + limiter + " is " + F(limiterPct) +
                "%, long-scoreboard is " + F(longSb) + "%, and wait is " + F(wait) +
                "%. Its " + schedule.SharedMemoryBytes.ToString(CultureInfo.InvariantCulture) +
                "-byte shared tile permits only one resident CTA/SM, so " +
                "neither L1 nor issue capacity can be filled by another block.",
                "Reduce the live transform/accumulator or shared-tile footprint enough for " +
                "two resident CTAs, or create more independent warp tiles inside the CTA. " +
                "A generic direct/implicit-GEMM recommendation is inapplicable because the " +
                "winning dispatch is already Winograd.");
        }

        if (!atRoofline && schedule.Winner.StartsWith(
                "tiled-contraction", StringComparison.Ordinal))
        {
            return new Diagnosis(
                phaseContext + "The exact contraction winner " + schedule.Shape +
                " remains under-filled after measured M/N/K tile search: " + limiter +
                " is " + F(limiterPct) + "%, LSU dependency stalls are " + F(mio) +
                "%, and long-scoreboard is " + F(longSb) + "%. This is not evidence that " +
                "an unmeasured larger scalar tile will win. The current " +
                schedule.BlockThreads.ToString(CultureInfo.InvariantCulture) + "-thread CTA uses " +
                schedule.SharedMemoryBytes.ToString(CultureInfo.InvariantCulture) +
                " shared bytes with " + schedule.Features + ".",
                "Vector shared fragments and a cp.async double buffer are already active. " +
                (schedule.Features.Contains("register-prefetch", StringComparison.Ordinal)
                    ? "Register prefetch across K is also active; next measure warp-specialized " +
                      "producer/consumer roles. Keep "
                    : "Register prefetch across K was included in the exact finite search and " +
                      "did not win; next measure warp-specialized producer/consumer roles. Keep ") +
                "geometry selection in the exact schedule search rather than hard-coding an " +
                "operation-specific tile.");
        }

        if (!atRoofline && schedule.Winner.StartsWith(
                "tiled-conv2d", StringComparison.Ordinal))
        {
            return new Diagnosis(
                phaseContext + "The measured dense-Conv2D schedule is under-filled at " +
                F(limiterPct) + "% " + limiter + " after the finite search already covered " +
                "M/row/channel tiles, warp-halo, direct-stream, asymmetric staging, and " +
                "reduction splits. Long-scoreboard is " + F(longSb) + "% and wait is " +
                F(wait) + "%, so no single existing pipe is the bottleneck. The winner already " +
                "uses " + schedule.Features + ".",
                "Add a genuinely different dataflow candidate—inline Winograd, implicit GEMM, " +
                "or a warp-specialized asynchronous pipeline—and let the same numerical and " +
                "paired-timing gates choose it. More tuning of the existing local knobs is " +
                "not the indicated lever.");
        }

        if (!atRoofline && (tiledSplit || tiledChunkedSplit))
        {
            return new Diagnosis(
                phaseContext + "The exact two-pass outer-product program is under-filled: " +
                limiter + " is " + F(limiterPct) + "%, long-scoreboard is " + F(longSb) +
                "%, wait is " + F(wait) + "%, and shared/LSU dependency stalls are " +
                F(mio) + "%. The partial pass, not the deterministic combine, is the " +
                "optimization target. Active lowering features: " + schedule.Features + ".",
                (tiledChunkedSplit
                    ? "The partial is already asynchronously double-buffered. Search " +
                      "CTA/thread fragments jointly with chunk count, vectorize shared loads, or " +
                      "increase the measured prefetch distance; keep the combine unchanged."
                    : "The partial is already asynchronously double-buffered. Expand the " +
                      "outer-product fragment search and vectorize its shared loads; keep the " +
                      "deterministic combine unchanged."));
        }

        if (!atRoofline && longSb >= 20.0 && wait >= 15.0)
        {
            return new Diagnosis(
                phaseContext + "No hardware unit is saturated (largest is " + limiter +
                " at " + F(limiterPct) +
                "%), but warps spend " + F(longSb) + "% on long scoreboards and " +
                F(wait) + "% on dependency wait. This is exposed memory latency plus " +
                "insufficient independent work, not bandwidth.",
                "Tile/prefetch reusable operands across " + schedule.Reuse +
                " and keep multiple independent accumulators in flight so global loads and " +
                "the reduction dependency chain overlap.");
        }

        if (!atRoofline && longSb >= 20.0)
        {
            if (schedule.SoftwarePrefetch)
            {
                return new Diagnosis(
                    phaseContext + "No unit is saturated (largest is " + limiter + " at " +
                    F(limiterPct) + "%), while long-scoreboard stalls consume " + F(longSb) +
                    "%. The current two-position software pipeline is active, so this is " +
                    "residual latency after prefetch rather than a missing prefetch stage.",
                    "Measure a deeper grid-stride prefetch distance and a mapping with multiple " +
                    "independent channel/tap groups per CTA; retain only variants that improve " +
                    "the paired timing and stability gates.");
            }
            return new Diagnosis(
                phaseContext + "No unit is saturated (largest is " + limiter + " at " +
                F(limiterPct) +
                "%), while long-scoreboard stalls consume " + F(longSb) +
                "% and LSU throttle only " + F(mio) + "%. Requests are waiting on latency; " +
                "the load pipe is not full.",
                "Prefetch or stage the operands reusable across " + schedule.Reuse +
                " and increase resident/independent work; reducing instruction count alone " +
                "does not address the measured stall.");
        }

        if (!atRoofline)
        {
            return new Diagnosis(
                phaseContext + "The direct schedule is balanced but under-filled: " +
                limiter + " is only " +
                F(limiterPct) + "%, long-scoreboard is " + F(longSb) + "%, wait is " +
                F(wait) + "%, and no measured resource reaches the 70% roofline.",
                "Stop tuning a single pipe and change the schedule: use a larger output/reduction " +
                "tile or implicit-GEMM mapping that exploits " + schedule.Reuse + ".");
        }

        return new Diagnosis(
            phaseContext + limiter + " is saturated at " + F(limiterPct) + "%.",
            "The current algorithm is at its measured hardware roofline; a win requires " +
            "changing its traffic or instruction mix rather than retuning the same lowering.");
    }

    private static void Print(IReadOnlyList<Result> results, DeviceCalibration.Rates rates)
    {
        Console.WriteLine();
        Console.WriteLine("CATALOG LOSS ORACLE - tuned dispatch vs cuDNN and measured counters");
        Console.WriteLine("protocol: " + CodegenMeasurementProtocol.Tag +
                          " | calibrated DRAM " +
                          (rates.DramBytesPerSecond / 1e9).ToString("0.0", CultureInfo.InvariantCulture) +
                          " GB/s");
        Console.WriteLine();
        Console.WriteLine("{0,-34} {1,5} {2,7} {3,9} {4,8} {5,9} {6,9}",
            "kernel", "state", "ratio", "ours us", "ceil us", "% ceil", "limiter");
        foreach (Result row in results)
        {
            Console.WriteLine("{0,-34} {1,5} {2,7} {3,9} {4,8} {5,9} {6,9}",
                row.Kernel,
                row.Outcome,
                row.Ratio.ToString("0.00x", CultureInfo.InvariantCulture),
                row.OursUs.ToString("0.0", CultureInfo.InvariantCulture),
                F(row.CeilingUs),
                row.SemanticEfficiency is double efficiency
                    ? F(efficiency) + "%"
                    : "-",
                row.Limiter + " " + row.LimiterPct.ToString("0", CultureInfo.InvariantCulture) + "%");
        }

        foreach (Result row in results)
        {
            Console.WriteLine();
            Console.WriteLine(row.Kernel + " — " + row.Outcome + " at " +
                              row.Ratio.ToString("0.000x", CultureInfo.InvariantCulture));
            Console.WriteLine("  lowering: " + row.Schedule.Winner + " (" + row.Schedule.Shape +
                              "), model " + F(row.Schedule.PredictedMicroseconds) + " us");
            Console.WriteLine("  competitor plan: " + row.CompetitorPlanStrategy +
                              ", cross-plan spread " + F(row.CompetitorPlanSpread) + "%");
            Console.WriteLine("  reuse: " + row.Schedule.Reuse);
            Console.WriteLine("  features: " + row.Schedule.Features + ", shared " +
                              row.Schedule.SharedMemoryBytes.ToString(CultureInfo.InvariantCulture) +
                              " B, " + row.Schedule.BlockThreads.ToString(CultureInfo.InvariantCulture) +
                              " threads");
            Console.WriteLine("  cause: " + row.Diagnosis.Cause);
            Console.WriteLine("  next:  " + row.Diagnosis.Action);
        }
    }

    private static void Write(string path, IReadOnlyList<Result> results)
    {
        var text = new StringBuilder();
        text.AppendLine("# protocol " + CodegenMeasurementProtocol.Tag + ": " +
                        CodegenMeasurementProtocol.Description);
        text.AppendLine("kernel\toutcome\tratio\tours_us\tcompetitor_us\tsemantic_ceiling_us" +
                        "\tcompetitor_plan_strategy\tcompetitor_plan_spread_pct" +
                        "\tsemantic_efficiency_pct\twinner\tschedule\tprogram_traffic_x" +
                        "\tthread_loads_per_mac\tmodel_us\tfeatures\tshared_bytes" +
                        "\tblock_threads\tlimiter\tlimiter_pct" +
                        "\tstall_long_sb\tstall_wait\tstall_mio\talu_share_pct" +
                        "\tfma_share_pct\tlsu_share_pct\tcontrol_share_pct\tprofiled_phase" +
                        "\tphase_count\tphase_share_pct\treuse\tcause\taction\tprotocol");
        foreach (Result row in results)
        {
            text.AppendLine(string.Join("\t",
                row.Kernel, row.Outcome,
                N(row.Ratio), N(row.OursUs), N(row.CompetitorUs), N(row.CeilingUs),
                row.CompetitorPlanStrategy, N(row.CompetitorPlanSpread),
                N(row.SemanticEfficiency), row.Schedule.Winner, Clean(row.Schedule.Shape),
                N(row.Schedule.MinimumTrafficAmplification), N(row.Schedule.WarpLoadsPerMac),
                N(row.Schedule.PredictedMicroseconds), row.Schedule.Features,
                row.Schedule.SharedMemoryBytes.ToString(CultureInfo.InvariantCulture),
                row.Schedule.BlockThreads.ToString(CultureInfo.InvariantCulture),
                row.Limiter, N(row.LimiterPct),
                N(row.LongScoreboard), N(row.Wait), N(row.Mio),
                N(row.AluShare), N(row.FmaShare), N(row.LsuShare), N(row.ControlShare),
                row.ProfiledPhase, row.PhaseCount.ToString(CultureInfo.InvariantCulture),
                N(row.PhaseShare),
                Clean(row.Schedule.Reuse), Clean(row.Diagnosis.Cause),
                Clean(row.Diagnosis.Action), CodegenMeasurementProtocol.Tag));
        }
        File.WriteAllText(path, text.ToString(), new UTF8Encoding(false));
    }

    private static Dictionary<string, EvidenceRow> ReadEvidence(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException("Required current-protocol evidence is missing.", path);

        string[]? header = null;
        var rows = new Dictionary<string, EvidenceRow>(StringComparer.Ordinal);
        foreach (string line in File.ReadAllLines(path))
        {
            if (line.Length == 0 || line[0] == '#') continue;
            string[] cells = line.Split('\t');
            if (header is null)
            {
                header = cells;
                continue;
            }
            if (cells.Length != header.Length ||
                !string.Equals(cells[^1], CodegenMeasurementProtocol.Tag, StringComparison.Ordinal))
                continue;

            var values = new Dictionary<string, string>(StringComparer.Ordinal);
            for (int i = 0; i < header.Length; i++) values[header[i]] = cells[i];
            rows[cells[0]] = new EvidenceRow(values);
        }
        if (header is null)
            throw new InvalidOperationException("Evidence file has no header: " + path);
        return rows;
    }

    private static void RequireColumns(
        string path, IReadOnlyDictionary<string, EvidenceRow> rows, params string[] columns)
    {
        EvidenceRow? sample = rows.Values.FirstOrDefault();
        if (sample is null)
            throw new InvalidOperationException("Evidence file has no current rows: " + path);
        foreach (string column in columns)
            _ = sample[column];
    }

    private static Exception Missing(string kernel, string path, string kind) =>
        new InvalidOperationException(
            kernel + " has no current " + kind + " row in " + path + ".");

    private static bool HasExactDivision(CodegenKernelSpec spec)
    {
        foreach (CodegenTensorBinding binding in AllBindings(spec))
        {
            foreach (CodegenAffineExpr map in binding.Map)
                if (map.RequiresExactDivision) return true;
            foreach (CodegenIndirectIndex? indirect in binding.Indirect)
                if (indirect?.Position.RequiresExactDivision == true) return true;
        }
        return false;
    }

    private static IEnumerable<CodegenTensorBinding> AllBindings(CodegenKernelSpec spec)
    {
        foreach (CodegenTensorBinding input in spec.Inputs) yield return input;
        yield return spec.Output;
        foreach (CodegenExtraOutput extra in spec.ExtraOutputs) yield return extra.Binding;
    }

    private static string ReuseText(CodegenKernelSpec spec)
    {
        var parts = new List<string>();
        foreach (var pair in CodegenPerformanceModel.ReuseAxes(spec))
            if (pair.Value.Count != 0)
                parts.Add(pair.Key + "{" + string.Join(",", pair.Value) + "}");
        return parts.Count == 0 ? "no cross-output reuse axis" : string.Join("; ", parts);
    }

    private static double? Ceiling(CodegenPerformancePrediction prediction) =>
        prediction.HasComputeCeiling
            ? Math.Max(prediction.DramMicroseconds, prediction.ComputeMicroseconds)
            : null;

    private static string F(double value) =>
        value.ToString("0.0", CultureInfo.InvariantCulture);

    private static string F(double? value) =>
        value?.ToString("0.0", CultureInfo.InvariantCulture) ?? "-";

    private static string N(double value) =>
        value.ToString("0.####", CultureInfo.InvariantCulture);

    private static string N(double? value) =>
        value?.ToString("0.####", CultureInfo.InvariantCulture) ?? "-";

    private static string Clean(string value) =>
        value.Replace('\t', ' ').Replace('\r', ' ').Replace('\n', ' ');
}
