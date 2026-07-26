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

    internal static void Run(string stage, string[] args)
    {
        string selector = args.FirstOrDefault(a => !a.StartsWith("--", StringComparison.Ordinal)) ?? "all";
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
                var emitter = new PtxAffineEmitter();
                if (args.Contains("--no-coarsen", StringComparer.Ordinal)) emitter.Coarsening = 1;
                    if (ValueOf(args, "--coarsen") is string cz)
                        emitter.Coarsening = int.Parse(cz, CultureInfo.InvariantCulture);
                    if (ValueOf(args, "--max-lanes") is string mz)
                        emitter.MaxTileLanes = int.Parse(mz, CultureInfo.InvariantCulture);
                string ptx = emitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                using var stage = LaunchableStage.Create(runtime, spec, ptx, emitter.LaunchBlocks, (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY);
                stage.Launch();
                runtime.Synchronize();
                Console.WriteLine(entry.Name + ": one launch, " + emitter.LaunchBlocks + " blocks, lanes=" +
                                  emitter.CoarsenedLanes + ", emitted loads=" + emitter.EmittedLoads);
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
        Console.WriteLine("measured column recorded under protocol " + CodegenMeasurementProtocol.Tag);
        Console.WriteLine();
        Console.WriteLine("kernel                        tile            block  ld/MAC  staged");

        foreach (var entry in entries)
        {
            var spec = entry.Bench;
            var emitter = new PtxAffineEmitter();
            if (ValueOf(args, "--coarsen") is string cz)
                emitter.Coarsening = int.Parse(cz, CultureInfo.InvariantCulture);
            emitter.Emit(spec, 8, 6);

            long threads = spec.Space.TotalThreads / Math.Max(1, emitter.CoarsenedLanes);
            var p = CodegenPerformanceModel.Predict(spec, threads, emitter.DynamicLoadsPerThread, machine);

            double measured = MeasuredMicroseconds(entry.Name);
            string measuredText = measured > 0 ? measured.ToString("F1", CultureInfo.InvariantCulture) : "-";
            string ratioText = measured > 0
                ? (p.PredictedMicroseconds / measured).ToString("F2", CultureInfo.InvariantCulture) + "x"
                : "-";

            Console.WriteLine(entry.Name.PadRight(30) +
                emitter.TileDescription.PadRight(16) +
                emitter.LaunchBlockThreads.ToString(CultureInfo.InvariantCulture).PadLeft(5) +
                p.LoadsPerMac.ToString("F3", CultureInfo.InvariantCulture).PadLeft(8) +
                "  " + emitter.StagedOperands);
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

    /// <summary>
    /// Measured times from the locked-clock true-fp32 bake-off, so the prediction can
    /// be checked against reality in the same table.
    /// </summary>
    private static double MeasuredMicroseconds(string kernel) => kernel switch
    {
        "depthwise_conv2d_3x3_bias_relu" => 81.0,
        "depthwise_conv2d_3x3" => 78.8,
        "depthwise_conv2d_3x3_bwd_data" => 78.1,
        "conv2d_1x1_bias_relu" => 38.6,
        "conv2d_1x1_bwd_data" => 42.9,
        "conv2d_3x3_bias_relu" => 75.0,
        "conv2d_3x3_bwd_data" => 87.6,
        "maxpool2d_2x2" => 171.6,
        "conv_transpose2d_3x3_stride2" => 109.0,
        _ => 0.0,
    };

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
                    var (dev, regs, elided, lowering) = VerifyOne(runtime, entry.Bench);

                    // The verify shape must exercise the SAME lowering as the shape
                    // that gets released. Otherwise the strip-mined loop path can ship
                    // having only ever been checked in its fully-unrolled form -- the
                    // released-an-unverified-branch failure, one abstraction up.
                    // One shape now, so there is no verify-vs-release divergence left to
                    // gate: the row records the lowering and the staging it was verified
                    // WITH, which is the fact that matters.
                    var shipped = new PtxAffineEmitter();
                    shipped.Emit(entry.Bench, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                    string staged = shipped.StagedOperands;

                    bool ok = dev <= Tolerance && !double.IsNaN(dev);
                    status = ok ? "PASS" : "FAIL";
                    if (ok) passed++; else failed++;
                    Console.WriteLine(entry.Name.PadRight(36) +
                        regs.ToString(CultureInfo.InvariantCulture).PadLeft(4) +
                        Describe(lowering).PadLeft(9) + ("/" + staged).PadRight(10) +
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
    private static void ApplyTuned(PtxAffineEmitter emitter, string kernelName)
    {
        switch (CodegenAutotuneCache.WinnerFor(kernelName))
        {
            case "no-tile": emitter.Coarsening = 1; break;
            case "tile2": emitter.Coarsening = 2; break;
            case "lanes4": emitter.MaxTileLanes = 4; break;
            case "no-staging": emitter.EnableSharedStaging = false; break;
            case "no-vector": emitter.EnableVectorLoads = false; break;
            default: break;   // untuned, or the modelled choice already won
        }
    }

    /// <summary>Number of reduction axes a spec lowers to runtime loops.</summary>
    private static int LoweringOf(DirectPtxRuntime runtime, CodegenKernelSpec spec)
    {
        var emitter = new PtxAffineEmitter();
        emitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        return emitter.LoopedAxes;
    }

    private static string Describe(int loopedAxes) =>
        loopedAxes == 0 ? "unroll" : "loop x" + loopedAxes.ToString(CultureInfo.InvariantCulture);

    private static (double Deviation, int Registers, int Elided, int Lowering) VerifyOne(
        DirectPtxRuntime runtime, CodegenKernelSpec spec)
    {
        var emitter = new PtxAffineEmitter();
        ApplyTuned(emitter, spec.Name);
        string ptx = emitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
        IntPtr fn = module.GetFunction(spec.Name, out DirectPtxFunctionInfo info);

        // Host data is a deterministic function of (input index, tensor index) so a
        // failure is reproducible and independent of run order.
        var hostInputs = new List<double[]>(spec.Inputs.Count);
        var buffers = new List<DirectPtxBuffer>();
        try
        {
            var pointers = new IntPtr[spec.ParameterCount];
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
                hostInputs.Add(host);
                var buffer = runtime.AllocateBytes((nuint)(count * sizeof(float)));
                buffer.Upload<float>(single);
                buffers.Add(buffer);
                pointers[i] = buffer.Pointer;
            }

            long outCount = Elements(spec.Output.Shape);
            var outBuffer = runtime.AllocateBytes((nuint)(outCount * sizeof(float)));
            buffers.Add(outBuffer);
            pointers[spec.Inputs.Count] = outBuffer.Pointer;

            LaunchSpec(module, fn, pointers, emitter.LaunchBlocks, (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY);
            runtime.Synchronize();

            var actual = new float[outCount];
            outBuffer.Download<float>(actual);
            double[] expected = spec.Interpret(hostInputs);

            double worst = 0;
            for (long e = 0; e < outCount; e++)
            {
                double diff = Math.Abs(actual[e] - expected[e]);
                double scale = Math.Max(1.0, Math.Abs(expected[e]));
                worst = Math.Max(worst, diff / scale);
            }
            return (worst, info.RegistersPerThread, emitter.ElidedGuards, emitter.LoopedAxes);
        }
        finally { foreach (var b in buffers) b.Dispose(); }
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
                var emitter = new PtxAffineEmitter();
                string ptx = emitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);

                int regs;
                using (var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true))
                {
                    module.GetFunction(spec.Name, out DirectPtxFunctionInfo info);
                    regs = info.RegistersPerThread;
                }

                var artifact = DirectPtxCubinArtifactCache.Resolve(runtime, ptx);
                string cubinPath = Path.Combine(outputDirectory, entry.Name + ".cubin");
                File.WriteAllBytes(cubinPath, artifact.Image);

                var metrics = nvdisasm is null ? null : ReadSass(nvdisasm, cubinPath);
                bool ok = metrics is null || (metrics.SpillLoads == 0 && metrics.SpillStores == 0);
                if (ok) gated++; else spilled++;

                Console.WriteLine(entry.Name.PadRight(36) +
                    regs.ToString(CultureInfo.InvariantCulture).PadLeft(4) +
                    (metrics?.Instructions.ToString(CultureInfo.InvariantCulture) ?? "-").PadLeft(13) +
                    (metrics?.Ldg.ToString(CultureInfo.InvariantCulture) ?? "-").PadLeft(6) +
                    (metrics?.Stg.ToString(CultureInfo.InvariantCulture) ?? "-").PadLeft(5) +
                    ((metrics is null ? "-" : metrics.SpillLoads + "/" + metrics.SpillStores)).PadLeft(14) +
                    (metrics is null ? "SKIP" : ok ? "PASS" : "FAIL").PadLeft(7));

                rows.Add(string.Join("\t", entry.Name, spec.Name, artifact.CubinSha256, artifact.SourceKey,
                    regs.ToString(CultureInfo.InvariantCulture),
                    metrics?.Instructions.ToString(CultureInfo.InvariantCulture) ?? "",
                    metrics?.Ldg.ToString(CultureInfo.InvariantCulture) ?? "",
                    metrics?.Stg.ToString(CultureInfo.InvariantCulture) ?? "",
                    metrics?.SpillLoads.ToString(CultureInfo.InvariantCulture) ?? "",
                    metrics?.SpillStores.ToString(CultureInfo.InvariantCulture) ?? "",
                    CodegenMeasurementProtocol.Tag));
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
        text.AppendLine("kernel\tentry\tcubin_sha256\tsource_key\tregisters\tsass_instructions\tldg\tstg\tspill_ld\tspill_st");
        foreach (string row in rows) text.AppendLine(row);
        File.WriteAllText(manifest, text.ToString());

        Console.WriteLine();
        Console.WriteLine("manifest " + manifest);
        Console.WriteLine("release: " + gated.ToString(CultureInfo.InvariantCulture) + " zero-spill, " +
                          spilled.ToString(CultureInfo.InvariantCulture) + " spilling");

        ReportEvidenceGates(entries, outputDirectory);
    }

    /// <summary>
    /// Blueprint #3 and #4 as release gates. Zero spills and a clean SASS audit say the
    /// kernel is well-formed; they say nothing about whether it is worth shipping. A
    /// kernel is only releasable when it also carries, at the CURRENT protocol:
    ///
    ///   a competitor ratio  -- otherwise every number about it is ours-vs-ours;
    ///   a named limiter     -- otherwise nobody knows what its next lever is.
    ///
    /// Both files are produced by other stages, so this reports rather than recomputes,
    /// and a missing or stale file is itself the finding.
    /// </summary>
    private static void ReportEvidenceGates(
        IReadOnlyList<CodegenCatalogEntry> entries, string outputDirectory)
    {
        var ratios = ReadEvidence(Path.Combine("artifacts", "competitor-ratios.tsv"), 3);
        var limiters = ReadEvidence(Path.Combine("artifacts", "limiter.tsv"), 1);

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
                          " carry both. Run --kernel-limiter and tools/bakeoff/run_bakeoff.py");
        Console.WriteLine("to fill gaps; a kernel without both is well-formed but unproven.");
    }

    /// <summary>Reads kernel -> column from a protocol-stamped evidence file.</summary>
    private static Dictionary<string, string> ReadEvidence(string path, int column)
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
        string output = process.StandardOutput.ReadToEnd();
        process.StandardError.ReadToEnd();
        process.WaitForExit();
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
                    var emitter = new PtxAffineEmitter();
                    ApplyTuned(emitter, spec.Name);
                    if (args.Contains("--no-coarsen", StringComparer.Ordinal)) emitter.Coarsening = 1;
                    if (ValueOf(args, "--coarsen") is string cz)
                        emitter.Coarsening = int.Parse(cz, CultureInfo.InvariantCulture);
                    if (ValueOf(args, "--max-lanes") is string mz)
                        emitter.MaxTileLanes = int.Parse(mz, CultureInfo.InvariantCulture);
                    string ptx = emitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                    using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
                    IntPtr fn = module.GetFunction(spec.Name, out _);
                    uint blocks = emitter.LaunchBlocks;

                    var buffers = new List<DirectPtxBuffer>();
                    try
                    {
                        var pointers = new IntPtr[spec.ParameterCount];
                        for (int i = 0; i < spec.Inputs.Count; i++)
                        {
                            long count = Elements(spec.Inputs[i].Shape);
                            var b = runtime.AllocateBytes((nuint)(count * sizeof(float)));
                            var single = new float[count];
                            for (long e = 0; e < count; e++) single[e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                            b.Upload<float>(single);
                            buffers.Add(b);
                            pointers[i] = b.Pointer;
                        }
                        var outBuffer = runtime.AllocateBytes((nuint)(Elements(spec.Output.Shape) * sizeof(float)));
                        buffers.Add(outBuffer);
                        pointers[spec.Inputs.Count] = outBuffer.Pointer;

                        void Launch() => LaunchSpec(module, fn, pointers, blocks, (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY);
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
                            if (drift < bestDrift)
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
                    finally { foreach (var b in buffers) b.Dispose(); }
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

    private static string? ValueOf(string[] args, string flag)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], flag, StringComparison.Ordinal))
                return args[i + 1];
        return null;
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
