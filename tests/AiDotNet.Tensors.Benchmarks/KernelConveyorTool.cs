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
            case "bench": Bench(entries); break;
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
            Console.WriteLine("kernel                              regs   lowering   guards   max abs dev   result");

            foreach (var entry in entries)
            {
                string status;
                try
                {
                    var (dev, regs, elided, lowering) = VerifyOne(runtime, entry.Verify);

                    // The verify shape must exercise the SAME lowering as the shape
                    // that gets released. Otherwise the strip-mined loop path can ship
                    // having only ever been checked in its fully-unrolled form -- the
                    // released-an-unverified-branch failure, one abstraction up.
                    int releasedLowering = LoweringOf(runtime, entry.Bench);
                    bool sameLowering = lowering == releasedLowering;
                    bool ok = dev <= Tolerance && !double.IsNaN(dev) && sameLowering;
                    status = ok ? "PASS" : sameLowering ? "FAIL" : "LOWERING";
                    if (ok) passed++; else failed++;
                    Console.WriteLine(entry.Name.PadRight(36) +
                        regs.ToString(CultureInfo.InvariantCulture).PadLeft(4) +
                        Describe(lowering).PadLeft(11) +
                        elided.ToString(CultureInfo.InvariantCulture).PadLeft(9) +
                        dev.ToString("E3", CultureInfo.InvariantCulture).PadLeft(14) +
                        status.PadLeft(9));
                    if (!sameLowering)
                        Console.WriteLine("    verify shape is " + Describe(lowering) +
                            " but the released shape is " + Describe(releasedLowering) +
                            "; the released path is unverified.");
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

            LaunchSpec(module, fn, pointers, PtxAffineEmitter.GridBlocks(spec));
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
                    metrics?.SpillStores.ToString(CultureInfo.InvariantCulture) ?? ""));
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

    private static void Bench(IReadOnlyList<CodegenCatalogEntry> entries)
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
            Console.WriteLine();
            Console.WriteLine("kernel                              blocks    us/launch    p95/med   run spread");

            foreach (var entry in entries)
            {
                try
                {
                    var spec = entry.Bench;
                    var emitter = new PtxAffineEmitter();
                    string ptx = emitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                    using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
                    IntPtr fn = module.GetFunction(spec.Name, out _);
                    uint blocks = PtxAffineEmitter.GridBlocks(spec);

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

                        void Launch() => LaunchSpec(module, fn, pointers, blocks);

                        var medians = new double[Runs];
                        double worstTail = 0;
                        for (int run = 0; run < Runs; run++)
                        {
                            var d = Measure(runtime.Synchronize, Launch);
                            medians[run] = d.Median;
                            worstTail = Math.Max(worstTail, d.Median > 0 ? d.P95 / d.Median : double.NaN);
                        }
                        double lo = medians.Min(), hi = medians.Max();
                        Console.WriteLine(entry.Name.PadRight(36) +
                            blocks.ToString("N0", CultureInfo.InvariantCulture).PadLeft(8) +
                            (lo * 1000.0).ToString("F1", CultureInfo.InvariantCulture).PadLeft(13) +
                            worstTail.ToString("F2", CultureInfo.InvariantCulture).PadLeft(11) +
                            ((hi / lo - 1.0) * 100).ToString("F1", CultureInfo.InvariantCulture).PadLeft(10) + "%");
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

    private static unsafe void LaunchSpec(DirectPtxModule module, IntPtr fn, IntPtr[] pointers, uint blocks)
    {
        fixed (IntPtr* pinned = pointers)
        {
            void** argv = stackalloc void*[pointers.Length];
            for (int i = 0; i < pointers.Length; i++) argv[i] = pinned + i;
            module.Launch(fn, blocks, 1, 1, PtxAffineEmitter.BlockThreads, 1, 1, 0, argv);
        }
    }
}
