// Copyright (c) AiDotNet. All rights reserved.
// --kernel-routes --static: the static-metric half of the #890 phase-4 gate. For each routed family, at its bench
// shape, the declared program as the backend lowers it (PtxProgramLowering, compiled by ptxas) against the reference
// hand-written kernels it replaces (their CUDA C, compiled by NVRTC with the backend's own source preparation and
// options): registers, local bytes (spills), static SASS instructions, global loads and stores and how many of each
// are 128-bit.
//
// THE RULE (decided on #1437, 2026-10-20; refined for #1439 the same day): the declared side FAILS on a spill (any
// local bytes) or on a smaller SHARE of its global loads being 128-bit than the hand-written kernel's. (It was the
// count; skinny_gemm, unrolled further, has more 128-bit loads than a declared GEMM whose every load is one.)
// Registers and static instructions are REPORTED, not gated: a held row keeps
// its elements in registers so x is read once -- 52-88 registers against 14-48, and the reason the declared kernels
// are 1.3-4.9x faster -- and static counts are code, not work (an unrolled row has more instructions than a loop
// over it; a fused program is one kernel standing in for several). The paired wall-clock gate is --kernel-routes.
//
// Options: --arch 75|86 (default 75), --cuda-bin <dir> (ptxas, nvdisasm, cuobjdump), --out <dir>.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Benchmarks;

internal static class KernelRouteStaticTool
{
    private sealed record Metrics(string Kernel, int Registers, int LocalBytes, int Instructions,
        int Loads, int WideLoads, int Stores, int WideStores);

    private sealed record Family(string Name, CodegenProgram Declared, string Module, string[] Kernels,
        Func<CodegenKernelSpec, PtxPassLowering?>? LowerPass = null);

    internal static void Run(string[] args)
    {
        string arch = Value(args, "--arch") ?? "75";
        int major = int.Parse(arch, CultureInfo.InvariantCulture) / 10, minor = int.Parse(arch, CultureInfo.InvariantCulture) % 10;
        string bin = Value(args, "--cuda-bin") ?? DefaultCudaBin()
            ?? throw new InvalidOperationException("No CUDA toolkit bin directory: pass --cuda-bin.");
        string Tool(string name) => Path.Combine(bin, RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? name + ".exe" : name);
        string work = Value(args, "--out") ?? Path.Combine(Path.GetTempPath(), "aidotnet-route-static");
        Directory.CreateDirectory(work);

        var modules = new Dictionary<string, Dictionary<string, Metrics>>(StringComparer.Ordinal)
        {
            ["normalization"] = HandWritten("normalization_kernels", CudaNormalizationKernels.GetSource(), major, minor, work, Tool),
            ["activation"] = HandWritten("activation_kernels", CudaActivationKernels.GetSource(), major, minor, work, Tool),
            ["neuralnet"] = HandWritten("neuralnet_kernels", CudaNeuralNetKernels.GetSource(), major, minor, work, Tool),
            ["skinny"] = HandWritten("skinny_gemm_kernels", CudaSkinnyGemmKernels.GetSource(), major, minor, work, Tool),
            ["attention"] = HandWritten("attention_kernels", CudaAttentionKernels.GetSource(), major, minor, work, Tool),
        };

        // A GEMM schedule's lowering, as CudaBackend.BuildUnderGemmSchedule builds it (implicit-GEMM tiles, optionally
        // split along the reduction with the planner's combine).
        Func<CodegenKernelSpec, PtxPassLowering?> ImplicitGemm(int tileM, int tileN, int split) => pass =>
        {
            PtxPassKernel Emit(CodegenKernelSpec spec)
            {
                var emitter = new PtxAffineEmitter { ImplicitGemm = true, ImplicitGemmTile = tileN, ImplicitGemmTileM = tileM };
                string text = emitter.Emit(spec, major, minor);
                return new PtxPassKernel(text, spec.Name, emitter.LaunchBlocks, (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY);
            }
            if (split <= 1) return new PtxPassLowering(new[] { Emit(pass) }, null, "implicit-gemm");
            var plan = CodegenSplitReduction.TryPlanChunkedForTiles(pass, split)
                ?? throw new NotSupportedException(pass.Name + " has no " + split + "-way split.");
            var combine = new PtxAffineEmitter { VectorTileReductions = true };
            string combineText = combine.Emit(plan.Combine, major, minor);
            return new PtxPassLowering(new[]
            {
                Emit(plan.Partial),
                new PtxPassKernel(combineText, plan.Combine.Name, combine.LaunchBlocks, (uint)combine.LaunchBlockX,
                    (uint)combine.LaunchBlockY),
            }, plan, "implicit-gemm split");
        };

        var families = new[]
        {
            new Family("LayerNorm 8192x1024", CodegenBackendPrograms.LayerNorm(8192, 1024, 1e-5), "normalization", new[] { "layernorm_forward" }),
            new Family("RmsNorm 8192x1024", CodegenBackendPrograms.RmsNorm(8192, 1024, 1e-5), "normalization", new[] { "rmsnorm_forward" }),
            new Family("GroupNorm 32x64/8x1024", CodegenBackendPrograms.GroupNorm(32, 64, 8, 1024, 1e-5), "normalization", new[] { "groupnorm_forward" }),
            new Family("BatchNorm 32x64x3136", CodegenBackendPrograms.BatchNorm2DTraining(32, 64, 3136, 1, 1e-5, 0.1), "normalization", new[] { "batchnorm_forward" }),
            new Family("LayerNormBackward 8192x1024", CodegenBackendPrograms.LayerNormBackward(8192, 1024), "normalization",
                new[] { "layernorm_backward", "layernorm_grad_params" }),
            new Family("RmsNormBackward 8192x1024", CodegenBackendPrograms.RmsNormBackward(8192, 1024), "normalization",
                new[] { "rmsnorm_backward", "rmsnorm_grad_gamma" }),
            new Family("BatchNormBackward 32x64x3136", CodegenBackendPrograms.BatchNorm2DBackward(32, 64, 3136, 1), "normalization",
                new[] { "batchnorm_backward" }),
            new Family("Softmax 8192x1024", CodegenFusedStatistics.Softmax(8192, 1024), "activation", new[] { "softmax" }),
            new Family("SoftmaxBackward 8192x1024", CodegenFusedStatistics.SoftmaxBackward(8192, 1024), "neuralnet", new[] { "softmax_backward" }),
            // The GEMM slot's declared schedules at shapes where they won, against the narrow kernel the entry ran there.
            new Family("Gemm 8192x128x256 implicit-gemm.m128n64", CodegenBackendPrograms.Gemm(1, 8192, 128, 256, false, false),
                "skinny", new[] { "skinny_gemm" }, ImplicitGemm(128, 64, 0)),
            new Family("GemmTB 4096x64x64 implicit-gemm.m64n64", CodegenBackendPrograms.Gemm(1, 4096, 64, 64, false, true),
                "skinny", new[] { "skinny_gemm" }, ImplicitGemm(64, 64, 0)),
            new Family("Gemm 512x512x16 implicit-gemm.m128n64", CodegenBackendPrograms.Gemm(1, 512, 512, 16, false, false),
                "skinny", new[] { "small_gemm_tinyk" }, ImplicitGemm(128, 64, 0)),
            new Family("GemmTA 128x128x2048 implicit-gemm.split8v.m64n64", CodegenBackendPrograms.Gemm(1, 128, 128, 2048, true, false),
                "skinny", new[] { "skinny_gemm_tn" }, ImplicitGemm(64, 64, 8)),
            // Attention: the two contractions as implicit GEMMs (CudaBackend.LowerAttentionPass), the rows by the planner.
            new Family("Attention 2x8x128x128x64", CodegenBackendPrograms.ScaledDotProductAttention(2, 8, 8, 128, 128, 64,
                0.125, CodegenAttentionMask.None, false, null), "attention", new[] { "scaled_dot_product_attention" },
                pass => pass.Name is "attention_scores" or "attention_output" ? ImplicitGemm(64, 64, 0)(pass) : null),
            new Family("Attention 2x8x128x128x64 flash", CodegenBackendPrograms.ScaledDotProductAttention(2, 8, 8, 128, 128, 64,
                0.125, CodegenAttentionMask.None, true, null), "attention", new[] { "flash_attention_v2" },
                pass => pass.Name is "attention_scores" or "attention_output" ? ImplicitGemm(64, 64, 0)(pass) : null),
            new Family("AttentionBackward 2x8x128x128x32", CodegenBackendPrograms.ScaledDotProductAttentionBackward(2, 8, 128, 128, 32,
                0.17), "neuralnet", new[] { "batched_transpose", "softmax_backward" },
                pass => pass.Name is "attention_dv" or "attention_dp" or "attention_dq" or "attention_dk"
                    ? ImplicitGemm(64, 64, 0)(pass) : null),
            new Family("Embedding 4096x32000x512", CodegenBackendPrograms.Embedding(4096, 32000, 512), "neuralnet",
                new[] { "embedding_forward" }),
            new Family("GemmTA 16x10x4096 planner", CodegenBackendPrograms.Gemm(1, 16, 10, 4096, true, false),
                "skinny", new[] { "small_gemm_rowreduce" }),
        };

        Console.WriteLine("STATIC METRICS - declared program (ptxas) vs hand-written reference (NVRTC), sm_" + arch);
        int flagged = 0;
        foreach (var family in families)
        {
            var lowering = PtxProgramLowering.Create(family.Declared, "audit", major, minor, family.LowerPass);
            var ours = new List<Metrics>();
            for (int s = 0; s < lowering.Steps.Count; s++)
            {
                var step = lowering.Steps[s];
                string ptx = Path.Combine(work, step.EntryPoint + ".ptx"), cubin = Path.ChangeExtension(ptx, ".cubin");
                File.WriteAllText(ptx, step.Ptx);
                Require(Run(Tool("ptxas"), $"-arch=sm_{arch} \"{ptx}\" -o \"{cubin}\""), "ptxas " + step.EntryPoint);
                ours.Add(Measure(cubin, Tool)[step.EntryPoint]);
            }
            var theirs = new List<Metrics>();
            foreach (string kernel in family.Kernels)
                theirs.Add(modules[family.Module].TryGetValue(kernel, out var m) ? m
                    : throw new InvalidOperationException(kernel + " is not in the " + family.Module + " module."));

            Console.WriteLine();
            Console.WriteLine(family.Name);
            Console.WriteLine("  kernel                                         regs  local  instrs   LDG (128b)   STG (128b)");
            foreach (var m in ours) Print("ours  ", m);
            foreach (var m in theirs) Print("theirs", m);
            var a = Total(ours, "declared");
            var b = Total(theirs, "hand-written");
            var failed = new List<string>();
            if (a.LocalBytes > 0) failed.Add("spills: " + a.LocalBytes + " local bytes");
            // The share of global loads that are 128-bit, not their count (user decision 2026-10-20): a hand-written
            // kernel unrolled further has more load instructions of either width, which is code, not vectorisation.
            double ShareWide(Metrics m) => m.Loads == 0 ? 1.0 : (double)m.WideLoads / m.Loads;
            if (ShareWide(a) < ShareWide(b))
                failed.Add("128-bit share of loads " + ShareWide(a).ToString("P0", CultureInfo.InvariantCulture) + " < " +
                    ShareWide(b).ToString("P0", CultureInfo.InvariantCulture));
            var reported = new List<string>();
            if (a.Registers > b.Registers) reported.Add("registers " + a.Registers + " > " + b.Registers);
            if (a.Instructions > b.Instructions) reported.Add("instructions " + a.Instructions + " > " + b.Instructions);
            Console.WriteLine("  totals: ours " + ours.Count + " kernel(s), max " + a.Registers + " regs, " + a.Instructions +
                " instrs, " + a.Loads + " LDG (" + a.WideLoads + " wide); hand-written " + theirs.Count + " kernel(s), max " +
                b.Registers + " regs, " + b.Instructions + " instrs, " + b.Loads + " LDG (" + b.WideLoads + " wide)");
            if (reported.Count != 0) Console.WriteLine("  higher (reported, not gated): " + string.Join("; ", reported));
            Console.WriteLine(failed.Count == 0 ? "  PASS" : "  FAIL: " + string.Join("; ", failed));
            if (failed.Count != 0) flagged++;
        }
        Console.WriteLine();
        Console.WriteLine(flagged == 0
            ? "All " + families.Length + " families pass: no spills, no smaller share of 128-bit loads than the hand-written kernels."
            : flagged + " of " + families.Length + " families fail the static rule.");
        if (flagged != 0) throw new InvalidOperationException(flagged + " families fail the static rule.");
    }

    private static void Print(string side, Metrics m) =>
        Console.WriteLine("  " + side + " " + m.Kernel.PadRight(38) + m.Registers.ToString(CultureInfo.InvariantCulture).PadLeft(6) +
            m.LocalBytes.ToString(CultureInfo.InvariantCulture).PadLeft(7) + m.Instructions.ToString(CultureInfo.InvariantCulture).PadLeft(8) +
            (m.Loads + " (" + m.WideLoads + ")").PadLeft(13) + (m.Stores + " (" + m.WideStores + ")").PadLeft(13));

    // Registers and local bytes per launch are the worst kernel's; code and memory instructions add up.
    private static Metrics Total(List<Metrics> all, string name)
    {
        int regs = 0, local = 0, instrs = 0, loads = 0, wideLoads = 0, stores = 0, wideStores = 0;
        foreach (var m in all)
        {
            regs = Math.Max(regs, m.Registers);
            local = Math.Max(local, m.LocalBytes);
            instrs += m.Instructions; loads += m.Loads; wideLoads += m.WideLoads; stores += m.Stores; wideStores += m.WideStores;
        }
        return new Metrics(name, regs, local, instrs, loads, wideLoads, stores, wideStores);
    }

    /// <summary>Compiles a hand-written module exactly as the backend does and measures every kernel in it.</summary>
    private static Dictionary<string, Metrics> HandWritten(string module, string source, int major, int minor, string work,
        Func<string, string> tool)
    {
        string prepared = CudaBackend.PrepareNvrtcSource(source);
        string[] options = CudaBackend.NvrtcOptions(major, minor, useFastMath: true);
        IntPtr program = IntPtr.Zero;
        Check(NvrtcNativeBindings.nvrtcCreateProgram(ref program, prepared, module + ".cu", 0, IntPtr.Zero, IntPtr.Zero), "create " + module);
        try
        {
            Check(NvrtcNativeBindings.nvrtcCompileProgram(program, options.Length, options), "compile " + module);
            Check(NvrtcNativeBindings.nvrtcGetCUBINSize(program, out UIntPtr size), "cubin size " + module);
            IntPtr buffer = Marshal.AllocHGlobal((int)size);
            try
            {
                Check(NvrtcNativeBindings.nvrtcGetCUBIN(program, buffer), "cubin " + module);
                var bytes = new byte[(int)size];
                Marshal.Copy(buffer, bytes, 0, bytes.Length);
                string cubin = Path.Combine(work, module + ".cubin");
                File.WriteAllBytes(cubin, bytes);
                return Measure(cubin, tool);
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }
        finally
        {
            NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
        }
    }

    /// <summary>Per function: resources from cuobjdump, instruction mix from nvdisasm.</summary>
    private static Dictionary<string, Metrics> Measure(string cubin, Func<string, string> tool)
    {
        var usage = Run(tool("cuobjdump"), $"--dump-resource-usage \"{cubin}\"");
        Require(usage, "cuobjdump " + cubin);
        var resources = new Dictionary<string, (int Regs, int Local)>(StringComparer.Ordinal);
        foreach (Match m in Regex.Matches(usage.StdOut, @"Function ([\w$]+):\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)"))
            resources[m.Groups[1].Value] = (int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture),
                int.Parse(m.Groups[3].Value, CultureInfo.InvariantCulture) + int.Parse(m.Groups[5].Value, CultureInfo.InvariantCulture));

        var sass = Run(tool("nvdisasm"), $"-c \"{cubin}\"");
        Require(sass, "nvdisasm " + cubin);
        var counts = new Dictionary<string, int[]>(StringComparer.Ordinal);
        int[]? current = null;
        foreach (string line in sass.StdOut.Split('\n'))
        {
            var section = Regex.Match(line, @"\.text\.([\w$]+):");
            if (section.Success)
            {
                if (!counts.TryGetValue(section.Groups[1].Value, out current))
                    counts[section.Groups[1].Value] = current = new int[5];
                continue;
            }
            if (current is null) continue;
            var instruction = Regex.Match(line, @"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z0-9_.]+)");
            if (!instruction.Success) continue;
            string opcode = instruction.Groups[1].Value;
            if (opcode.StartsWith("NOP", StringComparison.Ordinal)) continue;
            current[0]++;
            bool wide = opcode.Contains(".128", StringComparison.Ordinal);
            if (opcode.StartsWith("LDG", StringComparison.Ordinal)) { current[1]++; if (wide) current[2]++; }
            if (opcode.StartsWith("STG", StringComparison.Ordinal)) { current[3]++; if (wide) current[4]++; }
        }

        var result = new Dictionary<string, Metrics>(StringComparer.Ordinal);
        foreach (var pair in counts)
        {
            var (regs, local) = resources.TryGetValue(pair.Key, out var r) ? r : (0, 0);
            result[pair.Key] = new Metrics(pair.Key, regs, local, pair.Value[0], pair.Value[1], pair.Value[2], pair.Value[3], pair.Value[4]);
        }
        return result;
    }

    private static void Check(NvrtcResult result, string what)
    {
        if (result != NvrtcResult.Success)
            throw new InvalidOperationException("NVRTC " + what + ": " + NvrtcNativeBindings.GetErrorString(result));
    }

    private static void Require((int ExitCode, string StdOut, string StdErr) run, string what)
    {
        if (run.ExitCode != 0) throw new InvalidOperationException(what + " failed: " + run.StdErr.Trim());
    }

    private static (int ExitCode, string StdOut, string StdErr) Run(string exe, string arguments)
    {
        var info = new ProcessStartInfo(exe, arguments)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        using var process = Process.Start(info) ?? throw new InvalidOperationException("Could not start " + exe + ".");
        var stdout = process.StandardOutput.ReadToEndAsync();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        return (process.ExitCode, stdout.Result, stderr);
    }

    private static string? DefaultCudaBin()
    {
        string? path = Environment.GetEnvironmentVariable("CUDA_PATH");
        if (!string.IsNullOrEmpty(path) && Directory.Exists(Path.Combine(path, "bin"))) return Path.Combine(path, "bin");
        string root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "NVIDIA GPU Computing Toolkit", "CUDA");
        if (!Directory.Exists(root)) return null;
        var versions = Directory.GetDirectories(root, "v*");
        Array.Sort(versions, StringComparer.Ordinal);
        return versions.Length == 0 ? null : Path.Combine(versions[^1], "bin");
    }

    private static string? Value(string[] args, string flag)
    {
        for (int i = 0; i + 1 < args.Length; i++) if (args[i] == flag) return args[i + 1];
        return null;
    }
}
