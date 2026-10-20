// Copyright (c) AiDotNet. All rights reserved.
// --kernel-routes --static: the static-metric half of the #890 phase-4 gate. For each routed family, at its bench
// shape, the declared program as the backend lowers it (PtxProgramLowering, compiled by ptxas) against the reference
// hand-written kernels it replaces (their CUDA C, compiled by NVRTC with the backend's own source preparation and
// options): registers, local bytes (spills), static SASS instructions, global loads and stores and how many of each
// are 128-bit. Every metric where the declared side is worse is flagged.
//
// Static counts are code, not work: an unrolled row kernel has more instructions than a loop over the same row, and
// a fused program is one kernel standing in for several. Both sides are reported in full so a flag can be judged.
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

    private sealed record Family(string Name, CodegenProgram Declared, string Module, string[] Kernels);

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
        };

        Console.WriteLine("STATIC METRICS - declared program (ptxas) vs hand-written reference (NVRTC), sm_" + arch);
        int flagged = 0;
        foreach (var family in families)
        {
            var lowering = PtxProgramLowering.Create(family.Declared, "audit", major, minor);
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
            var worse = new List<string>();
            if (a.Registers > b.Registers) worse.Add("registers " + a.Registers + " > " + b.Registers);
            if (a.LocalBytes > b.LocalBytes) worse.Add("local bytes " + a.LocalBytes + " > " + b.LocalBytes);
            if (a.Instructions > b.Instructions) worse.Add("instructions " + a.Instructions + " > " + b.Instructions);
            if (a.WideLoads < b.WideLoads) worse.Add("128-bit loads " + a.WideLoads + " < " + b.WideLoads);
            Console.WriteLine("  totals: ours " + ours.Count + " kernel(s), max " + a.Registers + " regs, " + a.Instructions +
                " instrs, " + a.Loads + " LDG (" + a.WideLoads + " wide); hand-written " + theirs.Count + " kernel(s), max " +
                b.Registers + " regs, " + b.Instructions + " instrs, " + b.Loads + " LDG (" + b.WideLoads + " wide)");
            Console.WriteLine(worse.Count == 0 ? "  no metric worse" : "  WORSE: " + string.Join("; ", worse));
            if (worse.Count != 0) flagged++;
        }
        Console.WriteLine();
        Console.WriteLine(flagged + " of " + families.Length + " families have a metric worse than the hand-written reference.");
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
