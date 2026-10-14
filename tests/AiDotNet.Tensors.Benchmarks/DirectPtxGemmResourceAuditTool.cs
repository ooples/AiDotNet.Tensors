using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// #1261: static resources of the two direct-PTX FP32 GEMM kernels, per architecture, without a GPU. Each kernel's
/// emitted PTX goes through <c>ptxas -v</c> (registers, shared memory, stack, spills) and the resulting cubin through
/// <c>cuobjdump -sass</c> (global/shared load and store widths, FFMA count), and the theoretical occupancy follows
/// from the per-SM limits. A PTX the assembler rejects for an architecture is reported with the assembler's reason,
/// which is how the grid documents a kernel that cannot run on a GPU at all.
/// </summary>
internal static class DirectPtxGemmResourceAuditTool
{
    // A representative admitted shape; the resource footprint of both kernels does not depend on the shape.
    private const int AuditM = 1024, AuditK = 1024, AuditN = 1024;

    private static readonly Regex UsedLine = new(
        @"Used (?<registers>\d+) registers(?:, used \d+ barriers)?(?:, (?<smem>\d+) bytes smem)?",
        RegexOptions.CultureInvariant);

    private static readonly Regex SpillLine = new(
        @"(?<stack>\d+) bytes stack frame, (?<stores>\d+) bytes spill stores, (?<loads>\d+) bytes spill loads",
        RegexOptions.CultureInvariant);

    private static readonly Regex SassInstruction = new(
        @"^\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?(?<mnemonic>[A-Z][A-Z0-9_.]*)",
        RegexOptions.CultureInvariant);

    private sealed record KernelUnderAudit(string Name, int BlockThreads, Func<int, int, string> EmitPtx);

    private readonly record struct ArchitectureLimits(
        int Major, int Minor, int MaxThreadsPerSm, int MaxBlocksPerSm, int RegistersPerSm, int SharedBytesPerSm,
        int ReservedSharedBytesPerBlock)
    {
        internal string Name => $"sm_{Major}{Minor}";
    }

    // Turing TU116 (GTX 1660 Ti, this repo's local measurement GPU) and GA10x (the RTX 3080 the issue targets).
    private static readonly ArchitectureLimits[] Architectures =
    {
        new(7, 5, 1024, 16, 65536, 64 * 1024, 0),
        new(8, 6, 1536, 16, 65536, 100 * 1024, 1024),
    };

    private static readonly KernelUnderAudit[] Kernels =
    {
        new("gemm 64x64x8 (4x4/thread)", PtxGemmKernel.BlockThreads,
            (major, minor) => PtxGemmKernel.EmitPtx(major, minor, AuditM, AuditK, AuditN)),
        new("tiled 16x16x32 (2x2/thread), input-major B", PtxFusedLinearTiledKernel.BlockThreads,
            (major, minor) => PtxFusedLinearTiledKernel.EmitPtx(major, minor, AuditM, AuditK, AuditN,
                DirectPtxLinearActivation.None, DirectPtxLinearWeightLayout.InputMajor, hasBias: false)),
        new("tiled 16x16x32 (2x2/thread), output-major B", PtxFusedLinearTiledKernel.BlockThreads,
            (major, minor) => PtxFusedLinearTiledKernel.EmitPtx(major, minor, AuditM, AuditK, AuditN,
                DirectPtxLinearActivation.None, DirectPtxLinearWeightLayout.OutputMajor, hasBias: false)),
    };

    /// <summary>Runs the audit. Returns the process exit code: 0 when every PTX was audited or rejected with a reason.</summary>
    internal static int Run(string[] arguments)
    {
        if (arguments.Length < 1)
        {
            Console.Error.WriteLine(
                "Usage: --audit-direct-ptx-gemm-resources <CUDA bin directory holding ptxas and cuobjdump> [output directory]");
            return 2;
        }
        string toolDirectory = Path.GetFullPath(arguments[0]);
        string ptxas = Tool(toolDirectory, "ptxas");
        string cuobjdump = Tool(toolDirectory, "cuobjdump");
        string outputDirectory = Path.GetFullPath(arguments.Length > 1
            ? arguments[1]
            : Path.Combine("artifacts", "direct-ptx", "gemm-resources"));
        Directory.CreateDirectory(outputDirectory);
        Console.WriteLine("ptxas: " + FirstLine(RunTool(ptxas, "--version").Output, "Cuda compilation tools"));

        var rows = new List<string>
        {
            "kernel\tarch\tstatus\tregisters\tsmem_bytes\tstack_bytes\tspill_store_bytes\tspill_load_bytes\t" +
            "occupancy_warps\toccupancy_percent\tsass_instructions\tffma\tldg32\tldg64\tldg128\tldgsts128\t" +
            "lds32\tlds64\tlds128\tsts32\tsts64\tsts128\tstg32\tstg64\tstg128\treason",
        };
        foreach (KernelUnderAudit kernel in Kernels)
        foreach (ArchitectureLimits architecture in Architectures)
            rows.Add(Audit(kernel, architecture, ptxas, cuobjdump, outputDirectory));

        string report = Path.Combine(outputDirectory, "gemm-resources.tsv");
        File.WriteAllLines(report, rows);
        foreach (string row in rows) Console.WriteLine(row);
        Console.WriteLine("Evidence: " + report);
        return 0;
    }

    private static string Audit(
        KernelUnderAudit kernel, ArchitectureLimits architecture, string ptxas, string cuobjdump, string outputDirectory)
    {
        string stem = Path.Combine(outputDirectory,
            Regex.Replace(kernel.Name, "[^A-Za-z0-9]+", "-").Trim('-') + "-" + architecture.Name);
        string ptxPath = stem + ".ptx", cubinPath = stem + ".cubin";
        File.WriteAllText(ptxPath, kernel.EmitPtx(architecture.Major, architecture.Minor));

        ToolResult assembled = RunTool(ptxas, "-v", "-arch=" + architecture.Name, "-o", cubinPath, ptxPath);
        string assemblerLog = assembled.Output + assembled.Error;
        File.WriteAllText(stem + ".ptxas.log", assemblerLog);
        if (assembled.ExitCode != 0)
        {
            Match error = Regex.Match(assemblerLog, @"error\s*:\s*(?<reason>[^\r\n]+)", RegexOptions.CultureInvariant);
            string reason = error.Success ? "ptxas: " + error.Groups["reason"].Value.Trim() : "ptxas failed";
            return string.Join("\t", kernel.Name, architecture.Name, "ptxas-rejected",
                string.Join("\t", Enumerable.Repeat("n/a", 22)), reason.Replace('\t', ' '));
        }

        Match used = UsedLine.Match(assemblerLog);
        Match spill = SpillLine.Match(assemblerLog);
        if (!used.Success || !spill.Success)
            throw new InvalidDataException($"Could not parse ptxas -v for {kernel.Name} on {architecture.Name}:\n{assemblerLog}");
        int registers = Parse(used.Groups["registers"].Value);
        int sharedBytes = used.Groups["smem"].Success ? Parse(used.Groups["smem"].Value) : 0;

        ToolResult disassembled = RunTool(cuobjdump, "-sass", cubinPath);
        if (disassembled.ExitCode != 0)
            throw new InvalidOperationException($"cuobjdump failed for {cubinPath}:\n{disassembled.Error}");
        File.WriteAllText(stem + ".sass", disassembled.Output);
        SassCounts sass = CountSass(disassembled.Output);
        int warps = OccupancyWarps(architecture, kernel.BlockThreads, registers, sharedBytes);

        return string.Join("\t",
            kernel.Name, architecture.Name, "ok",
            Format(registers), Format(sharedBytes), spill.Groups["stack"].Value,
            spill.Groups["stores"].Value, spill.Groups["loads"].Value,
            Format(warps), (100d * warps * 32 / architecture.MaxThreadsPerSm).ToString("F0", CultureInfo.InvariantCulture),
            Format(sass.Instructions), Format(sass.Ffma),
            Format(sass.Get("LDG", 32)), Format(sass.Get("LDG", 64)), Format(sass.Get("LDG", 128)),
            Format(sass.Get("LDGSTS", 128)),
            Format(sass.Get("LDS", 32)), Format(sass.Get("LDS", 64)), Format(sass.Get("LDS", 128)),
            Format(sass.Get("STS", 32)), Format(sass.Get("STS", 64)), Format(sass.Get("STS", 128)),
            Format(sass.Get("STG", 32)), Format(sass.Get("STG", 64)), Format(sass.Get("STG", 128)),
            "");
    }

    /// <summary>Resident warps per SM, from the thread, block, register-file and shared-memory limits.</summary>
    private static int OccupancyWarps(ArchitectureLimits limits, int blockThreads, int registers, int sharedBytes)
    {
        int warpsPerBlock = (blockThreads + 31) / 32;
        // Registers are allocated per warp in 256-register units.
        int registersPerWarp = (registers * 32 + 255) / 256 * 256;
        int byThreads = limits.MaxThreadsPerSm / (warpsPerBlock * 32);
        int byRegisters = registersPerWarp == 0 ? int.MaxValue : limits.RegistersPerSm / (registersPerWarp * warpsPerBlock);
        int sharedPerBlock = sharedBytes + limits.ReservedSharedBytesPerBlock;
        int byShared = sharedPerBlock == 0 ? int.MaxValue : limits.SharedBytesPerSm / sharedPerBlock;
        int blocks = Math.Min(Math.Min(byThreads, byRegisters), Math.Min(byShared, limits.MaxBlocksPerSm));
        return blocks * warpsPerBlock;
    }


    private sealed class SassCounts
    {
        private readonly Dictionary<(string Family, int Bits), int> _memory = new();
        internal int Instructions { get; set; }
        internal int Ffma { get; set; }

        internal void AddMemory(string family, int bits) =>
            _memory[(family, bits)] = Get(family, bits) + 1;

        internal int Get(string family, int bits) => _memory.TryGetValue((family, bits), out int count) ? count : 0;
    }

    private static SassCounts CountSass(string sass)
    {
        var counts = new SassCounts();
        foreach (string line in sass.Split('\n'))
        {
            Match match = SassInstruction.Match(line);
            if (!match.Success) continue;
            string mnemonic = match.Groups["mnemonic"].Value;
            string[] parts = mnemonic.Split('.');
            counts.Instructions++;
            if (parts[0] == "FFMA") counts.Ffma++;
            if (parts[0] is "LDG" or "LDS" or "STS" or "STG" or "LDGSTS")
                counts.AddMemory(parts[0], AccessBits(parts));
        }
        return counts;
    }

    // Width modifiers are .64/.128 (and .U8/.S16 etc. for sub-word, counted as 32 here); none means 32 bits.
    private static int AccessBits(string[] parts)
    {
        foreach (string part in parts.Skip(1))
        {
            if (part == "128") return 128;
            if (part == "64") return 64;
        }
        return 32;
    }

    private readonly record struct ToolResult(int ExitCode, string Output, string Error);

    private static ToolResult RunTool(string fileName, params string[] arguments)
    {
        var start = new ProcessStartInfo
        {
            FileName = fileName,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (string argument in arguments) start.ArgumentList.Add(argument);
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Failed to start " + fileName);
        Task<string> error = process.StandardError.ReadToEndAsync();
        string output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        return new ToolResult(process.ExitCode, output, error.GetAwaiter().GetResult());
    }

    private static string Tool(string directory, string name)
    {
        string path = Path.Combine(directory, OperatingSystem.IsWindows() ? name + ".exe" : name);
        if (!File.Exists(path)) throw new FileNotFoundException($"{name} was not found in {directory}.", path);
        return path;
    }

    private static string FirstLine(string text, string containing)
    {
        foreach (string line in text.Split('\n'))
            if (line.Contains(containing, StringComparison.OrdinalIgnoreCase))
                return line.Trim();
        return "";
    }

    private static int Parse(string value) => int.Parse(value, NumberStyles.None, CultureInfo.InvariantCulture);

    private static string Format(int value) => value.ToString(CultureInfo.InvariantCulture);
}
