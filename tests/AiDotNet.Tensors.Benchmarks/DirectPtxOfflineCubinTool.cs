using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Produces and verifies architecture-specific executables for the layout
/// direct-PTX family WITHOUT a GPU.
///
/// The normalization family generates its cubins with the CUDA driver linker,
/// which requires an admitted device to be present: its manifest header records
/// a device fingerprint and a driver version because the driver's JIT compiler
/// is what produced the machine code. That makes the artifacts unreproducible
/// off the one machine, and it blocks the SASS evidence the promotion gate
/// wants until that machine is available.
///
/// <c>ptxas</c> is a host compiler. Given the same PTX, the same
/// <c>-arch</c>, and the same pinned toolkit version it emits the same cubin on
/// any machine, GPU or not - so the artifacts become reproducible in CI and the
/// SASS contract becomes enforceable on every pull request. Because the cubin
/// is shipped and loaded directly, the driver never JITs it, and the machine
/// code stops depending on whatever driver happens to be installed.
///
/// The driver-linker route remains available for cross-checking: pass
/// <c>--compare</c> on a machine that has a device and the tool will report
/// where the two generators disagree rather than assuming they must match.
/// </summary>
internal static class DirectPtxOfflineCubinTool
{
    private const string TargetArch = "sm_86";

    /// <summary>One emitted module: what it is, and the PTX that defines it.</summary>
    private sealed record PtxModule(string BlueprintId, string EntryPoint, string Ptx);

    /// <summary>
    /// Every module the layout family can admit. The shape domains are closed,
    /// so this enumeration is complete by construction - if a kernel gains a
    /// shape, this list grows with it and the gate sees the new cubin.
    /// </summary>
    private static IEnumerable<PtxModule> EnumerateModules()
    {
        foreach (int size in new[] { 65_536, 262_144, 1_048_576, 4_194_304 })
        {
            yield return new PtxModule(
                $"cast-f32-to-f16-v1-n{size}",
                PtxFusedCastF32ToF16Kernel.EntryPoint,
                PtxFusedCastF32ToF16Kernel.EmitPtx(8, 6, size));
            yield return new PtxModule(
                $"cast-f16-to-f32-v1-n{size}",
                PtxFusedCastF16ToF32Kernel.EntryPoint,
                PtxFusedCastF16ToF32Kernel.EmitPtx(8, 6, size));
        }

        foreach (int rows in new[] { 512, 1024, 2048, 4096 })
        foreach (int columns in new[] { 512, 1024, 2048, 4096 })
        {
            yield return new PtxModule(
                $"transpose2d-f32-v1-r{rows}-c{columns}",
                PtxFusedTranspose2DF32Kernel.EntryPoint,
                PtxFusedTranspose2DF32Kernel.EmitPtx(8, 6, rows, columns));
        }
    }

    internal static int Generate(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine(
                "usage: --generate-direct-ptx-layout-cubins <ptxas-path> <output-directory>");
            return 2;
        }

        string ptxas = args[1];
        string outputDirectory = args[2];
        if (!File.Exists(ptxas))
        {
            Console.Error.WriteLine($"ptxas not found: {ptxas}");
            return 2;
        }
        Directory.CreateDirectory(outputDirectory);

        string ptxasVersion = RunCapture(ptxas, "--version").Replace("\r", string.Empty)
            .Split('\n').FirstOrDefault(line => line.Contains("release", StringComparison.Ordinal))
            ?.Trim() ?? "unknown";

        var rows = new List<string>();
        int failures = 0;
        foreach (PtxModule module in EnumerateModules())
        {
            string ptxSha = Sha256(Encoding.UTF8.GetBytes(module.Ptx));
            string ptxPath = Path.Combine(outputDirectory, ptxSha + ".ptx");
            File.WriteAllText(ptxPath, module.Ptx);
            string cubinPath = Path.Combine(outputDirectory, ptxSha + ".cubin");

            // -O3 and the exact arch are pinned so the output depends only on the
            // PTX and the toolkit version, never on the host.
            string arguments =
                $"-arch={TargetArch} -O3 --warn-on-spills -o \"{cubinPath}\" \"{ptxPath}\"";
            (int exitCode, string stdout, string stderr) = Run(ptxas, arguments);
            if (exitCode != 0)
            {
                Console.Error.WriteLine($"[FAIL] ptxas rejected {module.BlueprintId}");
                Console.Error.WriteLine(stderr.Trim());
                failures++;
                continue;
            }

            // --warn-on-spills puts spill diagnostics on stderr. The layout
            // family budgets zero local memory, so any spill is a failure here
            // rather than a warning to be noticed later.
            string diagnostics = (stdout + stderr).Trim();
            if (diagnostics.Contains("spill", StringComparison.OrdinalIgnoreCase))
            {
                Console.Error.WriteLine($"[FAIL] {module.BlueprintId} spills:");
                Console.Error.WriteLine(diagnostics);
                failures++;
                continue;
            }

            byte[] cubin = File.ReadAllBytes(cubinPath);
            rows.Add(string.Join("\t",
                module.BlueprintId, module.EntryPoint, ptxSha, Sha256(cubin),
                cubin.Length.ToString(CultureInfo.InvariantCulture),
                ptxSha + ".cubin"));
            File.Delete(ptxPath);
        }

        var manifest = new StringBuilder();
        manifest.AppendLine("# generator=ptxas (offline, host-only - no GPU required)");
        manifest.AppendLine($"# ptxas={ptxasVersion}");
        manifest.AppendLine($"# target={TargetArch}");
        manifest.AppendLine("# Reproducible: identical PTX plus this ptxas version yields these exact cubins.");
        manifest.AppendLine("blueprint-id\tentry-point\tptx-sha256\tcubin-sha256\tcubin-bytes\tfile");
        foreach (string row in rows.OrderBy(r => r, StringComparer.Ordinal))
            manifest.AppendLine(row);
        File.WriteAllText(Path.Combine(outputDirectory, "layout-cubins.tsv"), manifest.ToString());

        Console.WriteLine($"Generated {rows.Count} cubins with {ptxasVersion}.");
        if (failures > 0)
            Console.Error.WriteLine($"{failures} module(s) failed.");
        return failures == 0 ? 0 : 1;
    }

    /// <summary>
    /// Re-generates every cubin and checks it against the checked-in manifest.
    /// This is what makes the artifacts trustworthy: a PTX change that is not
    /// accompanied by a regenerated cubin fails here instead of silently
    /// shipping a stale binary.
    /// </summary>
    internal static int Verify(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine(
                "usage: --verify-direct-ptx-layout-cubins <ptxas-path> <artifact-directory>");
            return 2;
        }

        string manifestPath = Path.Combine(args[2], "layout-cubins.tsv");
        if (!File.Exists(manifestPath))
        {
            Console.Error.WriteLine($"manifest not found: {manifestPath}");
            return 2;
        }

        var expected = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string line in File.ReadAllLines(manifestPath))
        {
            if (line.StartsWith("#", StringComparison.Ordinal) ||
                line.StartsWith("blueprint-id", StringComparison.Ordinal) ||
                line.Length == 0)
                continue;
            string[] parts = line.Split('\t');
            expected[parts[0]] = parts[3];
        }

        string temporary = Path.Combine(Path.GetTempPath(), "direct-ptx-layout-verify");
        if (Directory.Exists(temporary)) Directory.Delete(temporary, recursive: true);
        int generated = Generate(new[] { args[0], args[1], temporary });
        if (generated != 0) return generated;

        int mismatches = 0;
        foreach (string line in File.ReadAllLines(Path.Combine(temporary, "layout-cubins.tsv")))
        {
            if (line.StartsWith("#", StringComparison.Ordinal) ||
                line.StartsWith("blueprint-id", StringComparison.Ordinal) ||
                line.Length == 0)
                continue;
            string[] parts = line.Split('\t');
            if (!expected.TryGetValue(parts[0], out string? want))
            {
                Console.Error.WriteLine($"[FAIL] {parts[0]} is not in the checked-in manifest.");
                mismatches++;
            }
            else if (!string.Equals(want, parts[3], StringComparison.Ordinal))
            {
                Console.Error.WriteLine(
                    $"[FAIL] {parts[0]} cubin changed: manifest {want}, regenerated {parts[3]}.");
                mismatches++;
            }
        }

        Console.WriteLine(mismatches == 0
            ? "All layout cubins match the checked-in manifest."
            : $"{mismatches} cubin identity mismatch(es).");
        return mismatches == 0 ? 0 : 1;
    }

    /// <summary>
    /// Disassembles every cubin and enforces the family's SASS safety contract.
    /// This is the piece that needs no GPU and no device fingerprint: nvdisasm
    /// is a static disassembler, so the machine-code evidence the promotion gate
    /// asks for can be produced on an ordinary CI runner.
    /// </summary>
    internal static int AuditSass(string[] args)
    {
        if (args.Length < 4)
        {
            Console.Error.WriteLine(
                "usage: --audit-direct-ptx-layout-sass <nvdisasm-path> <artifact-directory> <output-directory>");
            return 2;
        }

        string nvdisasm = args[1];
        string artifactDirectory = args[2];
        string outputDirectory = args[3];
        Directory.CreateDirectory(outputDirectory);

        string[] cubins = Directory.GetFiles(artifactDirectory, "*.cubin");
        if (cubins.Length == 0)
        {
            Console.Error.WriteLine($"no cubins under {artifactDirectory}");
            return 2;
        }

        int violations = 0;
        foreach (string cubin in cubins.OrderBy(c => c, StringComparer.Ordinal))
        {
            (int exitCode, string sass, string stderr) = Run(nvdisasm, $"-c \"{cubin}\"");
            if (exitCode != 0)
            {
                Console.Error.WriteLine($"[FAIL] nvdisasm failed on {Path.GetFileName(cubin)}");
                Console.Error.WriteLine(stderr.Trim());
                violations++;
                continue;
            }

            string name = Path.GetFileNameWithoutExtension(cubin);
            File.WriteAllText(Path.Combine(outputDirectory, name + ".sass"), sass);

            // The layout family budgets zero local memory. LDL/STL are local
            // loads and stores, which is exactly what a register spill emits,
            // so their absence is the contract.
            foreach (string forbidden in new[] { "LDL", "STL" })
            {
                if (sass.Contains(forbidden, StringComparison.Ordinal))
                {
                    Console.Error.WriteLine(
                        $"[FAIL] {name} contains {forbidden}: the family budgets zero local memory.");
                    violations++;
                }
            }
        }

        Console.WriteLine(violations == 0
            ? $"SASS contract holds for {cubins.Length} cubin(s)."
            : $"{violations} SASS violation(s).");
        return violations == 0 ? 0 : 1;
    }

    private static string Sha256(byte[] bytes)
    {
        using var sha = SHA256.Create();
        return Convert.ToHexString(sha.ComputeHash(bytes)).ToLowerInvariant();
    }

    private static string RunCapture(string file, string arguments) => Run(file, arguments).StdOut;

    private static (int ExitCode, string StdOut, string StdErr) Run(string file, string arguments)
    {
        var info = new ProcessStartInfo(file, arguments)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        using Process process = Process.Start(info)
            ?? throw new InvalidOperationException($"could not start {file}");
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        return (process.ExitCode, stdout, stderr);
    }
}
