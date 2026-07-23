using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>One emitted module: what it is, and the PTX that defines it.</summary>
internal sealed record DirectPtxModuleSource(string BlueprintId, string EntryPoint, string Ptx);

/// <summary>
/// Shared machinery for producing and verifying architecture-specific
/// executables for a direct-PTX family WITHOUT a GPU.
///
/// The normalization family generates its cubins with the CUDA driver linker,
/// which requires an admitted device: its manifest records a device fingerprint
/// and a driver version because the driver's JIT compiler produced the machine
/// code. That makes the artifacts unreproducible off one machine and blocks the
/// SASS evidence the promotion gate wants until that machine is free.
///
/// <c>ptxas</c> is a host compiler. Given the same PTX, the same <c>-arch</c>,
/// and the same pinned toolkit version it emits the same cubin anywhere, so the
/// artifacts become reproducible in CI and the SASS contract becomes enforceable
/// on every pull request. Shipping the cubin also means the driver never JITs
/// it, so the machine code stops varying with whatever driver is installed.
///
/// This file is intentionally family-agnostic and identical on every branch that
/// carries it: a family contributes only its module enumeration, so the shared
/// half merges cleanly when the families land together.
/// </summary>
internal static class DirectPtxCubinToolCore
{
    internal const string TargetArch = "sm_86";

    /// <summary>
    /// Compiles every module to a cubin and writes a manifest. Any register
    /// spill is a failure rather than a warning: these families all budget zero
    /// local memory, so a spill means the budget was wrong.
    /// </summary>
    internal static int Generate(
        string family,
        IEnumerable<DirectPtxModuleSource> modules,
        string ptxas,
        string outputDirectory)
    {
        if (!File.Exists(ptxas))
        {
            Console.Error.WriteLine($"ptxas not found: {ptxas}");
            return 2;
        }
        Directory.CreateDirectory(outputDirectory);

        string ptxasVersion = Run(ptxas, "--version").StdOut
            .Replace("\r", string.Empty)
            .Split('\n')
            .FirstOrDefault(line => line.Contains("release", StringComparison.Ordinal))
            ?.Trim() ?? "unknown";

        var rows = new List<string>();
        int failures = 0;
        foreach (DirectPtxModuleSource module in modules)
        {
            string ptxSha = Sha256(Encoding.UTF8.GetBytes(module.Ptx));
            string ptxPath = Path.Combine(outputDirectory, ptxSha + ".ptx");
            string cubinPath = Path.Combine(outputDirectory, ptxSha + ".cubin");
            File.WriteAllText(ptxPath, module.Ptx);

            // -O3 and the exact arch are pinned so the output depends only on
            // the PTX and the toolkit version, never on the host.
            (int exitCode, string stdout, string stderr) = Run(
                ptxas, $"-arch={TargetArch} -O3 --warn-on-spills -o \"{cubinPath}\" \"{ptxPath}\"");
            if (exitCode != 0)
            {
                Console.Error.WriteLine($"[FAIL] ptxas rejected {module.BlueprintId}");
                Console.Error.WriteLine(stderr.Trim());
                failures++;
                File.Delete(ptxPath);
                continue;
            }

            string diagnostics = (stdout + stderr).Trim();
            if (diagnostics.Contains("spill", StringComparison.OrdinalIgnoreCase))
            {
                Console.Error.WriteLine($"[FAIL] {module.BlueprintId} spills:");
                Console.Error.WriteLine(diagnostics);
                failures++;
                File.Delete(ptxPath);
                continue;
            }

            byte[] cubin = File.ReadAllBytes(cubinPath);
            rows.Add(string.Join("\t",
                module.BlueprintId, module.EntryPoint, ptxSha, Sha256(cubin),
                cubin.Length.ToString(CultureInfo.InvariantCulture), ptxSha + ".cubin"));
            File.Delete(ptxPath);
        }

        var manifest = new StringBuilder();
        manifest.AppendLine("# generator=ptxas (offline, host-only - no GPU required)");
        manifest.AppendLine($"# ptxas={ptxasVersion}");
        manifest.AppendLine($"# target={TargetArch}");
        manifest.AppendLine($"# family={family}");
        manifest.AppendLine("# Reproducible: identical PTX plus this ptxas version yields these exact cubins.");
        manifest.AppendLine("blueprint-id\tentry-point\tptx-sha256\tcubin-sha256\tcubin-bytes\tfile");
        foreach (string row in rows.OrderBy(r => r, StringComparer.Ordinal))
            manifest.AppendLine(row);
        File.WriteAllText(Path.Combine(outputDirectory, ManifestName(family)), manifest.ToString());

        Console.WriteLine($"Generated {rows.Count} {family} cubins with {ptxasVersion}.");
        if (failures > 0) Console.Error.WriteLine($"{failures} module(s) failed.");
        return failures == 0 ? 0 : 1;
    }

    /// <summary>
    /// Regenerates every cubin and compares it to the checked-in manifest, so a
    /// kernel edit that is not accompanied by a regenerated artifact fails here
    /// instead of shipping a binary that no longer matches its source.
    /// </summary>
    internal static int Verify(
        string family,
        IEnumerable<DirectPtxModuleSource> modules,
        string ptxas,
        string artifactDirectory)
    {
        string manifestPath = Path.Combine(artifactDirectory, ManifestName(family));
        if (!File.Exists(manifestPath))
        {
            Console.Error.WriteLine($"manifest not found: {manifestPath}");
            return 2;
        }

        Dictionary<string, string> expected = ReadManifest(File.ReadAllLines(manifestPath));

        string temporary = Path.Combine(
            Path.GetTempPath(), "direct-ptx-verify-" + family);
        if (Directory.Exists(temporary)) Directory.Delete(temporary, recursive: true);
        int generated = Generate(family, modules, ptxas, temporary);
        if (generated != 0) return generated;

        Dictionary<string, string> actual = ReadManifest(
            File.ReadAllLines(Path.Combine(temporary, ManifestName(family))));

        int mismatches = 0;
        foreach (KeyValuePair<string, string> pair in actual)
        {
            if (!expected.TryGetValue(pair.Key, out string? want))
            {
                Console.Error.WriteLine($"[FAIL] {pair.Key} is not in the checked-in manifest.");
                mismatches++;
            }
            else if (!string.Equals(want, pair.Value, StringComparison.Ordinal))
            {
                Console.Error.WriteLine(
                    $"[FAIL] {pair.Key} cubin changed: manifest {want}, regenerated {pair.Value}.");
                mismatches++;
            }
        }
        foreach (string stale in expected.Keys.Where(k => !actual.ContainsKey(k)))
        {
            Console.Error.WriteLine($"[FAIL] {stale} is in the manifest but no longer emitted.");
            mismatches++;
        }

        Console.WriteLine(mismatches == 0
            ? $"All {family} cubins match the checked-in manifest."
            : $"{mismatches} {family} cubin identity mismatch(es).");
        return mismatches == 0 ? 0 : 1;
    }

    /// <summary>
    /// Disassembles every cubin and enforces the zero-local-memory contract.
    /// nvdisasm is a static disassembler, so this machine-code evidence needs no
    /// device and no device fingerprint.
    /// </summary>
    internal static int AuditSass(
        string family, string nvdisasm, string artifactDirectory, string outputDirectory)
    {
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

            // LDL and STL are local loads and stores, which is exactly what a
            // register spill emits, so their absence is the contract.
            foreach (string forbidden in new[] { "LDL", "STL" })
            {
                if (sass.Contains(forbidden, StringComparison.Ordinal))
                {
                    Console.Error.WriteLine(
                        $"[FAIL] {name} contains {forbidden}: this family budgets zero local memory.");
                    violations++;
                }
            }
        }

        Console.WriteLine(violations == 0
            ? $"SASS contract holds for {cubins.Length} {family} cubin(s)."
            : $"{violations} SASS violation(s).");
        return violations == 0 ? 0 : 1;
    }

    internal static string ManifestName(string family) => family + "-cubins.tsv";

    private static Dictionary<string, string> ReadManifest(IEnumerable<string> lines)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string line in lines)
        {
            if (line.Length == 0 ||
                line.StartsWith("#", StringComparison.Ordinal) ||
                line.StartsWith("blueprint-id", StringComparison.Ordinal))
                continue;
            string[] parts = line.Split('\t');
            if (parts.Length >= 4) map[parts[0]] = parts[3];
        }
        return map;
    }

    private static string Sha256(byte[] bytes)
    {
        using var sha = SHA256.Create();
        return Convert.ToHexString(sha.ComputeHash(bytes)).ToLowerInvariant();
    }

    internal static (int ExitCode, string StdOut, string StdErr) Run(string file, string arguments)
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
