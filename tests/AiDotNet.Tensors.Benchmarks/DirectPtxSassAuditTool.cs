using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Disassembles and rejects unsafe final machine code.</summary>
internal static class DirectPtxSassAuditTool
{
    internal static void Run(
        string nvdisasmPath,
        string artifactDirectory,
        string evidenceDirectory,
        string manifestFileName = "normalization-cubins.tsv",
        string reportFileName = "normalization-sass-audit.tsv",
        string label = "normalization")
    {
        nvdisasmPath = Path.GetFullPath(nvdisasmPath);
        artifactDirectory = Path.GetFullPath(artifactDirectory);
        evidenceDirectory = Path.GetFullPath(evidenceDirectory);
        if (!File.Exists(nvdisasmPath))
            throw new FileNotFoundException("nvdisasm is required for the release SASS gate.", nvdisasmPath);
        string manifestPath = Path.Combine(artifactDirectory, manifestFileName);
        if (!File.Exists(manifestPath))
            throw new FileNotFoundException("The " + label + " cubin manifest is missing.", manifestPath);
        Directory.CreateDirectory(evidenceDirectory);

        IReadOnlyDictionary<string, ManifestCell> manifest = ReadManifest(manifestPath);
        var report = new List<string>
        {
            "source-key\tblueprint-id\tcubin-sha256\tentry\tregisters\tinstructions\tldg\tstg\tlds\tsts\tasync-copy\ttensor-core\tlocal-load\tlocal-store"
        };
        foreach (ManifestCell cell in manifest.Values.OrderBy(value => value.SourceKey, StringComparer.Ordinal))
        {
            string cubinPath = Path.Combine(artifactDirectory, cell.FileName);
            if (!File.Exists(cubinPath))
                throw new FileNotFoundException("Manifest cubin is missing.", cubinPath);
            string actualHash = Sha256(File.ReadAllBytes(cubinPath));
            if (!string.Equals(actualHash, cell.CubinSha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Cubin hash mismatch: " + cubinPath);

            string sass = Disassemble(nvdisasmPath, cubinPath);
            string sassPath = Path.Combine(evidenceDirectory, cell.SourceKey + ".sass");
            File.WriteAllText(sassPath, sass);
            SassMetrics metrics = InspectDirectEntry(sass, cubinPath);
            if (metrics.LocalLoads != 0 || metrics.LocalStores != 0)
                throw new InvalidDataException(
                    "Final SASS uses local memory for " + cell.BlueprintId +
                    ": LDL=" + metrics.LocalLoads.ToString(CultureInfo.InvariantCulture) +
                    ", STL=" + metrics.LocalStores.ToString(CultureInfo.InvariantCulture));
            report.Add(string.Join("\t",
                cell.SourceKey,
                cell.BlueprintId,
                actualHash,
                metrics.EntryPoint,
                metrics.Registers.ToString(CultureInfo.InvariantCulture),
                metrics.Instructions.ToString(CultureInfo.InvariantCulture),
                metrics.GlobalLoads.ToString(CultureInfo.InvariantCulture),
                metrics.GlobalStores.ToString(CultureInfo.InvariantCulture),
                metrics.SharedLoads.ToString(CultureInfo.InvariantCulture),
                metrics.SharedStores.ToString(CultureInfo.InvariantCulture),
                metrics.AsyncCopies.ToString(CultureInfo.InvariantCulture),
                metrics.TensorCoreInstructions.ToString(CultureInfo.InvariantCulture),
                metrics.LocalLoads.ToString(CultureInfo.InvariantCulture),
                metrics.LocalStores.ToString(CultureInfo.InvariantCulture)));
        }

        string reportPath = Path.Combine(evidenceDirectory, reportFileName);
        File.WriteAllLines(reportPath, report);
        Console.WriteLine("SASS gate passed for " + manifest.Count.ToString(CultureInfo.InvariantCulture) +
            " compiled " + label + " cubins; no direct entry uses LDL/STL.");
        Console.WriteLine("Evidence: " + reportPath);
    }

    private static string Disassemble(string nvdisasmPath, string cubinPath)
    {
        var start = new ProcessStartInfo
        {
            FileName = nvdisasmPath,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        start.ArgumentList.Add("--print-code");
        start.ArgumentList.Add("--print-instruction-encoding");
        start.ArgumentList.Add(cubinPath);
        using Process? process = Process.Start(start);
        if (process == null)
            throw new InvalidOperationException("Failed to start nvdisasm.");
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                "nvdisasm failed for " + cubinPath + " with exit code " +
                process.ExitCode.ToString(CultureInfo.InvariantCulture) + ":\n" + error);
        return output;
    }

    private static SassMetrics InspectDirectEntry(string sass, string source)
    {
        string? entry = null;
        int registers = -1;
        int instructions = 0;
        int ldg = 0, stg = 0, lds = 0, sts = 0, async = 0, tensor = 0, ldl = 0, stl = 0;
        bool active = false;
        foreach (string rawLine in sass.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries))
        {
            string line = rawLine.Trim();
            if (line.StartsWith("//--------------------- .text.", StringComparison.Ordinal))
            {
                active = line.IndexOf(".text.aidotnet_", StringComparison.Ordinal) >= 0;
                if (active)
                {
                    if (entry != null)
                        throw new InvalidDataException("Cubin contains multiple direct entry points: " + source);
                    int start = line.IndexOf(".text.", StringComparison.Ordinal) + ".text.".Length;
                    int end = line.IndexOf(' ', start);
                    entry = end > start ? line.Substring(start, end - start) : line.Substring(start);
                }
                continue;
            }
            if (!active)
                continue;
            const string registerMarker = "SHI_REGISTERS=";
            int registerIndex = line.IndexOf(registerMarker, StringComparison.Ordinal);
            if (registerIndex >= 0)
            {
                int start = registerIndex + registerMarker.Length;
                int end = start;
                while (end < line.Length && char.IsDigit(line[end])) end++;
                if (int.TryParse(line.Substring(start, end - start), NumberStyles.None,
                        CultureInfo.InvariantCulture, out int parsed))
                    registers = parsed;
            }
            if (!line.StartsWith("/*", StringComparison.Ordinal))
                continue;
            instructions++;
            if (HasMnemonic(line, "LDL")) ldl++;
            if (HasMnemonic(line, "STL")) stl++;
            if (HasMnemonic(line, "LDGSTS") || HasMnemonic(line, "CPASYNC")) async++;
            if (HasMnemonic(line, "LDG")) ldg++;
            if (HasMnemonic(line, "STG")) stg++;
            if (HasMnemonic(line, "LDS")) lds++;
            if (HasMnemonic(line, "STS")) sts++;
            if (HasMnemonic(line, "HMMA") || HasMnemonic(line, "IMMA") || HasMnemonic(line, "MMA")) tensor++;
        }
        if (entry == null || registers < 0 || instructions == 0)
            throw new InvalidDataException("No auditable AiDotNet direct entry point was found in " + source);
        return new SassMetrics(entry, registers, instructions, ldg, stg, lds, sts, async, tensor, ldl, stl);
    }

    private static bool HasMnemonic(string instruction, string mnemonic)
    {
        int index = instruction.IndexOf(mnemonic, StringComparison.Ordinal);
        while (index >= 0)
        {
            bool before = index == 0 || !char.IsLetterOrDigit(instruction[index - 1]);
            int afterIndex = index + mnemonic.Length;
            bool after = afterIndex == instruction.Length ||
                instruction[afterIndex] == '.' || char.IsWhiteSpace(instruction[afterIndex]);
            if (before && after) return true;
            index = instruction.IndexOf(mnemonic, index + 1, StringComparison.Ordinal);
        }
        return false;
    }

    private static IReadOnlyDictionary<string, ManifestCell> ReadManifest(string path)
    {
        var result = new Dictionary<string, ManifestCell>(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line) || line[0] == '#' ||
                line.StartsWith("blueprint-id", StringComparison.Ordinal))
                continue;
            string[] columns = line.Split('\t');
            if (columns.Length != 5)
                throw new InvalidDataException("Malformed normalization cubin manifest row: " + line);
            var cell = new ManifestCell(columns[0], columns[2], columns[3], columns[4]);
            if (!result.TryAdd(cell.SourceKey, cell))
            {
                ManifestCell existing = result[cell.SourceKey];
                if (!string.Equals(existing.CubinSha256, cell.CubinSha256,
                        StringComparison.OrdinalIgnoreCase) ||
                    !string.Equals(existing.FileName, cell.FileName, StringComparison.Ordinal))
                    throw new InvalidDataException(
                        "Shared cubin source key has inconsistent manifest data: " + cell.SourceKey);
                result[cell.SourceKey] = existing with
                {
                    BlueprintId = existing.BlueprintId + "," + cell.BlueprintId
                };
            }
        }
        return result;
    }

    private static string Sha256(byte[] bytes)
    {
        using SHA256 sha = SHA256.Create();
        return PtxCompat.ToHexString(sha.ComputeHash(bytes)).ToLowerInvariant();
    }

    private sealed record ManifestCell(
        string BlueprintId, string SourceKey, string CubinSha256, string FileName);

    private sealed record SassMetrics(
        string EntryPoint,
        int Registers,
        int Instructions,
        int GlobalLoads,
        int GlobalStores,
        int SharedLoads,
        int SharedStores,
        int AsyncCopies,
        int TensorCoreInstructions,
        int LocalLoads,
        int LocalStores);
}
