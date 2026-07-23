using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Disassembles the exact manifest cubins and rejects local-memory machine
/// instructions. Multi-entry modules (notably fused-linear backward) are
/// audited entry-by-entry instead of hiding behind aggregate module metadata.
/// </summary>
internal static class DirectPtxDenseLinearSassAuditTool
{
    internal static void Run(
        string nvdisasmPath,
        string artifactDirectory,
        string evidenceDirectory)
    {
        nvdisasmPath = Path.GetFullPath(nvdisasmPath);
        artifactDirectory = Path.GetFullPath(artifactDirectory);
        evidenceDirectory = Path.GetFullPath(evidenceDirectory);
        if (!File.Exists(nvdisasmPath))
            throw new FileNotFoundException("nvdisasm is required for the release SASS gate.", nvdisasmPath);
        string manifestPath = Path.Combine(artifactDirectory, "dense-linear-cubins.tsv");
        if (!File.Exists(manifestPath))
            throw new FileNotFoundException("Dense-linear cubin manifest is missing.", manifestPath);
        Directory.CreateDirectory(evidenceDirectory);

        IReadOnlyList<ManifestCell> manifest = ReadManifest(manifestPath);
        var report = new List<string>
        {
            "source-key\tblueprint-id\tcubin-sha256\tentry\tregisters\tinstructions\tldg\tstg\tlds\tsts\tldmatrix\tasync-copy\ttensor-core\tlocal-load\tlocal-store"
        };
        foreach (ManifestCell cell in manifest.OrderBy(value => value.SourceKey, StringComparer.Ordinal))
        {
            string cubinPath = Path.Combine(artifactDirectory, cell.FileName);
            if (!File.Exists(cubinPath))
                throw new FileNotFoundException("Manifest cubin is missing.", cubinPath);
            string actualHash = Sha256(File.ReadAllBytes(cubinPath));
            if (!string.Equals(actualHash, cell.CubinSha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Cubin hash mismatch: " + cubinPath);
            string sass = Disassemble(nvdisasmPath, cubinPath);
            File.WriteAllText(Path.Combine(evidenceDirectory, cell.SourceKey + ".sass"), sass);
            foreach (SassMetrics metrics in InspectDirectEntries(sass, cubinPath))
            {
                if (metrics.LocalLoads != 0 || metrics.LocalStores != 0)
                    throw new InvalidDataException(
                        $"Final SASS uses local memory for {cell.BlueprintId}/{metrics.EntryPoint}: " +
                        $"LDL={metrics.LocalLoads}, STL={metrics.LocalStores}");
                if (cell.BlueprintId.Contains("tensorcore-async", StringComparison.Ordinal) &&
                    (metrics.AsyncCopies == 0 || metrics.TensorCoreInstructions == 0))
                    throw new InvalidDataException(
                        $"Tensor-Core SASS contract regressed for {cell.BlueprintId}/{metrics.EntryPoint}: " +
                        $"async={metrics.AsyncCopies}, tensor={metrics.TensorCoreInstructions}");
                if (cell.BlueprintId.Contains("async-ldmatrix", StringComparison.Ordinal) &&
                    metrics.MatrixLoads == 0)
                    throw new InvalidDataException(
                        $"Warp-collective shared-load contract regressed for " +
                        $"{cell.BlueprintId}/{metrics.EntryPoint}: LDSM=0");
                if ((cell.BlueprintId.StartsWith("gemm-tiled-v3-", StringComparison.Ordinal) ||
                     cell.BlueprintId.StartsWith("fused-linear-tiled-v3-", StringComparison.Ordinal)) &&
                    metrics.AsyncCopies == 0)
                    throw new InvalidDataException(
                        $"Async tiled-GEMM SASS contract regressed for {cell.BlueprintId}/{metrics.EntryPoint}.");
                report.Add(string.Join("\t",
                    cell.SourceKey, cell.BlueprintId, actualHash, metrics.EntryPoint,
                    metrics.Registers.ToString(CultureInfo.InvariantCulture),
                    metrics.Instructions.ToString(CultureInfo.InvariantCulture),
                    metrics.GlobalLoads.ToString(CultureInfo.InvariantCulture),
                    metrics.GlobalStores.ToString(CultureInfo.InvariantCulture),
                    metrics.SharedLoads.ToString(CultureInfo.InvariantCulture),
                    metrics.SharedStores.ToString(CultureInfo.InvariantCulture),
                    metrics.MatrixLoads.ToString(CultureInfo.InvariantCulture),
                    metrics.AsyncCopies.ToString(CultureInfo.InvariantCulture),
                    metrics.TensorCoreInstructions.ToString(CultureInfo.InvariantCulture),
                    metrics.LocalLoads.ToString(CultureInfo.InvariantCulture),
                    metrics.LocalStores.ToString(CultureInfo.InvariantCulture)));
            }
        }
        const int expectedEntryPoints = 18;
        if (report.Count - 1 != expectedEntryPoints)
            throw new InvalidDataException(
                $"Dense-linear SASS inventory expected {expectedEntryPoints} entry points; " +
                $"found {report.Count - 1}.");
        string reportPath = Path.Combine(evidenceDirectory, "dense-linear-sass-audit.tsv");
        File.WriteAllLines(reportPath, report);
        Console.WriteLine(
            $"SASS gate passed for {manifest.Count} dense-linear cubins and {report.Count - 1} entries; no LDL/STL.");
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
                $"nvdisasm failed for {cubinPath} with exit code {process.ExitCode}:\n{error}");
        return output;
    }

    private static IReadOnlyList<SassMetrics> InspectDirectEntries(string sass, string source)
    {
        var result = new List<SassMetrics>();
        MutableMetrics? current = null;
        foreach (string rawLine in sass.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries))
        {
            string line = rawLine.Trim();
            if (line.StartsWith("//--------------------- .text.", StringComparison.Ordinal))
            {
                Flush();
                int marker = line.IndexOf(".text.aidotnet_", StringComparison.Ordinal);
                if (marker >= 0)
                {
                    int start = marker + ".text.".Length;
                    int end = line.IndexOf(' ', start);
                    current = new MutableMetrics(
                        end > start ? line.Substring(start, end - start) : line.Substring(start));
                }
                continue;
            }
            if (current == null)
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
                    current.Registers = parsed;
            }
            if (!line.StartsWith("/*", StringComparison.Ordinal))
                continue;
            current.Instructions++;
            if (HasMnemonic(line, "LDL")) current.LocalLoads++;
            if (HasMnemonic(line, "STL")) current.LocalStores++;
            if (HasMnemonic(line, "LDGSTS") || HasMnemonic(line, "CPASYNC")) current.AsyncCopies++;
            if (HasMnemonic(line, "LDG")) current.GlobalLoads++;
            if (HasMnemonic(line, "STG")) current.GlobalStores++;
            if (HasMnemonic(line, "LDS")) current.SharedLoads++;
            if (HasMnemonic(line, "STS")) current.SharedStores++;
            if (HasMnemonic(line, "LDSM")) current.MatrixLoads++;
            if (HasMnemonic(line, "HMMA") || HasMnemonic(line, "IMMA") ||
                HasMnemonic(line, "MMA")) current.TensorCoreInstructions++;
        }
        Flush();
        if (result.Count == 0)
            throw new InvalidDataException("No auditable AiDotNet direct entry point was found in " + source);
        return result;

        void Flush()
        {
            if (current == null)
                return;
            if (current.Registers < 0 || current.Instructions == 0)
                throw new InvalidDataException(
                    "Incomplete SASS metadata for " + current.EntryPoint + " in " + source);
            result.Add(current.Freeze());
            current = null;
        }
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
            if (before && after)
                return true;
            index = instruction.IndexOf(mnemonic, index + 1, StringComparison.Ordinal);
        }
        return false;
    }

    private static IReadOnlyList<ManifestCell> ReadManifest(string path)
    {
        var result = new List<ManifestCell>();
        var keys = new HashSet<string>(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line) || line[0] == '#' ||
                line.StartsWith("blueprint-id", StringComparison.Ordinal))
                continue;
            string[] columns = line.Split('\t');
            if (columns.Length != 6)
                throw new InvalidDataException("Malformed dense-linear cubin manifest row: " + line);
            if (!keys.Add(columns[2]))
                throw new InvalidDataException("Duplicate manifest source key: " + columns[2]);
            result.Add(new ManifestCell(columns[0], columns[2], columns[3], columns[4]));
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

    private sealed class MutableMetrics(string entryPoint)
    {
        internal string EntryPoint { get; } = entryPoint;
        internal int Registers { get; set; } = -1;
        internal int Instructions { get; set; }
        internal int GlobalLoads { get; set; }
        internal int GlobalStores { get; set; }
        internal int SharedLoads { get; set; }
        internal int SharedStores { get; set; }
        internal int MatrixLoads { get; set; }
        internal int AsyncCopies { get; set; }
        internal int TensorCoreInstructions { get; set; }
        internal int LocalLoads { get; set; }
        internal int LocalStores { get; set; }

        internal SassMetrics Freeze() => new(
            EntryPoint, Registers, Instructions, GlobalLoads, GlobalStores,
            SharedLoads, SharedStores, MatrixLoads, AsyncCopies,
            TensorCoreInstructions, LocalLoads, LocalStores);
    }

    private sealed record SassMetrics(
        string EntryPoint,
        int Registers,
        int Instructions,
        int GlobalLoads,
        int GlobalStores,
        int SharedLoads,
        int SharedStores,
        int MatrixLoads,
        int AsyncCopies,
        int TensorCoreInstructions,
        int LocalLoads,
        int LocalStores);
}
