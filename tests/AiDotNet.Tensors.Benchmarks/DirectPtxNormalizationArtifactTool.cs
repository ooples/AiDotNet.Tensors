using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Materializes the exact SM-specific normalization cubins returned by the
/// CUDA driver linker. These files are embedded into AiDotNet.Tensors and are
/// also the inputs to nvdisasm/Nsight release auditing.
/// </summary>
internal static class DirectPtxNormalizationArtifactTool
{
    private readonly record struct ExpectedArtifact(
        string BlueprintId, string PtxSha256, string SourceKey);

    internal static void Run(string outputDirectory)
    {
        if (string.IsNullOrWhiteSpace(outputDirectory))
            throw new ArgumentException("An artifact output directory is required.", nameof(outputDirectory));
        outputDirectory = Path.GetFullPath(outputDirectory);
        Directory.CreateDirectory(outputDirectory);

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasValidatedResidualLayerNormGelu(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "Normalization cubins can currently be released only for validated GA10x/SM86 devices.");

        var exported = new HashSet<string>(StringComparer.Ordinal);
        var manifest = new List<string>
        {
            "# generator=CUDA Driver Linker",
            "# device-fingerprint=" + runtime.DeviceFingerprint,
            "# target=sm" + runtime.ComputeCapabilityMajor.ToString(CultureInfo.InvariantCulture) +
                runtime.ComputeCapabilityMinor.ToString(CultureInfo.InvariantCulture),
            "# driver-version=" + runtime.DriverVersion.ToString(CultureInfo.InvariantCulture),
            "blueprint-id\tptx-sha256\tsource-key\tcubin-sha256\tfile"
        };
        int[] rows = [256, 2_048, 8_192];
        foreach (DirectPtxRowNormalizationOperation operation in
                 Enum.GetValues(typeof(DirectPtxRowNormalizationOperation)))
        {
            foreach (int rowCount in rows)
            {
                using var kernel = new PtxRowNormalizationD64Kernel(
                    runtime, operation, rowCount, 1e-5f);
                Export(kernel.Audit, outputDirectory, exported, manifest);
            }
        }

        foreach (DirectPtxChannelNormalizationOperation operation in
                 Enum.GetValues(typeof(DirectPtxChannelNormalizationOperation)))
        {
            using var kernel = new PtxChannelNormalizationD64Kernel(
                runtime, operation, 1e-5f, 0.1f);
            Export(kernel.Audit, outputDirectory, exported, manifest);
        }

        foreach (string cubinPath in Directory.GetFiles(
                     outputDirectory, "*.cubin", SearchOption.TopDirectoryOnly))
        {
            string sourceKey = Path.GetFileNameWithoutExtension(cubinPath);
            if (!exported.Contains(sourceKey))
                File.Delete(cubinPath);
        }

        string manifestPath = Path.Combine(outputDirectory, "normalization-cubins.tsv");
        File.WriteAllLines(manifestPath, manifest);
        Console.WriteLine("Exported " + exported.Count.ToString(CultureInfo.InvariantCulture) +
            " unique SM86 cubins for " + (manifest.Count - 5).ToString(CultureInfo.InvariantCulture) +
            " normalization specializations.");
        Console.WriteLine("Manifest: " + manifestPath);
    }

    internal static void Verify(string artifactDirectory)
    {
        if (string.IsNullOrWhiteSpace(artifactDirectory))
            throw new ArgumentException("An artifact directory is required.", nameof(artifactDirectory));
        artifactDirectory = Path.GetFullPath(artifactDirectory);
        string manifestPath = Path.Combine(artifactDirectory, "normalization-cubins.tsv");
        if (!File.Exists(manifestPath))
            throw new FileNotFoundException("The normalization cubin manifest is missing.", manifestPath);

        IReadOnlyList<ExpectedArtifact> expected = CreateExpectedArtifacts();
        var expectedByBlueprint = new Dictionary<string, ExpectedArtifact>(
            expected.Count, StringComparer.Ordinal);
        var expectedSourceKeys = new HashSet<string>(StringComparer.Ordinal);
        foreach (ExpectedArtifact artifact in expected)
        {
            if (!expectedByBlueprint.TryAdd(artifact.BlueprintId, artifact))
                throw new InvalidDataException(
                    "Two specializations produced the same blueprint id: " + artifact.BlueprintId);
            expectedSourceKeys.Add(artifact.SourceKey);
        }
        var observedBlueprints = new HashSet<string>(StringComparer.Ordinal);
        var observedSourceKeys = new HashSet<string>(StringComparer.Ordinal);
        int manifestRows = 0;
        foreach (string line in File.ReadLines(manifestPath))
        {
            if (line.Length == 0 || line[0] == '#' ||
                line.StartsWith("blueprint-id", StringComparison.Ordinal))
                continue;
            string[] columns = line.Split('\t');
            if (columns.Length != 5)
                throw new InvalidDataException("Malformed normalization cubin manifest row: " + line);
            manifestRows++;
            string blueprintId = columns[0];
            string ptxSha256 = columns[1];
            string sourceKey = columns[2];
            string cubinSha256 = columns[3];
            string fileName = columns[4];
            if (!expectedByBlueprint.TryGetValue(blueprintId, out ExpectedArtifact specialization))
                throw new InvalidDataException(
                    "The manifest contains a stale or unknown blueprint id: " + blueprintId);
            if (!observedBlueprints.Add(blueprintId))
                throw new InvalidDataException("The manifest repeats blueprint id: " + blueprintId);
            if (!string.Equals(blueprintId, specialization.BlueprintId, StringComparison.Ordinal) ||
                !string.Equals(ptxSha256, specialization.PtxSha256, StringComparison.OrdinalIgnoreCase) ||
                !string.Equals(sourceKey, specialization.SourceKey, StringComparison.Ordinal))
                throw new InvalidDataException(
                    "The manifest PTX identity is stale for " + specialization.BlueprintId + ".");
            observedSourceKeys.Add(sourceKey);
            if (!string.Equals(fileName, sourceKey + ".cubin", StringComparison.Ordinal))
                throw new InvalidDataException("The manifest cubin filename is not content-addressed: " + fileName);
            string cubinPath = Path.Combine(artifactDirectory, fileName);
            if (!File.Exists(cubinPath))
                throw new FileNotFoundException("A manifest cubin is missing.", cubinPath);
            string actualCubinHash = Sha256(File.ReadAllBytes(cubinPath));
            if (!string.Equals(cubinSha256, actualCubinHash, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Cubin SHA-256 mismatch: " + fileName);
        }

        if (manifestRows != expected.Count || observedBlueprints.Count != expected.Count)
        {
            string missing = string.Join(", ", expectedByBlueprint.Keys.Where(
                key => !observedBlueprints.Contains(key)));
            throw new InvalidDataException(
                "Normalization cubin inventory mismatch: expected " +
                expected.Count.ToString(CultureInfo.InvariantCulture) + ", observed " +
                manifestRows.ToString(CultureInfo.InvariantCulture) + ". Missing: " + missing);
        }

        int diskCubins = Directory.GetFiles(artifactDirectory, "*.cubin", SearchOption.TopDirectoryOnly).Length;
        if (diskCubins != expectedSourceKeys.Count ||
            !observedSourceKeys.SetEquals(expectedSourceKeys))
            throw new InvalidDataException(
                "The artifact directory must contain exactly " +
                expectedSourceKeys.Count.ToString(CultureInfo.InvariantCulture) + " cubins; found " +
                diskCubins.ToString(CultureInfo.InvariantCulture) + ".");
        Console.WriteLine("Verified " + expected.Count.ToString(CultureInfo.InvariantCulture) +
            " current-source PTX identities and " +
            expectedSourceKeys.Count.ToString(CultureInfo.InvariantCulture) +
            " distinct content-addressed SM86 cubins.");
    }

    private static IReadOnlyList<ExpectedArtifact> CreateExpectedArtifacts()
    {
        var expected = new List<ExpectedArtifact>();
        foreach (DirectPtxRowNormalizationOperation operation in
                 Enum.GetValues(typeof(DirectPtxRowNormalizationOperation)))
        foreach (int rows in new[] { 256, 2_048, 8_192 })
        {
            string ptx = PtxRowNormalizationD64Kernel.EmitPtx(8, 6, operation, rows, 1e-5f);
            AddExpected(expected,
                PtxRowNormalizationD64Kernel.CreateBlueprint(
                    DirectPtxArchitectureFamily.Ampere, operation, rows).Id,
                ptx);
        }
        foreach (DirectPtxChannelNormalizationOperation operation in
                 Enum.GetValues(typeof(DirectPtxChannelNormalizationOperation)))
        {
            string ptx = PtxChannelNormalizationD64Kernel.EmitPtx(8, 6, operation, 1e-5f, 0.1f);
            AddExpected(expected,
                PtxChannelNormalizationD64Kernel.CreateBlueprint(
                    DirectPtxArchitectureFamily.Ampere, operation).Id,
                ptx);
        }
        return expected;
    }

    private static void AddExpected(
        List<ExpectedArtifact> expected, string blueprintId, string ptx)
    {
        string sourceKey = DirectPtxCubinArtifactCache.ComputeSourceKey(ptx, 8, 6);
        expected.Add(new ExpectedArtifact(
            blueprintId, Sha256(Encoding.UTF8.GetBytes(ptx)), sourceKey));
    }

    private static string Sha256(byte[] bytes)
    {
        using SHA256 sha = SHA256.Create();
        return PtxCompat.ToHexString(sha.ComputeHash(bytes)).ToLowerInvariant();
    }

    private static void Export(
        DirectPtxKernelAudit audit,
        string outputDirectory,
        HashSet<string> exported,
        List<string> manifest)
    {
        if (string.IsNullOrWhiteSpace(audit.CubinSha256) ||
            string.IsNullOrWhiteSpace(audit.CubinSourceKey))
            throw new InvalidDataException(
                "Normalization specialization did not produce a compiled cubin: " + audit.BlueprintId);
        string fileName = audit.CubinSourceKey + ".cubin";
        string destination = Path.Combine(outputDirectory, fileName);
        if (exported.Add(audit.CubinSourceKey))
        {
            if (audit.ImageKind == DirectPtxModuleImageKind.EmbeddedCubin)
            {
                if (!File.Exists(destination))
                    throw new FileNotFoundException(
                        "Embedded cubin is missing from the expected source artifact directory.", destination);
            }
            else
            {
                if (string.IsNullOrWhiteSpace(audit.CubinPath) || !File.Exists(audit.CubinPath))
                    throw new FileNotFoundException(
                        "Compiled cubin cache file is unavailable for export.", audit.CubinPath);
                File.Copy(audit.CubinPath, destination, overwrite: true);
            }
        }
        manifest.Add(string.Join("\t",
            audit.BlueprintId,
            audit.PtxSha256,
            audit.CubinSourceKey,
            audit.CubinSha256,
            fileName));
    }
}
