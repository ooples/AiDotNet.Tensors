using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Materializes the exact SM86 convolution cubin returned by the CUDA driver
/// linker for the #841 fused Conv2D+bias+ReLU specialization. The file is
/// embedded into AiDotNet.Tensors and is also the input to nvdisasm/Nsight
/// release auditing. Mirrors <see cref="DirectPtxNormalizationArtifactTool"/>;
/// the convolution family currently promotes exactly one exact specialization,
/// so the inventory is a single content-addressed cubin.
/// </summary>
internal static class DirectPtxConvolutionArtifactTool
{
    private const string ManifestFileName = "convolution-cubins.tsv";

    private readonly record struct ExpectedArtifact(
        string BlueprintId, string PtxSha256, string SourceKey);

    internal static void Run(string outputDirectory)
    {
        if (string.IsNullOrWhiteSpace(outputDirectory))
            throw new ArgumentException("An artifact output directory is required.", nameof(outputDirectory));
        outputDirectory = Path.GetFullPath(outputDirectory);
        Directory.CreateDirectory(outputDirectory);

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "Convolution cubins can currently be released only for the experimental GA102/SM86 device.");

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

        using (var kernel = new PtxFusedConv2DNchwK1Kernel(runtime))
            Export(kernel.Audit, outputDirectory, exported, manifest);

        // The promoted register-blocked ResNet c64 1x1 specialization (beats cuDNN 1.60x).
        using (var reg = new PtxConv2DNchwK1RegBlockedKernel(runtime, RegBlockedC64))
            Export(reg.Audit, outputDirectory, exported, manifest);

        foreach (string cubinPath in Directory.GetFiles(
                     outputDirectory, "*.cubin", SearchOption.TopDirectoryOnly))
        {
            string sourceKey = Path.GetFileNameWithoutExtension(cubinPath);
            if (!exported.Contains(sourceKey))
                File.Delete(cubinPath);
        }

        string manifestPath = Path.Combine(outputDirectory, ManifestFileName);
        File.WriteAllLines(manifestPath, manifest);
        Console.WriteLine("Exported " + exported.Count.ToString(CultureInfo.InvariantCulture) +
            " unique SM86 cubin(s) for " + (manifest.Count - 5).ToString(CultureInfo.InvariantCulture) +
            " convolution specialization(s).");
        Console.WriteLine("Manifest: " + manifestPath);
    }

    internal static void Verify(string artifactDirectory)
    {
        if (string.IsNullOrWhiteSpace(artifactDirectory))
            throw new ArgumentException("An artifact directory is required.", nameof(artifactDirectory));
        artifactDirectory = Path.GetFullPath(artifactDirectory);
        string manifestPath = Path.Combine(artifactDirectory, ManifestFileName);
        if (!File.Exists(manifestPath))
            throw new FileNotFoundException("The convolution cubin manifest is missing.", manifestPath);

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
                throw new InvalidDataException("Malformed convolution cubin manifest row: " + line);
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
                "Convolution cubin inventory mismatch: expected " +
                expected.Count.ToString(CultureInfo.InvariantCulture) + ", observed " +
                manifestRows.ToString(CultureInfo.InvariantCulture) + ". Missing: " + missing);
        }

        int diskCubins = Directory.GetFiles(artifactDirectory, "*.cubin", SearchOption.TopDirectoryOnly).Length;
        if (diskCubins != expectedSourceKeys.Count ||
            !observedSourceKeys.SetEquals(expectedSourceKeys))
            throw new InvalidDataException(
                "The artifact directory must contain exactly " +
                expectedSourceKeys.Count.ToString(CultureInfo.InvariantCulture) + " cubin(s); found " +
                diskCubins.ToString(CultureInfo.InvariantCulture) + ".");
        Console.WriteLine("Verified " + expected.Count.ToString(CultureInfo.InvariantCulture) +
            " current-source PTX identity(ies) and " +
            expectedSourceKeys.Count.ToString(CultureInfo.InvariantCulture) +
            " distinct content-addressed SM86 cubin(s).");
    }

    // The exact promoted register-blocked specialization: ResNet c64 1x1
    // (N32/C64/56x56/K64), BM64/BN64/BK16, TM4/TN4.
    private static readonly Conv2DRegBlockShape RegBlockedC64 =
        new(32, 64, 64, 3136, 64, 64, 16, 4, 4);

    private static IReadOnlyList<ExpectedArtifact> CreateExpectedArtifacts()
    {
        var expected = new List<ExpectedArtifact>();
        string ptx = PtxFusedConv2DNchwK1Kernel.EmitPtx(8, 6);
        expected.Add(new ExpectedArtifact(
            PtxFusedConv2DNchwK1Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(ptx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(ptx, 8, 6)));

        string regPtx = PtxConv2DNchwK1RegBlockedKernel.EmitPtx(8, 6, RegBlockedC64);
        expected.Add(new ExpectedArtifact(
            PtxConv2DNchwK1RegBlockedKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, RegBlockedC64).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(regPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(regPtx, 8, 6)));
        return expected;
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
                "Convolution specialization did not produce a compiled cubin: " + audit.BlueprintId);
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
