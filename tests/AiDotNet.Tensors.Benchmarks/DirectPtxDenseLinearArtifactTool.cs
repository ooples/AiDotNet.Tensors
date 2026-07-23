using System.Globalization;
using System.Security.Cryptography;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Materializes and verifies the exact SM-specific cubins for every #836
/// championship cell. The manifest is content-addressed from canonical PTX,
/// so a kernel edit makes checked-in machine-code evidence fail closed.
/// </summary>
internal static class DirectPtxDenseLinearArtifactTool
{
    private readonly record struct ExpectedArtifact(
        string BlueprintId, string PtxSha256, string SourceKey);

    internal static void Export(string outputDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);
        outputDirectory = Path.GetFullPath(outputDirectory);
        Directory.CreateDirectory(outputDirectory);
        string manifestPath = Path.Combine(outputDirectory, "dense-linear-cubins.tsv");
        HashSet<string> previousFiles = ReadManifestFiles(manifestPath);
        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "Dense-linear release cubins currently require validated GA10x/SM86 hardware.");
        using IDisposable freshCompileScope =
            DirectPtxCubinArtifactCache.EnterFreshCompileScope();

        var exported = new HashSet<string>(StringComparer.Ordinal);
        var manifest = new List<string>
        {
            "# generator=CUDA Driver Linker",
            "# pipeline-version=" + DirectPtxCubinArtifactCache.PipelineVersion.ToString(
                CultureInfo.InvariantCulture),
            "# device-fingerprint=" + runtime.DeviceFingerprint,
            "# target=sm" + runtime.ComputeCapabilityMajor.ToString(CultureInfo.InvariantCulture) +
                runtime.ComputeCapabilityMinor.ToString(CultureInfo.InvariantCulture),
            "# driver-version=" + runtime.DriverVersion.ToString(CultureInfo.InvariantCulture),
            "blueprint-id\tptx-sha256\tsource-key\tcubin-sha256\tfile\tlinker-log-sha256"
        };

        using (var kernel = new PtxFusedLinearGeluM1Kernel(runtime, 512, 2_048))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxFusedLinearTiledKernel(
            runtime, 64, 256, 256, DirectPtxLinearActivation.None,
            DirectPtxLinearWeightLayout.InputMajor, hasBias: false))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxFusedLinearTiledKernel(
            runtime, 64, 256, 256, DirectPtxLinearActivation.GeluTanh,
            DirectPtxLinearWeightLayout.InputMajor))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxFusedLinearTiledKernel(
            runtime, 64, 256, 256, DirectPtxLinearActivation.None,
            DirectPtxLinearWeightLayout.InputMajor, hasBias: false, batchCount: 4))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxFp16GemmKernel(runtime, 16, 16, 32))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxFusedLinearGeluFp16M16Kernel(runtime, 512, 2_048))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxFusedLinearGeluFp16M16Kernel(
            runtime, 512, 2_048, outputsPerBlock: 32))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxFusedLinearGeluFp16M16Kernel(runtime, 1_024, 4_096))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxFusedLinearGeluFp16M16Kernel(
            runtime, 1_024, 4_096, outputsPerBlock: 32))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxFusedLoRAKernel(runtime, 8, 256, 8, 256, 0.125f))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxFusedLinearCrossEntropyKernel(
            runtime, DirectPtxCrossEntropyTarget.Index, 4, 16, 32))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxFusedLinearBackwardKernel(
            runtime, 64, 256, 256, DirectPtxLinearActivation.Relu))
            Add(kernel.Audits[0], kernel.Ptx);
        using (var kernel = new PtxDenseVectorKernel(
            runtime, DirectPtxDenseVectorOperation.Dot, 4_096))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxDenseVectorKernel(
            runtime, DirectPtxDenseVectorOperation.Outer, 64, 128))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxBatchedVectorKernel(
            runtime, DirectPtxBatchedVectorOperation.Dot, 4, 512))
            Add(kernel.Audit, kernel.Ptx);
        using (var kernel = new PtxStridedDotKernel(runtime, 512, 512, 511, -1))
            Add(kernel.Audit, kernel.Ptx);

        var currentFiles = exported.Select(key => key + ".cubin")
            .ToHashSet(StringComparer.Ordinal);
        foreach (string previousFile in previousFiles)
        {
            if (currentFiles.Contains(previousFile) ||
                IsReferencedByAnotherManifest(
                    outputDirectory, manifestPath, previousFile))
                continue;
            string stalePath = Path.Combine(outputDirectory, previousFile);
            if (File.Exists(stalePath)) File.Delete(stalePath);
            string staleLogPath = Path.Combine(
                outputDirectory,
                Path.GetFileNameWithoutExtension(previousFile) + ".linker.txt");
            if (File.Exists(staleLogPath)) File.Delete(staleLogPath);
        }
        File.WriteAllLines(manifestPath, manifest);
        Console.WriteLine(
            $"Exported {exported.Count} exact dense-linear SM86 cubins. Manifest: {manifestPath}");
        return;

        void Add(DirectPtxKernelAudit audit, string ptx)
        {
            if (string.IsNullOrWhiteSpace(audit.CubinSourceKey))
                throw new InvalidDataException(
                    "Dense-linear specialization did not produce a compiled cubin: " + audit.BlueprintId);
            string fileName = audit.CubinSourceKey + ".cubin";
            string destination = Path.Combine(outputDirectory, fileName);
            DirectPtxCubinArtifact artifact =
                DirectPtxCubinArtifactCache.CompileExact(runtime, ptx);
            if (audit.ImageKind != DirectPtxModuleImageKind.DriverLinkedCubin ||
                !string.Equals(
                    artifact.CubinSha256, audit.CubinSha256,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Fresh-linked cubin was not deterministic for " +
                    audit.BlueprintId + ".");
            if (!string.Equals(
                    artifact.SourceKey, audit.CubinSourceKey,
                    StringComparison.Ordinal))
                throw new InvalidDataException(
                    "Fresh-linked cubin source identity changed during export: " +
                    audit.BlueprintId);
            string linkerLog = DirectPtxCubinArtifactCache.FormatLinkerLog(
                artifact.CompilerLog, runtime);
            byte[] linkerLogBytes = System.Text.Encoding.UTF8.GetBytes(linkerLog);
            if (exported.Add(audit.CubinSourceKey))
            {
                File.WriteAllBytes(destination, artifact.Image);
                File.WriteAllBytes(
                    Path.Combine(outputDirectory, audit.CubinSourceKey + ".linker.txt"),
                    linkerLogBytes);
            }
            manifest.Add(string.Join("\t", audit.BlueprintId, audit.PtxSha256,
                audit.CubinSourceKey, artifact.CubinSha256, fileName,
                Sha256(linkerLogBytes)));
        }
    }

    internal static void Verify(string artifactDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(artifactDirectory);
        artifactDirectory = Path.GetFullPath(artifactDirectory);
        string manifestPath = Path.Combine(artifactDirectory, "dense-linear-cubins.tsv");
        if (!File.Exists(manifestPath))
            throw new FileNotFoundException("Dense-linear cubin manifest is missing.", manifestPath);

        IReadOnlyList<ExpectedArtifact> expected = Expected();
        var expectedByKey = expected.ToDictionary(item => item.SourceKey, StringComparer.Ordinal);
        var observed = new HashSet<string>(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(manifestPath))
        {
            if (line.Length == 0 || line[0] == '#' ||
                line.StartsWith("blueprint-id", StringComparison.Ordinal))
                continue;
            string[] columns = line.Split('\t');
            if (columns.Length != 6)
                throw new InvalidDataException("Malformed dense-linear cubin manifest row: " + line);
            if (!expectedByKey.TryGetValue(columns[2], out ExpectedArtifact cell))
                throw new InvalidDataException("Stale or unknown PTX source key: " + columns[2]);
            if (!observed.Add(columns[2]))
                throw new InvalidDataException("Duplicate PTX source key: " + columns[2]);
            if (!string.Equals(columns[0], cell.BlueprintId, StringComparison.Ordinal))
                throw new InvalidDataException(
                    "Stale blueprint identity for " + cell.BlueprintId);
            if (!string.Equals(columns[1], cell.PtxSha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Stale PTX hash for " + cell.BlueprintId);
            if (!string.Equals(columns[4], columns[2] + ".cubin", StringComparison.Ordinal))
                throw new InvalidDataException("Cubin filename is not content-addressed: " + columns[4]);
            string cubinPath = Path.Combine(artifactDirectory, columns[4]);
            if (!File.Exists(cubinPath))
                throw new FileNotFoundException("Manifest cubin is missing.", cubinPath);
            if (!string.Equals(columns[3], Sha256(File.ReadAllBytes(cubinPath)),
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Cubin SHA-256 mismatch: " + columns[4]);
            string linkerPath = Path.Combine(
                artifactDirectory, columns[2] + ".linker.txt");
            if (!File.Exists(linkerPath))
                throw new FileNotFoundException(
                    "Manifest linker-log sidecar is missing.", linkerPath);
            byte[] linkerBytes = File.ReadAllBytes(linkerPath);
            if (!string.Equals(
                    columns[5], Sha256(linkerBytes),
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Linker-log SHA-256 mismatch: " + Path.GetFileName(linkerPath));
            ValidateLinkerLog(linkerBytes, linkerPath);
        }
        if (observed.Count != expected.Count ||
            !observed.SetEquals(expectedByKey.Keys))
            throw new InvalidDataException(
                $"Dense-linear cubin inventory mismatch: expected {expected.Count}, observed {observed.Count}.");
        Console.WriteLine(
            $"Verified {expected.Count} current-source dense-linear PTX identities and cubin hashes.");
    }

    private static HashSet<string> ReadManifestFiles(string manifestPath)
    {
        var files = new HashSet<string>(StringComparer.Ordinal);
        if (!File.Exists(manifestPath)) return files;
        foreach (string line in File.ReadLines(manifestPath))
        {
            if (line.Length == 0 || line[0] == '#' ||
                line.StartsWith("blueprint-id", StringComparison.Ordinal))
                continue;
            string[] columns = line.Split('\t');
            if (columns.Length >= 5 &&
                string.Equals(Path.GetFileName(columns[4]), columns[4],
                    StringComparison.Ordinal))
                files.Add(columns[4]);
        }
        return files;
    }

    private static bool IsReferencedByAnotherManifest(
        string directory,
        string currentManifest,
        string fileName)
    {
        foreach (string manifest in Directory.GetFiles(directory, "*-cubins.tsv"))
        {
            if (string.Equals(Path.GetFullPath(manifest), currentManifest,
                    StringComparison.OrdinalIgnoreCase))
                continue;
            if (ReadManifestFiles(manifest).Contains(fileName)) return true;
        }
        return false;
    }

    private static IReadOnlyList<ExpectedArtifact> Expected()
    {
        const int major = 8, minor = 6;
        return
        [
            Cell("fused-linear-bias-gelu-v1-Ampere-decode-fp32-m1-k512-n2048",
                PtxFusedLinearGeluM1Kernel.EmitPtx(major, minor, 512, 2_048)),
            Cell("gemm-tiled-v3-Ampere-gemm-fp32-b1-m64-k256-n256-none-inputmajor",
                PtxFusedLinearTiledKernel.EmitPtx(
                major, minor, 64, 256, 256, DirectPtxLinearActivation.None,
                DirectPtxLinearWeightLayout.InputMajor, false, 1)),
            Cell("fused-linear-tiled-v3-Ampere-gemm-fp32-b1-m64-k256-n256-gelutanh-inputmajor",
                PtxFusedLinearTiledKernel.EmitPtx(
                major, minor, 64, 256, 256, DirectPtxLinearActivation.GeluTanh,
                DirectPtxLinearWeightLayout.InputMajor, true, 1)),
            Cell("gemm-tiled-v3-Ampere-gemm-fp32-b4-m64-k256-n256-none-inputmajor",
                PtxFusedLinearTiledKernel.EmitPtx(
                major, minor, 64, 256, 256, DirectPtxLinearActivation.None,
                DirectPtxLinearWeightLayout.InputMajor, false, 4)),
            Cell("16-bit-gemm-v1-Ampere-Float16-to-Float32-b1-m16-n16-k32-taFalse-tbFalse-haFalse",
                PtxFp16GemmKernel.EmitPtx(major, minor, 16, 16, 32)),
            Cell("fused-linear-bias-gelu-v4-Ampere-tensorcore-async-ldmatrix-fp16-fp32acc-m16-nblock64-k512-n2048",
                PtxFusedLinearGeluFp16M16Kernel.EmitPtx(
                    major, minor, 512, 2_048)),
            Cell("fused-linear-bias-gelu-v4-Ampere-tensorcore-async-ldmatrix-fp16-fp32acc-m16-nblock32-k512-n2048",
                PtxFusedLinearGeluFp16M16Kernel.EmitPtx(
                    major, minor, 512, 2_048, outputsPerBlock: 32)),
            Cell("fused-linear-bias-gelu-v4-Ampere-tensorcore-async-ldmatrix-fp16-fp32acc-m16-nblock64-k1024-n4096",
                PtxFusedLinearGeluFp16M16Kernel.EmitPtx(
                    major, minor, 1_024, 4_096)),
            Cell("fused-linear-bias-gelu-v4-Ampere-tensorcore-async-ldmatrix-fp16-fp32acc-m16-nblock32-k1024-n4096",
                PtxFusedLinearGeluFp16M16Kernel.EmitPtx(
                    major, minor, 1_024, 4_096, outputsPerBlock: 32)),
            Cell("fused-lora-forward-v1-Ampere-fp32-b8-i256-r8-o256-s3e000000",
                PtxFusedLoRAKernel.EmitPtx(major, minor, 8, 256, 8, 256, 0.125f)),
            Cell("fused-linear-cross-entropy-index-v1-Ampere-fp32-b4-k16-v32",
                PtxFusedLinearCrossEntropyKernel.EmitPtx(
                major, minor, DirectPtxCrossEntropyTarget.Index, 4, 16, 32)),
            Cell("fused-linear-backward-v1-Ampere-fp32-m64-k256-n256-relu",
                PtxFusedLinearBackwardKernel.EmitPtx(
                major, minor, 64, 256, 256, DirectPtxLinearActivation.Relu)),
            Cell("dense-dot-v1-Ampere-fp32-k4096", PtxDenseVectorKernel.EmitPtx(
                major, minor, DirectPtxDenseVectorOperation.Dot, 4_096)),
            Cell("dense-outer-v1-Ampere-fp32-m64-n128", PtxDenseVectorKernel.EmitPtx(
                major, minor, DirectPtxDenseVectorOperation.Outer, 64, 128)),
            Cell("batched-dense-dot-v1-Ampere-fp32-b4-d512",
                PtxBatchedVectorKernel.EmitPtx(
                major, minor, DirectPtxBatchedVectorOperation.Dot, 4, 512)),
            Cell("strided-dot-v1-Ampere-fp32-a512-b512-o511-s-1",
                PtxStridedDotKernel.EmitPtx(
                major, minor, 512, 512, 511, -1))
        ];
    }

    private static ExpectedArtifact Cell(string blueprintId, string ptx) => new(
        blueprintId,
        DirectPtxCubinArtifactCache.ComputePtxSha256(ptx),
        DirectPtxCubinArtifactCache.ComputeSourceKey(ptx, 8, 6));

    private static string Sha256(byte[] bytes)
    {
        using SHA256 sha = SHA256.Create();
        return PtxCompat.ToHexString(sha.ComputeHash(bytes)).ToLowerInvariant();
    }

    private static void ValidateLinkerLog(byte[] bytes, string source)
    {
        string text = System.Text.Encoding.UTF8.GetString(bytes);
        string pipeline = "pipeline-version=" +
            DirectPtxCubinArtifactCache.PipelineVersion.ToString(
                CultureInfo.InvariantCulture) + "\n";
        if (!text.StartsWith(pipeline, StringComparison.Ordinal) ||
            !text.Contains("\ntarget=sm86\n", StringComparison.Ordinal) ||
            !text.Contains("\ndriver-version=", StringComparison.Ordinal) ||
            !text.Contains(
                "\ncuda-driver-linker-info-log:\n",
                StringComparison.Ordinal) ||
            !text.Contains("Function properties for 'aidotnet_", StringComparison.Ordinal))
            throw new InvalidDataException(
                "Dense-linear linker-log sidecar is incomplete: " + source);
    }
}
