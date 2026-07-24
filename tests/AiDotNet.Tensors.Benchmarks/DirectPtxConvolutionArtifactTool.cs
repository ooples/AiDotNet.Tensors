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

        // #841 coverage specializations (correctness-verified; perf timing pending idle GPU).
        using (var fusedConv3d = new PtxFusedConv3DKernel(
            runtime, FusedConv3D.Batch, FusedConv3D.InputChannels, FusedConv3D.OutputChannels,
            FusedConv3D.Depth, FusedConv3D.Height, FusedConv3D.Width,
            FusedConv3D.KernelD, FusedConv3D.KernelH, FusedConv3D.KernelW, FusedConv3D.Stride, FusedConv3D.Padding))
            Export(fusedConv3d.Audit, outputDirectory, exported, manifest);

        using (var fusedTranspose = new PtxFusedConvTranspose2DKernel(
            runtime, FusedConvTranspose2D.Batch, FusedConvTranspose2D.InputChannels, FusedConvTranspose2D.OutputChannels,
            FusedConvTranspose2D.Height, FusedConvTranspose2D.Width, FusedConvTranspose2D.KernelH, FusedConvTranspose2D.KernelW,
            FusedConvTranspose2D.Stride, FusedConvTranspose2D.Padding, FusedConvTranspose2D.OutputPadding))
            Export(fusedTranspose.Audit, outputDirectory, exported, manifest);

        using (var conv1d = new PtxConv1DKernel(
            runtime, Conv1D.Batch, Conv1D.InputChannels, Conv1D.OutputChannels, Conv1D.Length,
            Conv1D.KernelLength, Conv1D.Stride, Conv1D.Padding, Conv1D.Relu))
            Export(conv1d.Audit, outputDirectory, exported, manifest);

        using (var conv3d = new PtxConv3DKernel(
            runtime, Conv3D.Batch, Conv3D.InputChannels, Conv3D.OutputChannels, Conv3D.Depth, Conv3D.Height,
            Conv3D.Width, Conv3D.KernelD, Conv3D.KernelH, Conv3D.KernelW, Conv3D.Stride, Conv3D.Padding, Conv3D.Relu))
            Export(conv3d.Audit, outputDirectory, exported, manifest);

        using (var convTranspose = new PtxConvTranspose2DKernel(
            runtime, ConvTranspose2D.Batch, ConvTranspose2D.InputChannels, ConvTranspose2D.OutputChannels,
            ConvTranspose2D.Height, ConvTranspose2D.Width, ConvTranspose2D.KernelH, ConvTranspose2D.KernelW,
            ConvTranspose2D.Stride, ConvTranspose2D.Padding, ConvTranspose2D.OutputPadding, ConvTranspose2D.Relu))
            Export(convTranspose.Audit, outputDirectory, exported, manifest);

        using (var depthwise = new PtxDepthwiseConv2D3x3Kernel(runtime, DwN, DwC, DwH, DwW, DwRelu))
            Export(depthwise.Audit, outputDirectory, exported, manifest);

        using (var unfold = new PtxUnfold2DKernel(
            runtime, Unfold2D.Batch, Unfold2D.Channels, Unfold2D.Height, Unfold2D.Width,
            Unfold2D.KernelH, Unfold2D.KernelW, Unfold2D.Stride, Unfold2D.Padding))
            Export(unfold.Audit, outputDirectory, exported, manifest);

        using (var bwBias = new PtxConv2DBackwardBiasKernel(runtime, Bw2dN, Bw2dK, Bw2dH, Bw2dW))
            Export(bwBias.Audit, outputDirectory, exported, manifest);

        using (var bwWeight = new PtxConv2DBackwardWeight3x3Kernel(runtime, Bw2dN, Bw2dK, Bw2dC, Bw2dH, Bw2dW))
            Export(bwWeight.Audit, outputDirectory, exported, manifest);

        using (var bwInput = new PtxConv2DBackwardInput3x3Kernel(runtime, Bw2dN, Bw2dK, Bw2dC, Bw2dH, Bw2dW))
            Export(bwInput.Audit, outputDirectory, exported, manifest);

        using (var convTranspose3d = new PtxConvTranspose3DKernel(
            runtime, ConvTranspose3D.Batch, ConvTranspose3D.InputChannels, ConvTranspose3D.OutputChannels,
            ConvTranspose3D.Depth, ConvTranspose3D.Height, ConvTranspose3D.Width, ConvTranspose3D.KernelD,
            ConvTranspose3D.KernelH, ConvTranspose3D.KernelW, ConvTranspose3D.Stride, ConvTranspose3D.Padding,
            ConvTranspose3D.OutputPadding, ConvTranspose3D.Relu))
            Export(convTranspose3d.Audit, outputDirectory, exported, manifest);

        using (var dwBwInput = new PtxDepthwiseConv2D3x3BackwardInputKernel(runtime, DwBwN, DwBwC, DwBwH, DwBwW))
            Export(dwBwInput.Audit, outputDirectory, exported, manifest);

        using (var dwBwWeight = new PtxDepthwiseConv2D3x3BackwardWeightKernel(runtime, DwBwN, DwBwC, DwBwH, DwBwW))
            Export(dwBwWeight.Audit, outputDirectory, exported, manifest);

        using (var c1dBwInput = new PtxConv1DBackwardInputKernel(
            runtime, Conv1DBackward.Batch, Conv1DBackward.InputChannels, Conv1DBackward.OutputChannels,
            Conv1DBackward.Length, Conv1DBackward.KernelLength, Conv1DBackward.Stride, Conv1DBackward.Padding))
            Export(c1dBwInput.Audit, outputDirectory, exported, manifest);

        using (var c1dBwWeight = new PtxConv1DBackwardWeightKernel(
            runtime, Conv1DBackward.Batch, Conv1DBackward.InputChannels, Conv1DBackward.OutputChannels,
            Conv1DBackward.Length, Conv1DBackward.KernelLength, Conv1DBackward.Stride, Conv1DBackward.Padding))
            Export(c1dBwWeight.Audit, outputDirectory, exported, manifest);

        using (var dwc1dFwd = new PtxDepthwiseConv1DForwardKernel(
            runtime, DwConv1D.Batch, DwConv1D.Channels, DwConv1D.Length, DwConv1D.KernelLength, DwConv1D.Stride, DwConv1D.Padding))
            Export(dwc1dFwd.Audit, outputDirectory, exported, manifest);

        using (var dwc1dBwIn = new PtxDepthwiseConv1DBackwardInputKernel(
            runtime, DwConv1D.Batch, DwConv1D.Channels, DwConv1D.Length, DwConv1D.KernelLength, DwConv1D.Stride, DwConv1D.Padding))
            Export(dwc1dBwIn.Audit, outputDirectory, exported, manifest);

        using (var dwc1dBwW = new PtxDepthwiseConv1DBackwardWeightKernel(
            runtime, DwConv1D.Batch, DwConv1D.Channels, DwConv1D.Length, DwConv1D.KernelLength, DwConv1D.Stride, DwConv1D.Padding))
            Export(dwc1dBwW.Audit, outputDirectory, exported, manifest);

        using (var ct2dBwIn = new PtxConvTranspose2DBackwardInputKernel(
            runtime, ConvTranspose2DBackward.Batch, ConvTranspose2DBackward.InputChannels, ConvTranspose2DBackward.OutputChannels,
            ConvTranspose2DBackward.Height, ConvTranspose2DBackward.Width, ConvTranspose2DBackward.KernelH, ConvTranspose2DBackward.KernelW,
            ConvTranspose2DBackward.Stride, ConvTranspose2DBackward.Padding, ConvTranspose2DBackward.OutputPadding))
            Export(ct2dBwIn.Audit, outputDirectory, exported, manifest);

        using (var ct2dBwW = new PtxConvTranspose2DBackwardWeightKernel(
            runtime, ConvTranspose2DBackward.Batch, ConvTranspose2DBackward.InputChannels, ConvTranspose2DBackward.OutputChannels,
            ConvTranspose2DBackward.Height, ConvTranspose2DBackward.Width, ConvTranspose2DBackward.KernelH, ConvTranspose2DBackward.KernelW,
            ConvTranspose2DBackward.Stride, ConvTranspose2DBackward.Padding, ConvTranspose2DBackward.OutputPadding))
            Export(ct2dBwW.Audit, outputDirectory, exported, manifest);

        using (var ct3dBwIn = new PtxConvTranspose3DBackwardInputKernel(
            runtime, ConvTranspose3DBackward.Batch, ConvTranspose3DBackward.InputChannels, ConvTranspose3DBackward.OutputChannels,
            ConvTranspose3DBackward.Depth, ConvTranspose3DBackward.Height, ConvTranspose3DBackward.Width,
            ConvTranspose3DBackward.KernelD, ConvTranspose3DBackward.KernelH, ConvTranspose3DBackward.KernelW,
            ConvTranspose3DBackward.Stride, ConvTranspose3DBackward.Padding, ConvTranspose3DBackward.OutputPadding))
            Export(ct3dBwIn.Audit, outputDirectory, exported, manifest);

        using (var ct3dBwW = new PtxConvTranspose3DBackwardWeightKernel(
            runtime, ConvTranspose3DBackward.Batch, ConvTranspose3DBackward.InputChannels, ConvTranspose3DBackward.OutputChannels,
            ConvTranspose3DBackward.Depth, ConvTranspose3DBackward.Height, ConvTranspose3DBackward.Width,
            ConvTranspose3DBackward.KernelD, ConvTranspose3DBackward.KernelH, ConvTranspose3DBackward.KernelW,
            ConvTranspose3DBackward.Stride, ConvTranspose3DBackward.Padding, ConvTranspose3DBackward.OutputPadding))
            Export(ct3dBwW.Audit, outputDirectory, exported, manifest);

        using (var c3dBwIn = new PtxConv3DBackwardInputKernel(
            runtime, Conv3DBackward.Batch, Conv3DBackward.InputChannels, Conv3DBackward.OutputChannels,
            Conv3DBackward.Depth, Conv3DBackward.Height, Conv3DBackward.Width,
            Conv3DBackward.KernelD, Conv3DBackward.KernelH, Conv3DBackward.KernelW, Conv3DBackward.Stride, Conv3DBackward.Padding))
            Export(c3dBwIn.Audit, outputDirectory, exported, manifest);

        using (var c3dBwW = new PtxConv3DBackwardWeightKernel(
            runtime, Conv3DBackward.Batch, Conv3DBackward.InputChannels, Conv3DBackward.OutputChannels,
            Conv3DBackward.Depth, Conv3DBackward.Height, Conv3DBackward.Width,
            Conv3DBackward.KernelD, Conv3DBackward.KernelH, Conv3DBackward.KernelW, Conv3DBackward.Stride, Conv3DBackward.Padding))
            Export(c3dBwW.Audit, outputDirectory, exported, manifest);

        // Prune only STALE convolution cubins. This directory is SHARED with
        // sibling operators (e.g. normalization), so never delete a cubin that is
        // referenced by another operator's manifest — only our own stale ones.
        var protectedFiles = CubinsReferencedByOtherManifests(outputDirectory);
        foreach (string cubinPath in Directory.GetFiles(
                     outputDirectory, "*.cubin", SearchOption.TopDirectoryOnly))
        {
            string sourceKey = Path.GetFileNameWithoutExtension(cubinPath);
            if (!exported.Contains(sourceKey) &&
                !protectedFiles.Contains(Path.GetFileName(cubinPath)))
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

    // #841 coverage specialization shapes (match the on-device correctness tests).
    private static readonly FusedConv3DShape FusedConv3D =
        new(2, 2, 4, 4, 4, 4, 3, 3, 3, 1, 1);
    private static readonly FusedConvTranspose2DShape FusedConvTranspose2D =
        new(2, 3, 4, 4, 4, 3, 3, 2, 1, 1);
    private static readonly Conv1DShape Conv1D = new(2, 3, 4, 32, 3, 1, 1, true);
    private static readonly Conv3DShape Conv3D = new(2, 2, 4, 8, 8, 8, 3, 3, 3, 1, 1, true);
    private static readonly ConvTranspose2DShape ConvTranspose2D = new(2, 3, 4, 4, 4, 3, 3, 2, 1, 1, true);
    private const int DwN = 2, DwC = 8, DwH = 8, DwW = 8;
    private const bool DwRelu = true;
    private static readonly Unfold2DShape Unfold2D = new(2, 4, 8, 8, 3, 3, 1, 1);
    private const int Bw2dN = 2, Bw2dK = 8, Bw2dC = 4, Bw2dH = 8, Bw2dW = 8;
    private static readonly ConvTranspose3DShape ConvTranspose3D = new(2, 2, 4, 4, 4, 4, 3, 3, 3, 2, 1, 1, true);
    private const int DwBwN = 2, DwBwC = 8, DwBwH = 8, DwBwW = 8;
    private static readonly Conv1DBackwardShape Conv1DBackward = new(2, 4, 4, 32, 3, 1, 1);
    private static readonly DepthwiseConv1DShape DwConv1D = new(2, 4, 32, 3, 1, 1);
    private static readonly ConvTranspose2DBackwardShape ConvTranspose2DBackward = new(2, 4, 4, 8, 8, 3, 3, 1, 1, 0);
    private static readonly ConvTranspose3DBackwardShape ConvTranspose3DBackward = new(2, 2, 4, 4, 4, 4, 3, 3, 3, 1, 1, 0);
    private static readonly Conv3DBackwardShape Conv3DBackward = new(2, 2, 4, 4, 4, 4, 3, 3, 3, 1, 1);

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

        string fusedConv3dPtx = PtxFusedConv3DKernel.EmitPtx(8, 6, FusedConv3D);
        expected.Add(new ExpectedArtifact(
            PtxFusedConv3DKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, FusedConv3D).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(fusedConv3dPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(fusedConv3dPtx, 8, 6)));

        string fusedTransposePtx = PtxFusedConvTranspose2DKernel.EmitPtx(8, 6, FusedConvTranspose2D);
        expected.Add(new ExpectedArtifact(
            PtxFusedConvTranspose2DKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, FusedConvTranspose2D).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(fusedTransposePtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(fusedTransposePtx, 8, 6)));

        string conv1dPtx = PtxConv1DKernel.EmitPtx(8, 6, Conv1D);
        expected.Add(new ExpectedArtifact(
            PtxConv1DKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, Conv1D).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(conv1dPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(conv1dPtx, 8, 6)));

        string conv3dPtx = PtxConv3DKernel.EmitPtx(8, 6, Conv3D);
        expected.Add(new ExpectedArtifact(
            PtxConv3DKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, Conv3D).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(conv3dPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(conv3dPtx, 8, 6)));

        string convTransposePtx = PtxConvTranspose2DKernel.EmitPtx(8, 6, ConvTranspose2D);
        expected.Add(new ExpectedArtifact(
            PtxConvTranspose2DKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, ConvTranspose2D).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(convTransposePtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(convTransposePtx, 8, 6)));

        string depthwisePtx = PtxDepthwiseConv2D3x3Kernel.EmitPtx(8, 6, DwN, DwC, DwH, DwW, DwRelu);
        expected.Add(new ExpectedArtifact(
            PtxDepthwiseConv2D3x3Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, DwN, DwC, DwH, DwW, DwRelu).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(depthwisePtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(depthwisePtx, 8, 6)));

        string unfoldPtx = PtxUnfold2DKernel.EmitPtx(8, 6, Unfold2D);
        expected.Add(new ExpectedArtifact(
            PtxUnfold2DKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, Unfold2D).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(unfoldPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(unfoldPtx, 8, 6)));

        string bwBiasPtx = PtxConv2DBackwardBiasKernel.EmitPtx(8, 6, Bw2dN, Bw2dK, Bw2dH, Bw2dW);
        expected.Add(new ExpectedArtifact(
            PtxConv2DBackwardBiasKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, Bw2dN, Bw2dK, Bw2dH, Bw2dW).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(bwBiasPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(bwBiasPtx, 8, 6)));

        string bwWeightPtx = PtxConv2DBackwardWeight3x3Kernel.EmitPtx(8, 6, Bw2dN, Bw2dK, Bw2dC, Bw2dH, Bw2dW);
        expected.Add(new ExpectedArtifact(
            PtxConv2DBackwardWeight3x3Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, Bw2dN, Bw2dK, Bw2dC, Bw2dH, Bw2dW).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(bwWeightPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(bwWeightPtx, 8, 6)));

        string bwInputPtx = PtxConv2DBackwardInput3x3Kernel.EmitPtx(8, 6, Bw2dN, Bw2dK, Bw2dC, Bw2dH, Bw2dW);
        expected.Add(new ExpectedArtifact(
            PtxConv2DBackwardInput3x3Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, Bw2dN, Bw2dK, Bw2dC, Bw2dH, Bw2dW).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(bwInputPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(bwInputPtx, 8, 6)));

        string convTranspose3dPtx = PtxConvTranspose3DKernel.EmitPtx(8, 6, ConvTranspose3D);
        expected.Add(new ExpectedArtifact(
            PtxConvTranspose3DKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, ConvTranspose3D).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(convTranspose3dPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(convTranspose3dPtx, 8, 6)));

        string dwBwInputPtx = PtxDepthwiseConv2D3x3BackwardInputKernel.EmitPtx(8, 6, DwBwN, DwBwC, DwBwH, DwBwW);
        expected.Add(new ExpectedArtifact(
            PtxDepthwiseConv2D3x3BackwardInputKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, DwBwN, DwBwC, DwBwH, DwBwW).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(dwBwInputPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(dwBwInputPtx, 8, 6)));

        string dwBwWeightPtx = PtxDepthwiseConv2D3x3BackwardWeightKernel.EmitPtx(8, 6, DwBwN, DwBwC, DwBwH, DwBwW);
        expected.Add(new ExpectedArtifact(
            PtxDepthwiseConv2D3x3BackwardWeightKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, DwBwN, DwBwC, DwBwH, DwBwW).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(dwBwWeightPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(dwBwWeightPtx, 8, 6)));

        string c1dBwInputPtx = PtxConv1DBackwardInputKernel.EmitPtx(8, 6, Conv1DBackward);
        expected.Add(new ExpectedArtifact(
            PtxConv1DBackwardInputKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, Conv1DBackward).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(c1dBwInputPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(c1dBwInputPtx, 8, 6)));

        string c1dBwWeightPtx = PtxConv1DBackwardWeightKernel.EmitPtx(8, 6, Conv1DBackward);
        expected.Add(new ExpectedArtifact(
            PtxConv1DBackwardWeightKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, Conv1DBackward).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(c1dBwWeightPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(c1dBwWeightPtx, 8, 6)));

        string dwc1dFwdPtx = PtxDepthwiseConv1DForwardKernel.EmitPtx(8, 6, DwConv1D);
        expected.Add(new ExpectedArtifact(
            PtxDepthwiseConv1DForwardKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, DwConv1D).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(dwc1dFwdPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(dwc1dFwdPtx, 8, 6)));

        string dwc1dBwInPtx = PtxDepthwiseConv1DBackwardInputKernel.EmitPtx(8, 6, DwConv1D);
        expected.Add(new ExpectedArtifact(
            PtxDepthwiseConv1DBackwardInputKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, DwConv1D).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(dwc1dBwInPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(dwc1dBwInPtx, 8, 6)));

        string dwc1dBwWPtx = PtxDepthwiseConv1DBackwardWeightKernel.EmitPtx(8, 6, DwConv1D);
        expected.Add(new ExpectedArtifact(
            PtxDepthwiseConv1DBackwardWeightKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, DwConv1D).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(dwc1dBwWPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(dwc1dBwWPtx, 8, 6)));

        string ct2dBwInPtx = PtxConvTranspose2DBackwardInputKernel.EmitPtx(8, 6, ConvTranspose2DBackward);
        expected.Add(new ExpectedArtifact(
            PtxConvTranspose2DBackwardInputKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, ConvTranspose2DBackward).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(ct2dBwInPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(ct2dBwInPtx, 8, 6)));

        string ct2dBwWPtx = PtxConvTranspose2DBackwardWeightKernel.EmitPtx(8, 6, ConvTranspose2DBackward);
        expected.Add(new ExpectedArtifact(
            PtxConvTranspose2DBackwardWeightKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, ConvTranspose2DBackward).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(ct2dBwWPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(ct2dBwWPtx, 8, 6)));

        string ct3dBwInPtx = PtxConvTranspose3DBackwardInputKernel.EmitPtx(8, 6, ConvTranspose3DBackward);
        expected.Add(new ExpectedArtifact(
            PtxConvTranspose3DBackwardInputKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, ConvTranspose3DBackward).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(ct3dBwInPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(ct3dBwInPtx, 8, 6)));

        string ct3dBwWPtx = PtxConvTranspose3DBackwardWeightKernel.EmitPtx(8, 6, ConvTranspose3DBackward);
        expected.Add(new ExpectedArtifact(
            PtxConvTranspose3DBackwardWeightKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, ConvTranspose3DBackward).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(ct3dBwWPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(ct3dBwWPtx, 8, 6)));

        string c3dBwInPtx = PtxConv3DBackwardInputKernel.EmitPtx(8, 6, Conv3DBackward);
        expected.Add(new ExpectedArtifact(
            PtxConv3DBackwardInputKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, Conv3DBackward).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(c3dBwInPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(c3dBwInPtx, 8, 6)));

        string c3dBwWPtx = PtxConv3DBackwardWeightKernel.EmitPtx(8, 6, Conv3DBackward);
        expected.Add(new ExpectedArtifact(
            PtxConv3DBackwardWeightKernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere, Conv3DBackward).Id,
            DirectPtxCubinArtifactCache.ComputePtxSha256(c3dBwWPtx),
            DirectPtxCubinArtifactCache.ComputeSourceKey(c3dBwWPtx, 8, 6)));
        return expected;
    }

    // Cubin filenames owned by sibling operators (any *-cubins.tsv other than
    // ours), so a conv export never prunes normalization/other cubins.
    private static HashSet<string> CubinsReferencedByOtherManifests(string dir)
    {
        var referenced = new HashSet<string>(StringComparer.Ordinal);
        foreach (string manifest in Directory.GetFiles(dir, "*-cubins.tsv", SearchOption.TopDirectoryOnly))
        {
            if (string.Equals(Path.GetFileName(manifest), ManifestFileName, StringComparison.Ordinal))
                continue; // our own manifest — governed by `exported`
            foreach (string line in File.ReadLines(manifest))
            {
                if (line.Length == 0 || line[0] == '#' ||
                    line.StartsWith("blueprint-id", StringComparison.Ordinal))
                    continue;
                string[] cols = line.Split('\t');
                string file = cols[cols.Length - 1].Trim();
                if (file.EndsWith(".cubin", StringComparison.Ordinal)) referenced.Add(file);
            }
        }
        return referenced;
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
