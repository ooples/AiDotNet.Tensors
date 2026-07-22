#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class DirectPtxVisionBoxIouTests
{
    public static IEnumerable<object[]> VisionDefinitions()
    {
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.GeneralizedBoxIou, 256, 256)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.DistanceBoxIou, 1024, 256)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.CompleteBoxIou, 1024, 1024)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.BoxArea, 256)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.BoxConvert, 256, 0, 2)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.IoULoss, 256)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.GIoULoss, 256)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.DIoULoss, 256)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.CIoULoss, 256)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.IoULossBackward, 256)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.GIoULossBackward, 256)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.DIoULossBackward, 256)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.CIoULossBackward, 256)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.IouFamilyBackwardA, 256, 256, 0)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.IouFamilyBackwardB, 256, 256, 3)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.Nms, 256,
            Flags: 1, ScalarBits: BitConverter.SingleToInt32Bits(0.5f))];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.Nms, 256,
            ScalarBits: BitConverter.SingleToInt32Bits(0.5f))];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.MasksToBoxes, 256, 28, 28)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.RoiAlign,
            1, 256, 56, 56, 256, 7, 7, 256, 2 | 0x100,
            BitConverter.SingleToInt32Bits(0.25f))];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.RoiPool,
            1, 256, 56, 56, 256, 7, 7, 256, 0,
            BitConverter.SingleToInt32Bits(0.25f))];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.PsRoiAlign,
            1, 196, 56, 56, 256, 7, 7, 4, 2,
            BitConverter.SingleToInt32Bits(0.25f))];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.PsRoiPool,
            1, 196, 56, 56, 256, 7, 7, 4, 0,
            BitConverter.SingleToInt32Bits(0.25f))];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.Cross3, 256, 1)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.Meshgrid2D, 256, 256, Flags: 0)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.Meshgrid2D, 256, 256, Flags: 1)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.Meshgrid2D, 256, 256, Flags: 2)];
        yield return [new DirectPtxVisionSpec(DirectPtxVisionOperation.Meshgrid2D, 256, 256, Flags: 3)];
    }

    private static IEnumerable<DirectPtxVisionSpec> AllAdmittedDefinitions()
    {
        (int N, int M)[] pairShapes =
            [(256, 256), (1024, 256), (1024, 1024), (4096, 256)];
        DirectPtxVisionOperation[] metrics =
        [
            DirectPtxVisionOperation.GeneralizedBoxIou,
            DirectPtxVisionOperation.DistanceBoxIou,
            DirectPtxVisionOperation.CompleteBoxIou
        ];
        foreach (DirectPtxVisionOperation operation in metrics)
        foreach ((int n, int m) in pairShapes)
            yield return new(operation, n, m);

        DirectPtxVisionOperation[] vectors =
        [
            DirectPtxVisionOperation.BoxArea,
            DirectPtxVisionOperation.IoULoss,
            DirectPtxVisionOperation.GIoULoss,
            DirectPtxVisionOperation.DIoULoss,
            DirectPtxVisionOperation.CIoULoss,
            DirectPtxVisionOperation.IoULossBackward,
            DirectPtxVisionOperation.GIoULossBackward,
            DirectPtxVisionOperation.DIoULossBackward,
            DirectPtxVisionOperation.CIoULossBackward
        ];
        foreach (DirectPtxVisionOperation operation in vectors)
        foreach (int n in new[] { 256, 1024, 4096 })
            yield return new(operation, n);

        foreach (int n in new[] { 256, 1024, 4096 })
        for (int from = 0; from < 3; from++)
        for (int to = 0; to < 3; to++)
            yield return new(DirectPtxVisionOperation.BoxConvert, n, from, to);

        foreach (bool ownerA in new[] { true, false })
        foreach (int n in new[] { 256, 1024 })
        foreach (int m in new[] { 256, 1024 })
        for (int variant = 0; variant < 4; variant++)
            yield return new(
                ownerA ? DirectPtxVisionOperation.IouFamilyBackwardA :
                    DirectPtxVisionOperation.IouFamilyBackwardB,
                n, m, variant);

        foreach (int n in new[] { 256, 1024 })
        foreach (int flags in new[] { 0, 1 })
            yield return new(DirectPtxVisionOperation.Nms, n, Flags: flags,
                ScalarBits: BitConverter.SingleToInt32Bits(0.5f));

        yield return new(DirectPtxVisionOperation.MasksToBoxes, 256, 28, 28);
        yield return new(DirectPtxVisionOperation.MasksToBoxes, 64, 64, 64);
        foreach (bool aligned in new[] { false, true })
            yield return new(DirectPtxVisionOperation.RoiAlign,
                1, 256, 56, 56, 256, 7, 7, 256,
                2 | (aligned ? 0x100 : 0), BitConverter.SingleToInt32Bits(0.25f));
        yield return new(DirectPtxVisionOperation.RoiPool,
            1, 256, 56, 56, 256, 7, 7, 256, 0,
            BitConverter.SingleToInt32Bits(0.25f));
        yield return new(DirectPtxVisionOperation.PsRoiAlign,
            1, 196, 56, 56, 256, 7, 7, 4, 2,
            BitConverter.SingleToInt32Bits(0.25f));
        yield return new(DirectPtxVisionOperation.PsRoiPool,
            1, 196, 56, 56, 256, 7, 7, 4, 0,
            BitConverter.SingleToInt32Bits(0.25f));

        foreach ((int outer, int inner) in new[] { (256, 1), (1024, 1), (256, 64) })
            yield return new(DirectPtxVisionOperation.Cross3, outer, inner);
        foreach ((int n0, int n1) in new[] { (256, 256), (1024, 256) })
        for (int flags = 0; flags < 4; flags++)
            yield return new(DirectPtxVisionOperation.Meshgrid2D, n0, n1, Flags: flags);
    }

    [Fact]
    public void Emitter_BakesExactShapeAndUsesPointerOnlyAbi()
    {
        string ptx = PtxFusedPairwiseBoxIouF32Kernel.EmitPtx(8, 6, 1024, 256);
        Assert.Contains(".target sm_86", ptx);
        Assert.Contains("ld.global.v4.f32", ptx);
        Assert.Contains("div.rn.f32", ptx);
        Assert.Contains("shr.u32 %r3, %r2, 8", ptx);
        Assert.Contains("and.b32 %r4, %r2, 255", ptx);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("nvrtc", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("setp.ge.u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void SpecializationMatrix_IsExactSm86AndUnpromoted()
    {
        Assert.True(DirectPtxArchitecture.HasValidatedVisionBoxIou(8, 6));
        Assert.False(DirectPtxArchitecture.HasValidatedVisionBoxIou(8, 0));
        Assert.False(DirectPtxArchitecture.HasValidatedVisionBoxIou(8, 9));
        Assert.False(DirectPtxArchitecture.HasValidatedVisionBoxIou(9, 0));
        Assert.True(PtxFusedPairwiseBoxIouF32Kernel.IsSupportedShape(256, 256));
        Assert.True(PtxFusedPairwiseBoxIouF32Kernel.IsSupportedShape(1024, 1024));
        Assert.False(PtxFusedPairwiseBoxIouF32Kernel.IsSupportedShape(257, 256));
        Assert.False(PtxFusedPairwiseBoxIouF32Kernel.IsPromotedShape(256, 256));
        Assert.Throws<NotSupportedException>(() =>
            PtxFusedPairwiseBoxIouF32Kernel.EmitPtx(8, 9, 256, 256));
        Assert.Throws<NotSupportedException>(() => PtxVisionEmitter.Emit(
            new(DirectPtxVisionOperation.BoxArea, 256),
            DirectPtxArchitectureFamily.Ada, 8, 9));
    }

    [Fact]
    public void FamilyAdmissionTable_FailsClosedForEveryUnsupportedAxis()
    {
        Assert.True(DirectPtxVisionSpecializations.TryPairwise(
            DirectPtxVisionOperation.GeneralizedBoxIou, 256, 256, out _));
        Assert.False(DirectPtxVisionSpecializations.TryPairwise(
            DirectPtxVisionOperation.Nms, 256, 256, out _));
        Assert.False(DirectPtxVisionSpecializations.TryPairwise(
            DirectPtxVisionOperation.GeneralizedBoxIou, 257, 256, out _));

        Assert.True(DirectPtxVisionSpecializations.TryVector(
            DirectPtxVisionOperation.CIoULossBackward, 4096, out _));
        Assert.False(DirectPtxVisionSpecializations.TryVector(
            DirectPtxVisionOperation.Cross3, 4096, out _));
        Assert.False(DirectPtxVisionSpecializations.TryVector(
            DirectPtxVisionOperation.BoxArea, 255, out _));

        Assert.True(DirectPtxVisionSpecializations.TryBoxConvert(256, 0, 2, out _));
        Assert.False(DirectPtxVisionSpecializations.TryBoxConvert(256, -1, 2, out _));
        Assert.False(DirectPtxVisionSpecializations.TryBoxConvert(256, 0, 3, out _));
        Assert.False(DirectPtxVisionSpecializations.TryPairwiseBackward(
            true, 256, 256, 4, out _));
        Assert.False(DirectPtxVisionSpecializations.TryPairwiseBackward(
            true, 4096, 256, 0, out _));

        Assert.True(DirectPtxVisionSpecializations.TryNms(256, 0.5f, false, out _));
        Assert.False(DirectPtxVisionSpecializations.TryNms(255, 0.5f, false, out _));
        Assert.False(DirectPtxVisionSpecializations.TryNms(256, float.NaN, false, out _));
        Assert.False(DirectPtxVisionSpecializations.TryNms(256, 1.01f, false, out _));
        Assert.False(DirectPtxVisionSpecializations.TryNms(256, 0.25f, false, out _));
        Assert.False(DirectPtxVisionSpecializations.TryMasksToBoxes(256, 28, 27, out _));

        Assert.True(DirectPtxVisionSpecializations.TryRoi(
            DirectPtxVisionOperation.RoiAlign,
            1, 256, 56, 56, 256, 7, 7, 256, 0.25f, 2, true, out _));
        Assert.False(DirectPtxVisionSpecializations.TryRoi(
            DirectPtxVisionOperation.BoxArea,
            1, 256, 56, 56, 256, 7, 7, 256, 0.25f, 2, true, out _));
        Assert.False(DirectPtxVisionSpecializations.TryRoi(
            DirectPtxVisionOperation.RoiAlign,
            1, 256, 56, 56, 256, 7, 7, 256, 0.25f, 1, true, out _));
        Assert.False(DirectPtxVisionSpecializations.TryRoi(
            DirectPtxVisionOperation.RoiPool,
            1, 256, 56, 56, 256, 7, 7, 256, 0f, 0, false, out _));
        Assert.False(DirectPtxVisionSpecializations.TryRoi(
            DirectPtxVisionOperation.RoiPool,
            1, 256, 56, 56, 256, 7, 7, 256, 0.5f, 0, false, out _));
        Assert.False(DirectPtxVisionSpecializations.TryRoi(
            DirectPtxVisionOperation.RoiPool,
            1, 256, 56, 56, 256, 7, 7, 4, 0.25f, 0, false, out _));
        Assert.False(DirectPtxVisionSpecializations.TryRoi(
            DirectPtxVisionOperation.PsRoiAlign,
            1, 196, 56, 56, 256, 7, 7, 4, 0.25f, 2, true, out _));

        Assert.False(DirectPtxVisionSpecializations.TryCross3(1024, 2, out _));
        Assert.False(DirectPtxVisionSpecializations.TryMeshgrid2D(
            1024, 256, 2, false, out _));
        Assert.False(DirectPtxVisionSpecializations.TryMeshgrid2D(
            1024, 255, 0, false, out _));

        Assert.True(DirectPtxVisionSpecializations.IsAdmitted(
            new(DirectPtxVisionOperation.Nms, 256, Flags: 1,
                ScalarBits: BitConverter.SingleToInt32Bits(0.5f))));
        Assert.False(DirectPtxVisionSpecializations.IsAdmitted(
            new(DirectPtxVisionOperation.Nms, 256, D7: 1, Flags: 1,
                ScalarBits: BitConverter.SingleToInt32Bits(0.5f))));
        Assert.False(DirectPtxVisionSpecializations.IsAdmitted(
            new((DirectPtxVisionOperation)int.MaxValue, 256)));
        Assert.Throws<NotSupportedException>(() => PtxVisionEmitter.Emit(
            new(DirectPtxVisionOperation.BoxArea, 257),
            DirectPtxArchitectureFamily.Ampere, 8, 6));

        DirectPtxVisionDefinition plainNms = PtxVisionEmitter.Emit(
            new(DirectPtxVisionOperation.Nms, 256,
                ScalarBits: BitConverter.SingleToInt32Bits(0.5f)),
            DirectPtxArchitectureFamily.Ampere, 8, 6);
        Assert.Equal(new DirectPtxExtent(1), plainNms.Blueprint.Tensors[2].LogicalExtent);

        DirectPtxVisionDefinition roiPool = PtxVisionEmitter.Emit(
            new(DirectPtxVisionOperation.RoiPool,
                1, 256, 56, 56, 256, 7, 7, 256, 0,
                BitConverter.SingleToInt32Bits(0.25f)),
            DirectPtxArchitectureFamily.Ampere, 8, 6);
        Assert.DoesNotContain("cvt.rni.s32.f32", roiPool.Ptx, StringComparison.Ordinal);
        Assert.Contains("cvt.rmi.s32.f32", roiPool.Ptx, StringComparison.Ordinal);
        Assert.Contains("cvt.rpi.s32.f32", roiPool.Ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorContracts_RejectWrongExtentAndAlignment_AndPermitOnlyReadAliases()
    {
        DirectPtxVisionDefinition definition = PtxVisionEmitter.Emit(
            new(DirectPtxVisionOperation.GeneralizedBoxIou, 256, 256),
            DirectPtxArchitectureFamily.Ampere, 8, 6);
        DirectPtxTensorContract aContract = definition.Blueprint.Tensors[0];
        DirectPtxTensorContract bContract = definition.Blueprint.Tensors[1];
        DirectPtxTensorContract outputContract = definition.Blueprint.Tensors[2];

        Assert.Throws<ArgumentException>(() => DirectPtxTensorView.Create(
            new FakeGpuBuffer(new IntPtr(0x1000), checked((long)aContract.RequiredBytes + 4)),
            aContract));
        Assert.Throws<ArgumentException>(() => DirectPtxTensorView.Create(
            new FakeGpuBuffer(new IntPtr(0x1004), checked((long)aContract.RequiredBytes)),
            aContract));

        var offsetContract = new DirectPtxTensorContract(
            "offset-boxes", aContract.PhysicalType, aContract.Layout,
            aContract.LogicalExtent, aContract.PhysicalExtent,
            aContract.AlignmentBytes, aContract.Access,
            DirectPtxExtentMode.Exact, byteOffset: 16);
        var offsetBuffer = new FakeGpuBuffer(
            new IntPtr(0x1000), checked((long)offsetContract.RequiredBytes + 16));
        Assert.Throws<ArgumentException>(() =>
            DirectPtxTensorView.Create(offsetBuffer, offsetContract));
        DirectPtxTensorView offsetView = DirectPtxTensorView.Create(
            offsetBuffer, offsetContract, byteOffset: 16);
        Assert.Equal(new IntPtr(0x1010), offsetView.Pointer);

        DirectPtxTensorView a = DirectPtxTensorView.Create(
            new FakeGpuBuffer(new IntPtr(0x1000), checked((long)aContract.RequiredBytes)),
            aContract);
        DirectPtxTensorView bAlias = DirectPtxTensorView.Create(
            new FakeGpuBuffer(new IntPtr(0x1000), checked((long)bContract.RequiredBytes)),
            bContract);
        DirectPtxTensorView output = DirectPtxTensorView.Create(
            new FakeGpuBuffer(new IntPtr(0x100000), checked((long)outputContract.RequiredBytes)),
            outputContract);
        PtxVisionKernel.ValidateViews(definition.Blueprint, [a, bAlias, output]);

        DirectPtxTensorView outputAlias = DirectPtxTensorView.Create(
            new FakeGpuBuffer(new IntPtr(0x1000), checked((long)outputContract.RequiredBytes)),
            outputContract);
        Assert.Throws<ArgumentException>(() =>
            PtxVisionKernel.ValidateViews(definition.Blueprint, [a, bAlias, outputAlias]));
        Assert.Throws<ArgumentException>(() =>
            PtxVisionKernel.ValidateViews(definition.Blueprint, [a, bAlias]));
    }

    [Fact]
    public void FeatureOverrides_AreThreadLocalAndRestoreDisabledState()
    {
        bool? oldExperiment = DirectPtxFeatureGate.VisionBoxIouExperimentOverride;
        bool? oldRoute = DirectPtxFeatureGate.VisionBoxIouGateOverride;
        try
        {
            DirectPtxFeatureGate.VisionBoxIouExperimentOverride = true;
            Assert.True(DirectPtxFeatureGate.IsVisionBoxIouEnabled);
            DirectPtxFeatureGate.VisionBoxIouGateOverride = false;
            Assert.False(DirectPtxFeatureGate.IsVisionBoxIouEnabled);
        }
        finally
        {
            DirectPtxFeatureGate.VisionBoxIouGateOverride = oldRoute;
            DirectPtxFeatureGate.VisionBoxIouExperimentOverride = oldExperiment;
        }
    }

    [Fact]
    public void FamilyEmitter_UsesExactPointerOnlyAbi()
    {
        DirectPtxVisionSpec[] admitted = AllAdmittedDefinitions().ToArray();
        Assert.Equal(120, admitted.Length);
        Assert.Equal(admitted.Length, admitted.Distinct().Count());
        foreach (DirectPtxVisionSpec spec in admitted)
        {
            Assert.True(DirectPtxVisionSpecializations.IsAdmitted(spec));
            DirectPtxVisionDefinition definition = PtxVisionEmitter.Emit(
                spec, DirectPtxArchitectureFamily.Ampere, 8, 6);
            Assert.Contains(".target sm_86", definition.Ptx);
            Assert.Contains(PtxVisionEmitter.EntryPoint(spec.Operation), definition.Ptx);
            Assert.DoesNotContain(".param .u32", definition.Ptx, StringComparison.Ordinal);
            Assert.DoesNotContain(".local", definition.Ptx, StringComparison.Ordinal);
            Assert.DoesNotContain("stride", definition.Ptx, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("nvrtc", definition.Ptx, StringComparison.OrdinalIgnoreCase);
            AssertPtxRegisterAndLabelClosure(definition.Ptx);
            Assert.NotEmpty(definition.Blueprint.Tensors);
            Assert.All(definition.Blueprint.Tensors, contract =>
            {
                Assert.Equal(DirectPtxExtentMode.Exact, contract.ExtentMode);
                Assert.Equal(16, contract.AlignmentBytes);
                Assert.Equal((nuint)0, contract.ByteOffset);
                Assert.Equal(contract.LogicalExtent, contract.PhysicalExtent);
            });
        }
    }

    private static void AssertPtxRegisterAndLabelClosure(string ptx)
    {
        var limits = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (Match declaration in Regex.Matches(
                     ptx, @"\.reg\s+\.(?:pred|b32|b64|f32)\s+%(rd|r|f|p)<(\d+)>;"))
            limits[declaration.Groups[1].Value] = int.Parse(declaration.Groups[2].Value);
        Assert.Equal(4, limits.Count);
        foreach (Match reference in Regex.Matches(ptx, @"%(rd|r|f|p)(\d+)"))
        {
            string kind = reference.Groups[1].Value;
            int index = int.Parse(reference.Groups[2].Value);
            Assert.True(limits.TryGetValue(kind, out int limit) && index < limit,
                $"PTX register %{kind}{index} exceeds its declared range.");
        }

        string[] labels = Regex.Matches(ptx, @"(?m)^\s*([A-Z][A-Z0-9_]*)\s*:")
            .Select(match => match.Groups[1].Value).ToArray();
        Assert.Equal(labels.Length, labels.Distinct(StringComparer.Ordinal).Count());
        var defined = labels.ToHashSet(StringComparer.Ordinal);
        foreach (Match branch in Regex.Matches(ptx, @"\bbra\s+([A-Z][A-Z0-9_]*)"))
            Assert.Contains(branch.Groups[1].Value, defined);
        Assert.DoesNotMatch(
            new Regex(@"st\.global[^;]*,\s*(?:0f[0-9A-F]+|-?\d+)\s*;",
                RegexOptions.CultureInvariant), ptx);
    }

    [Fact]
    public void CoverageManifest_AssignsOnlyImplementedDirectPtxCells()
    {
        Assert.True(DirectPtxVisionCoverageManifest.All.Count >= 25);
        Assert.Equal(
            DirectPtxVisionCoverageManifest.All.Count,
            DirectPtxVisionCoverageManifest.All.Select(cell => cell.Api).Distinct(StringComparer.Ordinal).Count());
        DirectPtxVisionCoverageCell golden =
            DirectPtxVisionCoverageManifest.Get(
                "IEngine.BoxIou -> IDetectionBackend.BoxIou");
        Assert.Equal(DirectPtxVisionCoverageStatus.ExperimentalDirectPtx, golden.Status);
        Assert.All(DirectPtxVisionCoverageManifest.All, cell =>
            Assert.Equal(DirectPtxVisionCoverageStatus.ExperimentalDirectPtx, cell.Status));
        Assert.All(DirectPtxVisionCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
    }

    [SkippableTheory]
    [MemberData(nameof(VisionDefinitions))]
    public void DriverOnly_FamilyPrewarmsCapturesAndRetainsZeroLocalBytes(
        object value)
    {
        var spec = (DirectPtxVisionSpec)value;
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? old = DirectPtxFeatureGate.VisionBoxIouExperimentOverride;
        DirectPtxFeatureGate.VisionBoxIouExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.PrewarmDirectPtxVisionKernel(spec),
                backend.DirectPtxLastError ?? "Requires the exact SM86 specialization.");
            DirectPtxVisionDefinition definition = PtxVisionEmitter.Emit(
                spec, DirectPtxArchitectureFamily.Ampere, 8, 6);
            IGpuBuffer[] buffers = definition.Blueprint.Tensors
                .Select(contract => backend.AllocateBuffer(
                    new float[checked((int)(contract.RequiredBytes / 4))]))
                .ToArray();
            try
            {
                void Launch()
                {
                    if (!backend.TryDirectPtxVisionKernel(
                            spec, buffers[0], At(buffers, 1), At(buffers, 2),
                            At(buffers, 3), At(buffers, 4), At(buffers, 5)))
                        throw new InvalidOperationException(
                            backend.DirectPtxLastError ?? "Direct PTX family dispatch failed.");
                }
                Launch();
                IntPtr graph = backend.CaptureGraph(Launch);
                Assert.NotEqual(IntPtr.Zero, graph);
                try
                {
                    backend.EnqueueCapturedGraph(graph);
                    backend.Synchronize();
                }
                finally
                {
                    backend.DestroyCapturedGraph(graph);
                }
                Assert.True(backend.TryGetDirectPtxVisionAudit(spec, out var audit));
                Assert.Equal(0, audit.Function.LocalBytesPerThread);
                Assert.True(backend.DirectPtxVisionDispatchCount(spec.Operation) >= 2);
            }
            finally
            {
                foreach (IGpuBuffer buffer in buffers) buffer.Dispose();
            }
        }
        finally
        {
            DirectPtxFeatureGate.VisionBoxIouExperimentOverride = old;
        }
    }

    private static IGpuBuffer? At(IGpuBuffer[] buffers, int index) =>
        index < buffers.Length ? buffers[index] : null;

    [SkippableTheory]
    [InlineData(256, 256)]
    [InlineData(1024, 256)]
    [InlineData(1024, 1024)]
    [InlineData(4096, 256)]
    public void DriverOnly_PublicRoutePrewarmsCapturesAndMatchesOracle(int n, int m)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? old = DirectPtxFeatureGate.VisionBoxIouExperimentOverride;
        DirectPtxFeatureGate.VisionBoxIouExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.PrewarmDirectPtxVisionBoxIou(n, m),
                backend.DirectPtxLastError ?? "Requires the exact SM86 specialization.");
            float[] a = Boxes(n, 17);
            float[] b = Boxes(m, 31);
            using var aBuffer = backend.AllocateBuffer(a);
            using var bBuffer = backend.AllocateBuffer(b);
            using var output = backend.AllocateBuffer(checked(n * m));
            void Launch() => backend.BoxIou(aBuffer, bBuffer, output, n, m);
            IntPtr graph = backend.CaptureGraph(Launch);
            Assert.NotEqual(IntPtr.Zero, graph);
            try
            {
                backend.EnqueueCapturedGraph(graph);
                backend.Synchronize();
            }
            finally
            {
                backend.DestroyCapturedGraph(graph);
            }
            float[] actual = backend.DownloadBuffer(output);
            Assert.True(MaximumError(a, b, actual, n, m) <= 2e-6f);
            Assert.True(backend.DirectPtxVisionBoxIouDispatchCount >= 1);
            Assert.Equal(1, backend.DirectPtxVisionBoxIouPinnedKernelCount);
            Assert.True(backend.TryGetDirectPtxVisionBoxIouAudit(n, m, out var audit));
            Assert.Equal(0, audit.Function.LocalBytesPerThread);
        }
        finally
        {
            DirectPtxFeatureGate.VisionBoxIouExperimentOverride = old;
        }
    }

    private static float[] Boxes(int count, int seed)
    {
        var random = new Random(seed);
        var result = new float[count * 4];
        for (int i = 0; i < count; i++)
        {
            float x = random.NextSingle() * 100f;
            float y = random.NextSingle() * 100f;
            result[i * 4] = x;
            result[i * 4 + 1] = y;
            result[i * 4 + 2] = x + random.NextSingle() * 25f;
            result[i * 4 + 3] = y + random.NextSingle() * 25f;
        }
        return result;
    }

    private static double MaximumError(float[] a, float[] b, float[] actual, int n, int m)
    {
        double error = 0;
        for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
        {
            double ax1 = a[i * 4], ay1 = a[i * 4 + 1], ax2 = a[i * 4 + 2], ay2 = a[i * 4 + 3];
            double bx1 = b[j * 4], by1 = b[j * 4 + 1], bx2 = b[j * 4 + 2], by2 = b[j * 4 + 3];
            double areaA = Math.Max(ax2 - ax1, 0) * Math.Max(ay2 - ay1, 0);
            double areaB = Math.Max(bx2 - bx1, 0) * Math.Max(by2 - by1, 0);
            double intersection = Math.Max(Math.Min(ax2, bx2) - Math.Max(ax1, bx1), 0) *
                Math.Max(Math.Min(ay2, by2) - Math.Max(ay1, by1), 0);
            double union = areaA + areaB - intersection;
            double expected = union > 0 ? intersection / union : 0;
            error = Math.Max(error, Math.Abs(expected - actual[i * m + j]));
        }
        return error;
    }

    private sealed class FakeGpuBuffer(IntPtr handle, long sizeInBytes) : IGpuBuffer
    {
        public int Size => checked((int)(sizeInBytes / 4));
        public long SizeInBytes { get; } = sizeInBytes;
        public IntPtr Handle { get; } = handle;
        public void Dispose() { }
    }
}
#endif
