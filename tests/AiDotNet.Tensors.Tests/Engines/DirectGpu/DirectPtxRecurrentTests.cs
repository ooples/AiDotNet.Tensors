using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class DirectPtxRecurrentTests
{
    [Fact]
    public void RgLruEligibility_IsExactShapeLayoutDtypePhaseAndSm()
    {
        var request = ValidRequest();
        Assert.True(DirectPtxRecurrentEligibility.Evaluate(request).IsEligible);
        Assert.Equal("rglru-sm-version-not-implemented",
            DirectPtxRecurrentEligibility.Evaluate(request with { ComputeCapabilityMinor = 9 }).Reason);
        Assert.Equal("rglru-dtype-not-fp32",
            DirectPtxRecurrentEligibility.Evaluate(request with { PhysicalType = DirectPtxPhysicalType.Float16 }).Reason);
        Assert.Equal("rglru-layout-not-dense-bsf",
            DirectPtxRecurrentEligibility.Evaluate(request with { Layout = DirectPtxPhysicalLayout.RowMajor2D }).Reason);
        Assert.Equal("rglru-batch-not-1",
            DirectPtxRecurrentEligibility.Evaluate(request with { Batch = 2 }).Reason);
        Assert.Equal("rglru-sequence-not-128",
            DirectPtxRecurrentEligibility.Evaluate(request with { SequenceLength = 127 }).Reason);
        Assert.Equal("rglru-dimension-not-256",
            DirectPtxRecurrentEligibility.Evaluate(request with { RecurrentDimension = 128 }).Reason);
        Assert.Equal("rglru-backward-not-implemented",
            DirectPtxRecurrentEligibility.Evaluate(request with { IsTraining = true }).Reason);
    }

    [Fact]
    public void RgLruBufferAdmission_RejectsExtentPointerAlignmentAliasAndOverflow()
    {
        DirectPtxRgLruBufferRequest request = ValidBuffers();
        Assert.True(DirectPtxRecurrentEligibility.EvaluateBuffers(request).IsEligible);
        Assert.Equal("rglru-physical-extent-mismatch",
            DirectPtxRecurrentEligibility.EvaluateBuffers(request with { OutputBytes = request.OutputBytes + 4 }).Reason);
        Assert.Equal("rglru-invalid-device-pointer",
            DirectPtxRecurrentEligibility.EvaluateBuffers(request with { DecayPointer = 0 }).Reason);
        Assert.Equal("rglru-alignment-mismatch",
            DirectPtxRecurrentEligibility.EvaluateBuffers(request with { ValuePointer = request.ValuePointer + 4 }).Reason);
        Assert.Equal("rglru-alias-not-supported",
            DirectPtxRecurrentEligibility.EvaluateBuffers(request with { OutputPointer = request.ValuePointer }).Reason);
        Assert.Equal("rglru-address-range-overflow",
            DirectPtxRecurrentEligibility.EvaluateBuffers(request with { ValuePointer = MaxNuint - 127 }).Reason);
    }

    // nuint.MaxValue is net7+ generic math. Truncating ulong.MaxValue to nuint
    // yields all-ones at the native pointer width on both 32- and 64-bit, which
    // is exactly the value the address-range-overflow guard is probed with.
    private static readonly nuint MaxNuint = unchecked((nuint)ulong.MaxValue);

    [Fact]
    public void RgLruBlueprint_DeclaresExactNoWorkspaceAbi()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxFusedRgLruScan128x256Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);
        Assert.Equal("rglru-scan-forward-b1-s128-d256", blueprint.Operation);
        Assert.Equal(5, blueprint.Tensors.Count);
        Assert.All(blueprint.Tensors, tensor => Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
        Assert.All(blueprint.Tensors, tensor => Assert.Equal(128, tensor.AlignmentBytes));
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
        Assert.Equal("0", blueprint.Semantics["temporary-device-bytes"]);
        Assert.Equal(0, blueprint.ResourceBudget.MaxLocalBytesPerThread);
        Assert.Contains("pending-gpu-evidence", blueprint.Semantics["promotion"]);
    }

    [Fact]
    public void RgLruEmitter_IsExactShapeUnrolledAndHasNoDynamicStrideBranch()
    {
        string ptx = PtxFusedRgLruScan128x256Kernel.EmitPtx(8, 6);
        Assert.Contains(".target sm_86", ptx);
        Assert.Contains(PtxFusedRgLruScan128x256Kernel.EntryPoint, ptx);
        Assert.Equal(1 + 3 * PtxFusedRgLruScan128x256Kernel.SequenceLength,
            Count(ptx, "ld.global.f32"));
        Assert.Equal(PtxFusedRgLruScan128x256Kernel.SequenceLength,
            Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("%ctaid", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Throws<NotSupportedException>(
            () => PtxFusedRgLruScan128x256Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void RecurrentCoverageManifest_AssignsEveryScopedEntryPointExactlyOnce()
    {
        string[] expected =
        [
            "CpuEngine.LstmSequenceForward(inference)",
            "CpuEngine.LstmSequenceForward(training)",
            "CpuEngine.GlaScanForward",
            "CpuEngine.XLstmScanForward",
            "CpuEngine.GatedDeltaNetScanForward",
            "CpuEngine.RgLruScanForward",
            "CpuEngine.Rwkv4WkvForward",
            "CpuEngine.Rwkv7SequenceForward",
            "CpuEngine.MambaSelectiveScanForward",
            "CpuEngine.Mamba2SsdScanForward",
            "IDeviceRnn.ForwardRnn",
            "IDeviceRnn.ForwardLstm",
            "IDeviceRnn.BackwardRnn",
            "IDirectGpuBackend.LstmForwardSequence",
            "IDirectGpuBackend.LstmBackwardSequence",
            "IDirectGpuBackend.GruForwardSequence",
            "IDirectGpuBackend.GruBackwardSequence",
            "IDirectGpuBackend.GruCellBackward",
            "IDirectGpuBackend.GlaScanForward",
            "IDirectGpuBackend.GlaScanBackward",
            "IDirectGpuBackend.XLstmScanForward",
            "IDirectGpuBackend.GatedDeltaNetScanForward",
            "IDirectGpuBackend.RgLruScanForward",
            "IDirectGpuBackend.Rwkv4WkvForward",
            "IDirectGpuBackend.Rwkv7Forward",
            "IDirectGpuBackend.MambaSelectiveScanForward",
            "IDirectGpuBackend.Mamba2SsdScanForward"
        ];
        string[] actual = DirectPtxRecurrentCoverageManifest.All.Select(cell => cell.Api).ToArray();
        Assert.Equal(expected.OrderBy(value => value), actual.OrderBy(value => value));
        Assert.Equal(actual.Length, actual.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxRecurrentCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingCudaImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Equal(2, DirectPtxRecurrentCoverageManifest.All.Count(cell =>
            cell.Status == DirectPtxRecurrentCoverageStatus.ExperimentalDirectPtx));
    }

    [Fact]
    public void KernelCache_PinnedRecurrentModuleCannotBeEvicted()
    {
        var cache = new DirectPtxKernelCache<int, DisposableProbe>(1);
        var first = new DisposableProbe();
        Assert.Same(first, cache.GetOrAdd(1, () => first));
        Assert.True(cache.Pin(1));
        Assert.Equal(1, cache.PinnedCount);
        var second = new DisposableProbe();
        Assert.Throws<InvalidOperationException>(() => cache.GetOrAdd(2, () => second));
        Assert.False(first.Disposed);
        Assert.False(second.Disposed);
        cache.Dispose();
        Assert.True(first.Disposed);
    }

    private static DirectPtxRgLruRequest ValidRequest() =>
        new(8, 6, DirectPtxPhysicalType.Float32,
            DirectPtxPhysicalLayout.BatchSequenceFeature,
            PtxFusedRgLruScan128x256Kernel.Batch,
            PtxFusedRgLruScan128x256Kernel.SequenceLength,
            PtxFusedRgLruScan128x256Kernel.RecurrentDimension,
            IsTraining: false);

    private static DirectPtxRgLruBufferRequest ValidBuffers()
    {
        const long sequenceBytes = 1L * 128 * 256 * sizeof(float);
        const long decayBytes = 256L * sizeof(float);
        return new DirectPtxRgLruBufferRequest(
            0x100000, sequenceBytes,
            0x120000, sequenceBytes,
            0x140000, sequenceBytes,
            0x160000, decayBytes,
            0x180000, sequenceBytes);
    }

    private static int Count(string source, string value)
    {
        int count = 0;
        int offset = 0;
        while ((offset = source.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += value.Length;
        }
        return count;
    }

    private sealed class DisposableProbe : IDisposable
    {
        internal bool Disposed { get; private set; }
        public void Dispose() => Disposed = true;
    }
}
