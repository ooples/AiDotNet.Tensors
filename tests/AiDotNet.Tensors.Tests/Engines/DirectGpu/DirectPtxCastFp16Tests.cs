#if NET5_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape FP32-to-FP16 direct-PTX cast family
/// (issue #845). The emitter and shape-domain assertions run without a GPU; the
/// driver correctness assertion is skipped unless a validated Ampere device is
/// present. The specialization stays disabled by default and fails closed until
/// three clean promotion runs clear the release gate.
/// </summary>
public class DirectPtxCastFp16Tests
{
    [Fact]
    public void FusedCastEmitter_IsRegisterResidentAndPointerOnly()
    {
        string ptx = PtxFusedCastF32ToF16Kernel.EmitPtx(8, 6, 1_048_576);
        Assert.Contains(".maxntid 256, 1, 1", ptx);
        Assert.Contains("exact-shape size=1048576 block=256", ptx);
        Assert.Contains("op=cast-f32-f16", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        Assert.Equal(1, Count(ptx, "ld.global.ca.v4.f32"));
        Assert.Equal(4, Count(ptx, "cvt.rn.f16.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.u16"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("shfl.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void FusedCastShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedCastF32ToF16Kernel.IsSupportedShape(65_536));
        Assert.True(PtxFusedCastF32ToF16Kernel.IsSupportedShape(262_144));
        Assert.True(PtxFusedCastF32ToF16Kernel.IsSupportedShape(1_048_576));
        Assert.True(PtxFusedCastF32ToF16Kernel.IsSupportedShape(4_194_304));
        Assert.False(PtxFusedCastF32ToF16Kernel.IsSupportedShape(65_535));
        Assert.False(PtxFusedCastF32ToF16Kernel.IsSupportedShape(1_000_000));
        Assert.False(PtxFusedCastF32ToF16Kernel.IsPromotedShape(1_048_576));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedCastF32ToF16Kernel.EmitPtx(8, 6, 4_194_304, 192));
    }

    [Fact]
    public void CastCoverageManifest_AssignsEveryCellExactlyOnce()
    {
        Assert.NotEmpty(DirectPtxLayoutCoverageManifest.All);
        string[] names = DirectPtxLayoutCoverageManifest.All
            .Select(cell => cell.Api).OrderBy(name => name, StringComparer.Ordinal).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxLayoutCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Equal(
            DirectPtxLayoutCoverageStatus.ExperimentalDirectPtx,
            DirectPtxLayoutCoverageManifest.Get("CudaBackend.ConvertToFp16").Status);
        Assert.Single(DirectPtxLayoutCoverageManifest.All,
            cell => cell.Status == DirectPtxLayoutCoverageStatus.ExperimentalDirectPtx);
        Assert.All(DirectPtxLayoutCoverageManifest.All,
            cell => Assert.NotEqual(
                DirectPtxLayoutCoverageStatus.PromotedDirectPtx, cell.Status));
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxLayoutCoverageManifest.Get("UnassignedLayoutApi"));
    }

    [SkippableFact]
    public void BackendCast_PrewarmCaptureAndModuleLifetimeContractsHold()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.CastFp16ExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.CastFp16ExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxCastFp16Enabled, "Requires an Ampere CUDA backend.");
            const int size = 65_536;
            using var input = backend.AllocateBuffer(new float[size]);
            using var output = backend.AllocateBuffer(size / 2);

            Assert.True(backend.PrewarmDirectPtxCastFp16(size), backend.DirectPtxLastError);
            bool captured = true;
            IntPtr graph = backend.CaptureGraph(() =>
                captured &= backend.TryDirectPtxCastFp16(input, output, size));
            Assert.True(captured, backend.DirectPtxLastError);
            Assert.NotEqual(IntPtr.Zero, graph);
            Assert.Equal(1, backend.DirectPtxCastFp16PinnedKernelCount);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }
            backend.Synchronize();
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.CastFp16ExperimentOverride = previousExperiment;
        }
    }

    [SkippableTheory]
    [InlineData(65_536)]
    [InlineData(262_144)]
    [InlineData(1_048_576)]
    public void DriverOnlyFusedCast_MatchesRoundNearestReferenceAndHasZeroLocalBytes(int size)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in cast specialization is validated on Ampere.");
        using var kernel = new PtxFusedCastF32ToF16Kernel(runtime, size);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 3);
        Assert.Equal("cast-f32-to-f16", kernel.Blueprint.Operation);
        Assert.Equal(2, kernel.Blueprint.Tensors.Count);

        var random = new Random(20260722);
        float[] input = Enumerable.Range(0, size)
            .Select(_ => (random.NextSingle() * 2f - 1f) * 65_504f).ToArray();
        input[0] = 0f;
        input[1] = float.PositiveInfinity;
        input[2] = float.NegativeInfinity;
        input[3] = 1e-5f;

        using var inputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        inputBuffer.Upload<float>(input);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();

        var actual = new ushort[size];
        outputBuffer.Download<ushort>(actual);
        for (int i = 0; i < size; i++)
        {
            ushort expected = BitConverter.HalfToUInt16Bits((Half)input[i]);
            Assert.Equal(expected, actual[i]);
        }
    }

    private static int Count(string text, string value)
    {
        int count = 0, offset = 0;
        while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += value.Length;
        }
        return count;
    }
}
#endif
