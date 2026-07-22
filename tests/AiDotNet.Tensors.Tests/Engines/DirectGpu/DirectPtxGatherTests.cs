using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape FP32 direct-PTX embedding gather family
/// (issue #844). The emitter and shape-domain assertions run without a GPU; the
/// driver correctness assertion is skipped unless a validated Ampere device is
/// present. The specialization stays disabled by default and fails closed until
/// three clean promotion runs clear the release gate.
/// </summary>
public class DirectPtxGatherTests
{
    [Fact]
    public void FusedGatherEmitter_IsRegisterResidentAndPointerOnly()
    {
        string ptx = PtxFusedGatherF32Kernel.EmitPtx(8, 6, 2048, 128);
        Assert.Contains(".maxntid 128, 1, 1", ptx);
        Assert.Contains("exact-shape indices=2048 feature=128 block=128", ptx);
        Assert.Contains("op=gather", ptx);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.Equal(1, Count(ptx, "ld.global.ca.u32"));
        Assert.Equal(1, Count(ptx, "mul.wide.s32"));
        Assert.Equal(1, Count(ptx, "ld.global.ca.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.Contains("ld.global.ca.v2.f32",
            PtxFusedGatherF32Kernel.EmitPtx(8, 6, 2048, 64));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("shfl.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void FusedGatherShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedGatherF32Kernel.IsSupportedShape(256, 128));
        Assert.True(PtxFusedGatherF32Kernel.IsSupportedShape(2048, 64));
        Assert.True(PtxFusedGatherF32Kernel.IsSupportedShape(2048, 128));
        Assert.True(PtxFusedGatherF32Kernel.IsSupportedShape(8192, 128));
        Assert.False(PtxFusedGatherF32Kernel.IsSupportedShape(256, 64));
        Assert.False(PtxFusedGatherF32Kernel.IsSupportedShape(2048, 96));
        Assert.False(PtxFusedGatherF32Kernel.IsSupportedShape(8193, 128));
        Assert.False(PtxFusedGatherF32Kernel.IsPromotedShape(2048, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedGatherF32Kernel.EmitPtx(8, 6, 17, 128));
    }

    [Fact]
    public void GatherCoverageManifest_AssignsEveryCellExactlyOnce()
    {
        Assert.NotEmpty(DirectPtxGatherCoverageManifest.All);
        string[] names = DirectPtxGatherCoverageManifest.All
            .Select(cell => cell.Api).OrderBy(name => name, StringComparer.Ordinal).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxGatherCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Equal(
            DirectPtxGatherCoverageStatus.ExperimentalDirectPtx,
            DirectPtxGatherCoverageManifest.Get("CudaBackend.Gather").Status);
        Assert.Single(DirectPtxGatherCoverageManifest.All,
            cell => cell.Status == DirectPtxGatherCoverageStatus.ExperimentalDirectPtx);
        Assert.All(DirectPtxGatherCoverageManifest.All,
            cell => Assert.NotEqual(
                DirectPtxGatherCoverageStatus.PromotedDirectPtx, cell.Status));
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxGatherCoverageManifest.Get("UnassignedGatherApi"));
    }

    [SkippableFact]
    public void BackendGather_PrewarmCaptureAndModuleLifetimeContractsHold()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.GatherExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.GatherExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxGatherEnabled, "Requires an Ampere CUDA backend.");
            const int numIndices = 256, featureSize = 128, tableRows = 8;
            using var source = backend.AllocateBuffer(tableRows * featureSize);
            // IEEE-754 +0 has the same all-zero bit pattern as INT32 index 0.
            using var indices = backend.AllocateBuffer(new float[numIndices]);
            using var output = backend.AllocateBuffer(numIndices * featureSize);

            Assert.True(backend.PrewarmDirectPtxGather(numIndices, featureSize),
                backend.DirectPtxLastError);
            bool captured = true;
            IntPtr graph = backend.CaptureGraph(() =>
                captured &= backend.TryDirectPtxGather(
                    source, indices, output, numIndices, featureSize));
            Assert.True(captured, backend.DirectPtxLastError);
            Assert.NotEqual(IntPtr.Zero, graph);
            Assert.Equal(1, backend.DirectPtxGatherPinnedKernelCount);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }
            backend.Synchronize();
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.GatherExperimentOverride = previousExperiment;
        }
    }

    [SkippableTheory]
    [InlineData(256, 128)]
    [InlineData(2048, 64)]
    [InlineData(2048, 128)]
    [InlineData(8192, 128)]
    public void DriverOnlyFusedGather_MatchesReferenceAndHasZeroLocalBytes(
        int numIndices,
        int featureSize)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in gather specialization is validated on Ampere.");
        using var kernel = new PtxFusedGatherF32Kernel(runtime, numIndices, featureSize);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 3);
        Assert.Equal("gather-f32", kernel.Blueprint.Operation);
        Assert.Equal(3, kernel.Blueprint.Tensors.Count);

        const int tableRows = 1024;
        var random = RandomHelper.CreateSeededRandom(20260722);
        float[] table = Enumerable.Range(0, tableRows * featureSize)
            .Select(_ => (float)((random.NextDouble() * 2.0 - 1.0) * 8.0)).ToArray();
        int[] indices = Enumerable.Range(0, numIndices)
            .Select(_ => random.Next(tableRows)).ToArray();

        using var sourceBuffer = runtime.AllocateBytes((nuint)table.Length * sizeof(float));
        using var indexBuffer = runtime.AllocateBytes((nuint)numIndices * sizeof(int));
        using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        sourceBuffer.Upload<float>(table);
        indexBuffer.Upload<int>(indices);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(indexBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(sourceBuffer, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();

        var actual = new float[numIndices * featureSize];
        outputBuffer.Download<float>(actual);
        for (int i = 0; i < numIndices; i++)
        {
            int sourceRow = indices[i];
            for (int f = 0; f < featureSize; f++)
                Assert.Equal(table[sourceRow * featureSize + f], actual[i * featureSize + f]);
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
