#if NET5_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape FP32 global-average-pool direct-PTX
/// family (issue #842). The emitter and shape-domain assertions run without a
/// GPU; the driver correctness assertion is skipped unless a validated Ampere
/// device is present. Disabled by default; fails closed until three clean
/// promotion runs clear the release gate.
/// </summary>
public class DirectPtxGlobalAvgPoolTests
{
    [Fact]
    public void FusedGlobalAvgPoolEmitter_IsRegisterResidentAndPointerOnly()
    {
        string ptx = PtxFusedGlobalAvgPoolF32Kernel.EmitPtx(8, 6, 2048, 128);
        Assert.Contains(".maxntid 128, 1, 1", ptx);
        Assert.Contains("exact-shape rows=2048 spatial=128 block=128", ptx);
        Assert.Contains("op=global-avgpool", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        Assert.Equal(1, Count(ptx, "ld.global.ca.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(5, Count(ptx, "shfl.sync.bfly.b32"));
        Assert.Equal(1, Count(ptx, "mul.rn.f32"));
        // 1/128 = 0x3C000000
        Assert.Contains("0f3C000000", ptx);
        Assert.Contains("ld.global.ca.v2.f32",
            PtxFusedGlobalAvgPoolF32Kernel.EmitPtx(8, 6, 2048, 64));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void FusedGlobalAvgPoolShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedGlobalAvgPoolF32Kernel.IsSupportedShape(256, 128));
        Assert.True(PtxFusedGlobalAvgPoolF32Kernel.IsSupportedShape(2048, 64));
        Assert.True(PtxFusedGlobalAvgPoolF32Kernel.IsSupportedShape(2048, 128));
        Assert.True(PtxFusedGlobalAvgPoolF32Kernel.IsSupportedShape(8192, 128));
        Assert.False(PtxFusedGlobalAvgPoolF32Kernel.IsSupportedShape(256, 64));
        Assert.False(PtxFusedGlobalAvgPoolF32Kernel.IsSupportedShape(2048, 96));
        Assert.False(PtxFusedGlobalAvgPoolF32Kernel.IsPromotedShape(2048, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedGlobalAvgPoolF32Kernel.EmitPtx(8, 6, 17, 128));
    }

    [Fact]
    public void PoolingCoverageManifest_AssignsEveryCellExactlyOnce()
    {
        Assert.NotEmpty(DirectPtxPoolingCoverageManifest.All);
        string[] names = DirectPtxPoolingCoverageManifest.All
            .Select(cell => cell.Api).OrderBy(name => name, StringComparer.Ordinal).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxPoolingCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Equal(
            DirectPtxPoolingCoverageStatus.ExperimentalDirectPtx,
            DirectPtxPoolingCoverageManifest.Get("CudaBackend.GlobalAvgPool2D").Status);
        Assert.Single(DirectPtxPoolingCoverageManifest.All,
            cell => cell.Status == DirectPtxPoolingCoverageStatus.ExperimentalDirectPtx);
        Assert.All(DirectPtxPoolingCoverageManifest.All,
            cell => Assert.NotEqual(
                DirectPtxPoolingCoverageStatus.PromotedDirectPtx, cell.Status));
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxPoolingCoverageManifest.Get("UnassignedPoolingApi"));
    }

    [SkippableTheory]
    [InlineData(256, 128)]
    [InlineData(2048, 64)]
    [InlineData(2048, 128)]
    [InlineData(8192, 128)]
    public void DriverOnlyFusedGlobalAvgPool_MatchesReferenceAndHasZeroLocalBytes(int rows, int spatial)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in global-average-pool specialization is validated on Ampere.");
        using var kernel = new PtxFusedGlobalAvgPoolF32Kernel(runtime, rows, spatial);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 3);
        Assert.Equal("global-avgpool-f32", kernel.Blueprint.Operation);
        Assert.Equal(2, kernel.Blueprint.Tensors.Count);

        int elements = rows * spatial;
        var random = new Random(20260722);
        float[] input = Enumerable.Range(0, elements).Select(_ => (random.NextSingle() * 2f - 1f) * 4f).ToArray();
        var expected = new float[rows];
        for (int row = 0; row < rows; row++)
        {
            double sum = 0;
            for (int s = 0; s < spatial; s++)
                sum += input[row * spatial + s];
            expected[row] = (float)(sum / spatial);
        }

        using var inputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        inputBuffer.Upload<float>(input);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();

        var actual = new float[rows];
        outputBuffer.Download<float>(actual);
        for (int row = 0; row < rows; row++)
        {
            float tolerance = 1e-4f * (MathF.Abs(expected[row]) + 1f);
            Assert.True(MathF.Abs(actual[row] - expected[row]) <= tolerance,
                $"row {row}: actual {actual[row]:G9}, expected {expected[row]:G9}, tol {tolerance:G9}.");
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
