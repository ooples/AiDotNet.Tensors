#if NET5_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape FP32 direct-PTX row-sum reduction family
/// (issue #843). The emitter and shape-domain assertions run without a GPU; the
/// driver correctness assertion is skipped unless a validated Ampere device is
/// present. The specialization stays disabled by default and fails closed until
/// three clean promotion runs clear the release gate.
/// </summary>
public class DirectPtxReductionTests
{
    [Fact]
    public void FusedRowSumEmitter_IsRegisterResidentAndPointerOnly()
    {
        string ptx = PtxFusedRowReduceF32Kernel.EmitPtx(8, 6, 2048, 128);
        Assert.Contains(".maxntid 128, 1, 1", ptx);
        Assert.Contains("exact-shape rows=2048 columns=128 block=128", ptx);
        Assert.Contains("op=sum", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        Assert.Equal(1, Count(ptx, "ld.global.ca.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(0, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(5, Count(ptx, "shfl.sync.bfly.b32"));
        Assert.Contains("ld.global.ca.v2.f32",
            PtxFusedRowReduceF32Kernel.EmitPtx(8, 6, 2048, 64));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("ex2.approx", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void FusedRowSumShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedRowReduceF32Kernel.IsSupportedShape(256, 128));
        Assert.True(PtxFusedRowReduceF32Kernel.IsSupportedShape(2048, 64));
        Assert.True(PtxFusedRowReduceF32Kernel.IsSupportedShape(2048, 128));
        Assert.True(PtxFusedRowReduceF32Kernel.IsSupportedShape(8192, 128));
        Assert.False(PtxFusedRowReduceF32Kernel.IsSupportedShape(256, 64));
        Assert.False(PtxFusedRowReduceF32Kernel.IsSupportedShape(2048, 96));
        Assert.False(PtxFusedRowReduceF32Kernel.IsSupportedShape(8193, 128));
        Assert.False(PtxFusedRowReduceF32Kernel.IsPromotedShape(2048, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedRowReduceF32Kernel.EmitPtx(8, 6, 17, 128));
    }

    [SkippableFact]
    public void DriverOnlyFusedRowSum_MatchesReferenceAndHasZeroLocalBytes()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in row-sum specialization is validated on Ampere.");
        const int rows = 2048, columns = 128;
        using var kernel = new PtxFusedRowReduceF32Kernel(runtime, rows, columns);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 3);
        Assert.Equal("row-sum-f32", kernel.Blueprint.Operation);
        Assert.Equal(2, kernel.Blueprint.Tensors.Count);
        Assert.Equal("none", kernel.Blueprint.Semantics["global-intermediates"]);

        int elements = rows * columns;
        using var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        var random = new Random(20260722);
        float[] values = Enumerable.Range(0, elements)
            .Select(_ => (random.NextSingle() * 2f - 1f) * 4f).ToArray();
        var expected = new float[rows];
        for (int row = 0; row < rows; row++)
        {
            double sum = 0;
            for (int column = 0; column < columns; column++)
                sum += values[row * columns + column];
            expected[row] = (float)sum;
        }
        input.Upload<float>(values);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[rows];
        output.Download<float>(actual);
        for (int row = 0; row < rows; row++)
        {
            float tolerance = 1e-4f * (MathF.Abs(expected[row]) + 1f);
            Assert.True(MathF.Abs(actual[row] - expected[row]) <= tolerance,
                $"row {row}: actual {actual[row]:G9}, expected {expected[row]:G9}, tol {tolerance:G9}.");
        }
    }

    [Fact]
    public void ReductionCoverageManifest_AssignsEveryCellExactlyOnce()
    {
        Assert.NotEmpty(DirectPtxReductionCoverageManifest.All);
        string[] names = DirectPtxReductionCoverageManifest.All
            .Select(cell => cell.Api).OrderBy(name => name, StringComparer.Ordinal).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxReductionCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Equal(
            DirectPtxReductionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxReductionCoverageManifest.Get("CudaBackend.SumAxis").Status);
        Assert.Single(DirectPtxReductionCoverageManifest.All,
            cell => cell.Status == DirectPtxReductionCoverageStatus.ExperimentalDirectPtx);
        Assert.All(DirectPtxReductionCoverageManifest.All,
            cell => Assert.NotEqual(
                DirectPtxReductionCoverageStatus.PromotedDirectPtx, cell.Status));
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxReductionCoverageManifest.Get("UnassignedReductionApi"));
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
