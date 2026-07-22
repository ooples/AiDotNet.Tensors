#if NET5_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape FP32 per-sample MSE-loss direct-PTX
/// family (issue #847). The emitter and shape-domain assertions run without a
/// GPU; the driver correctness assertion is skipped unless a validated Ampere
/// device is present. The specialization stays disabled by default and fails
/// closed until three clean promotion runs clear the release gate.
/// </summary>
public class DirectPtxMseLossTests
{
    [Fact]
    public void FusedMseLossEmitter_IsRegisterResidentAndPointerOnly()
    {
        string ptx = PtxFusedMseLossF32Kernel.EmitPtx(8, 6, 2048, 128);
        Assert.Contains(".maxntid 128, 1, 1", ptx);
        Assert.Contains("exact-shape rows=2048 columns=128 block=128", ptx);
        Assert.Contains("op=mse-loss", ptx);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.Equal(2, Count(ptx, "ld.global.ca.v4.f32"));
        Assert.Equal(4, Count(ptx, "sub.rn.f32"));
        Assert.Equal(4, Count(ptx, "fma.rn.f32"));
        Assert.Equal(5, Count(ptx, "shfl.sync.bfly.b32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        // 1/128 = 0x3C000000
        Assert.Contains("0f3C000000", ptx);
        Assert.Contains("ld.global.ca.v2.f32",
            PtxFusedMseLossF32Kernel.EmitPtx(8, 6, 2048, 64));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void FusedMseLossShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedMseLossF32Kernel.IsSupportedShape(256, 128));
        Assert.True(PtxFusedMseLossF32Kernel.IsSupportedShape(2048, 64));
        Assert.True(PtxFusedMseLossF32Kernel.IsSupportedShape(2048, 128));
        Assert.True(PtxFusedMseLossF32Kernel.IsSupportedShape(8192, 128));
        Assert.False(PtxFusedMseLossF32Kernel.IsSupportedShape(256, 64));
        Assert.False(PtxFusedMseLossF32Kernel.IsSupportedShape(2048, 96));
        Assert.False(PtxFusedMseLossF32Kernel.IsPromotedShape(2048, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedMseLossF32Kernel.EmitPtx(8, 6, 17, 128));
    }

    [Fact]
    public void LossCoverageManifest_AssignsEveryCellExactlyOnce()
    {
        Assert.NotEmpty(DirectPtxLossCoverageManifest.All);
        string[] names = DirectPtxLossCoverageManifest.All
            .Select(cell => cell.Api).OrderBy(name => name, StringComparer.Ordinal).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxLossCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Equal(
            DirectPtxLossCoverageStatus.ExperimentalDirectPtx,
            DirectPtxLossCoverageManifest.Get("CudaBackend.MseLoss").Status);
        Assert.Single(DirectPtxLossCoverageManifest.All,
            cell => cell.Status == DirectPtxLossCoverageStatus.ExperimentalDirectPtx);
        Assert.All(DirectPtxLossCoverageManifest.All,
            cell => Assert.NotEqual(
                DirectPtxLossCoverageStatus.PromotedDirectPtx, cell.Status));
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxLossCoverageManifest.Get("UnassignedLossApi"));
    }

    [SkippableTheory]
    [InlineData(256, 128)]
    [InlineData(2048, 64)]
    [InlineData(2048, 128)]
    [InlineData(8192, 128)]
    public void DriverOnlyFusedMseLoss_MatchesReferenceAndHasZeroLocalBytes(int rows, int columns)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in MSE-loss specialization is validated on Ampere.");
        using var kernel = new PtxFusedMseLossF32Kernel(runtime, rows, columns);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 3);
        Assert.Equal("mse-loss-f32", kernel.Blueprint.Operation);
        Assert.Equal(3, kernel.Blueprint.Tensors.Count);

        int elements = rows * columns;
        var random = new Random(20260722);
        float[] pred = Enumerable.Range(0, elements).Select(_ => (random.NextSingle() * 2f - 1f) * 4f).ToArray();
        float[] target = Enumerable.Range(0, elements).Select(_ => (random.NextSingle() * 2f - 1f) * 4f).ToArray();
        var expected = new float[rows];
        for (int row = 0; row < rows; row++)
        {
            double sum = 0;
            for (int col = 0; col < columns; col++)
            {
                double diff = pred[row * columns + col] - target[row * columns + col];
                sum += diff * diff;
            }
            expected[row] = (float)(sum / columns);
        }

        using var predBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var targetBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var lossBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        predBuffer.Upload<float>(pred);
        targetBuffer.Upload<float>(target);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(predBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(targetBuffer, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(lossBuffer, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();

        var actual = new float[rows];
        lossBuffer.Download<float>(actual);
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
