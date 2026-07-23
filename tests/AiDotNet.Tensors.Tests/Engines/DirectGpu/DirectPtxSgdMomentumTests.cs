using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape FP32 fused SGD-with-momentum direct-PTX
/// family (issue #848). The emitter and shape-domain assertions run without a
/// GPU; the driver correctness assertion is skipped unless a validated Ampere
/// device is present. The learning rate, momentum, and weight decay are baked
/// module identity. Disabled by default; fails closed until three clean
/// promotion runs clear the release gate.
/// </summary>
public class DirectPtxSgdMomentumTests
{
    [Fact]
    public void FusedSgdMomentumEmitter_IsRegisterResidentAndPointerOnly()
    {
        string ptx = PtxFusedSgdMomentumF32Kernel.EmitPtx(8, 6, 1_048_576, hasWeightDecay: true);
        Assert.Contains(".maxntid 256, 1, 1", ptx);
        Assert.Contains("exact-shape size=1048576 block=256", ptx);
        Assert.Contains("op=sgd-momentum wd=1", ptx);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        // param and velocity are read-modify-write and stay cached; the
        // gradient is read once, so it uses the read-only data cache.
        Assert.Equal(2, Count(ptx, "ld.global.ca.v4.f32"));
        Assert.Equal(1, Count(ptx, "ld.global.nc.v4.f32"));
        Assert.Equal(2, Count(ptx, "st.global.v4.f32"));
        // With weight decay: 3 fma per element * 4 elements = 12.
        Assert.Equal(12, Count(ptx, "fma.rn.f32"));
        // Without weight decay: 2 fma per element * 4 = 8, and wd=0 marker.
        string noWd = PtxFusedSgdMomentumF32Kernel.EmitPtx(8, 6, 65_536, hasWeightDecay: false);
        Assert.Contains("op=sgd-momentum wd=0", noWd);
        Assert.Equal(8, Count(noWd, "fma.rn.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        // Three scalars travel as launch parameters, so the module key is the
        // shape plus the decay presence - one module for a whole run.
        Assert.Equal(3, Count(ptx, ".param .f32"));
        Assert.Equal(
            PtxFusedSgdMomentumF32Kernel.EmitPtx(8, 6, 1_048_576, hasWeightDecay: true),
            PtxFusedSgdMomentumF32Kernel.EmitPtx(8, 6, 1_048_576, hasWeightDecay: true));
    }

    [Fact]
    public void FusedSgdMomentumShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedSgdMomentumF32Kernel.IsSupportedShape(65_536));
        Assert.True(PtxFusedSgdMomentumF32Kernel.IsSupportedShape(262_144));
        Assert.True(PtxFusedSgdMomentumF32Kernel.IsSupportedShape(1_048_576));
        Assert.True(PtxFusedSgdMomentumF32Kernel.IsSupportedShape(4_194_304));
        Assert.False(PtxFusedSgdMomentumF32Kernel.IsSupportedShape(65_535));
        Assert.False(PtxFusedSgdMomentumF32Kernel.IsSupportedShape(1_000_000));
        Assert.False(PtxFusedSgdMomentumF32Kernel.IsPromotedShape(1_048_576));
        // The hyperparameters are launch parameters now, so they are validated
        // when the kernel is constructed rather than when the module is emitted;
        // the module itself no longer depends on their values.
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedSgdMomentumF32Kernel.EmitPtx(8, 6, 1_000_000, hasWeightDecay: false));
    }

    [Fact]
    public void OptimizerCoverageManifest_AssignsEveryCellExactlyOnce()
    {
        Assert.NotEmpty(DirectPtxOptimizerCoverageManifest.All);
        string[] names = DirectPtxOptimizerCoverageManifest.All
            .Select(cell => cell.Api).OrderBy(name => name, StringComparer.Ordinal).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxOptimizerCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Equal(
            DirectPtxOptimizerCoverageStatus.ExperimentalDirectPtx,
            DirectPtxOptimizerCoverageManifest.Get("CudaBackend.SgdMomentumUpdate").Status);
        Assert.Equal(
            DirectPtxOptimizerCoverageStatus.ExperimentalDirectPtx,
            DirectPtxOptimizerCoverageManifest.Get("CudaBackend.AdamUpdate").Status);
        // Name the live cells rather than counting them: this pins WHICH cells
        // are experimental, so a new cell cannot quietly take a live slot.
        Assert.Equal(
            new[] { "CudaBackend.AdamUpdate", "CudaBackend.SgdMomentumUpdate" },
            DirectPtxOptimizerCoverageManifest.All
                .Where(cell => cell.Status == DirectPtxOptimizerCoverageStatus.ExperimentalDirectPtx)
                .Select(cell => cell.Api)
                .OrderBy(api => api, StringComparer.Ordinal)
                .ToArray());
        // Eight cells named NVRTC kernels and backend ops that do not exist.
        // Each must now declare itself blocked so it cannot read as a port.
        Assert.All(
            DirectPtxOptimizerCoverageManifest.All
                .Where(cell => cell.ExistingImplementation.StartsWith("none", StringComparison.Ordinal)),
            cell => Assert.StartsWith("blocked:", cell.DirectPtxAssignment, StringComparison.Ordinal));
        Assert.Equal(8, DirectPtxOptimizerCoverageManifest.All
            .Count(cell => cell.ExistingImplementation.StartsWith("none", StringComparison.Ordinal)));
        Assert.All(DirectPtxOptimizerCoverageManifest.All,
            cell => Assert.NotEqual(
                DirectPtxOptimizerCoverageStatus.PromotedDirectPtx, cell.Status));
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxOptimizerCoverageManifest.Get("UnassignedOptimizerApi"));
    }

    [SkippableTheory]
    [InlineData(65_536, 0.1f, 0.9f, 0f)]
    [InlineData(262_144, 0.05f, 0.95f, 1e-4f)]
    [InlineData(1_048_576, 0.01f, 0.9f, 5e-4f)]
    public void DriverOnlyFusedSgdMomentum_MatchesReferenceAndHasZeroLocalBytes(
        int size, float lr, float momentum, float weightDecay)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in SGD-momentum specialization is validated on Ampere.");
        using var kernel = new PtxFusedSgdMomentumF32Kernel(runtime, size, lr, momentum, weightDecay);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 3);
        Assert.Equal("sgd-momentum-f32", kernel.Blueprint.Operation);
        Assert.Equal(3, kernel.Blueprint.Tensors.Count);

        var random = RandomHelper.CreateSeededRandom(20260722);
        float[] param = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 2.0 - 1.0)).ToArray();
        float[] grad = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 2.0 - 1.0) * 0.5f).ToArray();
        float[] velocity = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 2.0 - 1.0) * 0.25f).ToArray();

        var expectedParam = new float[size];
        var expectedVelocity = new float[size];
        for (int i = 0; i < size; i++)
        {
            float g = grad[i] + weightDecay * param[i];
            float v = momentum * velocity[i] + g;
            expectedVelocity[i] = v;
            expectedParam[i] = param[i] - lr * v;
        }

        using var paramBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var gradBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var velocityBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        paramBuffer.Upload<float>(param);
        gradBuffer.Upload<float>(grad);
        velocityBuffer.Upload<float>(velocity);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(paramBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(gradBuffer, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(velocityBuffer, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();

        var actualParam = new float[size];
        var actualVelocity = new float[size];
        paramBuffer.Download<float>(actualParam);
        velocityBuffer.Download<float>(actualVelocity);
        for (int i = 0; i < size; i++)
        {
            Assert.True(MathF.Abs(actualVelocity[i] - expectedVelocity[i]) <= 1e-5f * (MathF.Abs(expectedVelocity[i]) + 1f),
                $"velocity {i}: actual {actualVelocity[i]:G9}, expected {expectedVelocity[i]:G9}.");
            Assert.True(MathF.Abs(actualParam[i] - expectedParam[i]) <= 1e-5f * (MathF.Abs(expectedParam[i]) + 1f),
                $"param {i}: actual {actualParam[i]:G9}, expected {expectedParam[i]:G9}.");
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
