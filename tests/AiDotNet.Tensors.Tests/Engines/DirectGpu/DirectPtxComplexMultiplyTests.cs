using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>Static and opt-in driver contracts for issue #850.</summary>
public class DirectPtxComplexMultiplyTests
{
    [Fact]
    public void ComplexMultiplyEmitter_IsPointerOnlyAndRegisterResident()
    {
        string ptx = PtxFusedComplexMultiplyF32Kernel.EmitPtx(8, 6, 262144);
        Assert.Contains("exact-shape pairs=262144 block=256", ptx);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.Equal(2, Count(ptx, "ld.global.nc.v2.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v2.f32"));
        Assert.Equal(2, Count(ptx, "mul.rn.f32"));
        Assert.Equal(2, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "neg.f32"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void ComplexMultiplyDomain_IsClosedExactSmAndUnpromoted()
    {
        Assert.True(PtxFusedComplexMultiplyF32Kernel.IsSupportedShape(65536));
        Assert.True(PtxFusedComplexMultiplyF32Kernel.IsSupportedShape(262144));
        Assert.True(PtxFusedComplexMultiplyF32Kernel.IsSupportedShape(1048576));
        Assert.True(PtxFusedComplexMultiplyF32Kernel.IsSupportedShape(4194304));
        Assert.False(PtxFusedComplexMultiplyF32Kernel.IsSupportedShape(1024));
        Assert.False(PtxFusedComplexMultiplyF32Kernel.IsPromotedShape(262144));
        Assert.True(DirectPtxArchitecture.HasValidatedComplexMultiply(8, 6));
        Assert.False(DirectPtxArchitecture.HasValidatedComplexMultiply(8, 0));
        Assert.False(DirectPtxArchitecture.HasValidatedComplexMultiply(8, 9));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedComplexMultiplyF32Kernel.EmitPtx(8, 6, 1024));
    }

    [Fact]
    public void SpectralCoverageManifest_AssignsEveryCellExactlyOnce()
    {
        Assert.True(DirectPtxSpectralCoverageManifest.All.Count >= 36);
        string[] names = DirectPtxSpectralCoverageManifest.All
            .Select(cell => cell.Api).OrderBy(name => name, StringComparer.Ordinal).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxSpectralCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        // Name the live cells rather than counting them: this pins WHICH cells
        // are experimental, so a new cell cannot quietly take a live slot.
        Assert.Equal(
            new[]
            {
                "CudaBackend.ComplexConjugate",
                "CudaBackend.ComplexMagnitude",
                "CudaBackend.ComplexMultiply",
                "CudaBackend.SplitComplexMagnitude",
                "CudaBackend.SplitComplexMagnitudeSquared",
            },
            DirectPtxSpectralCoverageManifest.All
                .Where(cell => cell.Status == DirectPtxSpectralCoverageStatus.ExperimentalDirectPtx)
                .Select(cell => cell.Api)
                .OrderBy(api => api, StringComparer.Ordinal)
                .ToArray());
        Assert.DoesNotContain(DirectPtxSpectralCoverageManifest.All,
            cell => cell.Status == DirectPtxSpectralCoverageStatus.PromotedDirectPtx);
        Assert.Equal(
            DirectPtxSpectralCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSpectralCoverageManifest.Get("CudaBackend.ComplexMultiply").Status);
    }

    [Fact]
    public void ComplexMultiplyExperimentOverride_IsThreadLocal()
    {
        bool original = DirectPtxFeatureGate.ComplexMultiplyExperimentOverride;
        try
        {
            DirectPtxFeatureGate.ComplexMultiplyExperimentOverride = true;
            bool workerValue = true;
            var worker = new System.Threading.Thread(() =>
                workerValue = DirectPtxFeatureGate.ComplexMultiplyExperimentOverride);
            worker.Start();
            worker.Join();
            Assert.True(DirectPtxFeatureGate.ComplexMultiplyExperimentOverride);
            Assert.False(workerValue);
        }
        finally
        {
            DirectPtxFeatureGate.ComplexMultiplyExperimentOverride = original;
        }
    }

    [Fact]
    public void ComplexMultiplyGateOverride_IsThreadLocalAndNullable()
    {
        bool? original = DirectPtxFeatureGate.ComplexMultiplyGateOverride;
        try
        {
            DirectPtxFeatureGate.ComplexMultiplyGateOverride = false;
            bool? workerValue = true;
            var worker = new System.Threading.Thread(() =>
                workerValue = DirectPtxFeatureGate.ComplexMultiplyGateOverride);
            worker.Start();
            worker.Join();
            Assert.False(DirectPtxFeatureGate.ComplexMultiplyGateOverride);
            Assert.Null(workerValue);
        }
        finally
        {
            DirectPtxFeatureGate.ComplexMultiplyGateOverride = original;
        }
    }

    [SkippableTheory]
    [InlineData(65536)]
    [InlineData(262144)]
    [InlineData(1048576)]
    [InlineData(4194304)]
    public void DriverOnlyComplexMultiply_MatchesDoubleOracleAndRejectsLocalBytes(int numPairs)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexMultiply(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        using var kernel = new PtxFusedComplexMultiplyF32Kernel(runtime, numPairs);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);

        var left = new float[numPairs * 2];
        var right = new float[numPairs * 2];
        // Seeded so an oracle mismatch on the admitted SM86 machine reproduces exactly.
        // NextDouble (not NextSingle) keeps the harness compiling on net471.
        var random = RandomHelper.CreateSeededRandom(20260722 + numPairs);
        for (int i = 0; i < left.Length; i++)
        {
            left[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            right[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }
        using var leftBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var rightBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        leftBuffer.Upload<float>(left);
        rightBuffer.Upload<float>(right);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(leftBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(rightBuffer, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[left.Length];
        outputBuffer.Download<float>(actual);
        for (int pair = 0; pair < numPairs; pair++)
        {
            int offset = pair * 2;
            double ar = left[offset], ai = left[offset + 1];
            double br = right[offset], bi = right[offset + 1];
            float expectedReal = (float)(ar * br - ai * bi);
            float expectedImaginary = (float)(ar * bi + ai * br);
            Assert.True(MathF.Abs(actual[offset] - expectedReal) <= 3e-6f);
            Assert.True(MathF.Abs(actual[offset + 1] - expectedImaginary) <= 3e-6f);
        }
    }

    [Fact]
    public void SplitComplexMagnitudeEmitter_IsPointerOnlyAndRegisterResident()
    {
        string ptx = PtxSplitComplexUnaryF32Kernel.EmitPtx(8, 6, DirectPtxSplitComplexUnaryOp.Magnitude, 262144);
        Assert.Contains("exact-shape count=262144 block=256", ptx);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));   // re, im from the two split buffers
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(2, Count(ptx, "mul.rn.f32"));          // re*re, im*im
        Assert.Equal(1, Count(ptx, "add.rn.f32"));
        Assert.Equal(1, Count(ptx, "sqrt.rn.f32"));
        Assert.Equal(0, Count(ptx, "fma.rn.f32"));          // deliberately unfused to match sqrtf rounding
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void SplitComplexMagnitudeSquaredEmitter_OmitsSqrt()
    {
        string ptx = PtxSplitComplexUnaryF32Kernel.EmitPtx(8, 6, DirectPtxSplitComplexUnaryOp.MagnitudeSquared, 65536);
        Assert.Equal(2, Count(ptx, "mul.rn.f32"));
        Assert.Equal(1, Count(ptx, "add.rn.f32"));
        Assert.Equal(0, Count(ptx, "sqrt.rn.f32"));          // power sum, no root
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void SplitComplexUnaryDomain_IsClosedExactSmAndUnpromoted()
    {
        Assert.True(PtxSplitComplexUnaryF32Kernel.IsSupportedShape(65536));
        Assert.True(PtxSplitComplexUnaryF32Kernel.IsSupportedShape(262144));
        Assert.True(PtxSplitComplexUnaryF32Kernel.IsSupportedShape(1048576));
        Assert.True(PtxSplitComplexUnaryF32Kernel.IsSupportedShape(4194304));
        Assert.False(PtxSplitComplexUnaryF32Kernel.IsSupportedShape(1024));
        Assert.False(PtxSplitComplexUnaryF32Kernel.IsPromotedShape(262144));
        Assert.True(DirectPtxArchitecture.HasValidatedComplexUnary(8, 6));
        Assert.False(DirectPtxArchitecture.HasValidatedComplexUnary(8, 9));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxSplitComplexUnaryF32Kernel.EmitPtx(8, 6, DirectPtxSplitComplexUnaryOp.Magnitude, 1024));
    }

    [SkippableTheory]
    [InlineData(true, 65536)]
    [InlineData(false, 65536)]
    [InlineData(true, 262144)]
    [InlineData(false, 1048576)]
    public void DriverOnlySplitComplexUnary_MatchesDoubleOracle(bool magnitude, int count)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        var op = magnitude ? DirectPtxSplitComplexUnaryOp.Magnitude : DirectPtxSplitComplexUnaryOp.MagnitudeSquared;
        using var kernel = new PtxSplitComplexUnaryF32Kernel(runtime, op, count);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);

        var re = new float[count];
        var im = new float[count];
        var random = RandomHelper.CreateSeededRandom(20260850 + count + (magnitude ? 1 : 0));
        for (int i = 0; i < count; i++)
        {
            re[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            im[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }
        using var reBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var imBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        reBuffer.Upload<float>(re);
        imBuffer.Upload<float>(im);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(reBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(imBuffer, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[count];
        outputBuffer.Download<float>(actual);
        for (int i = 0; i < count; i++)
        {
            double r = re[i], m = im[i];
            float expected = magnitude ? (float)Math.Sqrt(r * r + m * m) : (float)(r * r + m * m);
            Assert.True(MathF.Abs(actual[i] - expected) <= 3e-6f);
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
