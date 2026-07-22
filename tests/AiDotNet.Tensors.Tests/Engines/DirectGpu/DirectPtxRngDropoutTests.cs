#if NET5_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class DirectPtxRngDropoutTests
{
    [Fact]
    public void EmitterBakesExactShapeAndHasNoTailOrStridePath()
    {
        string ptx = PtxFusedPhiloxDropoutF32Kernel.EmitPtx(8, 6, 65_536);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxFusedPhiloxDropoutF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Contains("exact_elements=65536", ptx, StringComparison.Ordinal);
        Assert.Contains("no_tail_branch=1", ptx, StringComparison.Ordinal);
        Assert.Contains("ld.global.v4.f32", ptx, StringComparison.Ordinal);
        Assert.Contains("st.global.v4.f32", ptx, StringComparison.Ordinal);
        Assert.Contains("0xD2511F53", ptx, StringComparison.Ordinal);
        Assert.Contains("0xCD9E8D57", ptx, StringComparison.Ordinal);
        Assert.Equal(10, Count(ptx, "// Philox4x32-10 round"));
        Assert.DoesNotContain(".param .u32 size", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("size_ptr", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("setp.ge", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(4_096, true)]
    [InlineData(65_536, true)]
    [InlineData(1_048_576, true)]
    [InlineData(4_095, false)]
    [InlineData(65_540, false)]
    public void ExactShapeBucketsAreExplicit(int elements, bool expected) =>
        Assert.Equal(expected, PtxFusedPhiloxDropoutF32Kernel.IsSupportedElementCount(elements));

    [Fact]
    public void EmitterRejectsEveryUnmeasuredSm()
    {
        Assert.Throws<PlatformNotSupportedException>(() =>
            PtxFusedPhiloxDropoutF32Kernel.EmitPtx(8, 9, 4_096));
    }

    [Theory]
    [InlineData((int)DirectPtxPhiloxFillKind.Uniform)]
    [InlineData((int)DirectPtxPhiloxFillKind.Normal)]
    [InlineData((int)DirectPtxPhiloxFillKind.BernoulliMask)]
    [InlineData((int)DirectPtxPhiloxFillKind.DropThresholdMask)]
    public void FillEmitterUsesExactFloat4PhiloxAbiWithoutTailOrLocalStorage(
        int kindValue)
    {
        var kind = (DirectPtxPhiloxFillKind)kindValue;
        string ptx = PtxPhiloxFillF32Kernel.EmitPtx(8, 6, kind, 65_536);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains("exact_elements=65536", ptx, StringComparison.Ordinal);
        Assert.Contains("values_per_thread=4", ptx, StringComparison.Ordinal);
        Assert.Contains("no_tail_branch=1", ptx, StringComparison.Ordinal);
        Assert.Equal(10, Count(ptx, "// Philox4x32-10 round"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.DoesNotContain("ld.global", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("setp.ge", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32 size", ptx, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void UniformFillEmitterScalesPhiloxWordsIntoRequestedRange()
    {
        string ptx = PtxPhiloxFillF32Kernel.EmitPtx(
            8, 6, DirectPtxPhiloxFillKind.Uniform, 4_096);

        Assert.Equal(4, Count(ptx, "cvt.rn.f32.u32"));
        Assert.Contains("sub.rn.f32 %f10, %f9, %f8", ptx, StringComparison.Ordinal);
        Assert.Equal(4, Count(ptx, "fma.rn.f32"));
        Assert.Contains("0f2F800000", ptx, StringComparison.Ordinal);
        Assert.Contains("0f3F7FFFFF", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void NormalFillEmitterUsesTwoBoxMullerPairs()
    {
        string ptx = PtxPhiloxFillF32Kernel.EmitPtx(
            8, 6, DirectPtxPhiloxFillKind.Normal, 4_096);

        Assert.Equal(2, Count(ptx, "lg2.approx.f32"));
        Assert.Equal(2, Count(ptx, "sqrt.approx.f32"));
        Assert.Equal(2, Count(ptx, "cos.approx.f32"));
        Assert.Equal(2, Count(ptx, "sin.approx.f32"));
        Assert.Equal(4, Count(ptx, "fma.rn.f32"));
    }

    [Fact]
    public void KeepAndDropThresholdMasksHaveOppositeSelectionSemantics()
    {
        string keep = PtxPhiloxFillF32Kernel.EmitPtx(
            8, 6, DirectPtxPhiloxFillKind.BernoulliMask, 4_096);
        string drop = PtxPhiloxFillF32Kernel.EmitPtx(
            8, 6, DirectPtxPhiloxFillKind.DropThresholdMask, 4_096);

        Assert.Contains("ld.param.u32 %r22, [threshold]", keep, StringComparison.Ordinal);
        Assert.Contains("ld.param.u32 %r22, [threshold]", drop, StringComparison.Ordinal);
        Assert.Contains("selp.f32 %f0, %f8, 0f00000000, %p0", keep, StringComparison.Ordinal);
        Assert.Contains("selp.f32 %f0, 0f00000000, %f8, %p0", drop, StringComparison.Ordinal);
    }

    [Fact]
    public void FillEmitterRejectsEveryUnmeasuredSm()
    {
        Assert.Throws<PlatformNotSupportedException>(() =>
            PtxPhiloxFillF32Kernel.EmitPtx(
                8, 9, DirectPtxPhiloxFillKind.Uniform, 4_096));
    }

    [Fact]
    public void DropoutBackwardEmitterIsOneExactFloat4Multiply()
    {
        string ptx = PtxDropoutBackwardF32Kernel.EmitPtx(8, 6, 65_536);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxDropoutBackwardF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Contains("exact_elements=65536", ptx, StringComparison.Ordinal);
        Assert.Contains("no_tail_branch=1", ptx, StringComparison.Ordinal);
        Assert.Equal(2, Count(ptx, "ld.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "mul.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.DoesNotContain(".param .u32 size", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("dropout_rate", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("setp.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void DropoutBackwardEmitterRejectsEveryUnmeasuredSm()
    {
        Assert.Throws<PlatformNotSupportedException>(() =>
            PtxDropoutBackwardF32Kernel.EmitPtx(8, 9, 4_096));
    }

    [Fact]
    public void GumbelSoftmaxEmitterFusesPhiloxAndWarpSoftmaxWithoutIntermediates()
    {
        string ptx = PtxFusedGumbelSoftmax32F32Kernel.EmitPtx(8, 6, 2_048);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxFusedGumbelSoftmax32F32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Contains("exact_shape=[2048,32]", ptx, StringComparison.Ordinal);
        Assert.Contains("one_warp_per_row=1", ptx, StringComparison.Ordinal);
        Assert.Contains("no_tail_branch=1", ptx, StringComparison.Ordinal);
        Assert.Equal(10, Count(ptx, "// Philox4x32-10 round"));
        Assert.Equal(10, Count(ptx, "shfl.sync.down.b32"));
        Assert.Equal(2, Count(ptx, "shfl.sync.idx.b32"));
        Assert.Equal(2, Count(ptx, "lg2.approx.f32"));
        Assert.Equal(1, Count(ptx, "ex2.approx.f32"));
        Assert.Equal(1, Count(ptx, "ld.global.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("outer_size", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("inner_size", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData(128, 32, true)]
    [InlineData(2_048, 32, true)]
    [InlineData(32_768, 32, true)]
    [InlineData(128, 31, false)]
    [InlineData(129, 32, false)]
    public void GumbelSoftmaxShapeBucketsAreExplicit(
        int outerSize, int innerSize, bool expected) =>
        Assert.Equal(expected,
            PtxFusedGumbelSoftmax32F32Kernel.IsSupportedShape(outerSize, innerSize));

    [Fact]
    public void GumbelSoftmaxEmitterRejectsEveryUnmeasuredSm()
    {
        Assert.Throws<PlatformNotSupportedException>(() =>
            PtxFusedGumbelSoftmax32F32Kernel.EmitPtx(8, 9, 128));
    }

    [Fact]
    public void ImportanceSamplingEmitterStagesEachRayOnceAndUnrollsCdf()
    {
        string ptx = PtxFusedImportanceSampling64F32Kernel.EmitPtx(8, 6, 1_024);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxFusedImportanceSampling64F32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Contains("exact_shape=[1024,64,64]", ptx, StringComparison.Ordinal);
        Assert.Contains("one_warp_per_ray=1", ptx, StringComparison.Ordinal);
        Assert.Contains("cdf_unrolled=64", ptx, StringComparison.Ordinal);
        Assert.Equal(20, Count(ptx, "// Philox4x32-10 round"));
        Assert.Equal(4, Count(ptx, "ld.global.f32"));
        Assert.Equal(4, Count(ptx, "st.shared.f32"));
        Assert.Equal(2, Count(ptx, "st.global.f32"));
        Assert.Equal(1, Count(ptx, "bar.sync 0"));
        Assert.Equal(128, Count(ptx, "ld.shared.f32 %f11, [weights_shared+"));
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("num_rays", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("num_coarse", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("num_fine", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData(64, 64, 64, true)]
    [InlineData(1_024, 64, 64, true)]
    [InlineData(16_384, 64, 64, true)]
    [InlineData(64, 32, 64, false)]
    [InlineData(64, 64, 32, false)]
    [InlineData(65, 64, 64, false)]
    public void ImportanceSamplingShapeBucketsAreExplicit(
        int rays, int coarse, int fine, bool expected) =>
        Assert.Equal(expected,
            PtxFusedImportanceSampling64F32Kernel.IsSupportedShape(rays, coarse, fine));

    [Fact]
    public void ImportanceSamplingEmitterRejectsEveryUnmeasuredSm()
    {
        Assert.Throws<PlatformNotSupportedException>(() =>
            PtxFusedImportanceSampling64F32Kernel.EmitPtx(8, 9, 64));
    }

    [Fact]
    public void BiasDropoutEmitterFusesBiasPhiloxMaskAndOutputWithoutTemporary()
    {
        string ptx = PtxFusedBiasPhiloxDropout256F32Kernel.EmitPtx(8, 6, 256);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxFusedBiasPhiloxDropout256F32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Contains("exact_shape=[256,256]", ptx, StringComparison.Ordinal);
        Assert.Contains("no_tail_branch=1", ptx, StringComparison.Ordinal);
        Assert.Equal(10, Count(ptx, "// Philox4x32-10 round"));
        Assert.Equal(2, Count(ptx, "ld.global.v4.f32"));
        Assert.Equal(2, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "add.rn.f32 %f1"));
        Assert.Equal(4, Count(ptx, "selp.f32"));
        Assert.Contains("and.b32 %r23, %r2, 63", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32 rows", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32 cols", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData(16, 256, true)]
    [InlineData(256, 256, true)]
    [InlineData(4_096, 256, true)]
    [InlineData(16, 128, false)]
    [InlineData(17, 256, false)]
    public void BiasDropoutShapeBucketsAreExplicit(int rows, int cols, bool expected) =>
        Assert.Equal(expected,
            PtxFusedBiasPhiloxDropout256F32Kernel.IsSupportedShape(rows, cols));

    [Fact]
    public void BiasDropoutEmitterRejectsEveryUnmeasuredSm()
    {
        Assert.Throws<PlatformNotSupportedException>(() =>
            PtxFusedBiasPhiloxDropout256F32Kernel.EmitPtx(8, 9, 16));
    }

    [Fact]
    public void DdimEmitterIsOneExactFloat4LinearCombination()
    {
        string ptx = PtxFusedDdimStepF32Kernel.EmitPtx(8, 6, 65_536);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxFusedDdimStepF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Contains("exact_elements=65536", ptx, StringComparison.Ordinal);
        Assert.Contains("no_tail_branch=1", ptx, StringComparison.Ordinal);
        Assert.Equal(2, Count(ptx, "ld.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.DoesNotContain("sqrt", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32 size", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void DdimEmitterRejectsEveryUnmeasuredSm()
    {
        Assert.Throws<PlatformNotSupportedException>(() =>
            PtxFusedDdimStepF32Kernel.EmitPtx(8, 9, 4_096));
    }

    [Fact]
    public void CategoricalEmitterFusesWarpCdfPhiloxAndOneHotWrite()
    {
        string ptx = PtxPhiloxCategorical32F32Kernel.EmitPtx(8, 6, 2_048);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxPhiloxCategorical32F32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Contains("exact_shape=[2048,32]", ptx, StringComparison.Ordinal);
        Assert.Contains("one_warp_per_row=1", ptx, StringComparison.Ordinal);
        Assert.Equal(10, Count(ptx, "// Philox4x32-10 round"));
        Assert.Equal(5, Count(ptx, "shfl.sync.up.b32"));
        Assert.Equal(1, Count(ptx, "shfl.sync.idx.b32"));
        Assert.Equal(1, Count(ptx, "ld.global.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32 rows", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32 classes", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void CpuCategoricalOracleIsSeededAndOneHot()
    {
        var engine = new CpuEngine();
        var probabilities = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(
            Enumerable.Repeat(1f, 64).ToArray(), new[] { 2, 32 });

        var first = engine.TensorCategoricalSample(probabilities, axis: -1, seed: 42);
        var repeat = engine.TensorCategoricalSample(probabilities, axis: -1, seed: 42);

        Assert.Equal(first.GetDataArray(), repeat.GetDataArray());
        Assert.Equal(1f, first.GetDataArray().Take(32).Sum());
        Assert.Equal(1f, first.GetDataArray().Skip(32).Take(32).Sum());
        Assert.All(first.GetDataArray(), value => Assert.True(value is 0f or 1f));
    }

    [Fact]
    public void CpuCategoricalOracleRejectsInvalidProbabilities()
    {
        var engine = new CpuEngine();
        var invalid = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(
            new float[] { 0f, -1f, 2f, 0f }, new[] { 2, 2 });

        Assert.Throws<ArgumentException>(() =>
            engine.TensorCategoricalSample(invalid, axis: -1, seed: 7));
    }

    [Theory]
    [InlineData(128, 32, true)]
    [InlineData(2_048, 32, true)]
    [InlineData(32_768, 32, true)]
    [InlineData(128, 16, false)]
    [InlineData(129, 32, false)]
    public void CategoricalShapeBucketsAreExplicit(int rows, int classes, bool expected) =>
        Assert.Equal(expected, PtxPhiloxCategorical32F32Kernel.IsSupportedShape(rows, classes));

    [Fact]
    public void CategoricalEmitterRejectsEveryUnmeasuredSm()
    {
        Assert.Throws<PlatformNotSupportedException>(() =>
            PtxPhiloxCategorical32F32Kernel.EmitPtx(8, 9, 128));
    }

    [Fact]
    public void GumbelBackwardEmitterFusesJacobianReductionAndTemperatureScale()
    {
        string ptx = PtxGumbelSoftmaxBackward32F32Kernel.EmitPtx(8, 6, 2_048);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxGumbelSoftmaxBackward32F32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Contains("exact_shape=[2048,32]", ptx, StringComparison.Ordinal);
        Assert.Equal(5, Count(ptx, "shfl.sync.down.b32"));
        Assert.Equal(1, Count(ptx, "shfl.sync.idx.b32"));
        Assert.Equal(2, Count(ptx, "ld.global.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32 rows", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32 classes", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void GumbelBackwardEmitterRejectsEveryUnmeasuredSm()
    {
        Assert.Throws<PlatformNotSupportedException>(() =>
            PtxGumbelSoftmaxBackward32F32Kernel.EmitPtx(8, 9, 128));
    }

    [Fact]
    public void FusedRreluEmitterKeepsPhiloxSlopesAndConsumerInOneLaunch()
    {
        string ptx = PtxFusedPhiloxRreluF32Kernel.EmitPtx(8, 6, 65_536);

        Assert.Contains(PtxFusedPhiloxRreluF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Contains("ld.global.v4.f32", ptx, StringComparison.Ordinal);
        Assert.Equal(2, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "setp.ge.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32 size", ptx, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData((int)DirectPtxRreluKind.Forward, 2, 1)]
    [InlineData((int)DirectPtxRreluKind.Backward, 3, 1)]
    public void SavedNoiseRreluEmittersAreExactFloat4Kernels(
        int kindValue,
        int expectedLoads,
        int expectedStores)
    {
        var kind = (DirectPtxRreluKind)kindValue;
        string ptx = PtxRreluF32Kernel.EmitPtx(8, 6, kind, 65_536);

        Assert.Equal(expectedLoads, Count(ptx, "ld.global.v4.f32"));
        Assert.Equal(expectedStores, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "setp.ge.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32 size", ptx, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RreluEmitterRejectsEveryUnmeasuredSm()
    {
        Assert.Throws<PlatformNotSupportedException>(() =>
            PtxFusedPhiloxRreluF32Kernel.EmitPtx(8, 9, 4_096));
        Assert.Throws<PlatformNotSupportedException>(() =>
            PtxRreluF32Kernel.EmitPtx(8, 9, DirectPtxRreluKind.Forward, 4_096));
    }

    [Fact]
    public void PhiloxReferenceMatchesPublishedZeroVector()
    {
        var actual = PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(0, 0, 0);

        Assert.Equal(0x6627E8D5u, actual.X0);
        Assert.Equal(0xE169C58Du, actual.X1);
        Assert.Equal(0xBC57AC4Cu, actual.X2);
        Assert.Equal(0x9B00DBD8u, actual.X3);
    }

    [Fact]
    public void SeedSubsequenceAndCounterFormDeterministicIndependentDomains()
    {
        var first = PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(
            0x0123_4567_89AB_CDEFul, 17, 29);
        var repeat = PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(
            0x0123_4567_89AB_CDEFul, 17, 29);

        Assert.Equal(first, repeat);
        Assert.NotEqual(first, PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(
            0x0123_4567_89AB_CDEEul, 17, 29));
        Assert.NotEqual(first, PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(
            0x0123_4567_89AB_CDEFul, 18, 29));
        Assert.NotEqual(first, PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(
            0x0123_4567_89AB_CDEFul, 17, 30));
    }

    [Fact]
    public void AdmissionProvesExactExtentAlignmentAndNumericalContract()
    {
        const int elements = 4_096;
        const long bytes = elements * sizeof(float);
        bool admitted = DirectPtxRngDropoutAdmission.TryValidate(
            (IntPtr)0x10000, bytes,
            (IntPtr)0x20000, bytes,
            (IntPtr)0x30000, bytes,
            elements, 0.1f, 8, 6,
            out DirectPtxRngDropoutParameters parameters,
            out string? rejection);

        Assert.True(admitted, rejection);
        Assert.Null(rejection);
        Assert.NotEqual(0u, parameters.KeepThreshold);
        Assert.Equal(1.0f / 0.9f, parameters.InverseKeep, 5);
    }

    [Theory]
    [InlineData(8, 9, "rng-dropout-exact-sm-not-supported")]
    [InlineData(9, 0, "rng-dropout-exact-sm-not-supported")]
    public void AdmissionRejectsUnmeasuredArchitectures(
        int major, int minor, string expectedReason)
    {
        Assert.False(DirectPtxRngDropoutAdmission.TryValidate(
            (IntPtr)0x10000, 16_384,
            (IntPtr)0x20000, 16_384,
            (IntPtr)0x30000, 16_384,
            4_096, 0.1f, major, minor,
            out _, out string? rejection));
        Assert.Equal(expectedReason, rejection);
    }

    [Fact]
    public void AdmissionRejectsAliasingBeforeDispatch()
    {
        Assert.False(DirectPtxRngDropoutAdmission.TryValidate(
            (IntPtr)0x10000, 16_384,
            (IntPtr)0x10000, 16_384,
            (IntPtr)0x30000, 16_384,
            4_096, 0.1f, 8, 6,
            out _, out string? rejection));
        Assert.Equal("rng-dropout-alias-not-supported", rejection);
    }

    [Fact]
    public void AdmissionRejectsOversizedViewsInsteadOfSilentlyAcceptingThem()
    {
        Assert.False(DirectPtxRngDropoutAdmission.TryValidate(
            (IntPtr)0x10000, 16_388,
            (IntPtr)0x20000, 16_384,
            (IntPtr)0x30000, 16_384,
            4_096, 0.1f, 8, 6,
            out _, out string? rejection));
        Assert.Equal("rng-dropout-physical-extent-mismatch", rejection);
    }

    [Theory]
    [InlineData("shape", "rng-dropout-exact-shape-not-supported")]
    [InlineData("rate", "rng-dropout-rate-not-supported")]
    [InlineData("pointer", "rng-dropout-invalid-device-pointer")]
    [InlineData("alignment", "rng-dropout-alignment-mismatch")]
    [InlineData("range", "rng-dropout-invalid-pointer-range")]
    public void AdmissionHasStableReasonsForEveryPhysicalRejection(
        string scenario, string expectedReason)
    {
        int elements = scenario == "shape" ? 4_100 : 4_096;
        long bytes = checked((long)elements * sizeof(float));
        IntPtr input = scenario switch
        {
            "pointer" => IntPtr.Zero,
            "alignment" => (IntPtr)0x10004,
            "range" => new IntPtr(-16),
            _ => (IntPtr)0x10000
        };
        float rate = scenario == "rate" ? 1.0f : 0.1f;

        Assert.False(DirectPtxRngDropoutAdmission.TryValidate(
            input, bytes,
            (IntPtr)0x20000, bytes,
            (IntPtr)0x30000, bytes,
            elements, rate, 8, 6,
            out _, out string? rejection));
        Assert.Equal(expectedReason, rejection);
    }

    [Fact]
    public void PinnedCaptureModuleCannotBeEvictedAndIsDisposedWithOwner()
    {
        using var cache = new DirectPtxKernelCache<int, DisposableProbe>(1);
        var pinned = new DisposableProbe();
        Assert.Same(pinned, cache.AddOrGetExisting(1, pinned));
        Assert.True(cache.Pin(1));
        var rejected = new DisposableProbe();

        Assert.Throws<InvalidOperationException>(() => cache.AddOrGetExisting(2, rejected));
        Assert.True(rejected.IsDisposed);
        Assert.False(pinned.IsDisposed);
        cache.Dispose();
        Assert.True(pinned.IsDisposed);
    }

    [Fact]
    public void CoverageManifestAssignsEachFamilyAndExcludesSecureRandom()
    {
        Assert.NotEmpty(DirectPtxRngCoverageManifest.All);
        Assert.Equal(
            DirectPtxRngCoverageManifest.All.Count,
            DirectPtxRngCoverageManifest.All.Select(cell => cell.Api).Distinct(StringComparer.Ordinal).Count());
        DirectPtxRngCoverageCell secure = DirectPtxRngCoverageManifest.Get(
            "Cryptographic and SecureSeededRandom APIs");
        Assert.Equal(DirectPtxRngCoverageStatus.ExplicitlyExcluded, secure.Status);
        Assert.Contains("fail closed", secure.DirectPtxAssignment, StringComparison.OrdinalIgnoreCase);

        string[] directApis =
        [
            "CudaBackend.Dropout", "IEngine.Dropout", "DirectGpuTensorEngine.DropoutGpu",
            "CudaBackend.DropoutBackward", "IEngine.DropoutBackward",
            "DirectGpuTensorEngine.DropoutBackwardGpu", "CudaBackend.DropoutMask",
            "CudaBackend.GenerateStatelessDropoutMask", "IEngine.TensorDropoutMask",
            "DirectGpuTensorEngine.FusedBiasDropout", "IEngine.GenerateDropoutMask",
            "CudaBackend.GaussianNoise", "IEngine.GenerateGaussianNoise",
            "CudaBackend.GenerateRandomUniform", "DirectGpuTensorEngine.RandomUniformGpu",
            "CudaBackend.GenerateRandomNormal", "DirectGpuTensorEngine.RandomNormalGpu",
            "CudaBackend.GumbelSoftmax", "IEngine.GumbelSoftmax",
            "IEngine.GumbelSoftmaxBackward", "CudaBackend.ImportanceSampling",
            "IEngine.ImportanceSampling", "IFusedAdvancedKernels.FusedDDIMStep",
            "CudaBackend.RRelu", "CudaBackend.RReluBackward", "IEngine.TensorRReLU",
            "IEngine.TensorCategoricalSample", "Tensor.CreateRandom",
            "IEngine.TensorRandomUniform", "IEngine.TensorRandomUniformRange",
            "IEngine.TensorRandomUniformRangeInto", "IEngine.TensorRandomNormal",
            "IEngine.TensorRandomNormalInto"
        ];
        Assert.All(directApis, api => Assert.Equal(
            DirectPtxRngCoverageStatus.ExperimentalDirectPtx,
            DirectPtxRngCoverageManifest.Get(api).Status));

        foreach (string hostApi in new[] { "CuRand.Uniform", "CuRand.Normal", "CuRand.Bernoulli" })
        {
            DirectPtxRngCoverageCell host = DirectPtxRngCoverageManifest.Get(hostApi);
            Assert.Equal(DirectPtxRngCoverageStatus.ExplicitlyExcluded, host.Status);
            Assert.Contains("host", host.Semantics, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("fail closed", host.DirectPtxAssignment, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void FeatureGateIsKernelSpecificAndOptIn()
    {
        Assert.Equal("AIDOTNET_DIRECT_PTX_RNG_DROPOUT",
            DirectPtxFeatureGate.RngDropoutEnvironmentVariable);
        Assert.True(DirectPtxArchitecture.HasExperimentalRngDropout(8, 6));
        Assert.False(DirectPtxArchitecture.HasExperimentalRngDropout(8, 9));
    }

    private static int Count(string text, string value)
    {
        int count = 0;
        int start = 0;
        while ((start = text.IndexOf(value, start, StringComparison.Ordinal)) >= 0)
        {
            count++;
            start += value.Length;
        }
        return count;
    }

    private sealed class DisposableProbe : IDisposable
    {
        internal bool IsDisposed { get; private set; }
        public void Dispose() => IsDisposed = true;
    }
}
#endif
