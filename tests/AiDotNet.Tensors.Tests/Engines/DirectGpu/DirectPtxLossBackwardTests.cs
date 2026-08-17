using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the FP32 loss-gradient kernels (issue #847): the MSE
/// backward pass and the MAE sign gradient. These replace kernels that feed
/// training, so the assertions below pin the arithmetic to the established
/// NVRTC kernels term for term - a gradient that merely "looks right" but
/// associates its multiplies differently would shift training results.
/// </summary>
public class DirectPtxLossBackwardTests
{
    [Fact]
    public void MseBackwardEmitter_HoistsTheBroadcastScalarAndKeepsTheAssociationOrder()
    {
        string ptx = PtxFusedLossBackwardF32Kernel.EmitPtx(
            8, 6, DirectPtxLossBackwardOp.MeanSquaredError, 65_536);

        Assert.Contains(".visible .entry aidotnet_fused_mse_loss_backward_f32(", ptx);
        Assert.Contains("op=mse-backward", ptx);
        // Four pointers: gradOutput, predictions, targets, gradInput.
        Assert.Equal(4, PtxText.CountOccurrences(ptx, "ld.param.u64"));

        // gradOutput[0] is a broadcast scalar, so it is read ONCE and doubled
        // once, outside the per-element work.
        Assert.Equal(1, PtxText.CountOccurrences(ptx, "ld.global.nc.f32 %f24, [%rd0];"));
        Assert.Equal(1, PtxText.CountOccurrences(ptx, "mul.rn.f32 %f24, %f24, 0f40000000;"));

        // Per element: subtract, multiply by the hoisted (g*2), then by invN -
        // exactly ((g * 2) * d) * invN. Only the SOURCE of invN changed: it is a
        // launch parameter now, so one module serves every batch size.
        Assert.Contains("ld.param.f32 %f25, [inv_n];", ptx);
        Assert.Equal(1, PtxText.CountOccurrences(ptx, "ld.param.f32"));
        // Eight elements per thread now: predictions %f0-7, targets %f8-15,
        // diffs %f16-23, with the hoisted scalars above them.
        for (int i = 0; i < 8; i++)
        {
            int diff = 16 + i;
            Assert.Contains($"sub.rn.f32 %f{diff}, %f{i}, %f{8 + i};", ptx);
            Assert.Contains($"mul.rn.f32 %f{diff}, %f24, %f{diff};", ptx);
            Assert.Contains($"mul.rn.f32 %f{diff}, %f{diff}, %f25;", ptx);
        }
        // The scale is a multiply, so the kernel never divides.
        Assert.DoesNotContain("div.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void MaeGradientEmitter_ReproducesTheTernarySignChainIncludingNaN()
    {
        string ptx = PtxFusedLossBackwardF32Kernel.EmitPtx(
            8, 6, DirectPtxLossBackwardOp.MeanAbsoluteError, 65_536);

        Assert.Contains(".visible .entry aidotnet_fused_mae_gradient_f32(", ptx);
        Assert.Contains("op=mae-gradient", ptx);
        // Three pointers only: the established mae_gradient takes no upstream
        // gradient and no scale.
        Assert.Equal(3, PtxText.CountOccurrences(ptx, "ld.param.u64"));
        Assert.DoesNotContain("grad_output_ptr", ptx, StringComparison.Ordinal);

        // Two predicates and two selects per element. Both predicates are false
        // for exact zero AND for NaN, so each yields +0 - matching
        // (d > 0) ? 1 : ((d < 0) ? -1 : 0).
        Assert.Equal(8, PtxText.CountOccurrences(ptx, "setp.gt.f32 %p1,"));
        Assert.Equal(8, PtxText.CountOccurrences(ptx, "setp.lt.f32 %p2,"));
        // Two selects per element: (+1 or 0), then (-1 or that).
        Assert.Equal(16, PtxText.CountOccurrences(ptx, "selp.f32"));
        Assert.Equal(8, PtxText.CountOccurrences(ptx, "0f3F800000"));    // +1.0, once per element
        Assert.Equal(8, PtxText.CountOccurrences(ptx, "0fBF800000"));    // -1.0, once per element
        // The sign gradient must not scale by anything.
        Assert.DoesNotContain("mul.rn.f32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void MseEmitter_ReadsScaleFromLaunchParameterWithoutBakedReciprocal()
    {
        string ptx = PtxFusedLossBackwardF32Kernel.EmitPtx(
            8, 6, DirectPtxLossBackwardOp.MeanSquaredError, 65_536);

        Assert.Contains(".param .f32 inv_n", ptx);
        Assert.Contains("ld.param.f32 %f25, [inv_n];", ptx);
        // 1 / 65,536 is 0x37800000. The scale must not be embedded in the module.
        Assert.DoesNotContain("0f37800000", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void Emitter_ReadsEachInputOnceAndWritesOneVector()
    {
        foreach (var op in new[]
                 {
                     DirectPtxLossBackwardOp.MeanSquaredError,
                     DirectPtxLossBackwardOp.MeanAbsoluteError
                 })
        {
            string ptx = PtxFusedLossBackwardF32Kernel.EmitPtx(8, 6, op, 262_144);
            // Two vectors each of predictions and targets, two stored.
            Assert.Equal(4, PtxText.CountOccurrences(ptx, "ld.global.nc.v4.f32"));
            Assert.Equal(2, PtxText.CountOccurrences(ptx, "st.global.v4.f32"));
            Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ShapeAndScaleDomain_AreClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedLossBackwardF32Kernel.IsSupportedShape(65_536));
        Assert.True(PtxFusedLossBackwardF32Kernel.IsSupportedShape(4_194_304));
        Assert.False(PtxFusedLossBackwardF32Kernel.IsSupportedShape(65_535));
        Assert.False(PtxFusedLossBackwardF32Kernel.IsPromotedShape(65_536));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedLossBackwardF32Kernel.EmitPtx(
                8, 6, DirectPtxLossBackwardOp.MeanSquaredError, 1_000));
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public async Task MseScaleDomain_RejectsNonPositiveAndNonFiniteValues(float invN)
    {
        await Task.Yield();

        var error = Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedLossBackwardF32Kernel.ValidateInvN(
                DirectPtxLossBackwardOp.MeanSquaredError, invN));
        Assert.Equal("invN", error.ParamName);
    }

    [Fact]
    public void ArchitectureGate_FailsClosedOutsideSm86()
    {
        Assert.True(DirectPtxArchitecture.HasValidatedLossBackward(8, 6));
        Assert.False(DirectPtxArchitecture.HasValidatedLossBackward(8, 0));
        Assert.False(DirectPtxArchitecture.HasValidatedLossBackward(8, 7));
        Assert.False(DirectPtxArchitecture.HasValidatedLossBackward(8, 9));
        Assert.False(DirectPtxArchitecture.HasValidatedLossBackward(9, 0));
    }

}
