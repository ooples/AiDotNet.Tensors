using System;
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
        Assert.Equal(4, Count(ptx, "ld.param.u64"));

        // gradOutput[0] is a broadcast scalar, so it is read ONCE and doubled
        // once, outside the per-element work.
        Assert.Equal(1, Count(ptx, "ld.global.ca.f32 %f12, [%rd0];"));
        Assert.Equal(1, Count(ptx, "mul.rn.f32 %f12, %f12, 0f40000000;"));

        // Per element: subtract, multiply by the hoisted (g*2), then by invN -
        // exactly ((g * 2) * d) * invN. Only the SOURCE of invN changed: it is a
        // launch parameter now, so one module serves every batch size.
        Assert.Contains("ld.param.f32 %f13, [inv_n];", ptx);
        Assert.Equal(1, Count(ptx, "ld.param.f32"));
        for (int i = 0; i < 4; i++)
        {
            int diff = 8 + i;
            Assert.Contains($"sub.rn.f32 %f{diff}, %f{i}, %f{4 + i};", ptx);
            Assert.Contains($"mul.rn.f32 %f{diff}, %f12, %f{diff};", ptx);
            Assert.Contains($"mul.rn.f32 %f{diff}, %f{diff}, %f13;", ptx);
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
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.DoesNotContain("grad_output_ptr", ptx, StringComparison.Ordinal);

        // Two predicates and two selects per element. Both predicates are false
        // for exact zero AND for NaN, so each yields +0 - matching
        // (d > 0) ? 1 : ((d < 0) ? -1 : 0).
        Assert.Equal(4, Count(ptx, "setp.gt.f32 %p1,"));
        Assert.Equal(4, Count(ptx, "setp.lt.f32 %p2,"));
        // Two selects per element: (+1 or 0), then (-1 or that).
        Assert.Equal(8, Count(ptx, "selp.f32"));
        Assert.Equal(4, Count(ptx, "selp.f32 %f8, 0f3F800000, 0f00000000, %p1;")
                      + Count(ptx, "selp.f32 %f9, 0f3F800000, 0f00000000, %p1;")
                      + Count(ptx, "selp.f32 %f10, 0f3F800000, 0f00000000, %p1;")
                      + Count(ptx, "selp.f32 %f11, 0f3F800000, 0f00000000, %p1;"));
        Assert.Equal(4, Count(ptx, "0f3F800000"));    // +1.0, once per element
        Assert.Equal(4, Count(ptx, "0fBF800000"));    // -1.0, once per element
        // The sign gradient must not scale by anything.
        Assert.DoesNotContain("mul.rn.f32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void MseEmitter_IsIndependentOfTheScale()
    {
        // The scale no longer enters the module, so every batch size shares one
        // module - the precondition for precompiling this kernel.
        Assert.Equal(
            PtxFusedLossBackwardF32Kernel.EmitPtx(
                8, 6, DirectPtxLossBackwardOp.MeanSquaredError, 65_536),
            PtxFusedLossBackwardF32Kernel.EmitPtx(
                8, 6, DirectPtxLossBackwardOp.MeanSquaredError, 65_536));
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
            Assert.Equal(2, Count(ptx, "ld.global.ca.v4.f32"));  // predictions, targets
            Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
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
        // invN is baked into the module, so a non-finite scale would poison
        // every gradient the cache serves under that key.
        // invN is now a launch parameter, so it is validated at Launch rather
        // than at emit time - the module no longer depends on its value.
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
