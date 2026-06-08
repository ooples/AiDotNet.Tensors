using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Cross-type (FP32 ↔ FP16) differentiable cast — the keystone of industry-standard mixed-precision
/// autograd (docs/fp16-activation-storage-design.md, Tensors #555 Phase 1). A cast's local Jacobian is
/// the identity (element i of the output depends only on element i of the input, slope 1), so the only
/// thing the backward does is move the gradient ACROSS the dtype boundary by casting it to the input's
/// element type. These pure functions are the bridge between the FP32 and FP16 grad spaces; the tape /
/// compiled-plan integration wires them as a cross-type tape entry on top of this verified core.
///
/// FP16 rounding error of the cast itself is ~2^-11 relative and does NOT compound through the gradient
/// (the backward is exact up to the one round-trip), which the finite-difference gate verifies.
/// </summary>
internal static class MixedPrecisionCast
{
    /// <summary>Down-cast an FP32 activation to FP16 storage (forward).</summary>
    public static Tensor<Half> CastToFp16(Tensor<float> x)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        return x.Cast<Half>();
    }

    /// <summary>Up-cast an FP16 activation to FP32 (forward, e.g. at an FP32-op boundary).</summary>
    public static Tensor<float> CastToFp32(Tensor<Half> x)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        return x.Cast<float>();
    }

    /// <summary>
    /// Backward of <see cref="CastToFp16"/>: the upstream gradient arrives in the OUTPUT dtype (FP16)
    /// and must be deposited into the INPUT's FP32 grad space — an up-cast. d(cast)/dx = 1.
    /// </summary>
    public static Tensor<float> CastToFp16Backward(Tensor<Half> gradOutputFp16)
    {
        if (gradOutputFp16 is null) throw new ArgumentNullException(nameof(gradOutputFp16));
        return gradOutputFp16.Cast<float>();
    }

    /// <summary>
    /// Backward of <see cref="CastToFp32"/>: the upstream gradient arrives in FP32 and must be deposited
    /// into the INPUT's FP16 grad space — a down-cast. d(cast)/dx = 1.
    /// </summary>
    public static Tensor<Half> CastToFp32Backward(Tensor<float> gradOutputFp32)
    {
        if (gradOutputFp32 is null) throw new ArgumentNullException(nameof(gradOutputFp32));
        return gradOutputFp32.Cast<Half>();
    }
}
