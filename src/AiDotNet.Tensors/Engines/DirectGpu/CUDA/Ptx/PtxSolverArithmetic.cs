using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Shared PTX arithmetic emission for register-resident solver kernels.</summary>
internal static class PtxSolverArithmetic
{
    private const string NegatedOperand = "%solver_neg";
    private const string Reciprocal = "%solver_recip";
    private const string ReciprocalCorrection = "%solver_recip_correction";

    /// <summary>
    /// Declares the reusable virtual register used to materialize a negated FMA operand.
    /// PTX does not accept a unary register modifier on an <c>fma</c> source operand.
    /// </summary>
    internal static void Declare(StringBuilder ptx)
    {
        if (ptx is null) throw new ArgumentNullException(nameof(ptx));
        ptx.AppendLine($"    .reg .f32 {NegatedOperand};");
        ptx.AppendLine($"    .reg .f32 {Reciprocal};");
        ptx.AppendLine($"    .reg .f32 {ReciprocalCorrection};");
    }

    /// <summary>Emits <c>destination -= left * right</c> with a single rounded FMA.</summary>
    internal static void EmitSubtractProduct(
        StringBuilder ptx,
        int destination,
        int left,
        int right)
    {
        if (ptx is null) throw new ArgumentNullException(nameof(ptx));
        if (destination < 0) throw new ArgumentOutOfRangeException(nameof(destination));
        if (left < 0) throw new ArgumentOutOfRangeException(nameof(left));
        if (right < 0) throw new ArgumentOutOfRangeException(nameof(right));

        ptx.AppendLine($"    neg.f32 {NegatedOperand}, %f{left};");
        ptx.AppendLine(
            $"    fma.rn.f32 %f{destination}, {NegatedOperand}, %f{right}, %f{destination};");
    }

    /// <summary>
    /// Computes a reusable reciprocal with one Newton correction. The approximate hardware
    /// reciprocal supplies the seed; <c>y = y * (2 - denominator * y)</c> restores FP32-level
    /// accuracy without repeating a full-precision divide for every numerator.
    /// </summary>
    internal static void EmitRefinedReciprocal(StringBuilder ptx, int denominator)
    {
        if (ptx is null) throw new ArgumentNullException(nameof(ptx));
        if (denominator < 0) throw new ArgumentOutOfRangeException(nameof(denominator));

        ptx.AppendLine($"    rcp.approx.f32 {Reciprocal}, %f{denominator};");
        ptx.AppendLine($"    neg.f32 {NegatedOperand}, %f{denominator};");
        ptx.AppendLine(
            $"    fma.rn.f32 {ReciprocalCorrection}, {NegatedOperand}, {Reciprocal}, 0f40000000;");
        ptx.AppendLine($"    mul.rn.f32 {Reciprocal}, {Reciprocal}, {ReciprocalCorrection};");
    }

    /// <summary>Multiplies a numerator by the most recently emitted refined reciprocal.</summary>
    internal static void EmitMultiplyByReciprocal(
        StringBuilder ptx,
        int destination,
        int numerator)
    {
        if (ptx is null) throw new ArgumentNullException(nameof(ptx));
        if (destination < 0) throw new ArgumentOutOfRangeException(nameof(destination));
        if (numerator < 0) throw new ArgumentOutOfRangeException(nameof(numerator));

        ptx.AppendLine($"    mul.rn.f32 %f{destination}, %f{numerator}, {Reciprocal};");
    }

    /// <summary>
    /// Replaces a positive radicand with its square root while retaining a reusable inverse.
    /// One Newton correction refines the hardware inverse-square-root seed before either value
    /// is consumed: <c>y = y * (1.5 - 0.5 * radicand * y * y)</c>.
    /// </summary>
    internal static void EmitRefinedSquareRootAndReciprocal(
        StringBuilder ptx,
        int radicand)
    {
        if (ptx is null) throw new ArgumentNullException(nameof(ptx));
        if (radicand < 0) throw new ArgumentOutOfRangeException(nameof(radicand));

        ptx.AppendLine($"    rsqrt.approx.f32 {Reciprocal}, %f{radicand};");
        ptx.AppendLine($"    mul.rn.f32 {ReciprocalCorrection}, {Reciprocal}, {Reciprocal};");
        ptx.AppendLine($"    mul.rn.f32 {NegatedOperand}, %f{radicand}, 0fBF000000;");
        ptx.AppendLine(
            $"    fma.rn.f32 {ReciprocalCorrection}, {NegatedOperand}, {ReciprocalCorrection}, 0f3FC00000;");
        ptx.AppendLine($"    mul.rn.f32 {Reciprocal}, {Reciprocal}, {ReciprocalCorrection};");
        ptx.AppendLine($"    mul.rn.f32 %f{radicand}, %f{radicand}, {Reciprocal};");
    }
}
