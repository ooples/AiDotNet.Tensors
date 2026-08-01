using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Shared PTX arithmetic emission for register-resident solver kernels.</summary>
internal static class PtxSolverArithmetic
{
    private const string NegatedOperand = "%solver_neg";

    /// <summary>
    /// Declares the reusable virtual register used to materialize a negated FMA operand.
    /// PTX does not accept a unary register modifier on an <c>fma</c> source operand.
    /// </summary>
    internal static void Declare(StringBuilder ptx)
    {
        if (ptx is null) throw new ArgumentNullException(nameof(ptx));
        ptx.AppendLine($"    .reg .f32 {NegatedOperand};");
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
}
