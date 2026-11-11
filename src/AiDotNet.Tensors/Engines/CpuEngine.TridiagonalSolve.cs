using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The batched tridiagonal solve (Thomas algorithm) behind <see cref="IEngine.TensorTridiagonalSolve{T}"/>: one tape
/// node whose backward is the same op on the transposed system, so a GPU engine runs both directions on the device
/// (<see cref="DirectGpuTensorEngine"/> overrides <see cref="TensorTridiagonalSolve{T}"/> with the backend's
/// <see cref="DirectGpu.INumericRecurrenceBackend"/>).
/// </summary>
public partial class CpuEngine
{
    /// <inheritdoc />
    public virtual Tensor<T> TensorTridiagonalSolve<T>(Tensor<T> lower, Tensor<T> diagonal, Tensor<T> upper, Tensor<T> rhs)
    {
        var (batch, n) = TridiagonalShape(lower, diagonal, upper, rhs);
        if (GraphMode.IsActive && GraphMode.Current is { } scope && scope.BindEngine(this))
        {
            var cl = lower; var cd = diagonal; var cu = upper; var cr = rhs;
            Action<IEngine, Tensor<T>> execute = (eng, result) =>
                DirectGpuTensorEngine.CopyResultInto(eng, eng.TensorTridiagonalSolve(cl, cd, cu, cr), result);
            return scope.RecordVariadic(LazyNodeType.Custom, "TridiagonalSolve", new[] { lower, diagonal, upper, rhs },
                (int[])diagonal._shape.Clone(), execute, TridiagonalSolveBackward<T>);
        }

        var x = RoIAlignPlan.FromDoubles<T>(
            TridiagonalSolveHost(RoIAlignPlan.ToDoubles(lower), RoIAlignPlan.ToDoubles(diagonal),
                RoIAlignPlan.ToDoubles(upper), RoIAlignPlan.ToDoubles(rhs), batch, n),
            (int[])diagonal._shape.Clone());
        RecordTridiagonalSolve(x, lower, diagonal, upper, rhs);
        return x;
    }

    /// <summary>Validates the four operands and returns (number of systems, system size).</summary>
    internal static (int Batch, int N) TridiagonalShape<T>(Tensor<T> lower, Tensor<T> diagonal, Tensor<T> upper, Tensor<T> rhs)
    {
        if (lower is null) throw new ArgumentNullException(nameof(lower));
        if (diagonal is null) throw new ArgumentNullException(nameof(diagonal));
        if (upper is null) throw new ArgumentNullException(nameof(upper));
        if (rhs is null) throw new ArgumentNullException(nameof(rhs));
        if (diagonal.Rank < 1)
            throw new ArgumentException("diagonal must have at least one dimension [..., n].", nameof(diagonal));
        int rank = diagonal.Rank, n = diagonal._shape[rank - 1];
        if (n < 1) throw new ArgumentException("Each tridiagonal system needs at least one unknown.", nameof(diagonal));
        if (!SameShape(rhs._shape, diagonal._shape))
            throw new ArgumentException($"rhs must have the diagonal's shape [{string.Join(", ", diagonal._shape)}]; got [{string.Join(", ", rhs._shape)}].", nameof(rhs));
        CheckOffDiagonal(lower, diagonal, nameof(lower));
        CheckOffDiagonal(upper, diagonal, nameof(upper));
        return (diagonal.Length / n, n);
    }

    private static bool SameShape(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void CheckOffDiagonal<T>(Tensor<T> band, Tensor<T> diagonal, string name)
    {
        int rank = diagonal.Rank;
        bool ok = band.Rank == rank && band._shape[rank - 1] == diagonal._shape[rank - 1] - 1;
        for (int d = 0; ok && d < rank - 1; d++) ok = band._shape[d] == diagonal._shape[d];
        if (!ok)
        {
            var expected = (int[])diagonal._shape.Clone();
            expected[rank - 1]--;
            throw new ArgumentException($"{name} must have shape [{string.Join(", ", expected)}] (the diagonal's leading dimensions and n - 1); got [{string.Join(", ", band._shape)}].", name);
        }
    }

    /// <summary>
    /// The Thomas algorithm on every system, systems in parallel: forward elimination keeps the scaled super-diagonal
    /// <c>c'[i] = upper[i] / beta[i]</c> and <c>x[i] = (rhs[i] - lower[i-1] x[i-1]) / beta[i]</c> with
    /// <c>beta[i] = diagonal[i] - lower[i-1] c'[i-1]</c>; back substitution then <c>x[i] -= c'[i] x[i+1]</c>. The device
    /// kernels (<see cref="DirectGpu.NumericRecurrenceKernelTemplate"/>) run the same steps in the same order.
    /// </summary>
    internal static double[] TridiagonalSolveHost(double[] lower, double[] diagonal, double[] upper, double[] rhs, int batch, int n)
    {
        var x = new double[batch * n];
        CpuParallelSettings.ParallelForOrSerial(0, batch, (long)batch * n * 8, b =>
        {
            int o = b * n, ob = b * (n - 1);
            var cp = new double[Math.Max(1, n - 1)];
            double beta = diagonal[o];
            x[o] = rhs[o] / beta;
            for (int i = 1; i < n; i++)
            {
                double c = upper[ob + i - 1] / beta;
                cp[i - 1] = c;
                double a = lower[ob + i - 1];
                beta = diagonal[o + i] - a * c;
                x[o + i] = (rhs[o + i] - a * x[o + i - 1]) / beta;
            }
            for (int i = n - 2; i >= 0; i--) x[o + i] -= cp[i] * x[o + i + 1];
        }, deterministicSafe: true);
        return x;
    }

    /// <summary>Records the solve's tape node (four inputs).</summary>
    private protected static void RecordTridiagonalSolve<T>(Tensor<T> x, Tensor<T> lower, Tensor<T> diagonal, Tensor<T> upper, Tensor<T> rhs) =>
        DifferentiableOps.RecordIfActive("TridiagonalSolve", x, new[] { lower, diagonal, upper, rhs }, TridiagonalSolveBackward<T>);

    /// <summary>
    /// Backward of <see cref="TensorTridiagonalSolve{T}"/>: <c>g = A^-T dx</c> is the solve of the transposed system
    /// (sub- and super-diagonal swapped), then <c>d rhs = g</c>, <c>d diagonal = -g x</c>,
    /// <c>d upper[i] = -g[i] x[i+1]</c> and <c>d lower[i] = -g[i+1] x[i]</c> (the tridiagonal band of <c>-g x^T</c>).
    /// Every step is an engine op, so a createGraph backward stays differentiable.
    /// </summary>
    internal static void TridiagonalSolveBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        Tensor<T> lower = inputs[0], diagonal = inputs[1], upper = inputs[2], rhs = inputs[3];
        var g = engine.TensorTridiagonalSolve(upper, diagonal, lower, gradOutput);
        if (DifferentiableOps.IsGradientRequired(rhs))
            DifferentiableOps.AccumulateGrad(grads, rhs, g, engine);
        if (DifferentiableOps.IsGradientRequired(diagonal))
            DifferentiableOps.AccumulateGrad(grads, diagonal, engine.TensorNegate(engine.TensorMultiply(g, output)), engine);
        int axis = output.Rank - 1, n = output._shape[axis];
        bool needUpper = DifferentiableOps.IsGradientRequired(upper), needLower = DifferentiableOps.IsGradientRequired(lower);
        if (n == 1)
        {
            // Empty bands: their gradient is the empty tensor of their shape.
            if (needUpper) DifferentiableOps.AccumulateGrad(grads, upper, new Tensor<T>((int[])upper._shape.Clone()), engine);
            if (needLower) DifferentiableOps.AccumulateGrad(grads, lower, new Tensor<T>((int[])lower._shape.Clone()), engine);
            return;
        }
        if (needUpper)
            DifferentiableOps.AccumulateGrad(grads, upper, engine.TensorNegate(engine.TensorMultiply(
                engine.TensorNarrow(g, axis, 0, n - 1), engine.TensorNarrow(output, axis, 1, n - 1))), engine);
        if (needLower)
            DifferentiableOps.AccumulateGrad(grads, lower, engine.TensorNegate(engine.TensorMultiply(
                engine.TensorNarrow(g, axis, 1, n - 1), engine.TensorNarrow(output, axis, 0, n - 1))), engine);
    }
}
