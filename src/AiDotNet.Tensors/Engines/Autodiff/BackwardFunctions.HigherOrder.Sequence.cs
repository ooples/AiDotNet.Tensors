using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Small building blocks for writing an op's forward as a composition of recorded engine ops, which the sequence-scan
/// and spectral backwards differentiate under <c>createGraph</c> (see
/// <see cref="BackwardFunctions{T}.TryCompositeBackward"/>). Every helper uses only ops whose own backward is recorded,
/// so the composition is differentiable to any order. Element-wise helpers broadcast operands of equal rank
/// explicitly (a recorded broadcast), since only the multiply broadcasts on its own.
/// </summary>
internal static class RecordedOps<T>
{
    private static readonly Interfaces.INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>The scalar <paramref name="value"/> in the element type.</summary>
    internal static T S(double value) => Ops.FromDouble(value);

    /// <summary>Slice <paramref name="index"/> of <paramref name="axis"/>, with that axis removed.</summary>
    internal static Tensor<T> At(IEngine e, Tensor<T> x, int axis, int index)
    {
        var shape = new int[x.Rank - 1];
        for (int d = 0, k = 0; d < x.Rank; d++) if (d != axis) shape[k++] = x._shape[d];
        return e.Reshape(e.TensorNarrow(x, axis, index, 1), shape);
    }

    /// <summary>Recorded reshape.</summary>
    internal static Tensor<T> R(IEngine e, Tensor<T> x, params int[] shape) => e.Reshape(x, shape);

    /// <summary>Stacks equal-shape tensors along a new <paramref name="axis"/>.</summary>
    internal static Tensor<T> Stack(IEngine e, IList<Tensor<T>> items, int axis)
    {
        var array = new Tensor<T>[items.Count];
        items.CopyTo(array, 0);
        return e.TensorStack(array, axis);
    }

    private static (Tensor<T>, Tensor<T>) Common(IEngine e, Tensor<T> a, Tensor<T> b)
    {
        if (a.Rank != b.Rank)
            throw new ArgumentException($"RecordedOps: operands must have equal rank; got [{string.Join(",", a._shape)}] and [{string.Join(",", b._shape)}].");
        bool same = true;
        var shape = new int[a.Rank];
        for (int d = 0; d < shape.Length; d++)
        {
            shape[d] = Math.Max(a._shape[d], b._shape[d]);
            if (a._shape[d] != b._shape[d]) same = false;
        }
        if (same) return (a, b);
        return (Expand(e, a, shape), Expand(e, b, shape));
    }

    private static Tensor<T> Expand(IEngine e, Tensor<T> x, int[] shape)
    {
        for (int d = 0; d < shape.Length; d++)
            if (x._shape[d] != shape[d]) return e.TensorBroadcastTo(x, shape);
        return x;
    }

    /// <summary>a + b (equal rank, broadcast over size-1 axes).</summary>
    internal static Tensor<T> Add(IEngine e, Tensor<T> a, Tensor<T> b)
    {
        var (x, y) = Common(e, a, b);
        return e.TensorAdd(x, y);
    }

    /// <summary>a - b (equal rank, broadcast over size-1 axes).</summary>
    internal static Tensor<T> Sub(IEngine e, Tensor<T> a, Tensor<T> b)
    {
        var (x, y) = Common(e, a, b);
        return e.TensorSubtract(x, y);
    }

    /// <summary>a ⊙ b (equal rank, broadcast over size-1 axes).</summary>
    internal static Tensor<T> Mul(IEngine e, Tensor<T> a, Tensor<T> b)
    {
        var (x, y) = Common(e, a, b);
        return e.TensorMultiply(x, y);
    }

    /// <summary>a / b (equal rank, broadcast over size-1 axes).</summary>
    internal static Tensor<T> Div(IEngine e, Tensor<T> a, Tensor<T> b)
    {
        var (x, y) = Common(e, a, b);
        return e.TensorDivide(x, y);
    }

    /// <summary>x · s.</summary>
    internal static Tensor<T> Scale(IEngine e, Tensor<T> x, double s) => e.TensorMultiplyScalar(x, S(s));

    /// <summary>x + s.</summary>
    internal static Tensor<T> Shift(IEngine e, Tensor<T> x, double s) => e.TensorAddScalar(x, S(s));

    /// <summary>Σ over <paramref name="axis"/>, kept as size 1 or removed.</summary>
    internal static Tensor<T> Sum(IEngine e, Tensor<T> x, int axis, bool keep) => e.ReduceSum(x, new[] { axis }, keep);

    /// <summary>
    /// A constant (unrecorded) tensor f(x) element-wise, read from <paramref name="x"/>'s current values: the masks and
    /// signs that select a branch of a piecewise op (a clamp, a max, an absolute value), whose derivative within the
    /// selected branch the recorded ops then carry.
    /// </summary>
    [ScalarKernel("Builds a constant branch mask from a few host values; nothing here is recorded or differentiated.")]
    internal static Tensor<T> Constant(Tensor<T> x, Func<double, double> f)
    {
        var src = x.GetFlattenedData();
        var dst = new T[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = Ops.FromDouble(f(Ops.ToDouble(src[i])));
        return new Tensor<T>(dst, (int[])x._shape.Clone());
    }

    /// <summary>A constant zero tensor (records nothing).</summary>
    internal static Tensor<T> Zeros(params int[] shape) => new Tensor<T>(shape);

    /// <summary>A constant tensor of <paramref name="value"/> (records nothing).</summary>
    internal static Tensor<T> Full(double value, params int[] shape)
    {
        var t = new Tensor<T>(shape);
        t.Fill(S(value));
        return t;
    }
}
