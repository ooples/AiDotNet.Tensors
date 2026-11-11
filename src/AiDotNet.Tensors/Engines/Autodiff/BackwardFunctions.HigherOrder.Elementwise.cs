using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Building blocks for the create-graph paths of element-wise backwards: a derivative written as recorded engine ops
/// (so the gradient it produces is differentiable again) mixed with host-computed constants — masks of where a
/// piecewise definition switches branch, and replacement arguments that keep the unused branch of a select finite.
/// </summary>
/// <remarks>
/// A constant here is a plain tensor no op produced: it carries no gradient, which is exactly right for a branch mask
/// (piecewise constant) and for torch's convention of no gradient through a non-differentiable argument.
/// </remarks>
internal static partial class BackwardFunctions<T>
{
    /// <summary>A constant tensor shaped like <paramref name="like"/> filled with <paramref name="value"/>.</summary>
    internal static Tensor<T> ConstantLike(Tensor<T> like, double value)
    {
        var result = new Tensor<T>((int[])like._shape.Clone());
        result.Fill(MathHelper.GetNumericOperations<T>().FromDouble(value));
        return result;
    }

    /// <summary>A constant 1/0 mask of where <paramref name="predicate"/> holds for each element of <paramref name="x"/>.</summary>
    internal static Tensor<T> HostMask(Tensor<T> x, Func<double, bool> predicate)
    {
        var values = RoIAlignPlan.ToDoubles(x);
        var mask = new double[values.Length];
        for (int i = 0; i < values.Length; i++) mask[i] = predicate(values[i]) ? 1.0 : 0.0;
        return RoIAlignPlan.FromDoubles<T>(mask, (int[])x._shape.Clone());
    }

    /// <summary>A constant 1/0 mask of where <paramref name="predicate"/> holds for each pair of equal-shaped elements.</summary>
    internal static Tensor<T> HostMask(Tensor<T> a, Tensor<T> b, Func<double, double, bool> predicate)
    {
        var av = RoIAlignPlan.ToDoubles(a);
        var bv = RoIAlignPlan.ToDoubles(b);
        var mask = new double[av.Length];
        for (int i = 0; i < av.Length; i++) mask[i] = predicate(av[i], bv[i]) ? 1.0 : 0.0;
        return RoIAlignPlan.FromDoubles<T>(mask, (int[])a._shape.Clone());
    }

    /// <summary>A constant tensor of <paramref name="map"/> applied to each element of <paramref name="x"/>.</summary>
    internal static Tensor<T> HostMap(Tensor<T> x, Func<double, double> map)
    {
        var values = RoIAlignPlan.ToDoubles(x);
        var result = new double[values.Length];
        for (int i = 0; i < values.Length; i++) result[i] = map(values[i]);
        return RoIAlignPlan.FromDoubles<T>(result, (int[])x._shape.Clone());
    }

    /// <summary>A constant tensor of <paramref name="map"/> applied to each pair of equal-shaped elements.</summary>
    internal static Tensor<T> HostMap(Tensor<T> a, Tensor<T> b, Func<double, double, double> map)
    {
        var av = RoIAlignPlan.ToDoubles(a);
        var bv = RoIAlignPlan.ToDoubles(b);
        var result = new double[av.Length];
        for (int i = 0; i < av.Length; i++) result[i] = map(av[i], bv[i]);
        return RoIAlignPlan.FromDoubles<T>(result, (int[])a._shape.Clone());
    }

    /// <summary>mask ? a : b, recorded (the gradient reaches a where the mask is set, b elsewhere).</summary>
    internal static Tensor<T> Select(IEngine engine, Tensor<T> mask, Tensor<T> a, Tensor<T> b) => engine.TensorWhere(mask, a, b);

    /// <summary>
    /// <paramref name="x"/> with the elements where <paramref name="predicate"/> holds replaced by the constant
    /// <paramref name="replacement"/>: the argument for a branch that is singular there (1/x at 0) and is masked out
    /// anyway, so its unused value stays finite and contributes an exact zero to every derivative.
    /// </summary>
    internal static Tensor<T> SafeArgument(IEngine engine, Tensor<T> x, Tensor<T> mask, double replacement)
        => engine.TensorWhere(mask, ConstantLike(x, replacement), x);

    /// <summary>t·c, recorded.</summary>
    internal static Tensor<T> Scale(IEngine engine, Tensor<T> t, double c)
        => engine.TensorMultiplyScalar(t, MathHelper.GetNumericOperations<T>().FromDouble(c));

    /// <summary>t + c, recorded.</summary>
    internal static Tensor<T> Shift(IEngine engine, Tensor<T> t, double c)
        => engine.TensorAddScalar(t, MathHelper.GetNumericOperations<T>().FromDouble(c));

    /// <summary>t², recorded.</summary>
    internal static Tensor<T> Squared(IEngine engine, Tensor<T> t) => engine.TensorMultiply(t, t);

    /// <summary>
    /// The clamp-with-tensor-bounds decision per element as three 1/0 masks (taken from the tensor, the min bound, the
    /// max bound), in the forward's order: min first, then max, so a max below the min wins.
    /// </summary>
    internal static double[][] ClampSelection(Tensor<T> tensor, Tensor<T>? min, Tensor<T>? max, int[]? minStrides,
        int[]? maxStrides, int[] shape)
    {
        var values = RoIAlignPlan.ToDoubles(tensor);
        var lows = min is null ? null : RoIAlignPlan.ToDoubles(min);
        var highs = max is null ? null : RoIAlignPlan.ToDoubles(max);
        var selection = new[] { new double[values.Length], new double[values.Length], new double[values.Length] };
        var at = new int[shape.Length];
        for (int i = 0; i < values.Length; i++)
        {
            double v = values[i];
            int target = 0;
            if (lows is not null)
            {
                int m = minStrides == null ? i : BroadcastLookupLocal(at, minStrides);
                if (v < lows[m]) { v = lows[m]; target = 1; }
            }
            if (highs is not null)
            {
                int m = maxStrides == null ? i : BroadcastLookupLocal(at, maxStrides);
                if (v > highs[m]) target = 2;
            }
            selection[target][i] = 1;
            for (int k = shape.Length - 1; k >= 0; k--)
            {
                at[k]++;
                if (at[k] < shape[k]) break;
                at[k] = 0;
            }
        }
        return selection;
    }

    /// <summary>Accumulates g ⊙ d into the gradient of <paramref name="input"/> (recorded).</summary>
    internal static void AccumulateScaled(Dictionary<Tensor<T>, Tensor<T>> grads, Tensor<T> input, Tensor<T> gradOutput,
        Tensor<T> derivative, IEngine engine)
        => DifferentiableOps.AccumulateGrad(grads, input, engine.TensorMultiply(gradOutput, derivative), engine);
}
