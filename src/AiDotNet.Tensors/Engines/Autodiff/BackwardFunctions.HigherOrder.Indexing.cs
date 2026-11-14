using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Double-backward support for the indexing, structural, selection and multilinear ops (see
/// <c>BackwardFunctions.HigherOrder.cs</c> for the shared mechanism).
/// </summary>
internal static partial class BackwardFunctions<T>
{
    /// <summary>
    /// A fixed sparse linear map as a recorded op: <c>out[to[k]] += coef[k] · in[from[k]]</c> (coef 1 when null), zero
    /// elsewhere. Its backward is the transposed map, itself a <see cref="SparseMap"/>, so a gather, a scatter-add or
    /// a weighted routing built from it is differentiable to any order. The backward functions use it under
    /// create-graph for the gradients of selections whose pattern (an argmax, a sort permutation, a quantile's order
    /// statistics) is fixed by the forward.
    /// </summary>
    internal static Tensor<T> SparseMap(Tensor<T> input, int[] outputShape, int[] to, int[] from, double[]? coef)
    {
        var values = RoIAlignPlan.ToDoubles(input);
        var result = new double[ShapeLength(outputShape)];
        for (int k = 0; k < to.Length; k++)
            result[to[k]] += (coef is null ? 1.0 : coef[k]) * values[from[k]];
        var output = RoIAlignPlan.FromDoubles<T>(result, (int[])outputShape.Clone());
        var inputShape = (int[])input._shape.Clone();
        DifferentiableOps.RecordUnary("SparseMap", output, input,
            (g, ins, _, _, eng, grads) => DifferentiableOps.AccumulateGrad(grads, ins[0], SparseMap(g, inputShape, from, to, coef), eng));
        return output;
    }

    private static int ShapeLength(int[] shape)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        return n;
    }

    /// <summary>A constant zero tensor of <paramref name="like"/>'s shape.</summary>
    internal static Tensor<T> ZerosLike(Tensor<T> like) => new Tensor<T>((int[])like._shape.Clone());

    /// <summary>
    /// The create-graph path of an op that is jointly linear in its differentiable inputs (an add, a block placement,
    /// a scatter or put of a source into a destination): every input's gradient is a fixed linear map of the upstream
    /// gradient, whose adjoint is the op's forward with that input set to the cotangent and the other differentiable
    /// inputs set to zero. <paramref name="forward"/> recomputes the op from a replacement input array (same order as
    /// <paramref name="inputs"/>) with recorded engine ops. Returns false when no graph is being created.
    /// </summary>
    internal static bool TryJointlyLinearBackward(string opName, BackwardFunction<T> backward, Tensor<T> gradOutput,
        Tensor<T>[] inputs, Tensor<T> output, object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads,
        Func<IEngine, Tensor<T>[], Tensor<T>> forward)
    {
        if (!CreatesGraph) return false;
        Tensor<T>?[] values;
        using (new NoGradScope<T>()) values = KernelGrads(backward, gradOutput, inputs, output, savedState, engine);
        for (int i = 0; i < inputs.Length; i++)
        {
            if (values[i] is not { } value) continue;
            int slot = i;
            var grad = LinearGrad(opName, gradOutput, () => value, (e, w) =>
            {
                var replaced = new Tensor<T>[inputs.Length];
                for (int j = 0; j < inputs.Length; j++) replaced[j] = j == slot ? w : ZerosLike(inputs[j]);
                return forward(e, replaced);
            });
            DifferentiableOps.AccumulateGrad(grads, inputs[i], grad, engine);
        }
        return true;
    }

    /// <summary>
    /// The create-graph path of an op that is linear in each differentiable input separately (a product: Kronecker,
    /// einsum/inner/dot, complex or octonion multiplication). Input i's gradient Bᵢ(g; x_{j≠i}) is linear in the
    /// upstream gradient g and in every other input, so its own backward, for a cotangent w, gives
    /// g ← f(x with xᵢ := w) (the forward) and x_j ← B_j(g; x with xᵢ := w) (this op's backward again, which records
    /// under a further create-graph); xᵢ itself gets nothing. The values come from the op's fast kernel.
    /// </summary>
    internal static bool TryMultilinearBackward(string opName, BackwardFunction<T> backward, Tensor<T> gradOutput,
        Tensor<T>[] inputs, Tensor<T> output, object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads,
        Func<IEngine, Tensor<T>[], Tensor<T>> forward)
    {
        if (!CreatesGraph) return false;
        Tensor<T>?[] values;
        using (new NoGradScope<T>()) values = KernelGrads(backward, gradOutput, inputs, output, savedState, engine);
        int n = inputs.Length;
        for (int i = 0; i < n; i++)
        {
            if (values[i] is not { } value) continue;
            int slot = i;
            var nodeInputs = new Tensor<T>[n + 1];
            nodeInputs[0] = gradOutput;
            Array.Copy(inputs, 0, nodeInputs, 1, n);
            DifferentiableOps.RecordIfActive(opName, value, nodeInputs, (w, ins, _, _, e, g) =>
            {
                var upstream = ins[0];
                var replaced = new Tensor<T>[n];
                Array.Copy(ins, 1, replaced, 0, n);
                replaced[slot] = w;
                DifferentiableOps.AccumulateGrad(g, upstream, forward(e, replaced), e);
                var others = KernelGrads(backward, upstream, replaced, output, savedState, e);
                for (int j = 0; j < n; j++)
                {
                    if (j == slot || others[j] is not { } gj) continue;
                    DifferentiableOps.AccumulateGrad(g, ins[j + 1], gj, e);
                }
            });
            DifferentiableOps.AccumulateGrad(grads, inputs[i], value, engine);
        }
        return true;
    }

    /// <summary>
    /// The create-graph path of a single-input op whose gradient is an element-wise mask of the upstream gradient (an
    /// overwrite's destination keeping the positions not written): the op's own backward is that mask and is its own
    /// adjoint. Returns false when no graph is being created.
    /// </summary>
    internal static bool TrySelfAdjointBackward(string opName, BackwardFunction<T> backward, Tensor<T> gradOutput,
        Tensor<T>[] inputs, Tensor<T> output, object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (!CreatesGraph) return false;
        var grad = DiagonalLinearGrad(opName, gradOutput, engine, (e, w) =>
            KernelGrads(backward, w, inputs, output, savedState, e)[0]
            ?? throw new InvalidOperationException($"{opName}: the backward gave its input no gradient."));
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
        return true;
    }

    /// <summary>Σ_{i ≥ k} v_i along <paramref name="axis"/> (a reverse cumulative sum) as recorded ops.</summary>
    internal static Tensor<T> ReverseCumSum(IEngine engine, Tensor<T> v, int axis)
    {
        var flip = new[] { axis };
        return engine.TensorFlip(engine.TensorCumSum(engine.TensorFlip(v, flip), axis), flip);
    }
}
