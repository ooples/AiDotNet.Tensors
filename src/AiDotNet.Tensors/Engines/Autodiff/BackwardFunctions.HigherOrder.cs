using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Higher-order (double-backward) support shared by the backward functions.
/// </summary>
/// <remarks>
/// <para>
/// Under <c>createGraph</c> the tape keeps recording while the backward runs, so a gradient computed with recorded
/// engine ops is itself a node of the graph and can be differentiated again (<c>torch.autograd.grad(...,
/// create_graph=True)</c>, Hessian-vector products, gradient penalties, the Laplacian of a physics-informed network).
/// A gradient computed by a raw kernel or a host loop records nothing: it is a constant to the second backward, which
/// then silently loses both the gradient's dependence on the op's inputs and on the upstream gradient.
/// </para>
/// <para>
/// The helpers below keep each op's fast kernel when nothing is recording and, under <c>createGraph</c>, either
/// (a) record the kernel's result as a node with an explicit derivative (<see cref="LinearGrad"/> for a gradient that
/// is a fixed linear map of the upstream gradient), or (b) differentiate a composition of recorded engine ops that
/// computes the same forward (<see cref="TryCompositeBackward"/>), which is twice- (and arbitrarily-) differentiable
/// because every op in it is.
/// </para>
/// </remarks>
internal static partial class BackwardFunctions<T>
{
    /// <summary>
    /// True when the running backward must itself be differentiable: <c>createGraph</c> is set and a tape is recording
    /// the backward's engine ops.
    /// </summary>
    internal static bool CreatesGraph => DifferentiableOps._isBackwardCreateGraph && DifferentiableOps.IsRecording<T>();

    /// <summary>
    /// A gradient that is a fixed linear map of the upstream gradient (a slice's, pad's, tile's, reduction's, gather's,
    /// pool's or selection's): runs <paramref name="kernel"/> and, under <c>createGraph</c>, records its result as a
    /// node of the tape whose own backward is the map's adjoint, <paramref name="adjoint"/> — for a linear op that is
    /// the op's forward applied to the cotangent; for a selection (max, sort, pooling) it is the gather at the saved
    /// positions.
    /// </summary>
    /// <remarks>
    /// The kernel runs with recording suppressed, so a kernel that happens to use recorded engine ops does not leave a
    /// second producer of the same tensor on the tape. <paramref name="adjoint"/> must be built from recorded engine
    /// ops (or from another <see cref="LinearGrad"/>) so third and higher orders stay connected.
    /// </remarks>
    internal static Tensor<T> LinearGrad(string opName, Tensor<T> gradOutput, Func<Tensor<T>> kernel,
        Func<IEngine, Tensor<T>, Tensor<T>> adjoint)
    {
        if (!CreatesGraph) return kernel();
        Tensor<T> grad;
        using (new NoGradScope<T>()) grad = kernel();
        if (ReferenceEquals(grad, gradOutput)) return grad;
        DifferentiableOps.RecordUnary(opName, grad, gradOutput,
            (w, ins, _, _, eng, g) => DifferentiableOps.AccumulateGrad(g, ins[0], adjoint(eng, w), eng));
        return grad;
    }

    /// <summary>
    /// Runs a backward function into a private gradient map (not the pass's own map or its indexed slots) and returns
    /// the gradient it gives each input, null where it gives none. Used to reuse a backward's fast kernel unchanged as
    /// the value of a <see cref="LinearGrad"/> node.
    /// </summary>
    internal static Tensor<T>?[] KernelGrads(BackwardFunction<T> backward, Tensor<T> gradOutput, Tensor<T>[] inputs,
        Tensor<T> output, object[] savedState, IEngine engine)
    {
        var scratch = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        var savedIndexed = DifferentiableOps._indexedGrads;
        DifferentiableOps._indexedGrads = null;
        try
        {
            backward(gradOutput, inputs, output, savedState, engine, scratch);
        }
        finally
        {
            DifferentiableOps._indexedGrads = savedIndexed;
        }
        var result = new Tensor<T>?[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
            if (inputs[i] is not null && scratch.TryGetValue(inputs[i], out var g)) result[i] = g;
        return result;
    }

    /// <summary>
    /// The create-graph path of a single-input op whose gradient is a fixed linear map of the upstream gradient
    /// (independent of the input's values): the op's own fast backward gives the value, and the recorded node maps a
    /// cotangent back with <paramref name="adjoint"/> — for a linear op, the op's forward. Returns false (doing
    /// nothing) when no graph is being created.
    /// </summary>
    internal static bool TryLinearBackward(string opName, BackwardFunction<T> backward, Tensor<T> gradOutput, Tensor<T>[] inputs,
        Tensor<T> output, object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads,
        Func<IEngine, Tensor<T>, Tensor<T>> adjoint)
    {
        if (!CreatesGraph) return false;
        var grad = LinearGrad(opName, gradOutput,
            () => KernelGrads(backward, gradOutput, inputs, output, savedState, engine)[0]
                  ?? throw new InvalidOperationException($"{opName}: the backward gave its input no gradient."),
            adjoint);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
        return true;
    }

    /// <summary>
    /// <see cref="LinearGrad"/> for a map that is its own adjoint — an element-wise mask or scale of the upstream
    /// gradient (ReLU, leaky ReLU, dropout): the same kernel maps the cotangent back.
    /// </summary>
    internal static Tensor<T> DiagonalLinearGrad(string opName, Tensor<T> gradOutput, IEngine engine,
        Func<IEngine, Tensor<T>, Tensor<T>> kernel)
        => LinearGrad(opName, gradOutput, () => kernel(engine, gradOutput), (e, w) => DiagonalLinearGrad(opName, w, e, kernel));

    /// <summary><paramref name="tensor"/> reshaped to <paramref name="shape"/> by a recorded reshape when they differ.</summary>
    internal static Tensor<T> ToShape(IEngine engine, Tensor<T> tensor, int[] shape)
        => ShapesEqual(tensor._shape, shape) ? tensor : engine.Reshape(tensor, shape);

    /// <summary>1 - x as recorded ops.</summary>
    internal static Tensor<T> OneMinus(IEngine engine, Tensor<T> x)
        => engine.TensorAddScalar(engine.TensorNegate(x), MathHelper.GetNumericOperations<T>().One);

    /// <summary>
    /// d gelu(x)/dx for the tanh approximation the GELU kernels use, as recorded ops:
    /// ½(1 + t) + ½·x·(1 - t²)·√(2/π)·(1 + 3·0.044715·x²), t = tanh(√(2/π)(x + 0.044715·x³)).
    /// </summary>
    internal static Tensor<T> GeluTanhDerivativeRecorded(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        const double k = 0.7978845608028654, c = 0.044715;
        var x2 = engine.TensorMultiply(x, x);
        var inner = engine.TensorMultiplyScalar(
            engine.TensorAdd(x, engine.TensorMultiplyScalar(engine.TensorMultiply(x2, x), ops.FromDouble(c))), ops.FromDouble(k));
        var t = engine.Tanh(inner);
        var half = ops.FromDouble(0.5);
        var first = engine.TensorMultiplyScalar(engine.TensorAddScalar(t, ops.One), half);
        var sech2 = OneMinus(engine, engine.TensorMultiply(t, t));
        var poly = engine.TensorAddScalar(engine.TensorMultiplyScalar(x2, ops.FromDouble(3 * c * k)), ops.FromDouble(k));
        var second = engine.TensorMultiplyScalar(engine.TensorMultiply(engine.TensorMultiply(x, sech2), poly), half);
        return engine.TensorAdd(first, second);
    }

    /// <summary>
    /// Under <c>createGraph</c>: computes the input gradients by differentiating <paramref name="composite"/> — the
    /// op's forward written with recorded engine ops, reading the SAME input tensors the op recorded — with the
    /// upstream gradient as its seed, and accumulates them into the running backward. Returns false (doing nothing)
    /// when no graph is being created, so the caller's fast kernel path runs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The composite is recorded on the SAME tape the create-graph backward records on, and its entries are walked
    /// right here, in reverse, with a private gradient map, so the backward's ops land on that tape too. Both matter:
    /// a later create-graph backward (a third derivative: a PINN differentiating a Laplacian residual) walks that
    /// tape's entries, and the GradFn walk of a plain backward follows the nodes either way. Differentiating the
    /// composite under a nested tape kept its entries off the outer tape, so the second create-graph backward found
    /// no path from the gradient back to the inputs and returned nothing.
    /// </para>
    /// <para>
    /// The private walk sets the running pass's indexed gradient slots aside (the inputs carry the outer pass's
    /// indices, which would alias), and appends after the outer walk's position, so the outer pass does not visit the
    /// composite's entries itself.
    /// </para>
    /// </remarks>
    internal static bool TryCompositeBackward(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, Func<IEngine, Tensor<T>> composite)
    {
        if (!CreatesGraph) return false;
        var tape = GradientTape<T>.Current;
        if (tape is null) return false;
        var sources = new List<Tensor<T>>(inputs.Length);
        foreach (var input in inputs)
            if (input is not null && !sources.Exists(s => ReferenceEquals(s, input))) sources.Add(input);

        var entries = tape.Entries;
        int start = entries.Count;
        var result = composite(engine);
        int end = entries.Count;

        var local = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance) { [result] = gradOutput };
        var savedIndexed = DifferentiableOps._indexedGrads;
        var savedSparse = DifferentiableOps._indexedSparseGrads;
        DifferentiableOps._indexedGrads = null;
        DifferentiableOps._indexedSparseGrads = null;
        try
        {
            for (int i = end - 1; i >= start; i--)
            {
                // Copy what the backward needs: the walk's own backward ops append to (and may grow) the arena.
                var entry = entries[i];
                if (entry.Output is null || !local.TryGetValue(entry.Output, out var upstream)) continue;
                var entryInputs = entry.GetInputsArray();
                var previousOwners = DifferentiableOps.BeginBackwardStep<T>();
                try
                {
                    entry.Backward(upstream, entryInputs, entry.Output, entry.SavedState ?? Array.Empty<object>(), engine, local);
                }
                finally
                {
                    DifferentiableOps.EndBackwardStep(previousOwners);
                }
            }
        }
        finally
        {
            DifferentiableOps._indexedGrads = savedIndexed;
            DifferentiableOps._indexedSparseGrads = savedSparse;
        }

        foreach (var input in sources)
        {
            if (local.TryGetValue(input, out var gi) && gi is not null)
                DifferentiableOps.AccumulateGrad(grads, input, gi, engine);
        }
        return true;
    }
}
