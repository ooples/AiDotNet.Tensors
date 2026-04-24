// Copyright (c) AiDotNet. All rights reserved.
// torch.func-style functional transforms over GradientTape.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff.ForwardAD;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff.Transforms;

/// <summary>
/// Functional autograd transforms mirroring <c>torch.func</c>
/// (formerly <c>functorch</c>). Each static method takes a user
/// function and returns a new function that applies an autograd
/// transformation — gradient, Jacobian, Hessian, batched execution,
/// or stateless parameter override.
/// </summary>
/// <remarks>
/// <para><b>Why these exist:</b></para>
/// <para>
/// Direct use of <see cref="GradientTape{T}"/> requires explicit
/// <c>using</c> blocks and manual watch/gradient calls. Transforms let
/// callers treat gradients as first-class function-to-function
/// mappings — essential for meta-learning (MAML), implicit layers,
/// physics-informed models, Hessian-free optimization, and any API
/// that needs to compose differentiation with other transformations
/// (vmap, jacfwd, jacrev).
/// </para>
/// <para><b>Design choice — function lists in/out:</b></para>
/// <para>
/// Transforms take <c>Func&lt;Tensor&lt;T&gt;[], Tensor&lt;T&gt;&gt;</c>
/// so they compose uniformly: a caller can chain <c>Grad(Vmap(fn))</c>
/// to get per-sample gradients without worrying about shape or argument
/// packing mismatches. Single-argument callers can use the overloads
/// that take <c>Func&lt;Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;</c> directly.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric element type of the tensors
/// participating in differentiation.</typeparam>
public static class TensorFunc<T>
{
    /// <summary>
    /// Returns a function that computes the gradient of
    /// <paramref name="fn"/> with respect to the input at
    /// <paramref name="argIndex"/> (default: 0). The returned function
    /// runs <paramref name="fn"/> on its inputs, then runs reverse-mode
    /// backward and returns the gradient tensor for the chosen input.
    /// </summary>
    /// <param name="fn">A scalar-valued function of one or more tensor
    /// inputs. The first return from <paramref name="fn"/> must be a
    /// tensor whose reduction to a scalar forms the loss; when the
    /// output is multi-dim, the sum is used as the loss seed.</param>
    /// <param name="argIndex">Index of the input tensor to
    /// differentiate with respect to. Defaults to <c>0</c> (the first
    /// input — matches <c>torch.func.grad(fn, argnums=0)</c>).</param>
    /// <param name="createGraph">If <c>true</c>, the returned
    /// gradient is itself differentiable (higher-order AD). Each call
    /// to the returned function allocates a fresh <see cref="GradientTape{T}"/>.</param>
    /// <returns>A function that takes the same inputs as
    /// <paramref name="fn"/> and returns the gradient w.r.t. the
    /// selected input.</returns>
    /// <exception cref="ArgumentNullException">Thrown if
    /// <paramref name="fn"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown on call if
    /// <paramref name="argIndex"/> is outside the input array bounds.</exception>
    public static Func<Tensor<T>[], Tensor<T>> Grad(
        Func<Tensor<T>[], Tensor<T>> fn,
        int argIndex = 0,
        bool createGraph = false)
    {
        if (fn is null) throw new ArgumentNullException(nameof(fn));
        if (argIndex < 0) throw new ArgumentOutOfRangeException(nameof(argIndex));

        return inputs =>
        {
            if (inputs is null) throw new ArgumentNullException(nameof(inputs));
            if (argIndex >= inputs.Length)
                throw new ArgumentOutOfRangeException(
                    nameof(argIndex),
                    $"argIndex {argIndex} is outside the input array of length {inputs.Length}.");

            using var tape = new GradientTape<T>();
            var output = fn(inputs);
            var grads = tape.ComputeGradients(output, new[] { inputs[argIndex] }, createGraph);
            if (!grads.TryGetValue(inputs[argIndex], out var grad))
            {
                // Input not used by fn — return a zero gradient of the input's shape.
                var ops = MathHelper.GetNumericOperations<T>();
                var zero = ops.Zero;
                var data = new T[inputs[argIndex].Length];
                for (int i = 0; i < data.Length; i++) data[i] = zero;
                grad = new Tensor<T>(data, inputs[argIndex]._shape);
            }
            return grad;
        };
    }

    /// <summary>
    /// Single-input convenience overload — wraps a unary function.
    /// </summary>
    /// <param name="fn">A scalar-valued function of a single tensor.</param>
    /// <param name="createGraph">If <c>true</c>, the returned gradient
    /// is itself differentiable.</param>
    /// <returns>A function that returns <c>d(fn)/d(input)</c>.</returns>
    /// <exception cref="ArgumentNullException">Thrown if
    /// <paramref name="fn"/> is null.</exception>
    public static Func<Tensor<T>, Tensor<T>> Grad(
        Func<Tensor<T>, Tensor<T>> fn,
        bool createGraph = false)
    {
        if (fn is null) throw new ArgumentNullException(nameof(fn));
        var lifted = Grad(args => fn(args[0]), 0, createGraph);
        return x => lifted(new[] { x });
    }

    /// <summary>
    /// Like <see cref="Grad(Func{Tensor{T}[], Tensor{T}}, int, bool)"/>
    /// but returns both the gradient and the forward output in a single
    /// call — saves an extra forward pass when the caller needs both
    /// (e.g., a training step that reports the loss alongside the
    /// gradient update).
    /// </summary>
    /// <param name="fn">The function to differentiate.</param>
    /// <param name="argIndex">Index of the input tensor to
    /// differentiate with respect to.</param>
    /// <param name="createGraph">If <c>true</c>, the returned gradient
    /// is itself differentiable.</param>
    /// <returns>A function that returns a <c>(Grad, Value)</c> tuple
    /// where <c>Value</c> is the forward output of <paramref name="fn"/>
    /// and <c>Grad</c> is the gradient of <c>Value</c> w.r.t. the
    /// selected input.</returns>
    public static Func<Tensor<T>[], (Tensor<T> Grad, Tensor<T> Value)> GradAndValue(
        Func<Tensor<T>[], Tensor<T>> fn,
        int argIndex = 0,
        bool createGraph = false)
    {
        if (fn is null) throw new ArgumentNullException(nameof(fn));
        if (argIndex < 0) throw new ArgumentOutOfRangeException(nameof(argIndex));

        return inputs =>
        {
            if (inputs is null) throw new ArgumentNullException(nameof(inputs));
            if (argIndex >= inputs.Length)
                throw new ArgumentOutOfRangeException(
                    nameof(argIndex),
                    $"argIndex {argIndex} is outside the input array of length {inputs.Length}.");

            using var tape = new GradientTape<T>();
            var output = fn(inputs);
            var grads = tape.ComputeGradients(output, new[] { inputs[argIndex] }, createGraph);
            if (!grads.TryGetValue(inputs[argIndex], out var grad))
            {
                var ops = MathHelper.GetNumericOperations<T>();
                var zero = ops.Zero;
                var data = new T[inputs[argIndex].Length];
                for (int i = 0; i < data.Length; i++) data[i] = zero;
                grad = new Tensor<T>(data, inputs[argIndex]._shape);
            }
            return (grad, output);
        };
    }

    /// <summary>
    /// Single-input convenience overload for <see cref="GradAndValue(Func{Tensor{T}[], Tensor{T}}, int, bool)"/>.
    /// </summary>
    public static Func<Tensor<T>, (Tensor<T> Grad, Tensor<T> Value)> GradAndValue(
        Func<Tensor<T>, Tensor<T>> fn,
        bool createGraph = false)
    {
        if (fn is null) throw new ArgumentNullException(nameof(fn));
        var lifted = GradAndValue(args => fn(args[0]), 0, createGraph);
        return x => lifted(new[] { x });
    }

    // ─── Forward-mode AD (Jvp) ────────────────────────────────────────

    /// <summary>
    /// Jacobian-vector product: runs <paramref name="dualFn"/> in
    /// forward-mode dual arithmetic. Given primals <c>x</c> and
    /// tangents <c>v</c>, returns <c>(f(x), ∂f/∂x · v)</c> in a single
    /// pass without building a Jacobian explicitly.
    /// </summary>
    /// <remarks>
    /// <para><b>How to write <paramref name="dualFn"/>:</b></para>
    /// <para>
    /// The user function must operate on <see cref="Dual{T}"/>
    /// tensors using <see cref="DualOps{T}"/> (e.g.
    /// <c>DualOps&lt;T&gt;.Add</c>, <c>DualOps&lt;T&gt;.MatMul</c>).
    /// Only ops that have a registered JVP rule participate in the
    /// forward-mode computation; using a raw
    /// <see cref="IEngine"/> op on <c>.Primal</c> inside the function
    /// would compute the primal correctly but drop the tangent.
    /// </para>
    /// <para><b>Why this is useful beyond Grad:</b></para>
    /// <para>
    /// Forward-mode computes directional derivatives at
    /// <c>O(output_dim)</c> cost independent of input dim. For short,
    /// wide functions (few inputs, many outputs) forward-mode is
    /// cheaper than reverse-mode and is the foundation of
    /// <see cref="JacFwd"/> and forward-over-reverse Hessians.
    /// </para>
    /// </remarks>
    /// <param name="engine">Engine used for primal + tangent
    /// arithmetic. Must be non-null.</param>
    /// <param name="dualFn">User function running over
    /// <see cref="Dual{T}"/> tensors via <see cref="DualOps{T}"/>.</param>
    /// <param name="primals">Primal input tensors.</param>
    /// <param name="tangents">Tangent (direction) tensors — must
    /// match <paramref name="primals"/> in count and shape.</param>
    /// <returns>Tuple <c>(primalOutput, tangentOutput)</c>.</returns>
    /// <exception cref="ArgumentNullException">Thrown if
    /// <paramref name="engine"/>, <paramref name="dualFn"/>,
    /// <paramref name="primals"/>, or <paramref name="tangents"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown if
    /// <paramref name="primals"/> and <paramref name="tangents"/> have
    /// different lengths.</exception>
    public static (Tensor<T> Primal, Tensor<T> Tangent) Jvp(
        IEngine engine,
        Func<Dual<T>[], Dual<T>> dualFn,
        Tensor<T>[] primals,
        Tensor<T>[] tangents)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (dualFn is null) throw new ArgumentNullException(nameof(dualFn));
        if (primals is null) throw new ArgumentNullException(nameof(primals));
        if (tangents is null) throw new ArgumentNullException(nameof(tangents));
        if (primals.Length != tangents.Length)
            throw new ArgumentException(
                $"primals and tangents must have the same length — got {primals.Length} and {tangents.Length}.");

        var duals = new Dual<T>[primals.Length];
        for (int i = 0; i < duals.Length; i++)
            duals[i] = new Dual<T>(primals[i], tangents[i]);
        var result = dualFn(duals);
        return (result.Primal, result.Tangent);
    }

    /// <summary>
    /// Single-input JVP overload.
    /// </summary>
    public static (Tensor<T> Primal, Tensor<T> Tangent) Jvp(
        IEngine engine,
        Func<Dual<T>, Dual<T>> dualFn,
        Tensor<T> primal,
        Tensor<T> tangent)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (dualFn is null) throw new ArgumentNullException(nameof(dualFn));
        return Jvp(engine, duals => dualFn(duals[0]), new[] { primal }, new[] { tangent });
    }
}
