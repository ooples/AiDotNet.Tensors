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
    /// <param name="dualFn">User function running over
    /// <see cref="Dual{T}"/> tensors via <see cref="DualOps{T}"/>.</param>
    /// <param name="primals">Primal input tensors.</param>
    /// <param name="tangents">Tangent (direction) tensors — must
    /// match <paramref name="primals"/> in count and shape.</param>
    /// <returns>Tuple <c>(primalOutput, tangentOutput)</c>.</returns>
    /// <remarks>
    /// The engine driving primal + tangent arithmetic is whatever
    /// <see cref="DualOps{T}"/> resolves at call time (currently
    /// <see cref="AiDotNetEngine.Current"/>). Earlier revisions of this
    /// API took an explicit <c>IEngine</c> parameter, but it was never
    /// threaded into the dual-number ops — we removed it so the call
    /// contract reflects what actually runs.
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown if
    /// <paramref name="dualFn"/>, <paramref name="primals"/>, or
    /// <paramref name="tangents"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown if
    /// <paramref name="primals"/> and <paramref name="tangents"/> have
    /// different lengths.</exception>
    public static (Tensor<T> Primal, Tensor<T> Tangent) Jvp(
        Func<Dual<T>[], Dual<T>> dualFn,
        Tensor<T>[] primals,
        Tensor<T>[] tangents)
    {
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
        Func<Dual<T>, Dual<T>> dualFn,
        Tensor<T> primal,
        Tensor<T> tangent)
    {
        if (dualFn is null) throw new ArgumentNullException(nameof(dualFn));
        return Jvp(duals => dualFn(duals[0]), new[] { primal }, new[] { tangent });
    }

    // ─── Reverse-mode (Vjp) ───────────────────────────────────────────

    /// <summary>
    /// Returns the forward output of <paramref name="fn"/> along with
    /// a VJP closure that computes <c>v<sup>T</sup> · J</c> for any
    /// cotangent <paramref name="fn"/>-output-shaped tensor supplied
    /// later. Matches <c>torch.func.vjp</c>.
    /// </summary>
    /// <remarks>
    /// The closure reuses the same <see cref="GradientTape{T}"/> that
    /// recorded the forward pass, so each invocation re-runs backward
    /// from scratch with a different cotangent seed. For repeated use
    /// the caller should cache the result of the closure.
    /// </remarks>
    public static (Tensor<T> Output, Func<Tensor<T>, Tensor<T>[]> VjpFn) Vjp(
        Func<Tensor<T>[], Tensor<T>> fn,
        params Tensor<T>[] primals)
    {
        if (fn is null) throw new ArgumentNullException(nameof(fn));
        if (primals is null) throw new ArgumentNullException(nameof(primals));

        // Freeze the primal *values* — not just the array shell — so
        // VjpClosure differentiates the same point in input space we
        // returned Output for. A shallow Array.Clone would still let a
        // caller mutate any primal's data buffer between Vjp() returning
        // and VjpFn() running; both the eager Output (already computed)
        // and the re-run inside VjpClosure would silently track those
        // post-hoc mutations, producing gradients that no longer match
        // the returned Output. PyTorch hides this by having the tape
        // pin tensor data — we make the contract explicit by deep
        // copying every primal up front.
        var primalsSnapshot = new Tensor<T>[primals.Length];
        for (int i = 0; i < primals.Length; i++)
            primalsSnapshot[i] = primals[i].Clone();

        // Run fn once in the ambient context to produce the forward
        // output the caller wants back. We deliberately do NOT wrap
        // this in NoGrad: NoGradScope<T>.IsSuppressed gates all nested
        // RecordIfActive calls, so wrapping here would break
        // composition with transforms like Vjp(Grad(...)) where fn
        // itself opens its own tape (Grad's closure expects to record).
        // Any side-effect on an ambient outer tape is accepted — the
        // closure below re-runs fn under a fresh dedicated tape, so
        // ambient pollution from this initial run is harmless to the
        // VJP semantic.
        Tensor<T> fwdOutput = fn(primalsSnapshot);

        Tensor<T>[] VjpClosure(Tensor<T> cotangent)
        {
            if (cotangent is null) throw new ArgumentNullException(nameof(cotangent));
            using var t = new GradientTape<T>();
            var output = fn(primalsSnapshot);
            // Seed backward with v^T by multiplying output by the
            // cotangent and summing — the chain rule then deposits
            // (∂output/∂primal)ᵀ · cotangent on each primal, which is
            // exactly the VJP.
            var weighted = AiDotNetEngine.Current.TensorMultiply(output, cotangent);
            var scalar = SumToScalarTensor(weighted);
            var grads = t.ComputeGradients(scalar, primalsSnapshot);
            var result = new Tensor<T>[primalsSnapshot.Length];
            for (int i = 0; i < primalsSnapshot.Length; i++)
                result[i] = grads.TryGetValue(primalsSnapshot[i], out var g) ? g : ZeroLike(primalsSnapshot[i]);
            return result;
        }

        return (fwdOutput, VjpClosure);
    }

    // ─── Jacobians ────────────────────────────────────────────────────

    /// <summary>
    /// Jacobian of <paramref name="fn"/> computed by reverse-mode —
    /// one backward pass per output element. For <c>f: ℝⁿ → ℝᵐ</c>
    /// this returns an <c>[m, n]</c> matrix. Preferred over
    /// <see cref="JacFwd"/> when the output dimension is smaller than
    /// the input dimension (cost is <c>O(m · backward)</c>).
    /// </summary>
    /// <remarks>
    /// Implementation: seed a unit cotangent on each output element,
    /// run <see cref="GradientTape{T}.ComputeGradients"/>, stack the
    /// resulting gradient rows. Matches <c>torch.func.jacrev</c>.
    /// </remarks>
    public static Func<Tensor<T>, Tensor<T>> JacRev(Func<Tensor<T>, Tensor<T>> fn)
    {
        if (fn is null) throw new ArgumentNullException(nameof(fn));

        return x =>
        {
            if (x is null) throw new ArgumentNullException(nameof(x));

            // Probe output shape by running fn once. We do NOT wrap the
            // probe in NoGrad — if fn's implementation happens to build
            // its own internal tape (e.g., when JacRev composes with
            // Grad to form Hessian), NoGrad would suppress that inner
            // tape too and produce a "no recorded operations" failure.
            // Any entries the probe adds to an outer tape are harmless:
            // the loop below creates a fresh tape per iteration.
            int m;
            int[] probeShape;
            {
                var probe = fn(x);
                m = probe.Length;
                probeShape = (int[])probe._shape.Clone();
            }
            int n = x.Length;

            var rows = new Tensor<T>[m];
            var ops = MathHelper.GetNumericOperations<T>();
            var zero = ops.Zero;
            var one = ops.One;

            for (int i = 0; i < m; i++)
            {
                using var tape = new GradientTape<T>();
                var output = fn(x);
                // Seed a unit cotangent at output position i by multiplying
                // by a selector that's 1 at position i and 0 elsewhere.
                // ComputeGradients seeds ones-of-loss-shape; zeros in the
                // selector contribute nothing to the gradient, leaving only
                // position i's contribution — equivalent to seeding e_i.
                // Using the tape multiply (not a plain ctor) keeps the
                // result connected to the computation graph so the backward
                // walk has an entry to traverse even when fn is the identity.
                var selectorData = new T[m];
                for (int k = 0; k < m; k++) selectorData[k] = (k == i) ? one : zero;
                var selector = new Tensor<T>(selectorData, probeShape);
                var masked = AiDotNetEngine.Current.TensorMultiply(output, selector);
                var grads = tape.ComputeGradients(masked, new[] { x });
                rows[i] = grads.TryGetValue(x, out var g) ? g : ZeroLike(x);
            }

            // Stack rows into an [m, n] matrix.
            var data = new T[m * n];
            for (int i = 0; i < m; i++)
            {
                var row = rows[i].AsSpan();
                for (int j = 0; j < n; j++) data[i * n + j] = row[j];
            }
            return new Tensor<T>(data, new[] { m, n });
        };
    }

    /// <summary>
    /// Jacobian of <paramref name="fn"/> computed by forward-mode —
    /// one JVP pass per input element. Preferred over
    /// <see cref="JacRev(Func{Tensor{T}, Tensor{T}})"/> when the input
    /// dimension is smaller than the output dimension (cost is
    /// <c>O(n · forward)</c>).
    /// </summary>
    /// <remarks>
    /// Requires the user function to be written using
    /// <see cref="DualOps{T}"/> so the tangent propagates through each
    /// op. Matches <c>torch.func.jacfwd</c>.
    /// </remarks>
    /// <remarks>
    /// As with <see cref="Jvp(Func{Dual{T}[], Dual{T}}, Tensor{T}[], Tensor{T}[])"/>,
    /// the engine driving primal + tangent arithmetic is the one resolved
    /// by <see cref="DualOps{T}"/> at call time — there is no
    /// per-invocation engine override.
    /// </remarks>
    public static Func<Tensor<T>, Tensor<T>> JacFwd(
        Func<Dual<T>, Dual<T>> dualFn)
    {
        if (dualFn is null) throw new ArgumentNullException(nameof(dualFn));

        return x =>
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            int n = x.Length;
            var ops = MathHelper.GetNumericOperations<T>();
            var zero = ops.Zero;
            var one = ops.One;

            // Probe output shape.
            var zeroData = new T[n];
            for (int k = 0; k < n; k++) zeroData[k] = zero;
            var zeroTangent = new Tensor<T>(zeroData, (int[])x._shape.Clone());
            var probeDual = new Dual<T>(x, zeroTangent);
            var probeOut = dualFn(probeDual);
            int m = probeOut.Primal.Length;

            var cols = new Tensor<T>[n];
            for (int j = 0; j < n; j++)
            {
                var tangentData = new T[n];
                for (int k = 0; k < n; k++) tangentData[k] = (k == j) ? one : zero;
                var tangent = new Tensor<T>(tangentData, (int[])x._shape.Clone());
                var result = dualFn(new Dual<T>(x, tangent));
                cols[j] = result.Tangent;
            }

            // Stack columns into [m, n] Jacobian.
            var data = new T[m * n];
            for (int j = 0; j < n; j++)
            {
                var col = cols[j].AsSpan();
                for (int i = 0; i < m; i++) data[i * n + j] = col[i];
            }
            return new Tensor<T>(data, new[] { m, n });
        };
    }

    // ─── Hessian ──────────────────────────────────────────────────────

    /// <summary>
    /// Hessian of a scalar-valued <paramref name="fn"/> — returns an
    /// <c>[n, n]</c> matrix of second partial derivatives at the
    /// input point. Implemented as <c>JacRev(Grad(fn))</c>, requiring
    /// only reverse-mode AD (no dual-number rewrite of the user fn).
    /// </summary>
    /// <remarks>
    /// Uses <see cref="GradientTape{T}.ComputeGradients"/> with
    /// <c>createGraph: true</c> under the hood so the gradient itself
    /// is differentiable. Cost: <c>O(n · backward²)</c>. Matches
    /// <c>torch.func.hessian</c>.
    /// </remarks>
    public static Func<Tensor<T>, Tensor<T>> Hessian(Func<Tensor<T>, Tensor<T>> fn)
    {
        if (fn is null) throw new ArgumentNullException(nameof(fn));
        var gradFn = Grad(fn, createGraph: true);
        return JacRev(gradFn);
    }

    // ─── Hessian-vector products ──────────────────────────────────────

    /// <summary>
    /// Hessian-vector product <c>(∇²f(x)) · v</c> for a scalar-valued
    /// <paramref name="fn"/> at point <paramref name="primal"/>, in the
    /// direction <paramref name="vec"/>. Matches
    /// <c>torch.autograd.functional.hvp</c>.
    /// </summary>
    /// <remarks>
    /// Implemented as forward-over-reverse: <c>jvp(grad(fn))(x, v)</c>.
    /// Cost is one forward + one backward + one tangent propagation —
    /// <c>O(forward + backward)</c>, independent of the input
    /// dimension. Materialising the full Hessian only to multiply by a
    /// vector would cost <c>O(n · backward²)</c>, so callers that just
    /// need <c>Hv</c> products (Hessian-free optimisation, Newton-CG,
    /// Lanczos) should reach for this directly rather than
    /// <see cref="Hessian"/>.
    /// </remarks>
    /// <param name="fn">Scalar-valued function (returns a 1-element
    /// <see cref="Tensor{T}"/>).</param>
    /// <param name="primal">Point at which to evaluate the Hessian.</param>
    /// <param name="vec">Direction vector — must have the same shape as
    /// <paramref name="primal"/>.</param>
    /// <returns>The Hessian-vector product, shaped like
    /// <paramref name="primal"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown if any argument
    /// is null.</exception>
    /// <exception cref="ArgumentException">Thrown if
    /// <paramref name="primal"/> and <paramref name="vec"/> shapes
    /// disagree.</exception>
    public static Tensor<T> Hvp(
        Func<Tensor<T>, Tensor<T>> fn,
        Tensor<T> primal,
        Tensor<T> vec)
    {
        if (fn is null) throw new ArgumentNullException(nameof(fn));
        if (primal is null) throw new ArgumentNullException(nameof(primal));
        if (vec is null) throw new ArgumentNullException(nameof(vec));
        if (!ShapesMatch(primal._shape, vec._shape))
            throw new ArgumentException(
                $"Hvp requires primal and vec to share shape; got [{string.Join(",", primal._shape)}] vs " +
                $"[{string.Join(",", vec._shape)}].");

        // Reverse-mode gradient with create_graph so the result is
        // itself differentiable — this is the Hessian's "row" we'll
        // push the tangent v through.
        var gradFn = Grad(fn, createGraph: true);

        // d(grad)/dx · v: a Jacobian-vector product on grad. JacFwd
        // would materialise the full Jacobian; the Jvp path threads v
        // directly and stops at the product.
        using var t = new GradientTape<T>();
        var grad = gradFn(primal);
        // Seed the backward with the cotangent v, then deposit its dot
        // with grad onto primal — that's exactly (∂grad/∂x)ᵀ · v = Hv
        // for a symmetric Hessian (which is always the case for a
        // C² f).
        var weighted = AiDotNetEngine.Current.TensorMultiply(grad, vec);
        var scalar = SumToScalarTensor(weighted);
        var grads = t.ComputeGradients(scalar, new[] { primal });
        return grads.TryGetValue(primal, out var hv) ? hv : ZeroLike(primal);
    }

    /// <summary>
    /// Vector-Hessian product <c>vᵀ · ∇²f(x)</c>. For symmetric Hessians
    /// (always the case for a C² <paramref name="fn"/>) this equals
    /// <see cref="Hvp"/> — kept as a separate method to match
    /// <c>torch.autograd.functional.vhp</c>.
    /// </summary>
    public static Tensor<T> Vhp(
        Func<Tensor<T>, Tensor<T>> fn,
        Tensor<T> primal,
        Tensor<T> vec) => Hvp(fn, primal, vec);

    // ─── vmap ─────────────────────────────────────────────────────────

    /// <summary>
    /// Batched function execution — runs <paramref name="fn"/>
    /// independently for each index along
    /// <paramref name="inDim"/> of the input, stacking the results
    /// along <paramref name="outDim"/>. Matches
    /// <c>torch.func.vmap</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>Current implementation:</b> explicit loop over the
    /// batch dimension. The graph-level fusion optimization
    /// (vmap-through-<see cref="Compilation.LazyTensorScope"/>) is a
    /// planned follow-up — this implementation is a correct baseline
    /// that matches the PyTorch semantics and keeps the rest of the
    /// torch.func surface usable today.</para>
    /// <para><b>Shape contract:</b> input must be at least rank-1 and
    /// have <c>size &gt;= 1</c> along <paramref name="inDim"/>. The
    /// output is stacked along <paramref name="outDim"/> with size
    /// equal to the input's size along <paramref name="inDim"/>.</para>
    /// </remarks>
    public static Func<Tensor<T>, Tensor<T>> Vmap(
        Func<Tensor<T>, Tensor<T>> fn,
        int inDim = 0,
        int outDim = 0)
    {
        if (fn is null) throw new ArgumentNullException(nameof(fn));

        return input =>
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            var inShape = input._shape;
            if (inShape.Length == 0)
                throw new ArgumentException("Vmap input must be at least rank-1.", nameof(input));
            int normIn = inDim < 0 ? inDim + inShape.Length : inDim;
            if ((uint)normIn >= (uint)inShape.Length)
                throw new ArgumentOutOfRangeException(nameof(inDim),
                    $"inDim {inDim} out of range for rank-{inShape.Length} input.");

            int batchSize = inShape[normIn];
            if (batchSize <= 0)
                throw new ArgumentException(
                    $"Vmap batch dimension has size {batchSize}; must be > 0.",
                    nameof(input));

            // Slice via the engine's TensorSliceAxis op so each slice
            // stays connected to `input` on the active gradient tape.
            // The previous implementation copied raw element data into
            // new Tensor<T>(...) instances, which severed the graph
            // and made Grad(Vmap(fn)) / JacRev(Vmap(fn)) return zero
            // gradients for the input — a user-visible composition
            // break for a torch.func-style transform.
            var engine = AiDotNetEngine.Current;
            var slices = new Tensor<T>[batchSize];
            for (int b = 0; b < batchSize; b++)
                slices[b] = engine.TensorSliceAxis(input, normIn, b);

            // Apply fn to each slice.
            var outs = new Tensor<T>[batchSize];
            for (int b = 0; b < batchSize; b++) outs[b] = fn(slices[b]);

            // Validate outDim before delegating to TensorStack so the
            // failure message stays in Vmap's argument-validation
            // surface rather than inside the engine.
            int outRank = outs[0]._shape.Length + 1;
            int normOut = outDim < 0 ? outDim + outRank : outDim;
            if ((uint)normOut >= (uint)outRank)
                throw new ArgumentOutOfRangeException(nameof(outDim),
                    $"outDim {outDim} out of range for rank-{outRank} output.");

            // Stack along outDim — TensorStack records on the tape so
            // gradients flow from the assembled output back through
            // every slice to the original input.
            return engine.TensorStack(outs, normOut);
        };
    }

    /// <summary>
    /// Compile-mode-aware vmap. When the caller's <paramref name="fn"/>
    /// is naturally batched (every op inside fn already broadcasts
    /// over the batch dimension), this overload bypasses the
    /// per-sample slice/stack and runs <paramref name="fn"/> on the
    /// full batched tensor exactly once. Recording cost drops from
    /// <c>O(batchSize × ops)</c> to <c>O(ops)</c>, and the resulting
    /// graph is the one the compile pipeline (LazyTensorScope,
    /// CodegenDispatcher, fusion passes) needs to fuse end-to-end.
    /// </summary>
    /// <remarks>
    /// <para><b>Contract on <paramref name="fn"/>:</b> must accept a
    /// tensor with the batch axis at <paramref name="inDim"/> and
    /// produce one whose batch axis sits at <paramref name="outDim"/>.
    /// Unlike <see cref="Vmap"/>, fn is NOT called per-slice — it
    /// sees the full batched tensor.</para>
    /// <para><b>Why a separate method:</b> the slice/stack
    /// <see cref="Vmap"/> is the conservative default that handles
    /// every fn whether it's broadcast-friendly or not (e.g. fn that
    /// uses ops not yet supporting a leading batch axis). The
    /// batched-fast-path overload is opt-in so callers explicitly
    /// confirm fn meets the contract — silently swapping
    /// implementations would break correctness for non-broadcast
    /// fns.</para>
    /// </remarks>
    public static Func<Tensor<T>, Tensor<T>> VmapBatched(
        Func<Tensor<T>, Tensor<T>> fn,
        int inDim = 0,
        int outDim = 0)
    {
        if (fn is null) throw new ArgumentNullException(nameof(fn));

        return input =>
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            var inShape = input._shape;
            if (inShape.Length == 0)
                throw new ArgumentException("VmapBatched input must be at least rank-1.", nameof(input));
            int normIn = inDim < 0 ? inDim + inShape.Length : inDim;
            if ((uint)normIn >= (uint)inShape.Length)
                throw new ArgumentOutOfRangeException(nameof(inDim),
                    $"inDim {inDim} out of range for rank-{inShape.Length} input.");

            // Reshape the batch axis into position 0 if it isn't already
            // there — most engine ops expect leading-batch layout. If
            // inDim != 0 we use a permute. outDim asymmetry is handled
            // by a final permute on the result.
            var engine = AiDotNetEngine.Current;
            Tensor<T> shaped = input;
            if (normIn != 0)
            {
                var perm = new int[inShape.Length];
                perm[0] = normIn;
                int next = 1;
                for (int i = 0; i < inShape.Length; i++)
                    if (i != normIn) perm[next++] = i;
                shaped = engine.TensorPermute(shaped, perm);
            }

            var output = fn(shaped);

            // If outDim != 0, permute the result so the batch axis
            // ends up at the requested position.
            int outRank = output._shape.Length;
            int normOut = outDim < 0 ? outDim + outRank : outDim;
            if ((uint)normOut >= (uint)outRank)
                throw new ArgumentOutOfRangeException(nameof(outDim),
                    $"outDim {outDim} out of range for rank-{outRank} output.");
            if (normOut == 0) return output;

            var outPerm = new int[outRank];
            int p = 0;
            for (int i = 1; i < outRank; i++)
            {
                if (p == normOut) p++;
                outPerm[p++] = i;
            }
            outPerm[normOut] = 0;
            return engine.TensorPermute(output, outPerm);
        };
    }

    // ─── functional_call ──────────────────────────────────────────────

    /// <summary>
    /// Stateless module application — runs <paramref name="fn"/> with
    /// <paramref name="parameters"/> substituted for the module's own
    /// parameters via <paramref name="parameterBuffer"/>. The
    /// parameter buffer's backing storage is overwritten for the
    /// duration of the call and restored afterwards. Matches
    /// <c>torch.func.functional_call</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>Zero-copy swap:</b> we copy the incoming flat
    /// parameter vector into the buffer's single contiguous storage
    /// and snapshot the original values for restore in the finally
    /// block. PyTorch has to clone the entire state dict; we get to
    /// reuse the existing buffer and its per-parameter views.</para>
    /// <para><b>Use case:</b> MAML-style meta-learning, implicit
    /// layers, weight ensembling — anything that needs to evaluate
    /// the same model under a different parameter vector without
    /// constructing a new module.</para>
    /// </remarks>
    public static Tensor<T> FunctionalCall(
        ParameterBuffer<T> parameterBuffer,
        Vector<T> parameters,
        Func<Tensor<T>> fn)
    {
        if (parameterBuffer is null) throw new ArgumentNullException(nameof(parameterBuffer));
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (fn is null) throw new ArgumentNullException(nameof(fn));

        // Hold the buffer's swap lock for the entire snapshot/swap/call/
        // restore sequence. Without this, two threads invoking
        // FunctionalCall on the same buffer can interleave their writes
        // and observe each other's temporary parameters — and the
        // restore order can leave the buffer holding the wrong "saved"
        // snapshot once both calls return. The lock makes the swap
        // atomic with respect to other FunctionalCall invocations on
        // the same buffer; cross-buffer parallelism is unaffected.
        lock (parameterBuffer.SwapLock)
        {
            var saved = parameterBuffer.AsVector();
            var savedCopy = new Vector<T>(saved.Length);
            for (int i = 0; i < saved.Length; i++) savedCopy[i] = saved[i];
            try
            {
                parameterBuffer.CopyFrom(parameters);
                return fn();
            }
            finally
            {
                parameterBuffer.CopyFrom(savedCopy);
            }
        }
    }

    // ─── helpers ──────────────────────────────────────────────────────

    /// <summary>
    /// Sums all elements of a tensor and returns a <c>[1]</c>-shaped
    /// tensor, using engine ops so the result participates in the
    /// current gradient tape. The plain <c>IEngine.TensorSum</c>
    /// returns a raw <c>T</c> (not a <see cref="Tensor{T}"/>), so
    /// wrapping the result in a fresh ctor would sever the graph —
    /// this helper avoids that by expressing "sum all" as an
    /// <c>[1, n] · [n, 1]</c> matmul with a ones vector, which does
    /// record on the tape.
    /// </summary>
    /// <remarks>
    /// Useful when writing scalar-valued functions for
    /// <see cref="Grad(Func{Tensor{T}, Tensor{T}}, bool)"/>,
    /// <see cref="Hessian"/>, or any transform that seeds the backward
    /// pass with a unit cotangent on the output.
    /// </remarks>
    public static Tensor<T> SumToScalarTensor(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        var engine = AiDotNetEngine.Current;
        int n = tensor.Length;
        var row = engine.Reshape(tensor, new[] { 1, n });
        var ops = MathHelper.GetNumericOperations<T>();
        var one = ops.One;
        var onesData = new T[n];
        for (int i = 0; i < n; i++) onesData[i] = one;
        var col = new Tensor<T>(onesData, new[] { n, 1 });
        var one_by_one = engine.TensorMatMul(row, col);
        return engine.Reshape(one_by_one, new[] { 1 });
    }

    private static Tensor<T> ZeroLike(Tensor<T> template)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var zero = ops.Zero;
        var data = new T[template.Length];
        for (int i = 0; i < data.Length; i++) data[i] = zero;
        return new Tensor<T>(data, (int[])template._shape.Clone());
    }

    private static bool ShapesMatch(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }
}
