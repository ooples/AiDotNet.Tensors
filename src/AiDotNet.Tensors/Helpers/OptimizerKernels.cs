using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Public, allocation-free optimizer-step primitives for eager-mode
/// training loops. Apply the parameter update directly to the
/// parameter buffer — no intermediate tensor allocation, no GC
/// pressure on the per-step hot path.
/// </summary>
/// <remarks>
/// <para>
/// The naive eager-mode optimizer pattern is:
/// <code>
/// foreach (var p in parameters)
/// {
///     var update = engine.TensorMultiplyScalar(grads[p], lr);
///     engine.TensorSubtractInPlace(p, update);
/// }
/// </code>
/// That allocates a fresh tensor (<c>update</c>) per parameter per
/// step. On ViT-Base-scale models this is ~9 MB / iter just for the
/// SGD step — visible as Gen-0 GC pressure and a measurable share of
/// total iteration time in <c>Issue319TrainingLoopPerfTests</c>.
/// </para>
/// <para>
/// <see cref="SgdInPlace{T}(Tensor{T}, Tensor{T}, T)"/> does the
/// equivalent <c>param -= lr · grad</c> in a single fused SIMD pass.
/// Float and double specialize to AVX2/FMA via the same
/// <c>FusedOptimizer.SgdUpdateSimd</c> kernel that powers the
/// compiled-training path; other numeric types use a scalar fallback
/// via <see cref="INumericOperations{T}"/>.
/// </para>
/// <para>
/// This API is intended for use OUTSIDE an active <c>GradientTape</c>
/// — i.e. after gradients have been computed, when the optimizer is
/// updating leaf parameters. It does not record on the tape.
/// </para>
/// <para>
/// <b>Contiguity contract (PR #322 review #3, #5, #10, #11, #14, #20, #22):</b>
/// the mutated tensors — <c>param</c> for SGD; <c>param</c>, <c>m</c>,
/// <c>v</c> for Adam/AdamW — MUST be contiguous. The SIMD kernels
/// walk the buffers linearly; passing a non-contiguous view (e.g.
/// the result of <c>Transpose</c>) would silently write through the
/// wrong stride and corrupt the optimizer state. Each method
/// validates <c>IsContiguous</c> up front and throws
/// <see cref="ArgumentException"/> with an actionable message when
/// the precondition is violated.
/// </para>
/// <para>
/// <b>"Allocation-free" — clarification (PR #322 review #3, #4, #13, #20):</b>
/// the kernel itself is genuinely allocation-free. The ONE exception
/// is the <c>grad</c> tensor: a non-contiguous <c>grad</c> view is
/// materialized via <c>grad.Contiguous()</c> before the kernel runs,
/// because the SIMD path requires linear memory. Allocate-free
/// behavior therefore requires that <c>grad</c> is contiguous; this
/// is the common case (most backward kernels produce contiguous
/// outputs). When <c>grad</c> is a non-contiguous view, one
/// shape-sized temporary tensor is allocated for the duration of the
/// call. Callers who need strict allocation-free behavior in all
/// cases should call <c>grad.Contiguous()</c> themselves once before
/// the optimizer loop, so the materialization (if any) happens
/// outside the hot path.
/// </para>
/// </remarks>
public static class OptimizerKernels
{
    /// <summary>
    /// In-place SGD parameter update: <c>param[i] -= lr · grad[i]</c>
    /// for every element. Single fused SIMD pass (AVX2/FMA on
    /// float/double); zero allocation.
    /// </summary>
    /// <typeparam name="T">Element type (float, double, or any type
    /// with an <see cref="INumericOperations{T}"/> implementation).</typeparam>
    /// <param name="param">Parameter tensor (mutated in place).
    /// Must be contiguous; if a non-contiguous view is passed, the
    /// caller is responsible for calling <c>Contiguous()</c> first.</param>
    /// <param name="grad">Gradient tensor. Must have the same length
    /// as <paramref name="param"/>. Read-only — not mutated.</param>
    /// <param name="lr">Learning rate. Applied as
    /// <c>param -= lr · grad</c>, so positive <paramref name="lr"/>
    /// performs a standard SGD step.</param>
    /// <exception cref="ArgumentNullException">If
    /// <paramref name="param"/> or <paramref name="grad"/> is null.</exception>
    /// <exception cref="ArgumentException">If the tensors have
    /// different lengths.</exception>
    public static unsafe void SgdInPlace<T>(Tensor<T> param, Tensor<T> grad, T lr)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (grad is null) throw new ArgumentNullException(nameof(grad));
        if (param.Length != grad.Length)
        {
            throw new ArgumentException(
                $"SgdInPlace requires matching lengths; param.Length={param.Length}, grad.Length={grad.Length}.");
        }
        // PR #322 review #22: explicit contiguity guard on `param`. The
        // SIMD kernel pins param.Data and walks it as a flat buffer; a
        // non-contiguous view (e.g. Transpose) would silently write
        // through the wrong stride and corrupt the parameter. Fail fast
        // with a clear message instead of letting Pin throw a generic
        // InvalidOperationException several frames deep.
        if (!param.IsContiguous)
        {
            throw new ArgumentException(
                "SgdInPlace requires param to be contiguous. Got a non-contiguous view; call param.Contiguous() before passing it to the optimizer, or materialize the parameter via its source.",
                nameof(param));
        }
        if (param.Length == 0) return;

        // Non-contiguous gradient views (e.g. permutations from
        // backward) get materialized — the SIMD kernel walks linearly.
        // Documented in the XML remarks as an exception to the
        // allocation-free contract (PR #322 review #3, #4, #13, #20).
        var gradContig = grad.IsContiguous ? grad : grad.Contiguous();

        if (typeof(T) == typeof(float))
        {
            var paramMem = AsFloatMemory(param.Data);
            var gradMem = AsFloatMemory(gradContig.Data);
            using var pinParam = paramMem.Pin();
            using var pinGrad = gradMem.Pin();
            // PR #322 review #12: Unsafe.As reinterprets the typed
            // generic value without boxing. The (T)(object)lr pattern
            // would box lr to System.Object (one heap alloc per call)
            // then unbox to float — wasted work since typeof(T) == float.
            float lrF = System.Runtime.CompilerServices.Unsafe.As<T, float>(ref lr);
            FusedOptimizer.SgdUpdateSimd(
                (float*)pinParam.Pointer, (float*)pinGrad.Pointer, param.Length, lrF);
            return;
        }

        if (typeof(T) == typeof(double))
        {
            var paramMem = AsDoubleMemory(param.Data);
            var gradMem = AsDoubleMemory(gradContig.Data);
            using var pinParam = paramMem.Pin();
            using var pinGrad = gradMem.Pin();
            double lrD = System.Runtime.CompilerServices.Unsafe.As<T, double>(ref lr);
            FusedOptimizer.SgdUpdateSimd(
                (double*)pinParam.Pointer, (double*)pinGrad.Pointer, param.Length, lrD);
            return;
        }

        // Generic-T scalar fallback for half/int/etc. The pinned-pointer
        // SIMD kernels above cover the dominant float/double cases;
        // everything else goes through the per-element numeric ops
        // dispatch. Still zero-allocation — no intermediate tensor.
        var numOps = MathHelper.GetNumericOperations<T>();
        var paramSpanT = param.AsWritableSpan();
        var gradSpanT = gradContig.AsSpan();
        for (int i = 0; i < paramSpanT.Length; i++)
        {
            paramSpanT[i] = numOps.Subtract(
                paramSpanT[i],
                numOps.Multiply(lr, gradSpanT[i]));
        }
    }

    /// <summary>
    /// In-place Adam parameter update with explicit step counter.
    /// Uses the AVX2/FMA-vectorized kernel in
    /// <c>FusedOptimizer.AdamUpdateSimd</c> for float/double; scalar
    /// fallback for other numeric types.
    /// </summary>
    /// <typeparam name="T">Element type (float, double, or any type
    /// with an <see cref="INumericOperations{T}"/> implementation).</typeparam>
    /// <param name="param">Parameter tensor (mutated in place).</param>
    /// <param name="grad">Gradient tensor. Must have the same length
    /// as <paramref name="param"/>.</param>
    /// <param name="m">First-moment buffer (mutated in place). Same
    /// length as <paramref name="param"/>. Caller persists this
    /// across optimizer steps.</param>
    /// <param name="v">Second-moment buffer (mutated in place). Same
    /// length as <paramref name="param"/>. Caller persists this
    /// across optimizer steps.</param>
    /// <param name="lr">Learning rate.</param>
    /// <param name="beta1">First-moment decay (default 0.9 in standard Adam).</param>
    /// <param name="beta2">Second-moment decay (default 0.999 in standard Adam).</param>
    /// <param name="eps">Numerical-stability epsilon (default 1e-8 in standard Adam).</param>
    /// <param name="step">Step counter, 1-indexed. Used for bias
    /// correction; caller increments per optimizer step.</param>
    public static unsafe void AdamInPlace<T>(
        Tensor<T> param, Tensor<T> grad,
        Tensor<T> m, Tensor<T> v,
        T lr, T beta1, T beta2, T eps, int step)
    {
        ValidateMoments(param, grad, m, v);
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step), "Adam step must be >= 1 (1-indexed for bias correction).");
        if (param.Length == 0) return;

        var gradContig = grad.IsContiguous ? grad : grad.Contiguous();

        if (typeof(T) == typeof(float))
        {
            var paramMem = AsFloatMemory(param.Data);
            var gradMem = AsFloatMemory(gradContig.Data);
            var mMem = AsFloatMemory(m.Data);
            var vMem = AsFloatMemory(v.Data);
            using var pinP = paramMem.Pin();
            using var pinG = gradMem.Pin();
            using var pinM = mMem.Pin();
            using var pinV = vMem.Pin();
            // PR #322 review #12: Unsafe.As — no boxing.
            FusedOptimizer.AdamUpdateSimd(
                (float*)pinP.Pointer, (float*)pinG.Pointer,
                (float*)pinM.Pointer, (float*)pinV.Pointer,
                param.Length,
                System.Runtime.CompilerServices.Unsafe.As<T, float>(ref lr),
                System.Runtime.CompilerServices.Unsafe.As<T, float>(ref beta1),
                System.Runtime.CompilerServices.Unsafe.As<T, float>(ref beta2),
                System.Runtime.CompilerServices.Unsafe.As<T, float>(ref eps),
                step);
            return;
        }

        if (typeof(T) == typeof(double))
        {
            var paramMem = AsDoubleMemory(param.Data);
            var gradMem = AsDoubleMemory(gradContig.Data);
            var mMem = AsDoubleMemory(m.Data);
            var vMem = AsDoubleMemory(v.Data);
            using var pinP = paramMem.Pin();
            using var pinG = gradMem.Pin();
            using var pinM = mMem.Pin();
            using var pinV = vMem.Pin();
            FusedOptimizer.AdamUpdateSimd(
                (double*)pinP.Pointer, (double*)pinG.Pointer,
                (double*)pinM.Pointer, (double*)pinV.Pointer,
                param.Length,
                System.Runtime.CompilerServices.Unsafe.As<T, double>(ref lr),
                System.Runtime.CompilerServices.Unsafe.As<T, double>(ref beta1),
                System.Runtime.CompilerServices.Unsafe.As<T, double>(ref beta2),
                System.Runtime.CompilerServices.Unsafe.As<T, double>(ref eps),
                step);
            return;
        }

        AdamFallback(param, gradContig, m, v, lr, beta1, beta2, eps, step);
    }

    /// <summary>
    /// In-place AdamW parameter update — Adam with decoupled weight
    /// decay (multiplies <paramref name="param"/> by
    /// <c>1 - weightDecay·lr</c> before the Adam step).
    /// AdamW is the standard optimizer for transformer training
    /// including ViT — vision-transformer reference configs use it
    /// almost universally over plain Adam.
    /// </summary>
    /// <param name="weightDecay">Decoupled weight-decay coefficient.</param>
    /// <inheritdoc cref="AdamInPlace{T}(Tensor{T}, Tensor{T}, Tensor{T}, Tensor{T}, T, T, T, T, int)"/>
    public static unsafe void AdamWInPlace<T>(
        Tensor<T> param, Tensor<T> grad,
        Tensor<T> m, Tensor<T> v,
        T lr, T beta1, T beta2, T eps, T weightDecay, int step)
    {
        ValidateMoments(param, grad, m, v);
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step), "AdamW step must be >= 1 (1-indexed for bias correction).");
        if (param.Length == 0) return;

        var gradContig = grad.IsContiguous ? grad : grad.Contiguous();

        if (typeof(T) == typeof(float))
        {
            var paramMem = AsFloatMemory(param.Data);
            var gradMem = AsFloatMemory(gradContig.Data);
            var mMem = AsFloatMemory(m.Data);
            var vMem = AsFloatMemory(v.Data);
            using var pinP = paramMem.Pin();
            using var pinG = gradMem.Pin();
            using var pinM = mMem.Pin();
            using var pinV = vMem.Pin();
            // PR #322 review #12: Unsafe.As — no boxing.
            FusedOptimizer.AdamWUpdateSimd(
                (float*)pinP.Pointer, (float*)pinG.Pointer,
                (float*)pinM.Pointer, (float*)pinV.Pointer,
                param.Length,
                System.Runtime.CompilerServices.Unsafe.As<T, float>(ref lr),
                System.Runtime.CompilerServices.Unsafe.As<T, float>(ref beta1),
                System.Runtime.CompilerServices.Unsafe.As<T, float>(ref beta2),
                System.Runtime.CompilerServices.Unsafe.As<T, float>(ref eps),
                System.Runtime.CompilerServices.Unsafe.As<T, float>(ref weightDecay),
                step);
            return;
        }

        if (typeof(T) == typeof(double))
        {
            var paramMem = AsDoubleMemory(param.Data);
            var gradMem = AsDoubleMemory(gradContig.Data);
            var mMem = AsDoubleMemory(m.Data);
            var vMem = AsDoubleMemory(v.Data);
            using var pinP = paramMem.Pin();
            using var pinG = gradMem.Pin();
            using var pinM = mMem.Pin();
            using var pinV = vMem.Pin();
            FusedOptimizer.AdamWUpdateSimd(
                (double*)pinP.Pointer, (double*)pinG.Pointer,
                (double*)pinM.Pointer, (double*)pinV.Pointer,
                param.Length,
                System.Runtime.CompilerServices.Unsafe.As<T, double>(ref lr),
                System.Runtime.CompilerServices.Unsafe.As<T, double>(ref beta1),
                System.Runtime.CompilerServices.Unsafe.As<T, double>(ref beta2),
                System.Runtime.CompilerServices.Unsafe.As<T, double>(ref eps),
                System.Runtime.CompilerServices.Unsafe.As<T, double>(ref weightDecay),
                step);
            return;
        }

        // Scalar AdamW = (decoupled decay) ∘ Adam
        var numOps = MathHelper.GetNumericOperations<T>();
        var paramSpan = param.AsWritableSpan();
        T decayFactor = numOps.Subtract(numOps.One, numOps.Multiply(weightDecay, lr));
        for (int i = 0; i < paramSpan.Length; i++)
            paramSpan[i] = numOps.Multiply(paramSpan[i], decayFactor);
        AdamFallback(param, gradContig, m, v, lr, beta1, beta2, eps, step);
    }

    private static void ValidateMoments<T>(Tensor<T> param, Tensor<T> grad, Tensor<T> m, Tensor<T> v)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (grad is null) throw new ArgumentNullException(nameof(grad));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (param.Length != grad.Length || param.Length != m.Length || param.Length != v.Length)
        {
            throw new ArgumentException(
                $"Adam/AdamW require matching lengths; param={param.Length}, grad={grad.Length}, m={m.Length}, v={v.Length}.");
        }
        // PR #322 review #11, #14, #23: explicit contiguity guards.
        // The SIMD kernels (AdamUpdateSimd / AdamWUpdateSimd) pin
        // param.Data, m.Data, v.Data and walk them linearly. A
        // non-contiguous view of any of these would silently write
        // through the wrong stride and corrupt the optimizer state.
        // Fail fast with actionable messages.
        if (!param.IsContiguous)
            throw new ArgumentException(
                "Adam/AdamW require param to be contiguous. Got a non-contiguous view.",
                nameof(param));
        if (!m.IsContiguous)
            throw new ArgumentException(
                "Adam/AdamW require the first-moment buffer m to be contiguous.",
                nameof(m));
        if (!v.IsContiguous)
            throw new ArgumentException(
                "Adam/AdamW require the second-moment buffer v to be contiguous.",
                nameof(v));
    }

    private static void AdamFallback<T>(
        Tensor<T> param, Tensor<T> grad, Tensor<T> m, Tensor<T> v,
        T lr, T beta1, T beta2, T eps, int step)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var pSpan = param.AsWritableSpan();
        var gSpan = grad.AsSpan();
        var mSpan = m.AsWritableSpan();
        var vSpan = v.AsWritableSpan();

        T oneMinusB1 = numOps.Subtract(numOps.One, beta1);
        T oneMinusB2 = numOps.Subtract(numOps.One, beta2);
        // Bias correction: 1 - beta^step. Computed via Pow for generality
        // (some T may not support efficient integer power; this is the
        // scalar fallback so polymorphism cost is amortized over the
        // per-element loop).
        T bc1 = numOps.Subtract(numOps.One, numOps.Power(beta1, numOps.FromDouble(step)));
        T bc2 = numOps.Subtract(numOps.One, numOps.Power(beta2, numOps.FromDouble(step)));

        for (int i = 0; i < pSpan.Length; i++)
        {
            mSpan[i] = numOps.Add(numOps.Multiply(beta1, mSpan[i]), numOps.Multiply(oneMinusB1, gSpan[i]));
            T g2 = numOps.Multiply(gSpan[i], gSpan[i]);
            vSpan[i] = numOps.Add(numOps.Multiply(beta2, vSpan[i]), numOps.Multiply(oneMinusB2, g2));
            T mHat = numOps.Divide(mSpan[i], bc1);
            T vHat = numOps.Divide(vSpan[i], bc2);
            T denom = numOps.Add(numOps.Sqrt(vHat), eps);
            pSpan[i] = numOps.Subtract(pSpan[i], numOps.Divide(numOps.Multiply(lr, mHat), denom));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Memory<float> AsFloatMemory<T>(Memory<T> data)
        => Unsafe.As<Memory<T>, Memory<float>>(ref data);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Memory<double> AsDoubleMemory<T>(Memory<T> data)
        => Unsafe.As<Memory<T>, Memory<double>>(ref data);
}
