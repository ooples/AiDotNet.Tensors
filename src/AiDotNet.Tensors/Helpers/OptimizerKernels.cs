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
        if (param.Length == 0) return;

        // Materialize non-contiguous gradient views — the SIMD kernel
        // walks linearly through memory. Param must already be
        // contiguous (it's a leaf parameter; non-contiguous leaves
        // would already break TensorSubtractInPlace).
        var gradContig = grad.IsContiguous ? grad : grad.Contiguous();

        if (typeof(T) == typeof(float))
        {
            var paramMem = AsFloatMemory(param.Data);
            var gradMem = AsFloatMemory(gradContig.Data);
            using var pinParam = paramMem.Pin();
            using var pinGrad = gradMem.Pin();
            float lrF = (float)(object)lr!;
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
            double lrD = (double)(object)lr!;
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
            FusedOptimizer.AdamUpdateSimd(
                (float*)pinP.Pointer, (float*)pinG.Pointer,
                (float*)pinM.Pointer, (float*)pinV.Pointer,
                param.Length,
                (float)(object)lr!, (float)(object)beta1!, (float)(object)beta2!, (float)(object)eps!,
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
                (double)(object)lr!, (double)(object)beta1!, (double)(object)beta2!, (double)(object)eps!,
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
            FusedOptimizer.AdamWUpdateSimd(
                (float*)pinP.Pointer, (float*)pinG.Pointer,
                (float*)pinM.Pointer, (float*)pinV.Pointer,
                param.Length,
                (float)(object)lr!, (float)(object)beta1!, (float)(object)beta2!, (float)(object)eps!,
                (float)(object)weightDecay!, step);
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
                (double)(object)lr!, (double)(object)beta1!, (double)(object)beta2!, (double)(object)eps!,
                (double)(object)weightDecay!, step);
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
