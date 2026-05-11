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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Memory<float> AsFloatMemory<T>(Memory<T> data)
        => Unsafe.As<Memory<T>, Memory<float>>(ref data);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Memory<double> AsDoubleMemory<T>(Memory<T> data)
        => Unsafe.As<Memory<T>, Memory<double>>(ref data);
}
