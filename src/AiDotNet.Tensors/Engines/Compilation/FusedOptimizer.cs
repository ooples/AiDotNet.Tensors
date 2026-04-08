using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Phase 4.4: Fused optimizer step — compiles SGD/Adam parameter updates
/// directly into the backward pass, eliminating a separate pass over all parameters.
///
/// Instead of: backward → collect gradients → loop params → update
/// The update is appended to the backward actions:
///   backward action writes gradient → immediately updates the parameter
///
/// For SGD: param -= lr * grad (single SIMD kernel per parameter)
/// For Adam: fused momentum + RMSprop + update with bias correction
/// </summary>
internal static class FusedOptimizer
{
    /// <summary>
    /// Creates a fused SGD update action for a single parameter.
    /// Reads from the pre-allocated gradient buffer and updates the parameter in-place.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Action<IEngine>? TryBuildFusedSGD<T>(
        Tensor<T> parameter, Tensor<T> gradient, T learningRate)
    {
        if (typeof(T) != typeof(float) || !parameter.IsContiguous || !gradient.IsContiguous)
            return null;

        float lr = Unsafe.As<T, float>(ref learningRate);
        int length = parameter.Length;

        return eng =>
        {
            var pSpan = parameter.AsWritableSpan();
            var gSpan = gradient.AsSpan();
            var numOps = MathHelper.GetNumericOperations<T>();

            // param -= lr * grad (element-wise)
            for (int i = 0; i < pSpan.Length; i++)
            {
                float p = Unsafe.As<T, float>(ref pSpan[i]);
                float g = Unsafe.As<T, float>(ref Unsafe.AsRef(in gSpan[i]));
                p -= lr * g;
                pSpan[i] = Unsafe.As<float, T>(ref p);
            }
        };
    }

    /// <summary>
    /// Creates a fused Adam update action for a single parameter.
    /// Maintains momentum (m) and squared gradient (v) state internally.
    /// </summary>
    internal static Action<IEngine>? TryBuildFusedAdam<T>(
        Tensor<T> parameter, Tensor<T> gradient,
        T learningRate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
    {
        if (typeof(T) != typeof(float) || !parameter.IsContiguous || !gradient.IsContiguous)
            return null;

        float lr = Unsafe.As<T, float>(ref learningRate);
        int length = parameter.Length;

        // Adam state: momentum and squared gradient accumulators
        var m = new float[length]; // First moment (momentum)
        var v = new float[length]; // Second moment (RMSprop)
        int step = 0;

        return eng =>
        {
            step++;
            float bc1 = 1f - MathF.Pow(beta1, step); // Bias correction for m
            float bc2 = 1f - MathF.Pow(beta2, step); // Bias correction for v

            var pSpan = parameter.AsWritableSpan();
            var gSpan = gradient.AsSpan();

            for (int i = 0; i < length; i++)
            {
                float g = Unsafe.As<T, float>(ref Unsafe.AsRef(in gSpan[i]));

                // Update biased first moment: m = beta1 * m + (1-beta1) * g
                m[i] = beta1 * m[i] + (1f - beta1) * g;
                // Update biased second moment: v = beta2 * v + (1-beta2) * g^2
                v[i] = beta2 * v[i] + (1f - beta2) * g * g;

                // Bias-corrected estimates
                float mHat = m[i] / bc1;
                float vHat = v[i] / bc2;

                // Update: param -= lr * mHat / (sqrt(vHat) + eps)
                float p = Unsafe.As<T, float>(ref pSpan[i]);
                p -= lr * mHat / (MathF.Sqrt(vHat) + epsilon);
                pSpan[i] = Unsafe.As<float, T>(ref p);
            }
        };
    }

    /// <summary>
    /// Appends fused optimizer update actions to a compiled training plan's backward actions.
    /// Each parameter gets its own fused update that runs immediately after its gradient is computed.
    /// </summary>
    internal static Action<IEngine>[] AppendFusedUpdates<T>(
        Action<IEngine>[] backwardActions,
        Tensor<T>[] parameters,
        Tensor<T>[] gradients,
        T learningRate,
        OptimizerType optimizerType = OptimizerType.SGD,
        float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
    {
        var result = new List<Action<IEngine>>(backwardActions);

        for (int i = 0; i < parameters.Length; i++)
        {
            if (gradients[i] is null) continue;

            Action<IEngine>? updateAction = optimizerType switch
            {
                OptimizerType.SGD => TryBuildFusedSGD(parameters[i], gradients[i], learningRate),
                OptimizerType.Adam => TryBuildFusedAdam(parameters[i], gradients[i], learningRate, beta1, beta2, epsilon),
                _ => null
            };

            if (updateAction is not null)
                result.Add(updateAction);
        }

        return result.ToArray();
    }
}

/// <summary>Optimizer type for fused parameter updates.</summary>
public enum OptimizerType
{
    /// <summary>Stochastic Gradient Descent: param -= lr * grad</summary>
    SGD = 0,

    /// <summary>Adam: adaptive learning rate with momentum + RMSprop</summary>
    Adam = 1
}
