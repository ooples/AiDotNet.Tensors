using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The LAMB update on any GPU backend, built from primitives every backend implements.
/// </summary>
/// <remarks>
/// <para>
/// LAMB scales each layer's Adam step by the trust ratio <c>||w|| / ||r + lambda w||</c>, which needs two whole-tensor
/// norms and therefore cannot be computed by one element-wise kernel. The per-backend <c>lamb_update</c> kernels took the
/// ratio as an argument and every backend passed a constant 1, so GPU "LAMB" was AdamW. This computes the real ratio
/// (with the optional clip) between an element-wise moment/step phase and the parameter update, so every backend runs
/// the same algorithm as the CPU kernel.
/// </para>
/// <para><b>Reference:</b> Y. You et al., "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes",
/// ICLR 2020, Algorithm 2.</para>
/// </remarks>
internal static class GpuLamb
{
    internal static void Step(
        IDirectGpuBackend backend,
        IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size,
        float maxTrustRatio, bool biasCorrection)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step), "Step must be at least 1.");
        if (!(epsilon > 0f)) throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");

        float bc1 = biasCorrection ? 1f - MathF.Pow(beta1, step) : 1f;
        float bc2 = biasCorrection ? 1f - MathF.Pow(beta2, step) : 1f;

        using var scratch = backend.AllocateBuffer(size);
        using var update = backend.AllocateBuffer(size);

        // m = beta1 m + (1 - beta1) g ;  v = beta2 v + (1 - beta2) g^2
        backend.AddScaled(m, gradient, m, beta1, 1f - beta1, size);
        backend.Multiply(gradient, gradient, scratch, size);
        backend.AddScaled(v, scratch, v, beta2, 1f - beta2, size);

        // r = (m / bc1) / (sqrt(v / bc2) + eps) ;  update = r + lambda w
        backend.Scale(v, scratch, 1f / bc2, size);
        backend.Sqrt(scratch, scratch, size);
        backend.AddScalar(scratch, scratch, epsilon, size);
        backend.Scale(m, update, 1f / bc1, size);
        backend.Divide(update, scratch, update, size);
        if (weightDecay != 0f)
            backend.AddScaled(update, param, update, 1f, weightDecay, size);

        float paramNorm = backend.L2Norm(param, size);
        float updateNorm = backend.L2Norm(update, size);
        float trustRatio = paramNorm > 0f && updateNorm > 0f ? paramNorm / updateNorm : 1f;
        if (maxTrustRatio > 0f && trustRatio > maxTrustRatio) trustRatio = maxTrustRatio;

        // w = w - lr * trust * update
        backend.AddScaled(param, update, param, 1f, -learningRate * trustRatio, size);
    }
}
