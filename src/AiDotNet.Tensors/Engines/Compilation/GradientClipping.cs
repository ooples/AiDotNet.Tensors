using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Gradient clipping utilities with AVX2 vectorization.
/// Prevents gradient explosion in transformer and deep network training.
///
/// PyTorch equivalents:
///   torch.nn.utils.clip_grad_norm_(params, max_norm)
///   torch.nn.utils.clip_grad_value_(params, clip_value)
/// </summary>
public static class GradientClipping
{
    /// <summary>
    /// Clips gradient tensor by global L2 norm. If the total norm exceeds maxNorm,
    /// all gradients are scaled down proportionally.
    ///
    /// This is the most common clipping strategy for transformer training.
    /// </summary>
    /// <param name="gradients">Gradient tensors to clip (modified in-place).</param>
    /// <param name="maxNorm">Maximum allowed L2 norm.</param>
    /// <returns>The total gradient norm before clipping.</returns>
    public static unsafe float ClipGradNorm(Tensor<float>[] gradients, float maxNorm)
    {
        if (gradients is null || gradients.Length == 0) return 0f;

        // Step 1: Compute total L2 norm across all gradient tensors
        float totalNormSq = 0f;
        for (int g = 0; g < gradients.Length; g++)
        {
            if (gradients[g] is null) continue;
            // Ensure contiguous — GetDataArray may return a copy for views
            var grad = gradients[g].IsContiguous ? gradients[g] : gradients[g].Contiguous();
            var data = grad.GetDataArray();
            int len = data.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Fma.IsSupported && len >= 8)
            {
                var acc = Vector256<float>.Zero;
                int simdLen = len & ~7;
                fixed (float* p = data)
                {
                    for (; i < simdLen; i += 8)
                    {
                        var v = Avx.LoadVector256(p + i);
                        acc = Fma.MultiplyAdd(v, v, acc);
                    }
                }
                totalNormSq += SimdKernels.HorizontalSum(acc);
            }
#endif
            for (; i < len; i++)
                totalNormSq += data[i] * data[i];
        }

        float totalNorm = MathF.Sqrt(totalNormSq);

        // Step 2: If norm exceeds maxNorm, scale all gradients down
        if (totalNorm > maxNorm)
        {
            float scale = maxNorm / (totalNorm + 1e-6f);
            for (int g = 0; g < gradients.Length; g++)
            {
                if (gradients[g] is null) continue;
                var data = gradients[g].GetDataArray();
                int len = data.Length;
                int i = 0;

#if NET5_0_OR_GREATER
                if (Avx.IsSupported && len >= 8)
                {
                    var vScale = Vector256.Create(scale);
                    int simdLen = len & ~7;
                    fixed (float* p = data)
                    {
                        for (; i < simdLen; i += 8)
                            Avx.Store(p + i, Avx.Multiply(Avx.LoadVector256(p + i), vScale));
                    }
                }
#endif
                for (; i < len; i++)
                    data[i] *= scale;
            }
        }

        return totalNorm;
    }

    /// <summary>
    /// Clips each gradient element to [-clipValue, +clipValue].
    /// Simpler but less commonly used than norm clipping.
    /// </summary>
    /// <param name="gradients">Gradient tensors to clip (modified in-place).</param>
    /// <param name="clipValue">Maximum absolute value per element.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void ClipGradValue(Tensor<float>[] gradients, float clipValue)
    {
        if (gradients is null) return;
        float negClip = -clipValue;

        for (int g = 0; g < gradients.Length; g++)
        {
            if (gradients[g] is null) continue;
            var data = gradients[g].GetDataArray();
            int len = data.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && len >= 8)
            {
                var vMax = Vector256.Create(clipValue);
                var vMin = Vector256.Create(negClip);
                int simdLen = len & ~7;
                fixed (float* p = data)
                {
                    for (; i < simdLen; i += 8)
                    {
                        var v = Avx.LoadVector256(p + i);
                        v = Avx.Max(v, vMin);
                        v = Avx.Min(v, vMax);
                        Avx.Store(p + i, v);
                    }
                }
            }
#endif
            for (; i < len; i++)
            {
                if (data[i] > clipValue) data[i] = clipValue;
                else if (data[i] < negClip) data[i] = negClip;
            }
        }
    }
}
