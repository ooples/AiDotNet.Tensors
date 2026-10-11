using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
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
        if (float.IsNaN(maxNorm) || maxNorm < 0f)
            throw new ArgumentOutOfRangeException(nameof(maxNorm), maxNorm,
                "Maximum gradient norm must be non-negative and not NaN.");
        if (gradients is null || gradients.Length == 0) return 0f;

        // Step 1: Compute total L2 norm across all gradient tensors
        // Squaring a finite float around 1e20 overflows float. Double covers
        // the entire float domain and still uses SIMD in SumSquares below.
        double totalNormSq = 0d;
        for (int g = 0; g < gradients.Length; g++)
        {
            if (gradients[g] is null) continue;
            var grad = gradients[g];
            if (grad.TryGetContiguousSpan(out var logical))
            {
                totalNormSq += SumSquares(logical);
            }
            else
            {
                // Non-contiguous views cannot expose one logical span. Copy into a
                // pooled scratch buffer so stride traversal remains correct without
                // creating a persistent tensor or a full-size GC allocation.
                var scratch = System.Buffers.ArrayPool<float>.Shared.Rent(grad.Length);
                try
                {
                    var logicalCopy = scratch.AsSpan(0, grad.Length);
                    grad.CopyLogicalTo(logicalCopy);
                    totalNormSq += SumSquares(logicalCopy);
                }
                finally
                {
                    System.Buffers.ArrayPool<float>.Shared.Return(scratch);
                }
            }
        }

        double totalNorm = Math.Sqrt(totalNormSq);

        // Step 2: If norm exceeds maxNorm, scale all gradients down
        if (totalNorm > maxNorm)
        {
            float scale = (float)(maxNorm / (totalNorm + 1e-6d));
            for (int g = 0; g < gradients.Length; g++)
            {
                if (gradients[g] is null) continue;
                var grad = gradients[g];
                if (grad.IsContiguous)
                {
                    // AsWritableSpan is the single write-intent gate: it applies COW,
                    // respects a view's offset, excludes ArrayPool padding, and works
                    // for managed and native CPU storage.
                    Scale(grad.AsWritableSpan(), scale);
                    grad.IncrementVersion();
                }
                else
                {
                    grad.ScaleLogicalInPlace(scale);
                }
            }
        }

        return (float)totalNorm;
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
        if (float.IsNaN(clipValue) || clipValue < 0f)
            throw new ArgumentOutOfRangeException(nameof(clipValue), clipValue,
                "Gradient clip value must be non-negative and not NaN.");
        if (gradients is null) return;
        float negClip = -clipValue;

        for (int g = 0; g < gradients.Length; g++)
        {
            if (gradients[g] is null) continue;
            var grad = gradients[g];
            if (!grad.IsContiguous)
            {
                grad.ClampLogicalInPlace(negClip, clipValue);
                continue;
            }

            var data = grad.AsWritableSpan();
            int len = data.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && len >= 8)
            {
                var vMax = Vector256.Create(clipValue);
                var vMin = Vector256.Create(negClip);
                int simdLen = len & ~7;
                fixed (float* p = &data.GetPinnableReference())
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
            grad.IncrementVersion();
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe double SumSquares(ReadOnlySpan<float> data)
    {
        double result = 0d;
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && data.Length >= 8)
        {
            var lowerAccumulator = Vector256<double>.Zero;
            var upperAccumulator = Vector256<double>.Zero;
            int simdLength = data.Length & ~7;
            fixed (float* p = &data.GetPinnableReference())
            {
                for (; i < simdLength; i += 8)
                {
                    var value = Avx.LoadVector256(p + i);
                    var lower = Avx.ConvertToVector256Double(value.GetLower());
                    var upper = Avx.ConvertToVector256Double(value.GetUpper());
                    if (Fma.IsSupported)
                    {
                        lowerAccumulator = Fma.MultiplyAdd(lower, lower, lowerAccumulator);
                        upperAccumulator = Fma.MultiplyAdd(upper, upper, upperAccumulator);
                    }
                    else
                    {
                        lowerAccumulator = Avx.Add(
                            lowerAccumulator, Avx.Multiply(lower, lower));
                        upperAccumulator = Avx.Add(
                            upperAccumulator, Avx.Multiply(upper, upper));
                    }
                }
            }
            var lanes = stackalloc double[4];
            Avx.Store(lanes, Avx.Add(lowerAccumulator, upperAccumulator));
            result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
        }
#endif
        for (; i < data.Length; i++)
        {
            double value = data[i];
            result += value * value;
        }
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Scale(Span<float> data, float scale)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && data.Length >= 8)
        {
            var vectorScale = Vector256.Create(scale);
            int simdLength = data.Length & ~7;
            fixed (float* p = &data.GetPinnableReference())
            {
                for (; i < simdLength; i += 8)
                    Avx.Store(p + i, Avx.Multiply(Avx.LoadVector256(p + i), vectorScale));
            }
        }
#endif
        for (; i < data.Length; i++)
            data[i] *= scale;
    }
}
