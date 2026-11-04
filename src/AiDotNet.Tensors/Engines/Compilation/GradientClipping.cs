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

        // The device clip's definition (DeterministicGradientNorm), so the CPU and every GPU backend produce the same
        // norm, coefficient and clipped gradients, bit for bit: exact and overflow-free, in integers. A non-finite
        // element leaves the gradients unscaled (NonFiniteGradientPolicy.LeaveUnscaled), as the device clip does.
        int total = 0;
        for (int g = 0; g < gradients.Length; g++)
            if (gradients[g] is { Length: > 0 } t) total += Gpu.DeterministicGradientNorm.ChunkCount(t.Length);
        var partials = new Gpu.DeterministicGradientNorm.Partial[total];
        var copies = new float[]?[gradients.Length];
        int offset = 0;
        uint index = 0;
        for (int g = 0; g < gradients.Length; g++)
        {
            var grad = gradients[g];
            if (grad is null || grad.Length == 0) continue;
            if (grad.TryGetContiguousSpan(out var logical))
            {
                fixed (float* p = logical)
                    Gpu.DeterministicGradientNorm.ChunkPartials(p, logical.Length, index, partials, offset);
            }
            else
            {
                // Non-contiguous views cannot expose one logical span: reduce (and later scale) a logical copy.
                var copy = copies[g] = new float[grad.Length];
                grad.CopyLogicalTo(copy);
                fixed (float* p = copy)
                    Gpu.DeterministicGradientNorm.ChunkPartials(p, copy.Length, index, partials, offset);
            }
            offset += Gpu.DeterministicGradientNorm.ChunkCount(grad.Length);
            index++;
        }
        var record = Gpu.DeterministicGradientNorm.Combine(partials, maxNorm);
        uint scale = record[Gpu.DeterministicGradientNorm.WordScale];

        if (scale != 0x3F800000u)
        {
            for (int g = 0; g < gradients.Length; g++)
            {
                var grad = gradients[g];
                if (grad is null || grad.Length == 0) continue;
                if (grad.IsContiguous)
                {
                    // AsWritableSpan is the single write-intent gate: it applies COW,
                    // respects a view's offset, excludes ArrayPool padding, and works
                    // for managed and native CPU storage.
                    using var gradKeep1 = new KeepAliveScope(grad);
                    var span = grad.AsWritableSpanUnmarked();
                    fixed (float* p = span) Gpu.DeterministicGradientNorm.Scale(p, span.Length, scale);
                    grad.IncrementVersion();
                }
                else
                {
                    grad.ScaleLogicalInPlace(Gpu.DeterministicGradientNorm.UInt32BitsToSingle(scale));
                }
            }
        }

        return Gpu.DeterministicGradientNorm.UInt32BitsToSingle(record[Gpu.DeterministicGradientNorm.WordNorm]);
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

            using var dataLease = grad.LeaseWritable();
            var data = dataLease.Span;
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
