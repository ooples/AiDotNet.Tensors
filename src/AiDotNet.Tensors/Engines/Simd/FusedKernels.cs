using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Hardcoded fused SIMD kernels for the most common pointwise operation chains.
/// Each kernel applies multiple ops in a single pass over memory, eliminating
/// intermediate buffer allocation and halving memory bandwidth.
///
/// These replace the delegate-based FusedPointwise composition for the hot paths.
/// Delegate composition is ~2ns/element overhead; these kernels have zero overhead.
/// </summary>
internal static class FusedKernels
{
    /// <summary>Swish/SiLU: x * sigmoid(x) — single pass AVX2</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void SwishUnsafe(float* input, float* output, int length)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported && length >= 8)
        {
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var x = Avx.LoadVector256(input + i);
                var sig = SimdKernels.FastSigmoid256(x);
                Avx.Store(output + i, Avx.Multiply(x, sig));
            }
        }
#endif
        for (; i < length; i++)
        {
            float x = input[i];
            output[i] = x / (1f + MathF.Exp(-x));
        }
    }

    /// <summary>GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) — single pass AVX2</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void GeluUnsafe(float* input, float* output, int length)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported && length >= 8)
        {
            var half = Vector256.Create(0.5f);
            var one = Vector256.Create(1f);
            var sqrt2OverPi = Vector256.Create(0.7978845608028654f);
            var coeff = Vector256.Create(0.044715f);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var x = Avx.LoadVector256(input + i);
                var xCubed = Avx.Multiply(Avx.Multiply(x, x), x);
                var inner = Avx.Multiply(sqrt2OverPi, Fma.MultiplyAdd(coeff, xCubed, x));
                var tanhVal = SimdKernels.FastTanh256(inner);
                Avx.Store(output + i, Avx.Multiply(Avx.Multiply(half, x), Avx.Add(one, tanhVal)));
            }
        }
#endif
        for (; i < length; i++)
        {
            float x = input[i];
            float xCubed = x * x * x;
            float inner = 0.7978845608028654f * (x + 0.044715f * xCubed);
            output[i] = 0.5f * x * (1f + MathF.Tanh(inner));
        }
    }

    /// <summary>Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x))) — single pass AVX2</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void MishUnsafe(float* input, float* output, int length)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported && length >= 8)
        {
            var one = Vector256.Create(1f);
            var threshold = Vector256.Create(20f);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var x = Avx.LoadVector256(input + i);
                // softplus = log(1 + exp(x)), but for x > 20 just use x
                var expX = SimdKernels.FastExp256(x);
                var softplus = SimdKernels.FastLog256(Avx.Add(one, expX));
                // Use x directly when x > threshold (softplus(x) ≈ x for large x)
                var mask = Avx.Compare(x, threshold, FloatComparisonMode.OrderedGreaterThanSignaling);
                softplus = Avx.BlendVariable(softplus, x, mask);
                var tanhSp = SimdKernels.FastTanh256(softplus);
                Avx.Store(output + i, Avx.Multiply(x, tanhSp));
            }
        }
#endif
        for (; i < length; i++)
        {
            float x = input[i];
            float sp = x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
            output[i] = x * MathF.Tanh(sp);
        }
    }

    /// <summary>Add + ReLU fused: max(0, a + b) — single pass AVX2</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AddReluUnsafe(float* a, float* b, float* output, int length)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && length >= 8)
        {
            var zero = Vector256<float>.Zero;
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
                Avx.Store(output + i, Avx.Max(zero, Avx.Add(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i))));
        }
#endif
        for (; i < length; i++)
            output[i] = MathF.Max(0f, a[i] + b[i]);
    }

    /// <summary>RMSNorm: x / sqrt(mean(x^2) + eps) * gamma — fused reduction + normalize</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void RMSNormUnsafe(float* input, float* gamma, float* output, int length, float eps)
    {
        // Pass 1: compute mean of squared values
        float sumSq = 0f;
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var acc = Vector256<float>.Zero;
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var x = Avx.LoadVector256(input + i);
                acc = Fma.MultiplyAdd(x, x, acc);
            }
            sumSq = SimdKernels.HorizontalSum(acc);
        }
#endif
        for (; i < length; i++)
            sumSq += input[i] * input[i];

        float rmsInv = 1f / MathF.Sqrt(sumSq / length + eps);

        // Pass 2: normalize and scale
        i = 0;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && length >= 8)
        {
            var vScale = Vector256.Create(rmsInv);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var x = Avx.LoadVector256(input + i);
                var g = Avx.LoadVector256(gamma + i);
                Avx.Store(output + i, Avx.Multiply(Avx.Multiply(x, vScale), g));
            }
        }
#endif
        for (; i < length; i++)
            output[i] = input[i] * rmsInv * gamma[i];
    }

    /// <summary>Sigmoid + Multiply (Swish building block): sigmoid(a) * b — single pass AVX2</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void SigmoidMulUnsafe(float* a, float* b, float* output, int length)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported && length >= 8)
        {
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var sig = SimdKernels.FastSigmoid256(Avx.LoadVector256(a + i));
                Avx.Store(output + i, Avx.Multiply(sig, Avx.LoadVector256(b + i)));
            }
        }
#endif
        for (; i < length; i++)
            output[i] = (1f / (1f + MathF.Exp(-a[i]))) * b[i];
    }

    /// <summary>
    /// Tries to match an op chain to a hardcoded fused kernel.
    /// Returns true if a match was found.
    /// </summary>
    internal static bool TryGetFusedKernel(string[] opChain, out string kernelName)
    {
        kernelName = string.Empty;
        if (opChain.Length == 1)
        {
            switch (opChain[0])
            {
                case "Swish": kernelName = "Swish"; return true;
                case "GELU": kernelName = "GELU"; return true;
                case "Mish": kernelName = "Mish"; return true;
            }
        }
        else if (opChain.Length == 2)
        {
            string combo = opChain[0] + "+" + opChain[1];
            switch (combo)
            {
                case "Sigmoid+TensorMultiply": kernelName = "SigmoidMul"; return true;
                case "TensorAdd+ReLU": kernelName = "AddReLU"; return true;
            }
        }
        return false;
    }
}
