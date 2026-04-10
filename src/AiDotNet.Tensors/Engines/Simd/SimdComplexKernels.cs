using System;
using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// SIMD-accelerated kernels for split real/imaginary complex operations.
/// Uses AVX/SSE on x86 and NEON on ARM for vectorized complex arithmetic.
/// Falls back to scalar for platforms without intrinsics.
/// </summary>
public static class SimdComplexKernels
{
    /// <summary>
    /// SIMD complex multiply: outR = aR*bR - aI*bI, outI = aR*bI + aI*bR
    /// </summary>
    [MethodImpl(HotInline)]
    public static void ComplexMultiply(ReadOnlySpan<float> aR, ReadOnlySpan<float> aI,
        ReadOnlySpan<float> bR, ReadOnlySpan<float> bI,
        Span<float> outR, Span<float> outI)
    {
        int n = aR.Length;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Avx.IsSupported && n >= 8)
        {
            int simdLen = n & ~7;
            for (; i < simdLen; i += 8)
            {
                var ar = SimdKernels.ReadVector256(aR, i);
                var ai = SimdKernels.ReadVector256(aI, i);
                var br = SimdKernels.ReadVector256(bR, i);
                var bi = SimdKernels.ReadVector256(bI, i);

                // outR = ar*br - ai*bi
                SimdKernels.WriteVector256(outR, i, Avx.Subtract(Avx.Multiply(ar, br), Avx.Multiply(ai, bi)));
                // outI = ar*bi + ai*br
                SimdKernels.WriteVector256(outI, i, Avx.Add(Avx.Multiply(ar, bi), Avx.Multiply(ai, br)));
            }
        }
#endif

        // Scalar fallback
        for (; i < n; i++)
        {
            outR[i] = aR[i] * bR[i] - aI[i] * bI[i];
            outI[i] = aR[i] * bI[i] + aI[i] * bR[i];
        }
    }

    /// <summary>
    /// SIMD complex conjugate: outR = inR, outI = -inI
    /// </summary>
    [MethodImpl(HotInline)]
    public static void ComplexConjugate(ReadOnlySpan<float> inR, ReadOnlySpan<float> inI,
        Span<float> outR, Span<float> outI)
    {
        int n = inR.Length;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Avx.IsSupported && n >= 8)
        {
            var negMask = Vector256.Create(-0.0f); // sign bit
            int simdLen = n & ~7;
            for (; i < simdLen; i += 8)
            {
                SimdKernels.WriteVector256(outR, i, SimdKernels.ReadVector256(inR, i));
                SimdKernels.WriteVector256(outI, i, Avx.Xor(SimdKernels.ReadVector256(inI, i), negMask));
            }
        }
#endif

        for (; i < n; i++)
        {
            outR[i] = inR[i];
            outI[i] = -inI[i];
        }
    }

    /// <summary>
    /// SIMD complex magnitude: out = sqrt(re^2 + im^2)
    /// </summary>
    [MethodImpl(HotInline)]
    public static void ComplexMagnitude(ReadOnlySpan<float> inR, ReadOnlySpan<float> inI, Span<float> output)
    {
        int n = inR.Length;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Avx.IsSupported && n >= 8)
        {
            int simdLen = n & ~7;
            for (; i < simdLen; i += 8)
            {
                var re = SimdKernels.ReadVector256(inR, i);
                var im = SimdKernels.ReadVector256(inI, i);
                var magSq = Avx.Add(Avx.Multiply(re, re), Avx.Multiply(im, im));
                SimdKernels.WriteVector256(output, i, Avx.Sqrt(magSq));
            }
        }
#endif

        for (; i < n; i++)
        {
            output[i] = MathF.Sqrt(inR[i] * inR[i] + inI[i] * inI[i]);
        }
    }

    /// <summary>
    /// SIMD complex magnitude squared: out = re^2 + im^2 (no sqrt)
    /// </summary>
    [MethodImpl(HotInline)]
    public static void ComplexMagnitudeSquared(ReadOnlySpan<float> inR, ReadOnlySpan<float> inI, Span<float> output)
    {
        int n = inR.Length;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Avx.IsSupported && n >= 8)
        {
            int simdLen = n & ~7;
            for (; i < simdLen; i += 8)
            {
                var re = SimdKernels.ReadVector256(inR, i);
                var im = SimdKernels.ReadVector256(inI, i);
                SimdKernels.WriteVector256(output, i, Avx.Add(Avx.Multiply(re, re), Avx.Multiply(im, im)));
            }
        }
#endif

        for (; i < n; i++)
        {
            output[i] = inR[i] * inR[i] + inI[i] * inI[i];
        }
    }

    /// <summary>
    /// SIMD complex scale: outR = inR * scalar, outI = inI * scalar
    /// </summary>
    [MethodImpl(HotInline)]
    public static void ComplexScale(ReadOnlySpan<float> inR, ReadOnlySpan<float> inI,
        Span<float> outR, Span<float> outI, float scalar)
    {
        int n = inR.Length;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Avx.IsSupported && n >= 8)
        {
            var s = Vector256.Create(scalar);
            int simdLen = n & ~7;
            for (; i < simdLen; i += 8)
            {
                SimdKernels.WriteVector256(outR, i, Avx.Multiply(SimdKernels.ReadVector256(inR, i), s));
                SimdKernels.WriteVector256(outI, i, Avx.Multiply(SimdKernels.ReadVector256(inI, i), s));
            }
        }
#endif

        for (; i < n; i++)
        {
            outR[i] = inR[i] * scalar;
            outI[i] = inI[i] * scalar;
        }
    }

    /// <summary>
    /// SIMD complex add: outR = aR + bR, outI = aI + bI
    /// </summary>
    [MethodImpl(HotInline)]
    public static void ComplexAdd(ReadOnlySpan<float> aR, ReadOnlySpan<float> aI,
        ReadOnlySpan<float> bR, ReadOnlySpan<float> bI,
        Span<float> outR, Span<float> outI)
    {
        int n = aR.Length;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Avx.IsSupported && n >= 8)
        {
            int simdLen = n & ~7;
            for (; i < simdLen; i += 8)
            {
                SimdKernels.WriteVector256(outR, i, Avx.Add(SimdKernels.ReadVector256(aR, i), SimdKernels.ReadVector256(bR, i)));
                SimdKernels.WriteVector256(outI, i, Avx.Add(SimdKernels.ReadVector256(aI, i), SimdKernels.ReadVector256(bI, i)));
            }
        }
#endif

        for (; i < n; i++)
        {
            outR[i] = aR[i] + bR[i];
            outI[i] = aI[i] + bI[i];
        }
    }

    /// <summary>
    /// SIMD cross-spectral density: X * conj(Y)
    /// outR = xR*yR + xI*yI, outI = xI*yR - xR*yI
    /// </summary>
    [MethodImpl(HotInline)]
    public static void ComplexCrossSpectral(ReadOnlySpan<float> xR, ReadOnlySpan<float> xI,
        ReadOnlySpan<float> yR, ReadOnlySpan<float> yI,
        Span<float> outR, Span<float> outI)
    {
        int n = xR.Length;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Avx.IsSupported && n >= 8)
        {
            int simdLen = n & ~7;
            for (; i < simdLen; i += 8)
            {
                var xr = SimdKernels.ReadVector256(xR, i);
                var xi = SimdKernels.ReadVector256(xI, i);
                var yr = SimdKernels.ReadVector256(yR, i);
                var yi = SimdKernels.ReadVector256(yI, i);

                SimdKernels.WriteVector256(outR, i, Avx.Add(Avx.Multiply(xr, yr), Avx.Multiply(xi, yi)));
                SimdKernels.WriteVector256(outI, i, Avx.Subtract(Avx.Multiply(xi, yr), Avx.Multiply(xr, yi)));
            }
        }
#endif

        for (; i < n; i++)
        {
            outR[i] = xR[i] * yR[i] + xI[i] * yI[i];
            outI[i] = xI[i] * yR[i] - xR[i] * yI[i];
        }
    }
}
