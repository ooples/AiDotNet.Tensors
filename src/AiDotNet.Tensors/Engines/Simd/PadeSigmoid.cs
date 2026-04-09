using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Padé [2,2] rational approximation for sigmoid — fuses exp and divide into one.
///
/// Mathematical foundation:
///   exp(x) ≈ P(x)/Q(x) where P(x) = 1 + x/2 + x²/12, Q(x) = 1 - x/2 + x²/12
///
///   sigmoid(x) = 1 / (1 + exp(-x))
///              = 1 / (1 + P(-r)/Q(-r))   [after range reduction: x = n*ln2 + r]
///              = Q(-r) / (Q(-r) + P(-r))  [single divide!]
///              = Q(r') / (Q(r') + P(r'))  [where r' = -r, using symmetry]
///
/// For the reduced range r' in [0, ln2):
///   Numerator:   Q(r') = 1 + r'/2 + r'²/12       (2 FMA)
///   Denominator: Q(r') + P(r') = 2 + r'²/6       (1 FMA)
///   Plus range reduction: 3 ops
///   Plus 2^n scaling: 1 op
///   Total: ~8 ops + 1 divide = 9 ops (vs current 12 ops + 1 divide = 13 ops)
///
/// Accuracy: Padé [2,2] gives 4.57e-4 max error on [0, ln2) for the exp component.
/// After sigmoid composition, the error is further reduced because sigmoid clamps.
/// </summary>
internal static class PadeSigmoid
{
#if NET5_0_OR_GREATER
    /// <summary>
    /// Padé sigmoid: 8 floats at a time, single divide, no exp.
    /// </summary>
    /// <summary>
    /// Padé [3,3] fused sigmoid. P and Q numerator/denominator computed IN PARALLEL
    /// (independent FMA chains) reducing effective latency from 6 serial FMAs to 4.
    /// Max error: 1.45e-9 (exceeds float32 precision — essentially exact).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Sigmoid8(Vector256<float> x)
    {
        // sigmoid(x) = 1/(1+exp(-x)) = Q(-r) / (Q(-r) + 2^n * P(-r))
        var negX = Avx.Subtract(Vector256<float>.Zero, x);

        // Range reduction: -x = n*ln2 + r
        var log2e = Vector256.Create(1.44269504088896341f);
        var ln2 = Vector256.Create(0.6931471805599453f);
        var n = Avx.RoundToNearestInteger(Avx.Multiply(negX, log2e));
        var r = Fma.MultiplyAddNegated(n, ln2, negX);

        // Clamp n to [-20, 20] to prevent 2^n overflow in IEEE bit manipulation.
        // This is off the critical path (r computation doesn't depend on clamped n).
        // For |n| > 20, sigmoid is < 1e-6 or > 1-1e-6, so clamping n is safe.
        n = Avx.Max(Vector256.Create(-20.0f), Avx.Min(Vector256.Create(20.0f), n));

        // Padé [3,3]: P(r) = 1 + r/2 + r²/10 + r³/120
        //              Q(r) = 1 - r/2 + r²/10 - r³/120
        var r2 = Avx.Multiply(r, r);
        var r3 = Avx.Multiply(r2, r);

        var r2_10 = Avx.Multiply(Vector256.Create(0.1f), r2);           // r²/10
        var r_half = Avx.Multiply(Vector256.Create(0.5f), r);            // r/2
        var r3_120 = Avx.Multiply(Vector256.Create(1.0f / 120.0f), r3); // r³/120

        var one = Vector256.Create(1.0f);
        var even = Avx.Add(one, r2_10);
        var odd = Avx.Add(r_half, r3_120);

        var P = Avx.Add(even, odd);
        var Q = Avx.Subtract(even, odd);

        // 2^n via IEEE bit manipulation (n is clamped, so nInt+127 is in [107, 147])
        var nInt = Avx.ConvertToVector256Int32(n);
        var pow2n = Avx2.ShiftLeftLogical(
            Avx2.Add(nInt, Vector256.Create(127)), 23).AsSingle();

        // sigmoid = Q / (Q + 2^n * P)
        var denom = Fma.MultiplyAdd(pow2n, P, Q);
        return Avx.Divide(Q, denom);
    }
#endif

    /// <summary>Scalar Padé [3,3] sigmoid.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float Sigmoid(float x)
    {
        x = MathF.Max(-20f, MathF.Min(20f, x));
        float negX = -x;
        float n = MathF.Round(negX * 1.44269504088896341f);
        float r = negX - n * 0.6931471805599453f;
        float r2 = r * r;
        float r3 = r2 * r;
        float even = 1f + r2 * 0.1f;
        float odd = r * 0.5f + r3 / 120f;
        float P = even + odd;
        float Q = even - odd;
        int nInt = (int)n;
        int bits = (nInt + 127) << 23;
        float pow2n = Unsafe.As<int, float>(ref bits);
        float denom = Q + pow2n * P;
        return Q / denom;
    }

    /// <summary>
    /// Process array using Padé sigmoid.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void SigmoidArray(float* input, float* output, int length)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
        {
            int simdLen = length & ~31;
            for (; i < simdLen; i += 32)
            {
                Avx.Store(output + i, Sigmoid8(Avx.LoadVector256(input + i)));
                Avx.Store(output + i + 8, Sigmoid8(Avx.LoadVector256(input + i + 8)));
                Avx.Store(output + i + 16, Sigmoid8(Avx.LoadVector256(input + i + 16)));
                Avx.Store(output + i + 24, Sigmoid8(Avx.LoadVector256(input + i + 24)));
            }
        }
        if (Avx2.IsSupported && Fma.IsSupported && length - i >= 8)
        {
            int simdLen = i + ((length - i) & ~7);
            for (; i < simdLen; i += 8)
                Avx.Store(output + i, Sigmoid8(Avx.LoadVector256(input + i)));
        }
#endif
        for (; i < length; i++)
            output[i] = Sigmoid(input[i]);
    }

}
