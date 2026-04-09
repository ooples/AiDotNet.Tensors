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
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Sigmoid8(Vector256<float> x)
    {
        // For sigmoid, we work with -x (since sigmoid(x) = 1/(1+exp(-x)))
        var negX = Avx.Subtract(Vector256<float>.Zero, x);

        // Range reduction: -x = n*ln2 + r where n = round(-x/ln2), r in [-ln2/2, ln2/2]
        var log2e = Vector256.Create(1.44269504088896341f); // 1/ln(2)
        var ln2 = Vector256.Create(0.6931471805599453f);
        var n = Avx.RoundToNearestInteger(Avx.Multiply(negX, log2e));
        var r = Fma.MultiplyAddNegated(n, ln2, negX); // r = -x - n*ln2

        // Padé [2,2] of exp(r): P(r)/Q(r)
        // P(r) = 1 + r/2 + r²/12 = ((r/12)*r + r/2) + 1
        // Q(r) = 1 - r/2 + r²/12 = ((r/12)*r - r/2) + 1
        var half = Vector256.Create(0.5f);
        var twelfth = Vector256.Create(1.0f / 12.0f);
        var one = Vector256.Create(1.0f);
        var r2_12 = Avx.Multiply(twelfth, Avx.Multiply(r, r)); // r²/12
        var r_half = Avx.Multiply(half, r); // r/2

        var P = Avx.Add(Avx.Add(r2_12, r_half), one); // 1 + r/2 + r²/12
        var Q = Avx.Add(Avx.Subtract(r2_12, r_half), one); // 1 - r/2 + r²/12

        // sigmoid(x) = 1/(1 + exp(-x)) = 1/(1 + 2^n * P/Q)
        // = Q / (Q + 2^n * P)

        // Compute 2^n via IEEE bit manipulation
        var nInt = Avx.ConvertToVector256Int32(n);
        var pow2n = Avx2.ShiftLeftLogical(
            Avx2.Add(nInt, Vector256.Create(127)), 23).AsSingle();

        // Denominator: Q + 2^n * P
        var denom = Fma.MultiplyAdd(pow2n, P, Q);

        // Result: Q / denom (single divide)
        var result = Avx.Divide(Q, denom);

        // Clamp to [0, 1] for numerical stability at extremes
        result = Avx.Max(Vector256<float>.Zero, Avx.Min(one, result));

        return result;
    }
#endif

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
        {
            float negX = -input[i];
            float n = MathF.Round(negX * 1.44269504088896341f);
            float r = negX - n * 0.6931471805599453f;
            float r2_12 = r * r / 12f;
            float P = 1f + r / 2f + r2_12;
            float Q = 1f - r / 2f + r2_12;
            int nInt = (int)n;
            int bits = (nInt + 127) << 23;
            float pow2n = Unsafe.As<int, float>(ref bits);
            float denom = Q + pow2n * P;
            output[i] = MathF.Max(0f, MathF.Min(1f, Q / denom));
        }
    }
}
