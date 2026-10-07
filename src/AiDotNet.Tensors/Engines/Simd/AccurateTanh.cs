#if NET5_0_OR_GREATER
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Vectorized float tanh accurate to a few ULP over the whole float range (verified exhaustively
/// against the double-precision tanh, see <c>AccurateTanhTests</c>).
/// </summary>
/// <remarks>
/// <para>The previous vector form, <c>2·sigmoid(2x) − 1</c>, cancels near zero and was hundreds of ULP
/// off there, so float tanh fell back to a scalar <see cref="MathF.Tanh"/> loop: 1.27 ms for 1M
/// elements against libtorch's 95 µs.</para>
/// <para>This follows Cephes <c>tanhf</c>: an odd minimax polynomial <c>x + x³·P(x²)</c> for
/// <c>|x| &lt; 0.625</c> (no cancellation, exact for tiny x), <c>1 − 2/(e^{2|x|} + 1)</c> with the sign
/// restored for <c>0.625 ≤ |x| &lt; 10</c>, and exactly ±1 beyond (tanh(10) rounds to 1 in float; tanh(9)
/// does not: 1 − 3.05·10⁻⁸ rounds to 0.99999994). The
/// exponential uses Cody–Waite range reduction and the Cephes degree-6 polynomial.</para>
/// </remarks>
internal static class AccurateTanh
{
    public static bool IsSupported => Avx2.IsSupported && Fma.IsSupported;

    public static unsafe void Tanh(float* input, float* output, int length)
    {
        int i = 0;
        for (; i + 16 <= length; i += 16)
        {
            Avx.Store(output + i, Tanh256(Avx.LoadVector256(input + i)));
            Avx.Store(output + i + 8, Tanh256(Avx.LoadVector256(input + i + 8)));
        }
        for (; i + 8 <= length; i += 8)
            Avx.Store(output + i, Tanh256(Avx.LoadVector256(input + i)));
        for (; i < length; i++)
            output[i] = MathF.Tanh(input[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Tanh256(Vector256<float> x)
    {
        var signMask = Vector256.Create(-0.0f);
        var ax = Avx.AndNot(signMask, x);                       // |x|
        var sign = Avx.And(signMask, x);

        // Small |x|: x + x^3 * P(x^2).
        var z = Avx.Multiply(x, x);
        var p = Vector256.Create(-5.70498872745E-3f);
        p = Fma.MultiplyAdd(p, z, Vector256.Create(2.06390887954E-2f));
        p = Fma.MultiplyAdd(p, z, Vector256.Create(-5.37397155531E-2f));
        p = Fma.MultiplyAdd(p, z, Vector256.Create(1.33314422036E-1f));
        p = Fma.MultiplyAdd(p, z, Vector256.Create(-3.33332819422E-1f));
        var small = Fma.MultiplyAdd(Avx.Multiply(p, z), x, x);

        // Large |x|: 1 - 2 / (exp(2|x|) + 1), with |x| clamped at 10 so exp cannot overflow.
        var v = Avx.Multiply(Avx.Min(ax, Vector256.Create(10.0f)), Vector256.Create(2.0f));
        var e = Exp256(v);
        var large = Avx.Subtract(Vector256.Create(1.0f),
            Avx.Divide(Vector256.Create(2.0f), Avx.Add(e, Vector256.Create(1.0f))));

        var isSmall = Avx.Compare(ax, Vector256.Create(0.625f), FloatComparisonMode.OrderedLessThanNonSignaling);
        // Both branches are non-negative for x >= 0 and carry x's sign otherwise, except that the small
        // branch loses the sign of -0; OR-ing x's sign bit in restores it and changes nothing else.
        var result = Avx.Or(Avx.BlendVariable(large, small, isSmall), sign);

        // NaN in, NaN out (the clamp above would otherwise turn it into ±1).
        var isNaN = Avx.Compare(x, x, FloatComparisonMode.UnorderedNonSignaling);
        return Avx.BlendVariable(result, x, isNaN);
    }

    /// <summary>exp(v) for v in [0, 20]: Cody–Waite reduction v = n·ln2 + r, then the Cephes polynomial.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Exp256(Vector256<float> v)
    {
        var n = Avx.RoundToNearestInteger(Avx.Multiply(v, Vector256.Create(1.44269504088896341f)));
        var r = Fma.MultiplyAddNegated(n, Vector256.Create(0.693359375f), v);          // v - n·ln2_hi
        r = Fma.MultiplyAddNegated(n, Vector256.Create(-2.12194440e-4f), r);            // - n·ln2_lo

        var p = Vector256.Create(1.9875691500E-4f);
        p = Fma.MultiplyAdd(p, r, Vector256.Create(1.3981999507E-3f));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(8.3334519073E-3f));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(4.1665795894E-2f));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(1.6666665459E-1f));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(5.0000001201E-1f));
        var r2 = Avx.Multiply(r, r);
        var poly = Avx.Add(Fma.MultiplyAdd(p, r2, r), Vector256.Create(1.0f));

        // Scale by 2^n: n is in [0, 29], so adding n to the exponent field cannot overflow.
        var scale = Avx2.ShiftLeftLogical(Avx2.Add(Avx.ConvertToVector256Int32(n), Vector256.Create(127)), 23);
        return Avx.Multiply(poly, scale.AsSingle());
    }
}
#endif
