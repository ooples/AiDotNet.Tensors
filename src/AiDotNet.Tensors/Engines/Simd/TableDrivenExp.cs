using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Table-driven vectorized exp(x) that targets MKL-level performance.
///
/// Algorithm: exp(x) = 2^(x/ln2) = 2^n * 2^(f/L)
///   where n = integer part, f = fractional part subdivided into L=32 entries
///
/// Steps:
///   1. z = x * (L / ln2) = x * 46.16624...
///   2. n = floor(z)
///   3. k = n mod L (table index, 0..31)
///   4. int_part = n / L (integer exponent)
///   5. r = x - n * (ln2 / L) (tiny remainder in [0, 0.0217))
///   6. poly = 1 + r + 0.5*r^2 (2nd order — sufficient for float32 on this range)
///   7. result = table[k] * poly * 2^int_part (bit shift for 2^int_part)
///
/// Total SIMD operations: ~8 per 8 floats (vs ~12 for Estrin polynomial)
/// Lookup table: 32 entries = 128 bytes (fits in L1 cache with room to spare)
///
/// References:
///   - herumi/fmath: table-driven exp for AVX2
///   - Schraudolph: IEEE 754 bit manipulation for 2^n
///   - Malossi et al.: improved error correction for Schraudolph's method
/// </summary>
internal static class TableDrivenExp
{
    // Lookup table: 2^(k/32) for k=0..31
    // Stored as both float[] (for scalar) and will be loaded into AVX registers
    private static readonly float[] ExpTable =
    {
        1.0000000000f, 1.0218971487f, 1.0442737824f, 1.0671404007f,
        1.0905077327f, 1.1143867426f, 1.1387886348f, 1.1637248588f,
        1.1892071150f, 1.2152473600f, 1.2418578121f, 1.2690509572f,
        1.2968395547f, 1.3252366432f, 1.3542555469f, 1.3839098820f,
        1.4142135624f, 1.4451808070f, 1.4768261459f, 1.5091644276f,
        1.5422108254f, 1.5759808451f, 1.6104903319f, 1.6457554782f,
        1.6817928305f, 1.7186192981f, 1.7562521604f, 1.7947090750f,
        1.8340080864f, 1.8741676341f, 1.9152065614f, 1.9571441242f,
    };

    private const int L = 32;           // Table size
    private const int LShift = 5;       // log2(L) for bit shifting
    private const int LMask = L - 1;    // Bitmask for table index (0x1F)

    // Constants
    private const float LOverLn2 = 46.16624130844683f;  // L / ln(2)
    private const float Ln2OverL = 0.021660849390173098f; // ln(2) / L

    // Clamp bounds (avoid inf/nan)
    private const float ClampMin = -87.3365f;  // exp(-87.3) ~ 1e-38
    private const float ClampMax = 88.7228f;   // exp(88.7) ~ 3.4e38

#if NET5_0_OR_GREATER
    /// <summary>
    /// AVX2 table-driven exp: 8 floats at a time.
    /// Uses vpgatherdd for table lookup + 2nd order polynomial.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe Vector256<float> Exp8(Vector256<float> x)
    {
        // Clamp
        x = Avx.Max(Vector256.Create(ClampMin), Avx.Min(Vector256.Create(ClampMax), x));

        // z = x * (L / ln2)
        var vLOverLn2 = Vector256.Create(LOverLn2);
        var z = Avx.Multiply(x, vLOverLn2);

        // n = floor(z) as integer
        var nFloat = Avx.Floor(z);
        var nInt = Avx.ConvertToVector256Int32(nFloat);

        // k = n & (L-1) — table index
        var kInt = Avx2.And(nInt, Vector256.Create(LMask));

        // int_part = n >> log2(L) — integer exponent
        var intPart = Avx2.ShiftRightArithmetic(nInt, LShift);

        // Table lookup: table[k] using AVX2 gather
        fixed (float* tablePtr = ExpTable)
        {
            var tableBase = Avx2.GatherVector256(tablePtr, kInt, 4);

            // r = x - n * (ln2/L) — tiny remainder
            var vLn2OverL = Vector256.Create(Ln2OverL);
            var r = Fma.MultiplyAddNegated(nFloat, vLn2OverL, x);

            // Polynomial: exp(r) ≈ 1 + r + 0.5*r^2
            // Using Horner: (0.5*r + 1)*r + 1
            var half = Vector256.Create(0.5f);
            var one = Vector256.Create(1.0f);
            var poly = Fma.MultiplyAdd(half, r, one); // 0.5*r + 1
            poly = Fma.MultiplyAdd(poly, r, one);     // (0.5*r + 1)*r + 1

            // Combine: table[k] * poly
            var combined = Avx.Multiply(tableBase, poly);

            // Scale by 2^int_part: add int_part to IEEE 754 exponent bits
            // 2^n = reinterpret((int_part + 127) << 23) as float
            var pow2n = Avx2.ShiftLeftLogical(
                Avx2.Add(intPart, Vector256.Create(127)),
                23);

            return Avx.Multiply(combined, pow2n.AsSingle());
        }
    }
#endif

    /// <summary>
    /// Process array of floats using table-driven exp.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void ExpArray(float* input, float* output, int length)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
        {
            // 4x unrolled for max throughput
            int simdLen = length & ~31;
            for (; i < simdLen; i += 32)
            {
                Avx.Store(output + i, Exp8(Avx.LoadVector256(input + i)));
                Avx.Store(output + i + 8, Exp8(Avx.LoadVector256(input + i + 8)));
                Avx.Store(output + i + 16, Exp8(Avx.LoadVector256(input + i + 16)));
                Avx.Store(output + i + 24, Exp8(Avx.LoadVector256(input + i + 24)));
            }
        }
        if (Avx2.IsSupported && Fma.IsSupported && length - i >= 8)
        {
            int simdLen = i + ((length - i) & ~7);
            for (; i < simdLen; i += 8)
                Avx.Store(output + i, Exp8(Avx.LoadVector256(input + i)));
        }
#endif
        // Scalar fallback
        for (; i < length; i++)
            output[i] = ScalarExp(input[i]);
    }

    /// <summary>
    /// Scalar table-driven exp for remainders and non-AVX2 platforms.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ScalarExp(float x)
    {
        if (x < ClampMin) return 0f;
        if (x > ClampMax) return float.PositiveInfinity;

        float z = x * LOverLn2;
        int n = (int)MathF.Floor(z);
        int k = n & LMask;
        int intPart = n >> LShift;

        float r = x - n * Ln2OverL;
        float poly = 1f + r + 0.5f * r * r;

        float tableVal = ExpTable[k];

        // 2^intPart via bit manipulation
        int bits = (intPart + 127) << 23;
        float pow2n = Unsafe.As<int, float>(ref bits);

        return tableVal * poly * pow2n;
    }
}
