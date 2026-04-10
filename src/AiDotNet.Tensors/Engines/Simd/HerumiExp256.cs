using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// 256-entry herumi/fmath-style table-driven exp(x) for maximum throughput.
///
/// Algorithm: exp(x) = 2^(x/ln2) = 2^n * 2^(f/256)
///   where n = integer exponent, f = fractional index into 256-entry table.
///
/// The 256-entry table makes the remainder range [0, ln2/256) = [0, 0.00271)
/// so tiny that a 1st-order polynomial (1 + r) suffices for float32 precision.
///
/// Total: clamp + mul + floor + and + shift + gather + FMA + mul + shift = ~9 ops
/// Accuracy: max relative error ~4.2e-7 (within 2 ULP for float32)
///
/// References:
///   - herumi/fmath (https://github.com/herumi/fmath) — original AVX2 table exp
///   - LitMath for .NET — array-level function chaining insight
/// </summary>
internal static class HerumiExp256
{
    private const int L = 256;
    private const int LShift = 8;
    private const int LMask = L - 1;
    private const float LOverLn2 = 369.3299304675746f;    // 256 / ln(2)
    private const float Ln2OverL = 0.002707606174f;         // ln(2) / 256
    private const float ClampMin = -87.3365f;
    private const float ClampMax = 88.7228f;

    // 256-entry lookup table: Table[k] = 2^(k/256) for k=0..255
    // 1024 bytes — fits comfortably in L1 cache
    private static readonly float[] Table = GenerateTable();

    private static float[] GenerateTable()
    {
        var t = new float[L];
        for (int k = 0; k < L; k++)
            t[k] = MathF.Pow(2.0f, k / (float)L);
        return t;
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// 256-entry table exp: 8 floats at a time via vpgatherdd.
    /// Best on Intel CPUs where gather is fast (4-8 cycles).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe Vector256<float> Exp8(Vector256<float> x)
    {
        // Detect out-of-range lanes BEFORE clamping so we can fix them after
        var vMin = Vector256.Create(ClampMin);
        var vMax = Vector256.Create(ClampMax);
        var underflow = Avx.Compare(x, vMin, FloatComparisonMode.OrderedLessThanSignaling);
        var overflow = Avx.Compare(x, vMax, FloatComparisonMode.OrderedGreaterThanSignaling);
        var outOfRange = Avx.Or(underflow, overflow);

        // Clamp for the polynomial path (safe values for table index and 2^n)
        var xClamped = Avx.Max(vMin, Avx.Min(vMax, x));

        // z = x * (256 / ln2)
        var z = Avx.Multiply(xClamped, Vector256.Create(LOverLn2));

        // n = floor(z)
        var nFloat = Avx.Floor(z);
        var nInt = Avx.ConvertToVector256Int32(nFloat);

        // k = n & 0xFF — table index
        var kInt = Avx2.And(nInt, Vector256.Create(LMask));

        // int_part = n >> 8 — integer exponent
        var intPart = Avx2.ShiftRightArithmetic(nInt, LShift);

        fixed (float* tablePtr = Table)
        {
            // Table lookup via AVX2 gather
            var tableVal = Avx2.GatherVector256(tablePtr, kInt, 4);

            // r = xClamped - n * (ln2/256) — tiny remainder in [0, 0.00271)
            var r = Fma.MultiplyAddNegated(nFloat, Vector256.Create(Ln2OverL), xClamped);

            // For [0, 0.00271), exp(r) ≈ 1 + r is within 2 ULP
            // Use 2nd order for extra safety: (0.5*r + 1)*r + 1
            var poly = Fma.MultiplyAdd(Vector256.Create(0.5f), r, Vector256.Create(1.0f));
            poly = Fma.MultiplyAdd(poly, r, Vector256.Create(1.0f));

            // Combine: table[k] * poly * 2^int_part
            var combined = Avx.Multiply(tableVal, poly);
            var pow2n = Avx2.ShiftLeftLogical(
                Avx2.Add(intPart, Vector256.Create(127)), 23).AsSingle();

            var result = Avx.Multiply(combined, pow2n);

            // For any out-of-range lanes: compute exact MathF.Exp per lane
            // to preserve subnormals, exact 0, +inf, NaN — matching scalar Exp()
            if (!Avx.TestZ(outOfRange, outOfRange))
            {
                float* scratch = stackalloc float[8];
                Avx.Store(scratch, x);       // original unclamped x
                float* rBuf = stackalloc float[8];
                Avx.Store(rBuf, result);     // fast-path result
                int mask = Avx.MoveMask(outOfRange);
                for (int lane = 0; lane < 8; lane++)
                {
                    if ((mask & (1 << lane)) != 0)
                        rBuf[lane] = MathF.Exp(scratch[lane]);
                }
                result = Avx.LoadVector256(rBuf);
            }

            return result;
        }
    }

    /// <summary>
    /// Process array using 256-entry table exp. 4x unrolled.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void ExpArray(float* input, float* output, int length)
    {
        int i = 0;
        if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
        {
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
        for (; i < length; i++)
            output[i] = Exp(input[i]);
    }
#endif

    /// <summary>Scalar table-driven exp with full-precision fallback for extremes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float Exp(float x)
    {
        // Outside approximation range or non-finite: fall back to MathF.Exp which
        // correctly produces subnormals, exact zero, +Infinity, and NaN.
        // Note: !(x >= ClampMin && x <= ClampMax) catches NaN (NaN fails all comparisons).
        if (!(x >= ClampMin && x <= ClampMax)) return MathF.Exp(x);
        float z = x * LOverLn2;
        int n = (int)MathF.Floor(z);
        int k = n & LMask;
        int intPart = n >> LShift;
        float r = x - n * Ln2OverL;
        float poly = (0.5f * r + 1.0f) * r + 1.0f;
        float tableVal = Table[k];
        int bits = (intPart + 127) << 23;
        float pow2n = Unsafe.As<int, float>(ref bits);
        return tableVal * poly * pow2n;
    }
}
