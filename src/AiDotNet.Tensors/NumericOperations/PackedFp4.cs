namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// 4-bit float with 1 sign / 2 exponent / 1 mantissa bit (E2M1). 16
/// possible values, shipped by NVIDIA Hopper+ via the <c>MXFP4</c>
/// microscaling spec. Compared to uniform int4, FP4 trades a denser
/// representation near zero for a sparser one near ±max — a good
/// match for activation distributions with heavy tails.
///
/// <para><b>Encoding:</b> bit 3 = sign, bits 2:1 = biased exponent
/// (bias 1), bit 0 = mantissa. Value magnitude =
/// <c>1.mantissa × 2^(exp - 1)</c> for normal exps, zero when exp=0.
/// Representable positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
/// Mirrored negatives for a total of 16 distinct codes.</para>
///
/// <para>Shares <see cref="PackedInt4"/> storage — two 4-bit codes per
/// byte — but the low nibble is interpreted via this table.</para>
/// </summary>
public static class Fp4E2M1
{
    // Private backing array so callers can't mutate the FP4 code table
    // at runtime and corrupt quantize/dequant process-wide.
    private static readonly float[] _table =
    {
        0.0f,   // 0000: +0
        0.5f,   // 0001: +0.5
        1.0f,   // 0010: +1.0
        1.5f,   // 0011: +1.5
        2.0f,   // 0100: +2.0
        3.0f,   // 0101: +3.0
        4.0f,   // 0110: +4.0
        6.0f,   // 0111: +6.0
       -0.0f,   // 1000: -0
       -0.5f,   // 1001
       -1.0f,
       -1.5f,
       -2.0f,
       -3.0f,
       -4.0f,
       -6.0f,
    };

    /// <summary>
    /// Read-only view of the precomputed FP4 E2M1 code values, indexed
    /// by nibble. Code 0 = +0 (positive zero), 8 = −0 (negative zero,
    /// collapsed).
    /// </summary>
    public static ReadOnlySpan<float> Table => _table;

    /// <summary>
    /// Snap <paramref name="value"/> to its nearest FP4 code [0, 15]
    /// with saturation — values beyond ±6 clamp to ±6 (no Inf
    /// encoding in MXFP4 E2M1). Ties break toward the entry of smaller
    /// magnitude.
    /// </summary>
    public static int ToIndex(float value)
    {
        if (float.IsNaN(value)) return 0; // no NaN encoding; collapse to +0
        if (value >= _table[7])  return 7;  // +6 cap
        if (value <= _table[15]) return 15; // -6 cap
        int best = 0;
        float bestErr = Math.Abs(value - _table[0]);
        for (int i = 1; i < _table.Length; i++)
        {
            float e = Math.Abs(value - _table[i]);
            // Tie-break toward the lower-magnitude entry so ±0 collisions
            // and midpoint values pick the "smaller" code deterministically.
            if (e < bestErr ||
                (e == bestErr && Math.Abs(_table[i]) < Math.Abs(_table[best])))
            {
                bestErr = e;
                best = i;
            }
        }
        return best;
    }

    /// <summary>Dequantize an FP4 code to float.</summary>
    public static float FromIndex(int index)
    {
        if ((uint)index >= (uint)_table.Length)
            throw new ArgumentOutOfRangeException(nameof(index));
        return _table[index];
    }
}
