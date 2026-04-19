namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// 4-bit non-uniform quantization with a fixed 16-entry lookup table
/// calibrated to the quantiles of a standard normal distribution —
/// QLoRA's "NF4" format. Higher fidelity than uniform Int4 on Gaussian
/// weights because the representation density matches the weight
/// density near zero where most activation-propagated weights live.
///
/// <para>Shares the <see cref="PackedInt4"/> two-nibbles-per-byte
/// storage; the interpretation is different (table lookup vs two's-
/// complement). We reuse the struct and expose the table + quant /
/// dequant helpers here.</para>
/// </summary>
public static class NormalFloat4
{
    /// <summary>
    /// The 16 quantile anchor values, sorted ascending. From Dettmers
    /// et al. (2023), "QLoRA: Efficient Finetuning of Quantized LLMs".
    /// These are the values a weight in [-1, +1] (after absmax scaling)
    /// is snapped to on quantize and recovered from on dequantize.
    /// </summary>
    public static readonly float[] Table =
    {
        -1.0f,
        -0.6961928f,
        -0.5250730f,
        -0.3949892f,
        -0.2844607f,
        -0.1848364f,
        -0.0911699f,
         0.0f,
         0.0795803f,
         0.1609302f,
         0.2461123f,
         0.3379029f,
         0.4407679f,
         0.5626925f,
         0.7229568f,
         1.0f,
    };

    /// <summary>
    /// Snap <paramref name="value"/> in [-1, 1] to its nearest table
    /// index [0..15]. Nearest-neighbor with tie-breaking by rounding
    /// toward zero in the table (arbitrary but deterministic).
    /// </summary>
    public static int ToIndex(float value)
    {
        // Clamp — values beyond ±1 saturate to the endpoints.
        if (value <= Table[0]) return 0;
        if (value >= Table[15]) return 15;
        // Linear scan; binary search is a micro-optimization that saves
        // a few cycles per call. The dominant cost at quantize time is
        // the absmax scan, not the lookup.
        int best = 0;
        float bestErr = Math.Abs(value - Table[0]);
        for (int i = 1; i < Table.Length; i++)
        {
            float e = Math.Abs(value - Table[i]);
            if (e < bestErr) { bestErr = e; best = i; }
        }
        return best;
    }

    /// <summary>
    /// Dequantize an index in [0, 15] back to its table value.
    /// </summary>
    public static float FromIndex(int index)
    {
        if ((uint)index >= (uint)Table.Length)
            throw new ArgumentOutOfRangeException(nameof(index));
        return Table[index];
    }
}
