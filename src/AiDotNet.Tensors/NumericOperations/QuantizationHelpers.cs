namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Per-tensor or per-group quantization scale. Produced by
/// <see cref="QuantizationHelpers.Quantize"/> variants and consumed by
/// <see cref="QuantizationHelpers.Dequantize"/> and the packed
/// matmul kernels.
///
/// <para>Two modes:</para>
/// <list type="bullet">
/// <item><b>Per-tensor:</b> <see cref="GroupSize"/> is 0; a single
/// <see cref="Scales"/> value covers the whole tensor.</item>
/// <item><b>Per-group:</b> <see cref="GroupSize"/> is the count of
/// consecutive elements sharing one scale (32 / 64 / 128 is typical;
/// 32 is llama.cpp Q4_0's choice). <see cref="Scales"/> is the scale
/// array of length <c>N / GroupSize</c>.</item>
/// </list>
///
/// <para>Per-group lets int4 recover most of the accuracy it would lose
/// as per-tensor — activations spanning &gt; 2 orders of magnitude are
/// common inside a tensor but rare within a 32-element group.</para>
/// </summary>
public sealed class QuantizationScale
{
    /// <summary>Scale factor(s). Length 1 for per-tensor; length
    /// <c>N / GroupSize</c> for per-group.</summary>
    public float[] Scales { get; }

    /// <summary>Number of consecutive elements that share one scale,
    /// or 0 for per-tensor.</summary>
    public int GroupSize { get; }

    /// <summary>Optional zero-point per scale (for asymmetric quant).
    /// Zero-length means symmetric quantization (zero-point = 0).</summary>
    public int[] ZeroPoints { get; }

    public QuantizationScale(float[] scales, int groupSize, int[]? zeroPoints = null)
    {
        Scales = scales ?? throw new ArgumentNullException(nameof(scales));
        // Negative GroupSize is meaningless — dequantizers either divide by it
        // or use it as an index base, both of which produce garbage. Fail at
        // construction rather than deep inside a dequant loop where the
        // symptom (OOB / DivideByZero) hides the cause (bad metadata). Zero
        // is still valid — it signals "per-tensor / single scale".
        if (groupSize < 0)
            throw new ArgumentOutOfRangeException(
                nameof(groupSize),
                $"groupSize must be non-negative (got {groupSize}). " +
                "Use 0 for per-tensor scale, or a positive value for per-group scale.");
        GroupSize = groupSize;
        ZeroPoints = zeroPoints ?? Array.Empty<int>();
        if (ZeroPoints.Length != 0 && ZeroPoints.Length != scales.Length)
            throw new ArgumentException(
                $"ZeroPoints length {ZeroPoints.Length} must match Scales length {scales.Length}.",
                nameof(zeroPoints));
    }
}

/// <summary>
/// Quantize / dequantize helpers for the <see cref="PackedInt1"/> and
/// <see cref="PackedInt4"/> types. Shipped as static methods so they
/// compose into any pipeline (CPU, GPU upload, disk serialization)
/// without forcing a framework choice.
/// </summary>
public static class QuantizationHelpers
{
    // ──────────── int4 symmetric per-group ────────────

    /// <summary>
    /// Quantize <paramref name="src"/> to int4 with per-group symmetric
    /// scales. Group size defaults to 32 (llama.cpp Q4_0) — match that
    /// for GGUF interop. Output buffer must hold
    /// <c>(src.Length + 1) / 2</c> packed bytes.
    /// </summary>
    /// <returns>The scale metadata needed by <see cref="DequantizeInt4"/>
    /// / matmul kernels.</returns>
    public static QuantizationScale QuantizeInt4(
        ReadOnlySpan<float> src,
        Span<PackedInt4> dst,
        int groupSize = 32)
    {
        if (groupSize <= 0 || (groupSize & 1) != 0)
            throw new ArgumentException(
                "groupSize must be positive and even.", nameof(groupSize));
        int expectedDst = (src.Length + 1) / 2;
        if (dst.Length < expectedDst)
            throw new ArgumentException(
                $"dst must hold at least {expectedDst} packed bytes.", nameof(dst));

        int groups = (src.Length + groupSize - 1) / groupSize;
        var scales = new float[groups];

        for (int g = 0; g < groups; g++)
        {
            int start = g * groupSize;
            int end = Math.Min(start + groupSize, src.Length);

            // Find absmax in this group → symmetric scale.
            float absMax = 0f;
            for (int i = start; i < end; i++)
            {
                float a = Math.Abs(src[i]);
                if (a > absMax) absMax = a;
            }
            // Max int4 magnitude is 7 (use 7 for symmetric, leaves -8
            // unused but avoids tipping the dequant into 8/-8 asymmetry).
            float scale = absMax == 0f ? 1f : absMax / 7f;
            scales[g] = scale;
            float invScale = 1f / scale;

            // Quantize pairs: each byte of dst holds two consecutive ints.
            for (int i = start; i < end; i++)
            {
                int q = (int)Math.Round(src[i] * invScale);
                if (q < PackedInt4.MinValue) q = PackedInt4.MinValue;
                if (q > PackedInt4.MaxValue) q = PackedInt4.MaxValue;

                int dstByte = i >> 1;
                bool hi = (i & 1) == 1;
                int existing = dst[dstByte].RawValue;
                int mask = hi ? 0x0F : 0xF0;
                int shift = hi ? 4 : 0;
                int merged = (existing & mask) | ((q & 0x0F) << shift);
                dst[dstByte] = new PackedInt4((byte)merged);
            }
        }

        return new QuantizationScale(scales, groupSize);
    }

    /// <summary>
    /// Inverse of <see cref="QuantizeInt4"/>. Writes reconstructed floats
    /// into <paramref name="dst"/>, one per packed nibble, using the
    /// supplied scale metadata.
    /// </summary>
    public static void DequantizeInt4(
        ReadOnlySpan<PackedInt4> src,
        QuantizationScale scale,
        Span<float> dst)
    {
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        int groupSize = scale.GroupSize;
        // Int4 dequant is strictly per-group — the `g = i / groupSize` index
        // below divides by groupSize, so 0 throws DivideByZeroException
        // deep in the loop. Reject at the boundary with a clear message
        // instead. Callers that really mean per-tensor should build an
        // appropriately sized Scales array; the shared QuantizationScale
        // zero-signifies-per-tensor convention doesn't apply to the int4 path.
        if (groupSize <= 0)
            throw new ArgumentException(
                $"scale.GroupSize must be positive for int4 dequantization (got {groupSize}).",
                nameof(scale));
        int n = dst.Length;
        int expectedSrc = (n + 1) / 2;
        if (src.Length < expectedSrc)
            throw new ArgumentException(
                $"src must hold at least {expectedSrc} packed bytes.", nameof(src));
        int groups = (n + groupSize - 1) / groupSize;
        if (scale.Scales.Length < groups)
            throw new ArgumentException(
                $"scale.Scales length {scale.Scales.Length} insufficient for {groups} groups.",
                nameof(scale));

        for (int i = 0; i < n; i++)
        {
            int g = i / groupSize;
            int nibble = (i & 1) == 0 ? src[i >> 1].LoNibble : src[i >> 1].HiNibble;
            dst[i] = nibble * scale.Scales[g];
        }
    }

    // ──────────── int1 (BitNet sign) ────────────

    /// <summary>
    /// Quantize <paramref name="src"/> to 1-bit sign encoding. Output
    /// buffer must hold <c>(src.Length + 7) / 8</c> packed bytes.
    /// </summary>
    /// <returns>A per-tensor scale (absmean, as in BitNet 1.58b) or
    /// per-group absmean if <paramref name="groupSize"/> &gt; 0.</returns>
    public static QuantizationScale QuantizeInt1(
        ReadOnlySpan<float> src,
        Span<PackedInt1> dst,
        int groupSize = 0)
    {
        int expectedDst = (src.Length + PackedInt1.ValuesPerByte - 1) / PackedInt1.ValuesPerByte;
        if (dst.Length < expectedDst)
            throw new ArgumentException(
                $"dst must hold at least {expectedDst} packed bytes.", nameof(dst));
        if (groupSize < 0 || (groupSize > 0 && groupSize % PackedInt1.ValuesPerByte != 0))
            throw new ArgumentException(
                "groupSize must be 0 (per-tensor) or a multiple of 8.", nameof(groupSize));

        int groups = groupSize == 0 ? 1 : (src.Length + groupSize - 1) / groupSize;
        int effectiveGroup = groupSize == 0 ? src.Length : groupSize;
        var scales = new float[groups];

        for (int g = 0; g < groups; g++)
        {
            int start = g * effectiveGroup;
            int end = Math.Min(start + effectiveGroup, src.Length);
            // BitNet scale: mean absolute value. Captures the average
            // magnitude of weights in the group — dequantizing sign(w)
            // by this scale recovers ~E[|w|] × sign(w).
            float absSum = 0f;
            int count = end - start;
            for (int i = start; i < end; i++) absSum += Math.Abs(src[i]);
            scales[g] = count == 0 ? 0f : absSum / count;
        }

        // Pack signs — lane i of byte b holds sign(src[b * 8 + i]).
        for (int b = 0; b < expectedDst; b++)
        {
            byte raw = 0;
            for (int i = 0; i < PackedInt1.ValuesPerByte; i++)
            {
                int idx = b * PackedInt1.ValuesPerByte + i;
                if (idx >= src.Length) break;
                if (src[idx] >= 0f) raw |= (byte)(1 << i);
            }
            dst[b] = new PackedInt1(raw);
        }

        return new QuantizationScale(scales, groupSize);
    }

    /// <summary>
    /// Inverse of <see cref="QuantizeInt1"/>. Dequantized value is
    /// <c>scale × sign(packed_bit)</c>.
    /// </summary>
    public static void DequantizeInt1(
        ReadOnlySpan<PackedInt1> src,
        QuantizationScale scale,
        Span<float> dst)
    {
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        int n = dst.Length;
        // scale.GroupSize == 0 means "per-tensor" (single scale applied to
        // every element); treat that as "one group of length n". Positive
        // values are real groupings.
        int groupSize = scale.GroupSize == 0 ? n : scale.GroupSize;

        // Validate src + scale.Scales lengths up front so a malformed caller
        // gets a deterministic ArgumentException instead of an
        // IndexOutOfRangeException from deep in the loop — matches the
        // fail-fast contract of DequantizeInt4 and the quantize entry points.
        int expectedSrc = (n + PackedInt1.ValuesPerByte - 1) / PackedInt1.ValuesPerByte;
        if (src.Length < expectedSrc)
            throw new ArgumentException(
                $"src must hold at least {expectedSrc} packed bytes (got {src.Length}).",
                nameof(src));
        int groups = scale.GroupSize == 0 ? 1 : (n + groupSize - 1) / groupSize;
        if (scale.Scales.Length < groups)
            throw new ArgumentException(
                $"scale.Scales length {scale.Scales.Length} insufficient for {groups} groups.",
                nameof(scale));

        for (int i = 0; i < n; i++)
        {
            int byteIdx = i / PackedInt1.ValuesPerByte;
            int laneIdx = i % PackedInt1.ValuesPerByte;
            int g = i / groupSize;
            dst[i] = src[byteIdx].GetLane(laneIdx) * scale.Scales[g];
        }
    }
}
