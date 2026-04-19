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

    // ──────────── Int2 symmetric per-group ────────────

    /// <summary>
    /// Quantize to 2-bit with per-group symmetric scaling. Step magnitude
    /// cap is 1 (range [-2, 1] on the dequant side); groupSize default 16
    /// matches GGUF Q2_K sub-block size.
    /// </summary>
    public static QuantizationScale QuantizeInt2(
        ReadOnlySpan<float> src,
        Span<PackedInt2> dst,
        int groupSize = 16)
    {
        if (groupSize <= 0 || groupSize % PackedInt2.ValuesPerByte != 0)
            throw new ArgumentException(
                $"groupSize must be positive and a multiple of {PackedInt2.ValuesPerByte}.",
                nameof(groupSize));
        int expectedDst = (src.Length + PackedInt2.ValuesPerByte - 1) / PackedInt2.ValuesPerByte;
        if (dst.Length < expectedDst)
            throw new ArgumentException(
                $"dst must hold at least {expectedDst} packed bytes.", nameof(dst));

        int groups = (src.Length + groupSize - 1) / groupSize;
        var scales = new float[groups];
        // Map absmax to 1 (Int2.MaxValue) so abs values saturate to ±1 × scale;
        // int2 -2 gives slight asymmetric headroom for negatives, tolerated.
        for (int g = 0; g < groups; g++)
        {
            int start = g * groupSize;
            int end = Math.Min(start + groupSize, src.Length);
            float absMax = 0f;
            for (int i = start; i < end; i++)
                absMax = Math.Max(absMax, Math.Abs(src[i]));
            scales[g] = absMax == 0f ? 1f : absMax;
        }

        // Pack — 4 int2 values per byte.
        Span<int> quads = stackalloc int[PackedInt2.ValuesPerByte];
        for (int b = 0; b < expectedDst; b++)
        {
            quads.Clear();
            for (int lane = 0; lane < PackedInt2.ValuesPerByte; lane++)
            {
                int idx = b * PackedInt2.ValuesPerByte + lane;
                if (idx >= src.Length) break;
                int g = idx / groupSize;
                float invScale = scales[g] == 0f ? 0f : 1f / scales[g];
                int q = (int)Math.Round(src[idx] * invScale);
                if (q < PackedInt2.MinValue) q = PackedInt2.MinValue;
                if (q > PackedInt2.MaxValue) q = PackedInt2.MaxValue;
                quads[lane] = q;
            }
            dst[b] = PackedInt2.FromInts(quads[0], quads[1], quads[2], quads[3]);
        }

        return new QuantizationScale(scales, groupSize);
    }

    /// <summary>Inverse of <see cref="QuantizeInt2"/>.</summary>
    public static void DequantizeInt2(
        ReadOnlySpan<PackedInt2> src, QuantizationScale scale, Span<float> dst)
    {
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        int gs = scale.GroupSize;
        if (gs <= 0)
            throw new ArgumentException(
                $"scale.GroupSize must be positive for int2 dequantization (got {gs}).",
                nameof(scale));
        int expectedSrc = (dst.Length + PackedInt2.ValuesPerByte - 1) / PackedInt2.ValuesPerByte;
        if (src.Length < expectedSrc)
            throw new ArgumentException(
                $"src must hold at least {expectedSrc} packed bytes (got {src.Length}).",
                nameof(src));
        int groups = (dst.Length + gs - 1) / gs;
        if (scale.Scales.Length < groups)
            throw new ArgumentException(
                $"scale.Scales length {scale.Scales.Length} insufficient for {groups} groups.",
                nameof(scale));
        for (int i = 0; i < dst.Length; i++)
        {
            int byteIdx = i / PackedInt2.ValuesPerByte;
            int laneIdx = i % PackedInt2.ValuesPerByte;
            int g = i / gs;
            dst[i] = src[byteIdx].GetLane(laneIdx) * scale.Scales[g];
        }
    }

    // ──────────── Int3 symmetric per-group ────────────

    /// <summary>
    /// Quantize to 3-bit per-group. Output is a 3-byte block for every
    /// 8 consecutive values (<see cref="PackedInt3Block"/>). Group size
    /// must be a multiple of 8.
    /// </summary>
    public static QuantizationScale QuantizeInt3(
        ReadOnlySpan<float> src,
        Span<PackedInt3Block> dst,
        int groupSize = 32)
    {
        if (groupSize <= 0 || groupSize % PackedInt3Block.ValuesPerBlock != 0)
            throw new ArgumentException(
                $"groupSize must be a positive multiple of {PackedInt3Block.ValuesPerBlock}.",
                nameof(groupSize));
        int expectedBlocks = (src.Length + PackedInt3Block.ValuesPerBlock - 1) / PackedInt3Block.ValuesPerBlock;
        if (dst.Length < expectedBlocks)
            throw new ArgumentException(
                $"dst must hold at least {expectedBlocks} 3-byte blocks.", nameof(dst));

        int groups = (src.Length + groupSize - 1) / groupSize;
        var scales = new float[groups];
        for (int g = 0; g < groups; g++)
        {
            int start = g * groupSize;
            int end = Math.Min(start + groupSize, src.Length);
            float absMax = 0f;
            for (int i = start; i < end; i++)
                absMax = Math.Max(absMax, Math.Abs(src[i]));
            // Max positive is 3; use it to map the range.
            scales[g] = absMax == 0f ? 1f : absMax / 3f;
        }

        Span<int> block = stackalloc int[PackedInt3Block.ValuesPerBlock];
        for (int bl = 0; bl < expectedBlocks; bl++)
        {
            block.Clear();
            for (int lane = 0; lane < PackedInt3Block.ValuesPerBlock; lane++)
            {
                int idx = bl * PackedInt3Block.ValuesPerBlock + lane;
                if (idx >= src.Length) break;
                int g = idx / groupSize;
                float invScale = scales[g] == 0f ? 0f : 1f / scales[g];
                int q = (int)Math.Round(src[idx] * invScale);
                if (q < PackedInt3Block.MinValue) q = PackedInt3Block.MinValue;
                if (q > PackedInt3Block.MaxValue) q = PackedInt3Block.MaxValue;
                block[lane] = q;
            }
            dst[bl] = PackedInt3Block.FromInts(block);
        }
        return new QuantizationScale(scales, groupSize);
    }

    /// <summary>Inverse of <see cref="QuantizeInt3"/>.</summary>
    public static void DequantizeInt3(
        ReadOnlySpan<PackedInt3Block> src, QuantizationScale scale, Span<float> dst)
    {
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        int gs = scale.GroupSize;
        if (gs <= 0)
            throw new ArgumentException(
                $"scale.GroupSize must be positive for int3 dequantization (got {gs}).",
                nameof(scale));
        int expectedSrc = (dst.Length + PackedInt3Block.ValuesPerBlock - 1) / PackedInt3Block.ValuesPerBlock;
        if (src.Length < expectedSrc)
            throw new ArgumentException(
                $"src must hold at least {expectedSrc} packed blocks (got {src.Length}).",
                nameof(src));
        int groups = (dst.Length + gs - 1) / gs;
        if (scale.Scales.Length < groups)
            throw new ArgumentException(
                $"scale.Scales length {scale.Scales.Length} insufficient for {groups} groups.",
                nameof(scale));
        for (int i = 0; i < dst.Length; i++)
        {
            int blockIdx = i / PackedInt3Block.ValuesPerBlock;
            int laneIdx = i % PackedInt3Block.ValuesPerBlock;
            int g = i / gs;
            dst[i] = src[blockIdx].GetLane(laneIdx) * scale.Scales[g];
        }
    }

    // ──────────── NF4 (QLoRA NormalFloat-4) ────────────

    /// <summary>
    /// Quantize to NF4 (non-uniform 4-bit) with per-group absmax
    /// scaling. Reuses <see cref="PackedInt4"/> storage layout
    /// (two nibbles per byte); the nibble value is a
    /// <see cref="NormalFloat4.Table"/> index in [0, 15].
    /// </summary>
    public static QuantizationScale QuantizeNF4(
        ReadOnlySpan<float> src,
        Span<PackedInt4> dst,
        int groupSize = 64)
    {
        if (groupSize <= 0 || (groupSize & 1) != 0)
            throw new ArgumentException("groupSize must be a positive even integer.", nameof(groupSize));
        int expectedDst = (src.Length + 1) / 2;
        if (dst.Length < expectedDst)
            throw new ArgumentException($"dst must hold at least {expectedDst} packed bytes.", nameof(dst));

        int groups = (src.Length + groupSize - 1) / groupSize;
        var scales = new float[groups];
        for (int g = 0; g < groups; g++)
        {
            int start = g * groupSize;
            int end = Math.Min(start + groupSize, src.Length);
            float absMax = 0f;
            for (int i = start; i < end; i++)
                absMax = Math.Max(absMax, Math.Abs(src[i]));
            scales[g] = absMax == 0f ? 1f : absMax;
        }

        for (int i = 0; i < src.Length; i++)
        {
            int g = i / groupSize;
            float normalized = scales[g] == 0f ? 0f : src[i] / scales[g];
            int index = NormalFloat4.ToIndex(normalized);

            int dstByte = i >> 1;
            bool hi = (i & 1) == 1;
            int existing = dst[dstByte].RawValue;
            int mask = hi ? 0x0F : 0xF0;
            int shift = hi ? 4 : 0;
            int merged = (existing & mask) | ((index & 0x0F) << shift);
            dst[dstByte] = new PackedInt4((byte)merged);
        }
        return new QuantizationScale(scales, groupSize);
    }

    /// <summary>Inverse of <see cref="QuantizeNF4"/>.</summary>
    public static void DequantizeNF4(
        ReadOnlySpan<PackedInt4> src, QuantizationScale scale, Span<float> dst)
    {
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        int gs = scale.GroupSize;
        if (gs <= 0)
            throw new ArgumentException(
                $"scale.GroupSize must be positive for NF4 dequantization (got {gs}).",
                nameof(scale));
        int expectedSrc = (dst.Length + 1) / 2;
        if (src.Length < expectedSrc)
            throw new ArgumentException(
                $"src must hold at least {expectedSrc} packed bytes (got {src.Length}).",
                nameof(src));
        int groups = (dst.Length + gs - 1) / gs;
        if (scale.Scales.Length < groups)
            throw new ArgumentException(
                $"scale.Scales length {scale.Scales.Length} insufficient for {groups} groups.",
                nameof(scale));
        for (int i = 0; i < dst.Length; i++)
        {
            int byteIdx = i >> 1;
            int nibble = (i & 1) == 0
                ? (src[byteIdx].RawValue & 0x0F)
                : ((src[byteIdx].RawValue >> 4) & 0x0F);
            int g = i / gs;
            dst[i] = NormalFloat4.FromIndex(nibble) * scale.Scales[g];
        }
    }

    // ──────────── FP4 (MXFP4 E2M1) ────────────

    /// <summary>
    /// Quantize to FP4 (1s/2e/1m, MXFP4). Like NF4 but uses a float
    /// value table instead of a normal-quantile table.
    /// </summary>
    public static QuantizationScale QuantizeFp4(
        ReadOnlySpan<float> src,
        Span<PackedInt4> dst,
        int groupSize = 32)
    {
        if (groupSize <= 0 || (groupSize & 1) != 0)
            throw new ArgumentException("groupSize must be a positive even integer.", nameof(groupSize));
        int expectedDst = (src.Length + 1) / 2;
        if (dst.Length < expectedDst)
            throw new ArgumentException($"dst must hold at least {expectedDst} packed bytes.", nameof(dst));

        int groups = (src.Length + groupSize - 1) / groupSize;
        var scales = new float[groups];
        for (int g = 0; g < groups; g++)
        {
            int start = g * groupSize;
            int end = Math.Min(start + groupSize, src.Length);
            float absMax = 0f;
            for (int i = start; i < end; i++)
                absMax = Math.Max(absMax, Math.Abs(src[i]));
            // FP4 max representable is 6.
            scales[g] = absMax == 0f ? 1f : absMax / 6f;
        }

        for (int i = 0; i < src.Length; i++)
        {
            int g = i / groupSize;
            float normalized = scales[g] == 0f ? 0f : src[i] / scales[g];
            int index = Fp4E2M1.ToIndex(normalized);

            int dstByte = i >> 1;
            bool hi = (i & 1) == 1;
            int existing = dst[dstByte].RawValue;
            int mask = hi ? 0x0F : 0xF0;
            int shift = hi ? 4 : 0;
            int merged = (existing & mask) | ((index & 0x0F) << shift);
            dst[dstByte] = new PackedInt4((byte)merged);
        }
        return new QuantizationScale(scales, groupSize);
    }

    /// <summary>Inverse of <see cref="QuantizeFp4"/>.</summary>
    public static void DequantizeFp4(
        ReadOnlySpan<PackedInt4> src, QuantizationScale scale, Span<float> dst)
    {
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        int gs = scale.GroupSize;
        if (gs <= 0)
            throw new ArgumentException(
                $"scale.GroupSize must be positive for FP4 dequantization (got {gs}).",
                nameof(scale));
        int expectedSrc = (dst.Length + 1) / 2;
        if (src.Length < expectedSrc)
            throw new ArgumentException(
                $"src must hold at least {expectedSrc} packed bytes (got {src.Length}).",
                nameof(src));
        int groups = (dst.Length + gs - 1) / gs;
        if (scale.Scales.Length < groups)
            throw new ArgumentException(
                $"scale.Scales length {scale.Scales.Length} insufficient for {groups} groups.",
                nameof(scale));
        for (int i = 0; i < dst.Length; i++)
        {
            int byteIdx = i >> 1;
            int nibble = (i & 1) == 0
                ? (src[byteIdx].RawValue & 0x0F)
                : ((src[byteIdx].RawValue >> 4) & 0x0F);
            int g = i / gs;
            dst[i] = Fp4E2M1.FromIndex(nibble) * scale.Scales[g];
        }
    }

    // ──────────── QAT: fake-quantize with straight-through estimator ────────────

    /// <summary>
    /// Fake-quantize forward: quantize + dequantize + write back to
    /// <paramref name="dst"/>. Used by quantization-aware training
    /// (QAT) where the forward pass sees a quantized approximation but
    /// the straight-through estimator (gradient identity through the
    /// fake-quant) lets backward flow the full-precision gradient.
    /// Caller is expected to wire the gradient itself (trivial — it's
    /// an identity copy).
    /// </summary>
    public static void FakeQuantizeInt4(
        ReadOnlySpan<float> src, Span<float> dst, int groupSize = 32)
    {
        // The inner Dequantize only fills dst.Length elements, so a
        // shorter dst silently truncates the QAT forward pass and hides
        // caller bugs. Fail fast so mismatched buffers surface immediately.
        if (dst.Length < src.Length)
            throw new ArgumentException(
                $"dst length {dst.Length} must be at least src length {src.Length} for fake-quant.",
                nameof(dst));
        int packedLen = (src.Length + 1) / 2;
        Span<PackedInt4> tmp = packedLen <= 256
            ? stackalloc PackedInt4[packedLen]
            : new PackedInt4[packedLen];
        var scale = QuantizeInt4(src, tmp, groupSize);
        DequantizeInt4(tmp, scale, dst.Slice(0, src.Length));
    }

    /// <summary>FP8-style fake-quantize for NF4.</summary>
    public static void FakeQuantizeNF4(
        ReadOnlySpan<float> src, Span<float> dst, int groupSize = 64)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException(
                $"dst length {dst.Length} must be at least src length {src.Length} for fake-quant.",
                nameof(dst));
        int packedLen = (src.Length + 1) / 2;
        Span<PackedInt4> tmp = packedLen <= 256
            ? stackalloc PackedInt4[packedLen]
            : new PackedInt4[packedLen];
        var scale = QuantizeNF4(src, tmp, groupSize);
        DequantizeNF4(tmp, scale, dst.Slice(0, src.Length));
    }

    /// <summary>Fake-quantize for Int1 (BitNet's sign-STE).</summary>
    public static void FakeQuantizeInt1(
        ReadOnlySpan<float> src, Span<float> dst, int groupSize = 0)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException(
                $"dst length {dst.Length} must be at least src length {src.Length} for fake-quant.",
                nameof(dst));
        int packedLen = (src.Length + 7) / 8;
        Span<PackedInt1> tmp = packedLen <= 256
            ? stackalloc PackedInt1[packedLen]
            : new PackedInt1[packedLen];
        var scale = QuantizeInt1(src, tmp, groupSize);
        DequantizeInt1(tmp, scale, dst.Slice(0, src.Length));
    }
}
