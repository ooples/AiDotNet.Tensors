using AiDotNet.Tensors.NumericOperations;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Reference matmul kernels for the packed sub-byte quantization types.
/// These are the scalar / correctness path; SIMD-accelerated versions
/// (AVX-512 VNNI / AVX2 / AVX-512 VPOPCNT) plug in behind the same
/// signature.
///
/// <para><b>Convention:</b> C (M × N) += A (M × K, quantized weights)
/// × B (K × N, float activations). The weight tensor is the quantized
/// operand; activations stay in float. This matches the AWQ / GPTQ /
/// GGUF weight-only quantization pattern used by LLM inference:
/// weights are 4-bit, activations and KV cache are fp16/fp32.</para>
///
/// <para>For symmetric matmul (both sides quantized), see
/// <see cref="Int1MatMulXnor"/> which is the BitNet pattern.</para>
/// </summary>
internal static class PackedMatMul
{
    /// <summary>
    /// Weight-only int4 matmul: <c>C = dequant(A) × B</c> with A stored
    /// as packed int4 + per-group scales.
    /// </summary>
    /// <param name="a">Packed int4 weight, shape <c>[M × K]</c> with K
    /// elements per row.</param>
    /// <param name="aScale">Per-group scales from
    /// <see cref="QuantizationHelpers.QuantizeInt4"/>. The group axis
    /// runs along K within each M row; scale layout is
    /// <c>[M × (K / groupSize)]</c>.</param>
    /// <param name="b">Float activations, shape <c>[K × N]</c> row-major.</param>
    /// <param name="c">Output float, shape <c>[M × N]</c>. Cleared by this call.</param>
    /// <param name="m">Rows of A / C.</param>
    /// <param name="k">Cols of A / rows of B (must be even).</param>
    /// <param name="n">Cols of B / C.</param>
    public static void Int4WeightMatMul(
        ReadOnlySpan<PackedInt4> a, QuantizationScale aScale,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m, int k, int n)
    {
        if (aScale is null) throw new ArgumentNullException(nameof(aScale));
        if ((k & 1) != 0)
            throw new ArgumentException("K must be even for int4 matmul (2 nibbles per byte).", nameof(k));
        int packedRowLen = k / 2;
        if (a.Length < m * packedRowLen)
            throw new ArgumentException("a is too small for M × (K/2) packed bytes.", nameof(a));
        if (b.Length < k * n)
            throw new ArgumentException("b is too small for K × N floats.", nameof(b));
        if (c.Length < m * n)
            throw new ArgumentException("c is too small for M × N floats.", nameof(c));

        int groupSize = aScale.GroupSize;
        if (groupSize <= 0)
            throw new ArgumentException("Per-group scales required (aScale.GroupSize > 0).", nameof(aScale));
        int groupsPerRow = (k + groupSize - 1) / groupSize;
        int expectedScales = m * groupsPerRow;
        if (aScale.Scales.Length < expectedScales)
            throw new ArgumentException(
                $"aScale.Scales length {aScale.Scales.Length} < M × groupsPerRow = {expectedScales}.",
                nameof(aScale));

        c.Clear();

        for (int i = 0; i < m; i++)
        {
            int aRowStart = i * packedRowLen;
            int cRowStart = i * n;
            int scaleRowStart = i * groupsPerRow;

            for (int p = 0; p < k; p++)
            {
                // Unpack A[i, p] nibble.
                int packedIdx = aRowStart + (p >> 1);
                int nibble = (p & 1) == 0
                    ? a[packedIdx].LoNibble
                    : a[packedIdx].HiNibble;
                float scale = aScale.Scales[scaleRowStart + p / groupSize];
                float aVal = nibble * scale;

                // Accumulate aVal × B[p, :] into C[i, :].
                int bRowStart = p * n;
                for (int j = 0; j < n; j++)
                    c[cRowStart + j] += aVal * b[bRowStart + j];
            }
        }
    }

    /// <summary>
    /// BitNet 1-bit matmul: both operands signed 1-bit, result accumulated
    /// as int32, scaled by <paramref name="aScale"/> and
    /// <paramref name="bScale"/>. Uses popcount of <c>a XNOR b</c> to
    /// count agreements → inner-product in {-1, +1}.
    /// </summary>
    /// <param name="a">Packed 1-bit weights, shape <c>[M × K]</c>.</param>
    /// <param name="aScale">Per-row or per-tensor weight scale.</param>
    /// <param name="b">Packed 1-bit activations, shape <c>[K × N]</c>
    /// stored column-major (byte b, lane i = activation at (k-byte-base + i, col)).
    /// In practice callers preprocess via <see cref="PackBTransposed"/>.</param>
    /// <param name="bScale">Per-col or per-tensor activation scale.</param>
    /// <param name="c">Output float, shape <c>[M × N]</c>.</param>
    /// <param name="m">Rows of A / C.</param>
    /// <param name="k">Inner dim; must be multiple of 8.</param>
    /// <param name="n">Cols of B / C.</param>
    public static void Int1MatMulXnor(
        ReadOnlySpan<PackedInt1> a, QuantizationScale aScale,
        ReadOnlySpan<PackedInt1> b, QuantizationScale bScale,
        Span<float> c,
        int m, int k, int n)
    {
        if (aScale is null) throw new ArgumentNullException(nameof(aScale));
        if (bScale is null) throw new ArgumentNullException(nameof(bScale));
        if ((k & 0x7) != 0)
            throw new ArgumentException("K must be a multiple of 8 for int1 matmul.", nameof(k));
        int kBytes = k / PackedInt1.ValuesPerByte;
        if (a.Length < m * kBytes)
            throw new ArgumentException("a is too small.", nameof(a));
        if (b.Length < n * kBytes)
            throw new ArgumentException("b is too small (expects [N × K/8]).", nameof(b));
        if (c.Length < m * n)
            throw new ArgumentException("c is too small.", nameof(c));

        c.Clear();

        for (int i = 0; i < m; i++)
        {
            int aRowStart = i * kBytes;
            int cRowStart = i * n;
            float aS = aScale.Scales.Length == 1 ? aScale.Scales[0] : aScale.Scales[i];

            for (int j = 0; j < n; j++)
            {
                int bColStart = j * kBytes;
                int agree = 0;
                for (int bIdx = 0; bIdx < kBytes; bIdx++)
                {
                    // XNOR = ~(a XOR b). Popcount gives count of agreeing
                    // signs (+1 × +1 or -1 × -1). Inner product in ±1
                    // algebra = 2 × agree − k.
                    byte xn = (byte)~(a[aRowStart + bIdx].RawValue ^ b[bColStart + bIdx].RawValue);
                    agree += PopCount(xn);
                }
                int dot = 2 * agree - k;
                float bS = bScale.Scales.Length == 1 ? bScale.Scales[0] : bScale.Scales[j];
                c[cRowStart + j] = dot * aS * bS;
            }
        }
    }

    /// <summary>
    /// Pack a float matrix B of shape [K × N] into the column-major
    /// 1-bit layout <see cref="Int1MatMulXnor"/> expects: output is
    /// [N × K/8] where byte (j, b) holds K-lanes k=8b..8b+7 of column j.
    /// </summary>
    public static QuantizationScale PackBTransposed(
        ReadOnlySpan<float> b, int k, int n,
        Span<PackedInt1> packed,
        int groupSize = 0)
    {
        if ((k & 0x7) != 0)
            throw new ArgumentException("K must be a multiple of 8.", nameof(k));
        int kBytes = k / PackedInt1.ValuesPerByte;
        if (b.Length < k * n)
            throw new ArgumentException("b too small for K × N.", nameof(b));
        if (packed.Length < n * kBytes)
            throw new ArgumentException("packed too small for N × K/8.", nameof(packed));

        // Per-column scale (BitNet absmean).
        var scales = new float[n];
        for (int j = 0; j < n; j++)
        {
            float sum = 0f;
            for (int kk = 0; kk < k; kk++) sum += Math.Abs(b[kk * n + j]);
            scales[j] = sum / k;
        }

        for (int j = 0; j < n; j++)
        {
            for (int bIdx = 0; bIdx < kBytes; bIdx++)
            {
                byte raw = 0;
                int laneBase = bIdx * PackedInt1.ValuesPerByte;
                for (int i = 0; i < PackedInt1.ValuesPerByte; i++)
                {
                    int kk = laneBase + i;
                    if (b[kk * n + j] >= 0f) raw |= (byte)(1 << i);
                }
                packed[j * kBytes + bIdx] = new PackedInt1(raw);
            }
        }
        return new QuantizationScale(scales, groupSize);
    }

    private static int PopCount(byte v)
    {
        // Software popcount — x86 POPCNT intrinsic would replace this on
        // the AVX-512 VPOPCNT path. Still one cycle on modern CPUs via
        // the compiler recognizing the pattern.
        v = (byte)(v - ((v >> 1) & 0x55));
        v = (byte)((v & 0x33) + ((v >> 2) & 0x33));
        return (v + (v >> 4)) & 0x0F;
    }
}
