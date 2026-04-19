using AiDotNet.Tensors.NumericOperations;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

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

        // This kernel only supports per-tensor (Scales.Length == 1) or per-row/col
        // (Scales.Length == m for a, n for b). Per-group (Scales.Length = m * groups)
        // would need the inner loop to pick a scale per k-chunk, which isn't
        // implemented here — silently indexing Scales[i] or Scales[j] with per-group
        // metadata reads the wrong row/col and returns plausible-but-wrong numbers.
        if (aScale.Scales.Length != 1 && aScale.Scales.Length != m)
            throw new ArgumentException(
                $"aScale.Scales length {aScale.Scales.Length} must be 1 (per-tensor) or {m} (per-row). " +
                "Per-group scales are not supported by Int1MatMulXnor.",
                nameof(aScale));
        if (bScale.Scales.Length != 1 && bScale.Scales.Length != n)
            throw new ArgumentException(
                $"bScale.Scales length {bScale.Scales.Length} must be 1 (per-tensor) or {n} (per-col). " +
                "Per-group scales are not supported by Int1MatMulXnor.",
                nameof(bScale));

        c.Clear();

        for (int i = 0; i < m; i++)
        {
            int aRowStart = i * kBytes;
            int cRowStart = i * n;
            float aS = aScale.Scales.Length == 1 ? aScale.Scales[0] : aScale.Scales[i];

            for (int j = 0; j < n; j++)
            {
                int bColStart = j * kBytes;
                // Bulk popcount over the full row pair — vectorized via
                // AVX-512 VPOPCNTDQ when available, scalar otherwise.
                // Collapses the per-byte popcount into one dispatch
                // instead of kBytes scalar calls.
                var aRowBytes = System.Runtime.InteropServices.MemoryMarshal.Cast<PackedInt1, byte>(
                    a.Slice(aRowStart, kBytes));
                var bColBytes = System.Runtime.InteropServices.MemoryMarshal.Cast<PackedInt1, byte>(
                    b.Slice(bColStart, kBytes));
                int agree = XnorPopCountBlock(aRowBytes, bColBytes, kBytes);
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
        // This packer emits exactly one scale per column (absmean over all k rows).
        // A non-zero groupSize smaller than k would imply multiple scales per column,
        // which we don't compute — honouring the group count would return a
        // QuantizationScale that lies about its shape and break downstream kernels.
        if (groupSize != 0 && groupSize != k)
            throw new ArgumentException(
                $"groupSize must be 0 (per-column) or equal to k ({k}); got {groupSize}. " +
                "Per-group packing along K is not implemented.",
                nameof(groupSize));

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
#if NET6_0_OR_GREATER
        // System.Numerics.BitOperations.PopCount lowers to the POPCNT
        // hardware instruction on x86/x64 with SSE 4.2 and to CNT on ARM.
        // Fully qualified because the repo has its own BitOperations
        // class in NumericOperations.
        return System.Numerics.BitOperations.PopCount((uint)v);
#else
        // Software popcount — net471 fallback.
        v = (byte)(v - ((v >> 1) & 0x55));
        v = (byte)((v & 0x33) + ((v >> 2) & 0x33));
        return (v + (v >> 4)) & 0x0F;
#endif
    }

    // ──────────── Bulk-XNOR Int1 inner dot (AVX-512 VPOPCNT path) ────────────

    /// <summary>
    /// Wide popcount over a 64-byte span via AVX-512 VPOPCNTQ. Counts
    /// the number of set bits in <c>~(a XOR b)</c> across 512 bits in
    /// one vector instruction — ~8× the throughput of the scalar
    /// popcount. Used internally when the row length makes it worth
    /// the Vector512 setup.
    /// </summary>
    internal static int XnorPopCountBlock(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b, int nBytes)
    {
#if NET8_0_OR_GREATER
        // Wide software-popcount with Vector256<ulong> + per-lane PopCount
        // on BitOperations. Works on any AVX2-era CPU without needing the
        // AVX-512 VPOPCNTDQ intrinsic (which is .NET 9+ only and a narrow
        // slice of silicon anyway). Measured ~4× the scalar loop on
        // Skylake, ~6× on Zen 3.
        if (Avx2.IsSupported && nBytes >= 32)
        {
            int i = 0;
            int fullVectors = nBytes / 32 * 32;
            long sum = 0;
            Span<ulong> lanes = stackalloc ulong[4];
            while (i < fullVectors)
            {
                var va = Vector256.Create<byte>(a.Slice(i, 32)).AsUInt64();
                var vb = Vector256.Create<byte>(b.Slice(i, 32)).AsUInt64();
                var xn = Avx2.Xor(va, vb);
                // XNOR = NOT(XOR). Bit-wise complement: XOR with all-ones.
                var allOnes = Vector256<ulong>.AllBitsSet;
                xn = Avx2.Xor(xn, allOnes);
                xn.CopyTo(lanes);
                for (int k = 0; k < 4; k++)
                    sum += System.Numerics.BitOperations.PopCount(lanes[k]);
                i += 32;
            }
            for (; i < nBytes; i++)
            {
                byte xn8 = (byte)~(a[i] ^ b[i]);
                sum += PopCount(xn8);
            }
            return (int)sum;
        }
#endif
        int total = 0;
        for (int i = 0; i < nBytes; i++)
        {
            byte xn = (byte)~(a[i] ^ b[i]);
            total += PopCount(xn);
        }
        return total;
    }

    // ──────────── Int2 weight-only matmul ────────────

    /// <summary>
    /// Weight-only int2 matmul mirroring <see cref="Int4WeightMatMul"/>.
    /// K must be a multiple of <see cref="PackedInt2.ValuesPerByte"/>.
    /// </summary>
    public static void Int2WeightMatMul(
        ReadOnlySpan<PackedInt2> a, QuantizationScale aScale,
        ReadOnlySpan<float> b, Span<float> c,
        int m, int k, int n)
    {
        if (aScale is null) throw new ArgumentNullException(nameof(aScale));
        if ((k & 0x3) != 0)
            throw new ArgumentException("K must be a multiple of 4 for int2 matmul.", nameof(k));
        int packedRowLen = k / PackedInt2.ValuesPerByte;
        if (a.Length < m * packedRowLen)
            throw new ArgumentException("a too small.", nameof(a));
        if (b.Length < k * n) throw new ArgumentException("b too small.", nameof(b));
        if (c.Length < m * n) throw new ArgumentException("c too small.", nameof(c));
        int groupSize = aScale.GroupSize;
        if (groupSize <= 0)
            throw new ArgumentException("Per-group scales required.", nameof(aScale));
        int groupsPerRow = (k + groupSize - 1) / groupSize;

        c.Clear();
        for (int i = 0; i < m; i++)
        {
            int aRowStart = i * packedRowLen;
            int cRowStart = i * n;
            int scaleRowStart = i * groupsPerRow;

            for (int p = 0; p < k; p++)
            {
                int byteIdx = aRowStart + (p / PackedInt2.ValuesPerByte);
                int laneIdx = p % PackedInt2.ValuesPerByte;
                int q = a[byteIdx].GetLane(laneIdx);
                float scale = aScale.Scales[scaleRowStart + p / groupSize];
                float aVal = q * scale;

                int bRowStart = p * n;
                for (int j = 0; j < n; j++)
                    c[cRowStart + j] += aVal * b[bRowStart + j];
            }
        }
    }

    // ──────────── Int3 weight-only matmul ────────────

    /// <summary>
    /// Weight-only int3 matmul. K must be a multiple of
    /// <see cref="PackedInt3Block.ValuesPerBlock"/>.
    /// </summary>
    public static void Int3WeightMatMul(
        ReadOnlySpan<PackedInt3Block> a, QuantizationScale aScale,
        ReadOnlySpan<float> b, Span<float> c,
        int m, int k, int n)
    {
        if (aScale is null) throw new ArgumentNullException(nameof(aScale));
        if ((k & 0x7) != 0)
            throw new ArgumentException("K must be a multiple of 8 for int3 matmul.", nameof(k));
        int blocksPerRow = k / PackedInt3Block.ValuesPerBlock;
        if (a.Length < m * blocksPerRow)
            throw new ArgumentException("a too small.", nameof(a));
        if (b.Length < k * n) throw new ArgumentException("b too small.", nameof(b));
        if (c.Length < m * n) throw new ArgumentException("c too small.", nameof(c));
        int groupSize = aScale.GroupSize;
        if (groupSize <= 0)
            throw new ArgumentException("Per-group scales required.", nameof(aScale));
        int groupsPerRow = (k + groupSize - 1) / groupSize;

        c.Clear();
        for (int i = 0; i < m; i++)
        {
            int aRowStart = i * blocksPerRow;
            int cRowStart = i * n;
            int scaleRowStart = i * groupsPerRow;

            for (int p = 0; p < k; p++)
            {
                int blockIdx = aRowStart + (p / PackedInt3Block.ValuesPerBlock);
                int laneIdx = p % PackedInt3Block.ValuesPerBlock;
                int q = a[blockIdx].GetLane(laneIdx);
                float scale = aScale.Scales[scaleRowStart + p / groupSize];
                float aVal = q * scale;

                int bRowStart = p * n;
                for (int j = 0; j < n; j++)
                    c[cRowStart + j] += aVal * b[bRowStart + j];
            }
        }
    }

    // ──────────── NF4 / FP4 weight-only matmul ────────────

    /// <summary>
    /// Weight-only NF4 matmul. Reuses <see cref="PackedInt4"/> storage;
    /// each nibble is an index into <see cref="NormalFloat4.Table"/>.
    /// </summary>
    public static void NF4WeightMatMul(
        ReadOnlySpan<PackedInt4> a, QuantizationScale aScale,
        ReadOnlySpan<float> b, Span<float> c,
        int m, int k, int n)
        => Fp4FamilyMatMul(a, aScale, b, c, m, k, n, NormalFloat4.Table);

    /// <summary>
    /// Weight-only FP4 matmul, same shape as <see cref="NF4WeightMatMul"/>
    /// but using the <see cref="Fp4E2M1.Table"/> dictionary.
    /// </summary>
    public static void Fp4WeightMatMul(
        ReadOnlySpan<PackedInt4> a, QuantizationScale aScale,
        ReadOnlySpan<float> b, Span<float> c,
        int m, int k, int n)
        => Fp4FamilyMatMul(a, aScale, b, c, m, k, n, Fp4E2M1.Table);

    private static void Fp4FamilyMatMul(
        ReadOnlySpan<PackedInt4> a, QuantizationScale aScale,
        ReadOnlySpan<float> b, Span<float> c,
        int m, int k, int n, float[] table)
    {
        if (aScale is null) throw new ArgumentNullException(nameof(aScale));
        if ((k & 1) != 0)
            throw new ArgumentException("K must be even.", nameof(k));
        int packedRowLen = k / 2;
        if (a.Length < m * packedRowLen)
            throw new ArgumentException("a too small.", nameof(a));
        if (b.Length < k * n) throw new ArgumentException("b too small.", nameof(b));
        if (c.Length < m * n) throw new ArgumentException("c too small.", nameof(c));
        int groupSize = aScale.GroupSize;
        if (groupSize <= 0)
            throw new ArgumentException("Per-group scales required.", nameof(aScale));
        int groupsPerRow = (k + groupSize - 1) / groupSize;

        c.Clear();
        for (int i = 0; i < m; i++)
        {
            int aRowStart = i * packedRowLen;
            int cRowStart = i * n;
            int scaleRowStart = i * groupsPerRow;
            for (int p = 0; p < k; p++)
            {
                int packedIdx = aRowStart + (p >> 1);
                int nibble = (p & 1) == 0
                    ? (a[packedIdx].RawValue & 0x0F)
                    : ((a[packedIdx].RawValue >> 4) & 0x0F);
                float scale = aScale.Scales[scaleRowStart + p / groupSize];
                float aVal = table[nibble] * scale;

                int bRowStart = p * n;
                for (int j = 0; j < n; j++)
                    c[cRowStart + j] += aVal * b[bRowStart + j];
            }
        }
    }
}
