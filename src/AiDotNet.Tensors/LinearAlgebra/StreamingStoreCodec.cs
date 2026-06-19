// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using System.IO.Compression;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Encodes/decodes weight buffers to bf16 for the streaming pool's backing store
/// (2x smaller disk + resident bytes at ~0.17% RMS error). bf16 keeps the full
/// 8-bit fp32 exponent + 7 mantissa bits, so the dynamic range is identical to
/// fp32 and only low-order precision is lost.
///
/// <para>Two rounding modes: round-to-nearest-even (deterministic, for inference
/// where the quantization happens once) and STOCHASTIC (unbiased — round up with
/// probability equal to the dropped fraction, so accumulating many bf16 stores
/// during training doesn't bias the weights, the standard fix for bf16 masters).</para>
/// </summary>
internal static class StreamingStoreCodec
{
    /// <summary>bf16 element size in bytes (stored little-endian).</summary>
    internal const int Bf16ElementSize = 2;

    // Fast, non-crypto per-thread PRNG for stochastic rounding. Stochastic
    // rounding is a NUMERICAL technique (PyTorch/CUDA use Philox counters) — it
    // needs a cheap high-rate source, not cryptographic randomness, so a
    // thread-static xorshift is correct here. Seeded off the managed thread id so
    // threads diverge; never used for security.
    [ThreadStatic] private static ulong _rngState;

    private static uint NextRand16()
    {
        ulong x = _rngState;
        if (x == 0) x = ((ulong)System.Threading.Thread.CurrentThread.ManagedThreadId * 0x9E3779B97F4A7C15UL) | 1UL;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        _rngState = x;
        return (uint)(x >> 48) & 0xFFFFu; // top 16 bits → uniform [0, 0xFFFF]
    }

    /// <summary>
    /// Pins the per-thread stochastic-rounding PRNG to a fixed seed for the CURRENT thread.
    /// Production never calls this — the stream auto-seeds from the managed thread id and stochastic
    /// rounding is unbiased either way. It exists so convergence tests can make the rounding sequence
    /// reproducible instead of depending on which pool thread xUnit scheduled them on (and on RNG
    /// state left by earlier tests on that thread) — the source of intermittent failures.
    /// </summary>
    internal static void SeedStochasticRng(ulong seed) => _rngState = seed | 1UL;

    private static unsafe uint F32Bits(float v) => *(uint*)&v;
    private static unsafe float BitsF32(uint b) => *(float*)&b;

    private static ushort EncodeOne(float value, bool stochastic)
    {
        uint bits = F32Bits(value);
        // Non-finite (Inf / NaN): take the top 16 bits directly; force the bf16
        // quiet-NaN bit so a NaN stays a NaN through the round-trip.
        if ((bits & 0x7F800000u) == 0x7F800000u)
        {
            ushort hi = (ushort)(bits >> 16);
            if ((bits & 0x007FFFFFu) != 0) hi |= 0x0040; // quiet NaN
            return hi;
        }
        uint rounded;
        if (stochastic)
        {
            // Round up iff the dropped low-16 fraction + a uniform [0,2^16) draw
            // carries into bit 16. P(round up) = low16 / 2^16 → unbiased.
            rounded = (bits + NextRand16()) >> 16;
        }
        else
        {
            // Round-to-nearest-even: bias 0x7FFF + LSB-of-target, then truncate.
            uint lsb = (bits >> 16) & 1u;
            rounded = (bits + 0x7FFFu + lsb) >> 16;
        }
        return (ushort)rounded;
    }

    /// <summary>fp32 → bf16 little-endian bytes. <paramref name="dst"/> must be
    /// <c>src.Length * 2</c> bytes.</summary>
    internal static void EncodeFloat(ReadOnlySpan<float> src, Span<byte> dst, bool stochastic)
    {
        for (int i = 0; i < src.Length; i++)
        {
            ushort raw = EncodeOne(src[i], stochastic);
            dst[i * 2] = (byte)(raw & 0xFF);
            dst[i * 2 + 1] = (byte)((raw >> 8) & 0xFF);
        }
    }

    /// <summary>fp64 → bf16 little-endian bytes (via fp32). <paramref name="dst"/>
    /// must be <c>src.Length * 2</c> bytes.</summary>
    internal static void EncodeDouble(ReadOnlySpan<double> src, Span<byte> dst, bool stochastic)
    {
        for (int i = 0; i < src.Length; i++)
        {
            ushort raw = EncodeOne((float)src[i], stochastic);
            dst[i * 2] = (byte)(raw & 0xFF);
            dst[i * 2 + 1] = (byte)((raw >> 8) & 0xFF);
        }
    }

    /// <summary>bf16 little-endian bytes → fp32. <paramref name="src"/> is
    /// <c>dst.Length * 2</c> bytes.</summary>
    internal static void DecodeFloat(ReadOnlySpan<byte> src, Span<float> dst)
    {
        for (int i = 0; i < dst.Length; i++)
        {
            uint raw = (uint)(src[i * 2] | (src[i * 2 + 1] << 8));
            dst[i] = BitsF32(raw << 16);
        }
    }

    /// <summary>bf16 little-endian bytes → fp64. <paramref name="src"/> is
    /// <c>dst.Length * 2</c> bytes.</summary>
    internal static void DecodeDouble(ReadOnlySpan<byte> src, Span<double> dst)
    {
        for (int i = 0; i < dst.Length; i++)
        {
            uint raw = (uint)(src[i * 2] | (src[i * 2 + 1] << 8));
            dst[i] = BitsF32(raw << 16);
        }
    }

    // ── int8 PER-ROW symmetric quantization (4x vs fp32) ──────────────────────
    // Layout: [int32 rows][rows × fp32 scale][count × signed int8]. One scale per ROW
    // (output channel) — each row uses its own max, so a small-magnitude row isn't
    // clipped by a global max one large row dominates. That preserves much more SNR than
    // a single per-tensor scale on trained weights whose row magnitudes vary widely (the
    // common transformer case), and it's the layout SgemmWithInt8RowScaledCachedB consumes
    // directly (sbyte[n,k] + per-row scales) so the quantized weight can feed the int8 GEMM
    // with NO upcast. `rows` = the weight's leading dim (output channels); k = count/rows.
    // ~1.1% per-tensor RMSE drops to noticeably less per-row; still opt-in (never Auto).

    /// <summary>int8 header bytes (4-byte row count + one 4-byte fp32 scale per row).</summary>
    internal static int Int8HeaderBytes(int rows) => 4 + 4 * rows;
    /// <summary>int8 buffer size for <paramref name="count"/> elements in
    /// <paramref name="rows"/> rows: header + 1 byte/element.</summary>
    internal static int Int8BufferBytes(int count, int rows) => Int8HeaderBytes(rows) + count;

    private static void WriteInt32(Span<byte> dst, int off, int v)
    {
        dst[off] = (byte)v; dst[off + 1] = (byte)(v >> 8); dst[off + 2] = (byte)(v >> 16); dst[off + 3] = (byte)(v >> 24);
    }
    private static int ReadInt32(ReadOnlySpan<byte> src, int off)
        => src[off] | (src[off + 1] << 8) | (src[off + 2] << 16) | (src[off + 3] << 24);
    private static void WriteScaleAt(Span<byte> dst, int off, float scale)
    {
        uint sb = F32Bits(scale);
        dst[off] = (byte)sb; dst[off + 1] = (byte)(sb >> 8); dst[off + 2] = (byte)(sb >> 16); dst[off + 3] = (byte)(sb >> 24);
    }
    private static float ReadScaleAt(ReadOnlySpan<byte> src, int off)
        => BitsF32((uint)(src[off] | (src[off + 1] << 8) | (src[off + 2] << 16) | (src[off + 3] << 24)));

    private static byte QuantOne(double v, double inv)
    {
        int q = (int)Math.Round(v * inv);
        if (q > 127) q = 127; else if (q < -127) q = -127; // symmetric, avoid -128
        return (byte)(sbyte)q;
    }

    /// <summary>fp32 → per-row int8. <paramref name="dst"/> must be
    /// <c>Int8BufferBytes(src.Length, rows)</c>; <paramref name="rows"/> must divide the length.</summary>
    internal static void EncodeInt8Float(ReadOnlySpan<float> src, Span<byte> dst, int rows)
    {
        int count = src.Length;
        if (rows <= 0 || count % rows != 0)
            throw new ArgumentException($"int8 encode: rows ({rows}) must be > 0 and divide length ({count}).", nameof(rows));
        int k = count / rows;
        int dataOff = Int8HeaderBytes(rows);
        WriteInt32(dst, 0, rows);
        for (int r = 0; r < rows; r++)
        {
            int baseI = r * k;
            float amax = 0f;
            for (int j = 0; j < k; j++)
            {
                float v = src[baseI + j];
                // Reject NaN/Infinity at the boundary: a poisoned amax → 1/scale = ±Inf →
                // (int)Math.Round(±Inf) = an architecture-dependent sentinel. Fail fast.
                if (float.IsNaN(v) || float.IsInfinity(v))
                    throw new ArgumentException(
                        $"int8 streaming-store encoder cannot encode non-finite value at index {baseI + j} (got {v}).",
                        nameof(src));
                float a = Math.Abs(v); if (a > amax) amax = a;
            }
            float scale = amax > 0f ? amax / 127f : 1f;
            WriteScaleAt(dst, 4 + r * 4, scale);
            double inv = 1.0 / scale;
            for (int j = 0; j < k; j++) dst[dataOff + baseI + j] = QuantOne(src[baseI + j], inv);
        }
    }

    /// <summary>fp64 → per-row int8.</summary>
    internal static void EncodeInt8Double(ReadOnlySpan<double> src, Span<byte> dst, int rows)
    {
        int count = src.Length;
        if (rows <= 0 || count % rows != 0)
            throw new ArgumentException($"int8 encode: rows ({rows}) must be > 0 and divide length ({count}).", nameof(rows));
        int k = count / rows;
        int dataOff = Int8HeaderBytes(rows);
        WriteInt32(dst, 0, rows);
        for (int r = 0; r < rows; r++)
        {
            int baseI = r * k;
            double amax = 0.0;
            for (int j = 0; j < k; j++)
            {
                double v = src[baseI + j];
                if (double.IsNaN(v) || double.IsInfinity(v))
                    throw new ArgumentException(
                        $"int8 streaming-store encoder cannot encode non-finite value at index {baseI + j} (got {v}).",
                        nameof(src));
                double a = Math.Abs(v); if (a > amax) amax = a;
            }
            float scale = amax > 0.0 ? (float)(amax / 127.0) : 1f;
            WriteScaleAt(dst, 4 + r * 4, scale);
            double inv = 1.0 / scale;
            for (int j = 0; j < k; j++) dst[dataOff + baseI + j] = QuantOne(src[baseI + j], inv);
        }
    }

    /// <summary>per-row int8 → fp32 (dequant fallback). <paramref name="src"/> is the encoded buffer.</summary>
    internal static void DecodeInt8Float(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int rows = ReadInt32(src, 0);
        int count = dst.Length;
        int k = count / rows;
        int dataOff = Int8HeaderBytes(rows);
        for (int r = 0; r < rows; r++)
        {
            float scale = ReadScaleAt(src, 4 + r * 4);
            int baseI = r * k;
            for (int j = 0; j < k; j++) dst[baseI + j] = (sbyte)src[dataOff + baseI + j] * scale;
        }
    }

    /// <summary>per-row int8 → fp64.</summary>
    internal static void DecodeInt8Double(ReadOnlySpan<byte> src, Span<double> dst)
    {
        int rows = ReadInt32(src, 0);
        int count = dst.Length;
        int k = count / rows;
        int dataOff = Int8HeaderBytes(rows);
        for (int r = 0; r < rows; r++)
        {
            float scale = ReadScaleAt(src, 4 + r * 4);
            int baseI = r * k;
            for (int j = 0; j < k; j++) dst[baseI + j] = (sbyte)src[dataOff + baseI + j] * (double)scale;
        }
    }

    // ── Compute-path extraction: pull the int8 weight + per-row scales out of the encoded
    //    buffer WITHOUT dequantizing, so a streaming int8 weight can feed the int8 GEMM directly.
    /// <summary>Row count stored in an int8 buffer header.</summary>
    internal static int Int8RowsOf(ReadOnlySpan<byte> src) => ReadInt32(src, 0);
    /// <summary>Per-row scales from an int8 buffer.</summary>
    internal static float[] Int8ScalesOf(ReadOnlySpan<byte> src)
    {
        int rows = ReadInt32(src, 0);
        var s = new float[rows];
        for (int r = 0; r < rows; r++) s[r] = ReadScaleAt(src, 4 + r * 4);
        return s;
    }
    /// <summary>The <paramref name="count"/> signed int8 weights from an int8 buffer (row-major [rows,k]).</summary>
    internal static sbyte[] Int8DataOf(ReadOnlySpan<byte> src, int count)
    {
        int rows = ReadInt32(src, 0);
        int dataOff = Int8HeaderBytes(rows);
        var d = new sbyte[count];
        for (int i = 0; i < count; i++) d[i] = (sbyte)src[dataOff + i];
        return d;
    }

    // ── int4 GROUP symmetric quantization (8x vs fp32) ───────────────────────
    // Layout: [int32 count][int32 groupSize][numGroups × fp32 scale][ceil(count/2) packed nibbles].
    // AWQ/GPTQ-style group quantization: contiguous runs of `groupSize` elements share one
    // symmetric scale (amax/7). 4 bits/weight = 8x compression vs fp32 — the most aggressive
    // rung of the quant ladder, required to make the very largest (>~20B) models RESIDENT in a
    // 16 GiB budget. Group (not per-tensor) scaling keeps the int4 RMSE bounded despite the tiny
    // 4-bit range; smaller groups = better SNR at a small header cost (one fp32 per group).
    // Signed 4-bit range is [-7, 7] (8 is excluded to stay symmetric, mirroring int8's -127).
    // Each byte packs two elements: low nibble = even index, high nibble = odd index.
    // Explicit opt-in only — Auto never picks int4 (far lossier than bf16/int8).

    /// <summary>Default int4 group size (AWQ/GPTQ convention). One fp32 scale per 128 weights.</summary>
    internal const int DefaultInt4GroupSize = 128;

    /// <summary>Number of int4 groups covering <paramref name="count"/> elements at
    /// <paramref name="groupSize"/> (the last group may be partial).</summary>
    internal static int Int4NumGroups(int count, int groupSize) => (count + groupSize - 1) / groupSize;

    /// <summary>int4 header bytes: 4 (count) + 4 (groupSize) + one 4-byte fp32 scale per group.</summary>
    internal static int Int4HeaderBytes(int numGroups) => 8 + 4 * numGroups;

    /// <summary>int4 buffer size for <paramref name="count"/> elements at <paramref name="groupSize"/>:
    /// header + ceil(count/2) packed-nibble bytes.</summary>
    internal static int Int4BufferBytes(int count, int groupSize)
        => Int4HeaderBytes(Int4NumGroups(count, groupSize)) + (count + 1) / 2;

    private static sbyte QuantOneInt4(double v, double inv)
    {
        int q = (int)Math.Round(v * inv);
        if (q > 7) q = 7; else if (q < -7) q = -7; // symmetric, avoid -8
        return (sbyte)q;
    }

    // Write a signed 4-bit value into the low (even i) or high (odd i) nibble of dst[dataOff + i/2].
    private static void PackNibble(Span<byte> dst, int dataOff, int i, sbyte q)
    {
        int b = dataOff + (i >> 1);
        byte nib = (byte)(q & 0x0F);
        if ((i & 1) == 0) dst[b] = (byte)((dst[b] & 0xF0) | nib);
        else dst[b] = (byte)((dst[b] & 0x0F) | (nib << 4));
    }

    // Read + sign-extend the 4-bit value at element index i.
    private static int UnpackNibble(ReadOnlySpan<byte> src, int dataOff, int i)
    {
        byte by = src[dataOff + (i >> 1)];
        int n = (i & 1) == 0 ? (by & 0x0F) : (by >> 4) & 0x0F;
        return (n & 0x8) != 0 ? n - 16 : n; // sign-extend -8..7
    }

    /// <summary>fp32 → int4 group-quant. <paramref name="dst"/> must be
    /// <c>Int4BufferBytes(src.Length, groupSize)</c>.</summary>
    internal static void EncodeInt4Float(ReadOnlySpan<float> src, Span<byte> dst, int groupSize)
    {
        int count = src.Length;
        if (groupSize <= 0)
            throw new ArgumentException($"int4 encode: groupSize ({groupSize}) must be > 0.", nameof(groupSize));
        int numGroups = Int4NumGroups(count, groupSize);
        int dataOff = Int4HeaderBytes(numGroups);
        WriteInt32(dst, 0, count);
        WriteInt32(dst, 4, groupSize);
        for (int g = 0; g < numGroups; g++)
        {
            int baseI = g * groupSize;
            int len = Math.Min(groupSize, count - baseI);
            float amax = 0f;
            for (int j = 0; j < len; j++)
            {
                float v = src[baseI + j];
                if (float.IsNaN(v) || float.IsInfinity(v))
                    throw new ArgumentException(
                        $"int4 streaming-store encoder cannot encode non-finite value at index {baseI + j} (got {v}).",
                        nameof(src));
                float a = Math.Abs(v); if (a > amax) amax = a;
            }
            float scale = amax > 0f ? amax / 7f : 1f;
            WriteScaleAt(dst, 8 + g * 4, scale);
            double inv = 1.0 / scale;
            for (int j = 0; j < len; j++) PackNibble(dst, dataOff, baseI + j, QuantOneInt4(src[baseI + j], inv));
        }
    }

    /// <summary>fp64 → int4 group-quant.</summary>
    internal static void EncodeInt4Double(ReadOnlySpan<double> src, Span<byte> dst, int groupSize)
    {
        int count = src.Length;
        if (groupSize <= 0)
            throw new ArgumentException($"int4 encode: groupSize ({groupSize}) must be > 0.", nameof(groupSize));
        int numGroups = Int4NumGroups(count, groupSize);
        int dataOff = Int4HeaderBytes(numGroups);
        WriteInt32(dst, 0, count);
        WriteInt32(dst, 4, groupSize);
        for (int g = 0; g < numGroups; g++)
        {
            int baseI = g * groupSize;
            int len = Math.Min(groupSize, count - baseI);
            double amax = 0.0;
            for (int j = 0; j < len; j++)
            {
                double v = src[baseI + j];
                if (double.IsNaN(v) || double.IsInfinity(v))
                    throw new ArgumentException(
                        $"int4 streaming-store encoder cannot encode non-finite value at index {baseI + j} (got {v}).",
                        nameof(src));
                double a = Math.Abs(v); if (a > amax) amax = a;
            }
            float scale = amax > 0.0 ? (float)(amax / 7.0) : 1f;
            WriteScaleAt(dst, 8 + g * 4, scale);
            double inv = 1.0 / scale;
            for (int j = 0; j < len; j++) PackNibble(dst, dataOff, baseI + j, QuantOneInt4(src[baseI + j], inv));
        }
    }

    /// <summary>int4 group-quant → fp32 (dequant). <paramref name="src"/> is the encoded buffer.</summary>
    internal static void DecodeInt4Float(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int count = ReadInt32(src, 0);
        int groupSize = ReadInt32(src, 4);
        int numGroups = Int4NumGroups(count, groupSize);
        int dataOff = Int4HeaderBytes(numGroups);
        for (int g = 0; g < numGroups; g++)
        {
            float scale = ReadScaleAt(src, 8 + g * 4);
            int baseI = g * groupSize;
            int len = Math.Min(groupSize, count - baseI);
            for (int j = 0; j < len; j++) dst[baseI + j] = UnpackNibble(src, dataOff, baseI + j) * scale;
        }
    }

    /// <summary>int4 group-quant → fp64.</summary>
    internal static void DecodeInt4Double(ReadOnlySpan<byte> src, Span<double> dst)
    {
        int count = ReadInt32(src, 0);
        int groupSize = ReadInt32(src, 4);
        int numGroups = Int4NumGroups(count, groupSize);
        int dataOff = Int4HeaderBytes(numGroups);
        for (int g = 0; g < numGroups; g++)
        {
            float scale = ReadScaleAt(src, 8 + g * 4);
            int baseI = g * groupSize;
            int len = Math.Min(groupSize, count - baseI);
            for (int j = 0; j < len; j++) dst[baseI + j] = UnpackNibble(src, dataOff, baseI + j) * (double)scale;
        }
    }

    // ── int4 compute-path extraction (for the no-upcast int4 GEMM) ──
    /// <summary>Element count stored in an int4 buffer header.</summary>
    internal static int Int4CountOf(ReadOnlySpan<byte> src) => ReadInt32(src, 0);
    /// <summary>Group size stored in an int4 buffer header.</summary>
    internal static int Int4GroupSizeOf(ReadOnlySpan<byte> src) => ReadInt32(src, 4);
    /// <summary>Per-group scales from an int4 buffer.</summary>
    internal static float[] Int4ScalesOf(ReadOnlySpan<byte> src)
    {
        int count = ReadInt32(src, 0);
        int groupSize = ReadInt32(src, 4);
        int numGroups = Int4NumGroups(count, groupSize);
        var s = new float[numGroups];
        for (int g = 0; g < numGroups; g++) s[g] = ReadScaleAt(src, 8 + g * 4);
        return s;
    }
    /// <summary>The <paramref name="count"/> sign-extended int4 weights (one per sbyte) from an int4 buffer.</summary>
    internal static sbyte[] Int4DataOf(ReadOnlySpan<byte> src, int count)
    {
        int groupSize = ReadInt32(src, 4);
        int numGroups = Int4NumGroups(count, groupSize);
        int dataOff = Int4HeaderBytes(numGroups);
        var d = new sbyte[count];
        for (int i = 0; i < count; i++) d[i] = (sbyte)UnpackNibble(src, dataOff, i);
        return d;
    }

    // ── Lossless: byte-plane shuffle + DEFLATE (EXACT, ~1.18x on fp weights) ──────────
    // Dense fp weights don't compress raw (~100%): the high-entropy mantissa bytes are
    // interleaved with the structured sign/exponent bytes. Byte-plane shuffle (SIMD, see
    // above) groups byte-k of every element together, so the sign+exponent plane — highly
    // repetitive for near-Gaussian weights — becomes a long low-entropy run. An ENTROPY coder
    // (Deflate's Huffman stage) then actually shrinks it (~1.18x); LZ4, being match-only with
    // no entropy stage, can't (~1.08x). Output = [1-byte flag][payload]: flag 1 = Deflate
    // (shuffled), 0 = raw shuffled (when it didn't shrink). This is the EXACT (lossless) store
    // — bit-for-bit — used by default in TRAINING where weights must stay exact; bf16/int8
    // give more (2x/4x) in inference at a precision cost.

    // Shuffle = transpose the [count × elem] byte matrix to [elem × count] (group byte-k of
    // every element). The naive whole-array strided pass thrashes cache/TLB (~256 MiB/s) and
    // was the reason lossless couldn't be a default. Fast paths:
    //   • fp32 (elem 4): AVX2 byte-plane transpose (unpack/permute), 32 elements/iter.
    //   • everything else / net471 / SIMD remainder: a CACHE-TILED scalar pass — same result
    //     as the naive loop but the strided source stays within an L1/L2-sized tile, which
    //     alone is several× faster than the whole-array version.
    private const int ShuffleTileElems = 2048; // tile * elem stays in L1/L2

    private static void BytePlaneShuffle(ReadOnlySpan<byte> src, Span<byte> dst, int elem)
    {
        int count = src.Length / elem;
#if NET5_0_OR_GREATER
        if (elem == 4 && Ssse3.IsSupported && count >= 4)
        {
            int vec = count & ~3; // largest multiple of 4
            Shuffle4Ssse3(src, dst, count, vec);
            ShuffleTailScalar(src, dst, elem, count, vec);
            return;
        }
#endif
        BytePlaneShuffleScalar(src, dst, elem, count, 0);
    }

    private static void BytePlaneUnshuffle(ReadOnlySpan<byte> src, Span<byte> dst, int elem)
    {
        int count = dst.Length / elem;
#if NET5_0_OR_GREATER
        if (elem == 4 && Ssse3.IsSupported && count >= 4)
        {
            int vec = count & ~3;
            Unshuffle4Ssse3(src, dst, count, vec);
            UnshuffleTailScalar(src, dst, elem, count, vec);
            return;
        }
#endif
        BytePlaneUnshuffleScalar(src, dst, elem, count, 0);
    }

    // Cache-tiled scalar transpose, from element `from` to `count`. Correctness reference
    // for the SIMD paths (the SIMD tests assert bit-identical output).
    private static void BytePlaneShuffleScalar(ReadOnlySpan<byte> src, Span<byte> dst, int elem, int count, int from)
    {
        for (int b0 = from; b0 < count; b0 += ShuffleTileElems)
        {
            int b1 = Math.Min(b0 + ShuffleTileElems, count);
            for (int k = 0; k < elem; k++)
            {
                int planeBase = k * count;
                for (int i = b0; i < b1; i++) dst[planeBase + i] = src[i * elem + k];
            }
        }
    }

    private static void BytePlaneUnshuffleScalar(ReadOnlySpan<byte> src, Span<byte> dst, int elem, int count, int from)
    {
        for (int b0 = from; b0 < count; b0 += ShuffleTileElems)
        {
            int b1 = Math.Min(b0 + ShuffleTileElems, count);
            for (int k = 0; k < elem; k++)
            {
                int planeBase = k * count;
                for (int i = b0; i < b1; i++) dst[i * elem + k] = src[planeBase + i];
            }
        }
    }

#if NET5_0_OR_GREATER
    // Scalar remainder for the SIMD prefix [vec, count). Element i's plane writes are at
    // dst[k*count + i] (shuffle) / read from there (unshuffle) — same layout as the SIMD body.
    private static void ShuffleTailScalar(ReadOnlySpan<byte> src, Span<byte> dst, int elem, int count, int vec)
    {
        for (int k = 0; k < elem; k++)
        {
            int planeBase = k * count;
            for (int i = vec; i < count; i++) dst[planeBase + i] = src[i * elem + k];
        }
    }

    private static void UnshuffleTailScalar(ReadOnlySpan<byte> src, Span<byte> dst, int elem, int count, int vec)
    {
        for (int k = 0; k < elem; k++)
        {
            int planeBase = k * count;
            for (int i = vec; i < count; i++) dst[i * elem + k] = src[planeBase + i];
        }
    }

    // SSSE3 byte-plane transpose for 4-byte elements. 16 bytes (4 floats) per iteration:
    // one pshufb groups the 4 elements' bytes by plane within a single 128-bit register
    // — no cross-lane ops, no second unpack pass — and the resulting four 32-bit lanes
    // scatter to four contiguous plane offsets. Bit-identical to BytePlaneShuffleScalar
    // (asserted by tests).
    // Self-inverse per-128-bit byte gather [0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15]:
    // applied to 4 elements' interleaved bytes it groups them by plane; applied to 4 planes'
    // grouped bytes it scatters them back to element order. Same mask both directions.
    private static readonly Vector128<byte> ByteGroupMask128 = Vector128.Create(
        (byte)0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    // SSSE3 byte-plane shuffle for 4-byte elements, 4 elements (16 bytes) per iteration. One
    // 128-bit pshufb groups the 4 elements' bytes by plane → [p0:4 p1:4 p2:4 p3:4]; the four
    // 32-bit lanes are then written to the four plane offsets. No cross-lane ops, so the output
    // is bit-identical to BytePlaneShuffleScalar (the SIMD prefix + scalar tail compose into
    // the exact scalar layout). Asserted by the round-trip + scalar-equality tests.
    private static unsafe void Shuffle4Ssse3(ReadOnlySpan<byte> src, Span<byte> dst, int count, int vec)
    {
        Vector128<byte> mask = ByteGroupMask128;
        fixed (byte* ps = src)
        fixed (byte* pd = dst)
        {
            for (int i = 0; i < vec; i += 4)
            {
                Vector128<int> g = Ssse3.Shuffle(Sse2.LoadVector128(ps + i * 4), mask).AsInt32();
                *(int*)(pd + 0 * count + i) = g.GetElement(0);
                *(int*)(pd + 1 * count + i) = g.GetElement(1);
                *(int*)(pd + 2 * count + i) = g.GetElement(2);
                *(int*)(pd + 3 * count + i) = g.GetElement(3);
            }
        }
    }

    // Inverse: gather one 32-bit lane from each of the 4 planes, pshufb back to element order.
    private static unsafe void Unshuffle4Ssse3(ReadOnlySpan<byte> src, Span<byte> dst, int count, int vec)
    {
        Vector128<byte> mask = ByteGroupMask128;
        fixed (byte* ps = src)
        fixed (byte* pd = dst)
        {
            for (int i = 0; i < vec; i += 4)
            {
                Vector128<int> g = Vector128.Create(
                    *(int*)(ps + 0 * count + i),
                    *(int*)(ps + 1 * count + i),
                    *(int*)(ps + 2 * count + i),
                    *(int*)(ps + 3 * count + i));
                Sse2.Store(pd + i * 4, Ssse3.Shuffle(g.AsByte(), mask));
            }
        }
    }
#endif

    // DEFLATE (zlib's raw deflate) is the entropy coder: unlike LZ4 (match-only, no entropy
    // stage → ~1.08x on shuffled fp weights), Deflate's Huffman stage compresses the highly
    // repetitive sign/exponent byte-plane that the shuffle exposes → ~1.18x, bit-exact, at
    // ~1.1 GiB/s decode. It's in the BCL on both net471 and net5+ (no extra dependency).
    private static byte[] DeflateCompress(byte[] src)
    {
        using var ms = new MemoryStream(src.Length);
        using (var ds = new DeflateStream(ms, CompressionLevel.Optimal, leaveOpen: true))
            ds.Write(src, 0, src.Length);
        return ms.ToArray();
    }

    private static int DeflateDecompress(byte[] comp, byte[] dst, int dstLen)
    {
        using var ms = new MemoryStream(comp);
        using var ds = new DeflateStream(ms, CompressionMode.Decompress);
        int read = 0, r;
        while (read < dstLen && (r = ds.Read(dst, read, dstLen - read)) > 0) read += r;
        return read;
    }

    private static byte[] EncodeLosslessBytes(ReadOnlySpan<byte> raw, int elem)
    {
        var shuffled = new byte[raw.Length];
        BytePlaneShuffle(raw, shuffled, elem);
        var comp = DeflateCompress(shuffled);
        if (comp.Length < shuffled.Length)
        {
            var outp = new byte[1 + comp.Length];
            outp[0] = 1; // Deflate-compressed
            Buffer.BlockCopy(comp, 0, outp, 1, comp.Length);
            return outp;
        }
        // Didn't shrink (rare — tiny or high-entropy) — store the raw shuffled bytes so the
        // round-trip is still exact and never larger than shuffled + 1.
        var rawOut = new byte[1 + shuffled.Length];
        rawOut[0] = 0;
        Buffer.BlockCopy(shuffled, 0, rawOut, 1, shuffled.Length);
        return rawOut;
    }

    private static void DecodeLosslessBytes(ReadOnlySpan<byte> src, Span<byte> dstRaw, int elem)
    {
        if (src.Length < 1) throw new ArgumentException("Lossless payload too short.", nameof(src));
        int shuffledLen = dstRaw.Length;
        byte flag = src[0];
        var payload = src.Slice(1);
        byte[] shuffled;
        if (flag == 1)
        {
            shuffled = new byte[shuffledLen];
            int dec = DeflateDecompress(payload.ToArray(), shuffled, shuffledLen);
            if (dec != shuffledLen)
                throw new InvalidOperationException($"Lossless decode: Deflate produced {dec} bytes, expected {shuffledLen}.");
        }
        else
        {
            shuffled = payload.ToArray();
            if (shuffled.Length != shuffledLen)
                throw new InvalidOperationException($"Lossless decode: raw payload {shuffled.Length} bytes, expected {shuffledLen}.");
        }
        BytePlaneUnshuffle(shuffled, dstRaw, elem);
    }

    // ── Test hooks (InternalsVisibleTo) — exercise the shuffle transform directly so tests
    //    can assert the SIMD path is bit-identical to the scalar reference + invertible. ──
    internal static byte[] ShuffleForTest(ReadOnlySpan<byte> raw, int elem)
    {
        var d = new byte[raw.Length];
        BytePlaneShuffle(raw, d, elem);
        return d;
    }

    internal static byte[] ShuffleScalarForTest(ReadOnlySpan<byte> raw, int elem)
    {
        var d = new byte[raw.Length];
        BytePlaneShuffleScalar(raw, d, elem, raw.Length / elem, 0);
        return d;
    }

    internal static byte[] UnshuffleForTest(ReadOnlySpan<byte> shuffled, int elem)
    {
        var d = new byte[shuffled.Length];
        BytePlaneUnshuffle(shuffled, d, elem);
        return d;
    }

    /// <summary>fp32 → lossless (byte-shuffle + Deflate). Returns a variable-size buffer.</summary>
    internal static byte[] EncodeLosslessFloat(ReadOnlySpan<float> src)
        => EncodeLosslessBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(src), 4);

    /// <summary>fp64 → lossless.</summary>
    internal static byte[] EncodeLosslessDouble(ReadOnlySpan<double> src)
        => EncodeLosslessBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(src), 8);

    /// <summary>lossless → fp32 (exact round-trip).</summary>
    internal static void DecodeLosslessFloat(ReadOnlySpan<byte> src, Span<float> dst)
        => DecodeLosslessBytes(src, System.Runtime.InteropServices.MemoryMarshal.AsBytes(dst), 4);

    /// <summary>lossless → fp64 (exact round-trip).</summary>
    internal static void DecodeLosslessDouble(ReadOnlySpan<byte> src, Span<double> dst)
        => DecodeLosslessBytes(src, System.Runtime.InteropServices.MemoryMarshal.AsBytes(dst), 8);
}
