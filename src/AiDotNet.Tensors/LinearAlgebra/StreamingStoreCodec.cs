// Copyright (c) AiDotNet. All rights reserved.

using System;
using K4os.Compression.LZ4;
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

    // ── int8 per-tensor symmetric quantization (4x vs fp32) ───────────────────
    // Layout: a 4-byte little-endian fp32 scale prefix, then one signed int8 per
    // element. Dequant = q * scale. ~1.1% RMS error on Gaussian weights — much more
    // lossy than bf16, so it's explicit opt-in (never the Auto default).

    /// <summary>int8 buffer size for <paramref name="count"/> elements: 4-byte scale
    /// prefix + 1 byte each.</summary>
    internal const int Int8ScaleBytes = 4;
    internal static int Int8BufferBytes(int count) => Int8ScaleBytes + count;

    private static void WriteScale(Span<byte> dst, float scale)
    {
        uint sb = F32Bits(scale);
        dst[0] = (byte)sb; dst[1] = (byte)(sb >> 8); dst[2] = (byte)(sb >> 16); dst[3] = (byte)(sb >> 24);
    }
    private static float ReadScale(ReadOnlySpan<byte> src)
        => BitsF32((uint)(src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24)));

    private static byte QuantOne(double v, double inv)
    {
        int q = (int)Math.Round(v * inv);
        if (q > 127) q = 127; else if (q < -127) q = -127; // symmetric, avoid -128
        return (byte)(sbyte)q;
    }

    /// <summary>fp32 → int8 (+scale). <paramref name="dst"/> must be
    /// <c>Int8BufferBytes(src.Length)</c>.</summary>
    internal static void EncodeInt8Float(ReadOnlySpan<float> src, Span<byte> dst)
    {
        float amax = 0f;
        for (int i = 0; i < src.Length; i++) { float a = Math.Abs(src[i]); if (a > amax) amax = a; }
        float scale = amax > 0f ? amax / 127f : 1f;
        WriteScale(dst, scale);
        double inv = 1.0 / scale;
        for (int i = 0; i < src.Length; i++) dst[Int8ScaleBytes + i] = QuantOne(src[i], inv);
    }

    /// <summary>fp64 → int8 (+scale).</summary>
    internal static void EncodeInt8Double(ReadOnlySpan<double> src, Span<byte> dst)
    {
        double amax = 0.0;
        for (int i = 0; i < src.Length; i++) { double a = Math.Abs(src[i]); if (a > amax) amax = a; }
        float scale = amax > 0.0 ? (float)(amax / 127.0) : 1f;
        WriteScale(dst, scale);
        double inv = 1.0 / scale;
        for (int i = 0; i < src.Length; i++) dst[Int8ScaleBytes + i] = QuantOne(src[i], inv);
    }

    /// <summary>int8 (+scale) → fp32. <paramref name="src"/> is
    /// <c>Int8BufferBytes(dst.Length)</c>.</summary>
    internal static void DecodeInt8Float(ReadOnlySpan<byte> src, Span<float> dst)
    {
        float scale = ReadScale(src);
        for (int i = 0; i < dst.Length; i++) dst[i] = (sbyte)src[Int8ScaleBytes + i] * scale;
    }

    /// <summary>int8 (+scale) → fp64.</summary>
    internal static void DecodeInt8Double(ReadOnlySpan<byte> src, Span<double> dst)
    {
        float scale = ReadScale(src);
        for (int i = 0; i < dst.Length; i++) dst[i] = (sbyte)src[Int8ScaleBytes + i] * (double)scale;
    }

    // ── Lossless: byte-plane shuffle + LZ4 (EXACT, ~1.08x on fp weights) ──────────
    // Raw LZ4 can't compress dense fp weights (~100%) because the high-entropy mantissa
    // bytes are interleaved with the structured sign/exponent bytes. Byte-plane shuffle
    // groups byte-k of every element together, so the high (sign+exponent) plane — highly
    // repetitive for near-Gaussian weights — becomes compressible and LZ4 then shrinks it.
    // Output = [1-byte flag][payload]: flag 1 = LZ4(shuffled), 0 = raw shuffled (when LZ4
    // didn't shrink). LZ4 decode is ~6.7 GB/s so it never bottlenecks rehydrate. This is
    // the EXACT (lossless) store option; bf16/int8 give more (2x/4x) at a precision cost.

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

    // AVX2 byte-plane transpose for 4-byte elements. 128 bytes (32 floats) per iteration:
    // pshufb groups bytes by plane within each 128-bit lane, two unpack stages transpose the
    // 32-/64-bit groups, and a permute fixes the lane crossing — yielding 4 contiguous
    // 32-byte plane outputs. Bit-identical to BytePlaneShuffleScalar (asserted by tests).
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

    private static byte[] EncodeLosslessBytes(ReadOnlySpan<byte> raw, int elem)
    {
        var shuffled = new byte[raw.Length];
        BytePlaneShuffle(raw, shuffled, elem);
        int maxOut = LZ4Codec.MaximumOutputSize(shuffled.Length);
        var lz = new byte[maxOut];
        int enc = LZ4Codec.Encode(shuffled, 0, shuffled.Length, lz, 0, maxOut);
        if (enc > 0 && enc < shuffled.Length)
        {
            var outp = new byte[1 + enc];
            outp[0] = 1; // LZ4-compressed
            Buffer.BlockCopy(lz, 0, outp, 1, enc);
            return outp;
        }
        // Didn't shrink — store the raw shuffled bytes so decode is still exact.
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
            int dec = LZ4Codec.Decode(payload.ToArray(), 0, payload.Length, shuffled, 0, shuffledLen);
            if (dec != shuffledLen)
                throw new InvalidOperationException($"Lossless decode: LZ4 produced {dec} bytes, expected {shuffledLen}.");
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

    /// <summary>fp32 → lossless (byte-shuffle + LZ4). Returns a variable-size buffer.</summary>
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
