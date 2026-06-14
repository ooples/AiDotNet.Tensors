// Copyright (c) AiDotNet. All rights reserved.

using System;

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
}
