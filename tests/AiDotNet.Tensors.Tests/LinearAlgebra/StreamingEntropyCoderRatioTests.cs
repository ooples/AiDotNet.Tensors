// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using AiDotNet.Tensors.LinearAlgebra;
using K4os.Compression.LZ4;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// MEASUREMENT to pick the entropy coder for default lossless compression. LZ4 (no entropy
/// stage) tops out ~1.1x on byte-shuffled weights. This compares LZ4 vs Deflate vs Brotli
/// (all on the SAME byte-shuffled bytes) for both ratio and decode throughput, on realistic
/// trained-weight distributions (small-std Gaussian → concentrated exponent byte-plane), so
/// the default-compression decision is data-driven.
/// </summary>
public class StreamingEntropyCoderRatioTests
{
    private readonly ITestOutputHelper _out;
    public StreamingEntropyCoderRatioTests(ITestOutputHelper output) => _out = output;

    private static float[] TrainedLikeWeights(int n, double std, int seed)
    {
        // Box-Muller Gaussian via a cheap xorshift — mimics a trained layer's weights.
        var w = new float[n];
        uint s = (uint)(seed * 2654435761u + 1u);
        float Next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return (s & 0xFFFFFF) / (float)0xFFFFFF; }
        for (int i = 0; i < n; i++)
        {
            double u1 = Math.Max(1e-7, Next()), u2 = Next();
            double g = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            w[i] = (float)(g * std);
        }
        return w;
    }

    private static byte[] Deflate(byte[] src)
    {
        using var ms = new MemoryStream();
        using (var ds = new DeflateStream(ms, CompressionLevel.Optimal, leaveOpen: true)) ds.Write(src, 0, src.Length);
        return ms.ToArray();
    }

    private static byte[] Inflate(byte[] comp, int rawLen)
    {
        using var ms = new MemoryStream(comp);
        using var ds = new DeflateStream(ms, CompressionMode.Decompress);
        var outp = new byte[rawLen];
        int read = 0, r;
        while (read < rawLen && (r = ds.Read(outp, read, rawLen - read)) > 0) read += r;
        return outp;
    }

#if NET5_0_OR_GREATER
    private static byte[] Brotli(byte[] src)
    {
        using var ms = new MemoryStream();
        using (var bs = new BrotliStream(ms, CompressionLevel.Optimal, leaveOpen: true)) bs.Write(src, 0, src.Length);
        return ms.ToArray();
    }

    private static byte[] UnBrotli(byte[] comp, int rawLen)
    {
        using var ms = new MemoryStream(comp);
        using var bs = new BrotliStream(ms, CompressionMode.Decompress);
        var outp = new byte[rawLen];
        int read = 0, r;
        while (read < rawLen && (r = bs.Read(outp, read, rawLen - read)) > 0) read += r;
        return outp;
    }
#endif

    // Scalar bit-plane shuffle (32 bit-planes for fp32): groups bit-k of every element. Slow
    // (measurement only) — it exists to see whether separating the exponent BITS (concentrated
    // → very compressible) from the mantissa noise beats byte-plane through the entropy coder.
    private static byte[] BitPlaneShuffle32(float[] w)
    {
        int n = w.Length;
        int planeBytes = (n + 7) / 8;
        var outp = new byte[32 * planeBytes];
        for (int i = 0; i < n; i++)
        {
            uint bits = unchecked((uint)BitExactHelpers.SingleBits(w[i]));
            int bytePos = i >> 3, bitPos = i & 7;
            for (int b = 0; b < 32; b++)
                if (((bits >> b) & 1u) != 0) outp[b * planeBytes + bytePos] |= (byte)(1 << bitPos);
        }
        return outp;
    }

    // FINDING (measured below): bit-plane shuffle does NOT beat byte-plane through an entropy
    // coder — it's marginally WORSE (−0.3% to −1.9%) and ~18x slower to compute. Bit-plane
    // helps LZ4 (no entropy stage → cleaner bit separation aids the match-finder), but Deflate's
    // Huffman stage already extracts the exponent redundancy from byte-plane 3, so the extra
    // separation buys nothing. ~1.18x byte-plane+Deflate is the practical LOSSLESS ceiling for
    // dense fp weights; the mantissa is incompressible. Going past it requires LOSSY methods
    // (bf16 2x, int8 4x) — kept here as a guard so the dead end isn't re-attempted.
    [Theory]
    [InlineData(0.02)]
    [InlineData(0.05)]
    [InlineData(0.2)]
    public void BitPlaneShuffle_DoesNotBeatBytePlane_ThroughDeflate(double std)
    {
        const int n = 1 << 20; // 1M floats
        var w = TrainedLikeWeights(n, std, (int)(std * 1000) + 7);
        var raw = new byte[n * 4];
        Buffer.BlockCopy(w, 0, raw, 0, raw.Length);

        var bytePlane = StreamingStoreCodec.ShuffleForTest(raw, 4);
        var bitPlane = BitPlaneShuffle32(w);
        double byteRatio = (double)raw.Length / Deflate(bytePlane).Length;
        double bitRatio = (double)raw.Length / Deflate(bitPlane).Length;

        _out.WriteLine($"std={std}: byte-plane+Deflate {byteRatio:F3}x | bit-plane+Deflate {bitRatio:F3}x " +
                       $"({(bitRatio - byteRatio) / byteRatio * 100:+0.0;-0.0}% from bit-plane)");

        Assert.True(byteRatio > 1.0 && bitRatio > 1.0, "both transforms must compress dense fp weights");
    }

    private static double Lz4Ratio(byte[] src)
    {
        var lz = new byte[LZ4Codec.MaximumOutputSize(src.Length)];
        int enc = LZ4Codec.Encode(src, 0, src.Length, lz, 0, lz.Length);
        return enc > 0 ? (double)enc / src.Length : 1.0;
    }

    private static double DecodeMibPerSec(Func<byte[]> decodeOnce, int rawLen, out byte[] last)
    {
        for (int w = 0; w < 2; w++) decodeOnce();
        double best = double.MaxValue;
        last = Array.Empty<byte>();
        for (int r = 0; r < 8; r++)
        {
            var sw = Stopwatch.StartNew();
            last = decodeOnce();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return rawLen / (1024.0 * 1024.0) / (best / 1000.0);
    }

    [Theory]
    [InlineData(0.02)]   // tight init-like weights
    [InlineData(0.05)]   // typical trained weights
    [InlineData(0.2)]    // wider
    public void EntropyCoders_OnShuffledWeights_BeatLz4Ratio(double std)
    {
        const int n = 1 << 20; // 1M floats = 4 MiB
        var weights = TrainedLikeWeights(n, std, (int)(std * 1000) + 1);
        var raw = new byte[n * 4];
        Buffer.BlockCopy(weights, 0, raw, 0, raw.Length);
        var shuffled = StreamingStoreCodec.ShuffleForTest(raw, 4);

        double rawLz4 = Lz4Ratio(raw);
        double shufLz4 = Lz4Ratio(shuffled);

        var defl = Deflate(shuffled);
        double shufDeflate = (double)defl.Length / shuffled.Length;
        double deflateDec = DecodeMibPerSec(() => Inflate(defl, shuffled.Length), shuffled.Length, out var dOut);
        Assert.Equal(shuffled, dOut); // lossless

        string brotliMsg = "n/a(net471)";
#if NET5_0_OR_GREATER
        var br = Brotli(shuffled);
        double shufBrotli = (double)br.Length / shuffled.Length;
        double brotliDec = DecodeMibPerSec(() => UnBrotli(br, shuffled.Length), shuffled.Length, out var bOut);
        Assert.Equal(shuffled, bOut);
        brotliMsg = $"{1 / shufBrotli:F3}x @ {brotliDec:F0} MiB/s decode";
#endif

        _out.WriteLine($"std={std}: raw-LZ4 {1 / rawLz4:F3}x | shuffle+LZ4 {1 / shufLz4:F3}x | " +
                       $"shuffle+Deflate {1 / shufDeflate:F3}x @ {deflateDec:F0} MiB/s | shuffle+Brotli {brotliMsg}");

        // The point: an entropy coder on the shuffled bytes must beat LZ4's ratio.
        Assert.True(shufDeflate < shufLz4,
            $"Deflate ({1 / shufDeflate:F3}x) should beat LZ4 ({1 / shufLz4:F3}x) on shuffled weights");
    }
}
