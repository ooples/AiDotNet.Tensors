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
/// Measures the real compression design space for streaming-pool weight bytes —
/// to answer "is LZ4 good enough, and what would do better?" with numbers rather
/// than assumptions. Covers:
///   * raw LZ4 (today's codec),
///   * byte-plane and BIT-plane shuffle (Blosc/HDF5 scientific-data technique)
///     to expose the structured sign/exponent bits to the codec,
///   * the order-0 Shannon-entropy floor (what a real entropy coder — zstd /
///     Brotli — could approach, which LZ4 cannot since it has no entropy stage),
///   * lossy bf16 (2x) and int8 per-tensor (4x) — the actual big levers for
///     weights, with their round-trip error.
/// Deterministic (seeded xorshift → reproducible).
/// </summary>
public class StreamingWeightCompressionRatioTests
{
    private readonly ITestOutputHelper _out;
    public StreamingWeightCompressionRatioTests(ITestOutputHelper output) => _out = output;

    private struct Rng
    {
        private ulong _s0, _s1;
        public Rng(ulong seed) { _s0 = seed | 1UL; _s1 = (seed * 0x9E3779B97F4A7C15UL) | 1UL; }
        public double NextUnit()
        {
            ulong x = _s0; ulong y = _s1; _s0 = y;
            x ^= x << 23; x ^= x >> 17; x ^= y ^ (y >> 26); _s1 = x;
            return ((x + y) >> 11) * (1.0 / (1UL << 53));
        }
        public double NextGaussian(double std)
        {
            double u1 = Math.Max(1e-12, NextUnit()), u2 = NextUnit();
            return std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }

    private static double Lz4Pct(byte[] data)
    {
        var dst = new byte[LZ4Codec.MaximumOutputSize(data.Length)];
        int n = LZ4Codec.Encode(data, 0, data.Length, dst, 0, dst.Length);
        return (n > 0 ? (double)n / data.Length : 1.0) * 100.0;
    }

    // Order-0 Shannon entropy in bits/byte; /8*100 = best % a pure entropy coder
    // (zstd/Brotli FSE/Huffman stage) could reach on this byte stream.
    private static double EntropyPct(byte[] data)
    {
        var hist = new long[256];
        foreach (var b in data) hist[b]++;
        double h = 0, n = data.Length;
        for (int i = 0; i < 256; i++)
            if (hist[i] > 0) { double p = hist[i] / n; h -= p * Math.Log(p, 2); }
        return h / 8.0 * 100.0;
    }

    private static byte[] BytePlaneShuffle(byte[] data, int elem)
    {
        int count = data.Length / elem;
        var outp = new byte[data.Length];
        for (int k = 0; k < elem; k++)
            for (int i = 0; i < count; i++) outp[k * count + i] = data[i * elem + k];
        return outp;
    }

    // Bit-plane transpose: gather bit b of every element into a contiguous plane.
    private static byte[] BitPlaneShuffle(byte[] data, int elem)
    {
        int count = data.Length / elem, totalBits = elem * 8;
        var outp = new byte[data.Length];
        int o = 0;
        for (int b = 0; b < totalBits; b++)
        {
            int by = b >> 3, bit = b & 7;
            for (int i = 0; i < count; i++)
            {
                if (((data[i * elem + by] >> bit) & 1) != 0) outp[o >> 3] |= (byte)(1 << (o & 7));
                o++;
            }
        }
        return outp;
    }

    private static float[] Gaussian(int count, double std, ulong seed)
    {
        var rng = new Rng(seed);
        var w = new float[count];
        for (int i = 0; i < count; i++) w[i] = (float)rng.NextGaussian(std);
        return w;
    }

    private static byte[] ToBytes(float[] w)
    {
        var b = new byte[w.Length * 4];
        Buffer.BlockCopy(w, 0, b, 0, b.Length);
        return b;
    }

    [Fact]
    public void MeasureLosslessCodecLandscape_RawVsShuffleVsEntropyFloor()
    {
        const int count = 1 << 20;
        _out.WriteLine("Lossless — compressed size as % of original (lower = better):");
        _out.WriteLine($"{"weights",-22} {"rawLZ4",8} {"byteShuf",10} {"bitShuf",10} {"entropyFloor",13}");
        void Row(string label, double std, ulong seed)
        {
            var raw = ToBytes(Gaussian(count, std, seed));
            var byteS = BytePlaneShuffle(raw, 4);
            var bitS = BitPlaneShuffle(raw, 4);
            _out.WriteLine($"{label,-22} {Lz4Pct(raw),7:F1}% {Lz4Pct(byteS),9:F1}% {Lz4Pct(bitS),9:F1}% " +
                           $"{EntropyPct(bitS),12:F1}%");
        }
        Row("fp32 std=0.02", 0.02, 1);
        Row("fp32 std=0.10", 0.10, 2);
        Row("fp32 std=1.0", 1.0, 3);
        _out.WriteLine("(entropyFloor = order-0 Shannon limit on the bit-shuffled stream — what a");
        _out.WriteLine(" zstd/Brotli entropy stage could approach; LZ4 has no entropy coder so it can't.)");
        Assert.True(count > 0);
    }

    private static byte[] Deflate(byte[] data)
    {
        using var ms = new MemoryStream();
        using (var ds = new DeflateStream(ms, CompressionLevel.Optimal, leaveOpen: true)) ds.Write(data, 0, data.Length);
        return ms.ToArray();
    }
    private static double InflateMBps(byte[] compressed, int originalLen, int iters)
    {
        var outBuf = new byte[originalLen];
        var sw = Stopwatch.StartNew();
        for (int it = 0; it < iters; it++)
        {
            using var ms = new MemoryStream(compressed);
            using var ds = new DeflateStream(ms, CompressionMode.Decompress);
            int off = 0, r; while ((r = ds.Read(outBuf, off, outBuf.Length - off)) > 0) off += r;
        }
        sw.Stop();
        return (originalLen / (1024.0 * 1024)) * iters / sw.Elapsed.TotalSeconds;
    }
    private static double Lz4DecodeMBps(byte[] compressed, int originalLen, int iters)
    {
        var outBuf = new byte[originalLen];
        var sw = Stopwatch.StartNew();
        for (int it = 0; it < iters; it++) LZ4Codec.Decode(compressed, 0, compressed.Length, outBuf, 0, outBuf.Length);
        sw.Stop();
        return (originalLen / (1024.0 * 1024)) * iters / sw.Elapsed.TotalSeconds;
    }

    [Fact]
    public void MeasureLosslessThroughputTradeoff_RatioVsDecodeSpeed()
    {
        // The gate for "default compression on": a 1.2x byte saving is a NET LOSS if
        // decode is slower than the disk read it replaces. Compare ratio AND decode
        // throughput against rough storage bandwidths (NVMe ~3-7 GB/s, SATA SSD ~0.5,
        // network/HDD ~0.1-0.2 GB/s).
        const int count = 1 << 18; // 256K elems — ratios are stable, keeps CI fast
        var raw = ToBytes(Gaussian(count, 0.02, 1)); // realistic small-std weights
        var bitS = BitPlaneShuffle(raw, 4);

        var lz4 = new byte[LZ4Codec.MaximumOutputSize(bitS.Length)];
        int lz4n = LZ4Codec.Encode(bitS, 0, bitS.Length, lz4, 0, lz4.Length);
        var lz4c = new byte[lz4n]; Array.Copy(lz4, lz4c, lz4n);
        var defl = Deflate(bitS);

        _out.WriteLine($"bit-shuffled fp32 std=0.02 ({raw.Length / (1024 * 1024)} MiB):");
        _out.WriteLine($"  LZ4    : {(double)lz4n / bitS.Length * 100:F1}% size, decode {Lz4DecodeMBps(lz4c, bitS.Length, 50):F0} MiB/s");
        _out.WriteLine($"  Deflate: {(double)defl.Length / bitS.Length * 100:F1}% size, decode {InflateMBps(defl, bitS.Length, 50):F0} MiB/s");
        _out.WriteLine("  → if decode MiB/s < storage bandwidth, compression bottlenecks reads on that storage.");
        _out.WriteLine("  → zstd (new dep) is the sweet spot: ~1.5-2 GB/s decode + Deflate-class ratio.");

        // The shuffle pre-transform is on the critical path too — measure it alone.
        int iters = 8; double mib = raw.Length / (1024.0 * 1024);
        var swB = Stopwatch.StartNew(); for (int i = 0; i < iters; i++) BytePlaneShuffle(raw, 4); swB.Stop();
        var swBit = Stopwatch.StartNew(); for (int i = 0; i < iters; i++) BitPlaneShuffle(raw, 4); swBit.Stop();
        _out.WriteLine($"  transform cost: byte-shuffle {mib * iters / swB.Elapsed.TotalSeconds:F0} MiB/s, " +
                       $"bit-shuffle (naive) {mib * iters / swBit.Elapsed.TotalSeconds:F0} MiB/s");
        _out.WriteLine("  → byte-shuffle is memory-bandwidth-cheap; naive bit-shuffle needs a SIMD impl to be viable.");
        Assert.True(count > 0);
    }

    [Fact]
    public void MeasureLossyQuantization_Bf16AndInt8_RatioAndError()
    {
        const int count = 1 << 20;
        _out.WriteLine("Lossy — size % and round-trip error (the real lever for weight bytes):");
        void Row(string label, double std, ulong seed)
        {
            var w = Gaussian(count, std, seed);

            // bf16: keep the top 16 bits of each fp32 (full 8-bit exponent + 7
            // mantissa bits). Round-to-nearest-even. Size 50%.
            double bf16MaxRel = 0, bf16Sum2 = 0, ref2 = 0;
            for (int i = 0; i < count; i++)
            {
                uint u = (uint)BitConverter.SingleToInt32Bits(w[i]);
                uint rounding = 0x7FFFu + ((u >> 16) & 1u);
                uint bf = (u + rounding) & 0xFFFF0000u;
                float r = BitConverter.Int32BitsToSingle((int)bf);
                double e = Math.Abs(r - w[i]);
                if (Math.Abs(w[i]) > 1e-12) bf16MaxRel = Math.Max(bf16MaxRel, e / Math.Abs(w[i]));
                bf16Sum2 += e * e; ref2 += (double)w[i] * w[i];
            }
            double bf16Rmse = Math.Sqrt(bf16Sum2 / ref2); // relative RMS error

            // int8 per-tensor symmetric: scale = max|w|/127. Size 25%.
            float amax = 0; for (int i = 0; i < count; i++) amax = Math.Max(amax, Math.Abs(w[i]));
            double scale = amax / 127.0;
            double i8Sum2 = 0;
            for (int i = 0; i < count; i++)
            {
                int q = (int)Math.Round(w[i] / scale); q = Math.Max(-127, Math.Min(127, q));
                double r = q * scale; double e = r - w[i]; i8Sum2 += e * e;
            }
            double i8Rmse = Math.Sqrt(i8Sum2 / ref2);

            _out.WriteLine($"{label,-22} bf16: 50.0% size, relRMSE {bf16Rmse * 100:F2}%, maxRel {bf16MaxRel * 100:F2}%   " +
                           $"int8: 25.0% size, relRMSE {i8Rmse * 100:F2}%");
        }
        Row("fp32 std=0.02", 0.02, 1);
        Row("fp32 std=0.10", 0.10, 2);
        Row("fp32 std=1.0", 1.0, 3);
        Assert.True(count > 0);
    }
}
