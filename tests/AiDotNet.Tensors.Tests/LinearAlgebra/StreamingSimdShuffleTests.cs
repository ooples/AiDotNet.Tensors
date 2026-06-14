// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// The SIMD byte-plane shuffle that makes lossless cheap enough to be a default. Two things
/// must hold: (1) the shuffle/unshuffle are an exact INVERSE PAIR — at sizes that hit the
/// AVX2 body, the scalar tail, AND the body+tail mix — so the backing-file round-trip is
/// bit-exact (the only contract that matters; the shuffled layout is an internal, ephemeral
/// format read back in the same process); (2) the transform is fast enough that it stops
/// being the bottleneck (the naive whole-array strided pass was ~256 MiB/s). The throughput
/// test prints the measured rate so the default decision is data-driven, not assumed.
/// </summary>
public class StreamingSimdShuffleTests
{
    private readonly ITestOutputHelper _out;
    public StreamingSimdShuffleTests(ITestOutputHelper output) => _out = output;

    private static byte[] RandomBytes(int n, int seed)
    {
        var b = new byte[n];
        uint s = (uint)(seed * 2654435761u + 1u);
        for (int i = 0; i < n; i++) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; b[i] = (byte)s; }
        return b;
    }

    [Theory]
    // Sizes around the 32-element AVX2 block boundary: pure body, body+tail, tail-only.
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(63)]
    [InlineData(95)]
    [InlineData(1000)]
    [InlineData(4096)]
    [InlineData(4097)]
    [InlineData(100000)]
    public void Shuffle4_SimdMatchesScalar_AndRoundTrips_AcrossBodyAndTail(int count)
    {
        var raw = RandomBytes(count * 4, count);

        // (1) SIMD output is bit-identical to the scalar reference (so the SIMD prefix +
        // scalar tail compose into one consistent, portable layout).
        var simd = StreamingStoreCodec.ShuffleForTest(raw, 4);
        var scalar = StreamingStoreCodec.ShuffleScalarForTest(raw, 4);
        Assert.Equal(scalar, simd);

        // (2) Unshuffle reproduces the input byte for byte.
        var round = StreamingStoreCodec.UnshuffleForTest(simd, 4);
        Assert.Equal(raw, round);
    }

    [Fact]
    public void Shuffle4_GroupsBytesByPlane()
    {
        // Elements whose 4 bytes are (i, i+64, i+128, i+192): each plane must come out a
        // known, contiguous run — a layout bug would scramble these.
        const int count = 64;
        var raw = new byte[count * 4];
        for (int i = 0; i < count; i++)
        {
            raw[i * 4 + 0] = (byte)i;
            raw[i * 4 + 1] = (byte)(i + 64);
            raw[i * 4 + 2] = (byte)(i + 128);
            raw[i * 4 + 3] = (byte)(i + 192);
        }
        var shuffled = StreamingStoreCodec.ShuffleForTest(raw, 4);
        for (int i = 0; i < count; i++)
        {
            Assert.Equal((byte)i, shuffled[0 * count + i]);
            Assert.Equal((byte)(i + 64), shuffled[1 * count + i]);
            Assert.Equal((byte)(i + 128), shuffled[2 * count + i]);
            Assert.Equal((byte)(i + 192), shuffled[3 * count + i]);
        }
    }

    [Fact]
    public void Lossless_FullCodec_RoundTripsExactly_OnSimdSizedFloats()
    {
        // Exercise the SIMD shuffle through the real encode→LZ4→decode→unshuffle path at a
        // size well into the vectorized regime.
        const int n = 50_000;
        var src = new float[n];
        uint s = 12345;
        for (int i = 0; i < n; i++) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; src[i] = (float)((int)s) * 1e-6f; }

        var enc = StreamingStoreCodec.EncodeLosslessFloat(src);
        var dec = new float[n];
        StreamingStoreCodec.DecodeLosslessFloat(enc, dec);
        for (int i = 0; i < n; i++)
            Assert.Equal(BitExactHelpers.SingleBits(src[i]), BitExactHelpers.SingleBits(dec[i]));
    }

#if NET5_0_OR_GREATER
    // The slow "naive" version this replaced: a single whole-array strided pass (no SIMD, no
    // tiling). Measured in-test so the throughput comparison is a RATIO on the SAME CPU/run —
    // runner-independent, not a fixed MiB/s bar that flakes on slow/shared CI VMs.
    private static void NaiveShuffle(byte[] src, byte[] dst, int elem)
    {
        int count = src.Length / elem;
        for (int k = 0; k < elem; k++)
            for (int i = 0; i < count; i++)
                dst[k * count + i] = src[i * elem + k];
    }

    [Fact]
    public void Shuffle4_Throughput_BeatsNaiveBottleneck()
    {
        bool ssse3 = System.Runtime.Intrinsics.X86.Ssse3.IsSupported;
        // 16 MiB of fp32 — large enough to be memory-bound, the realistic regime.
        const int count = 4 * 1024 * 1024;
        var raw = RandomBytes(count * 4, 7);

        double MeasureMiBps(Func<byte[]> shuffle)
        {
            for (int w = 0; w < 3; w++) { var _ = shuffle(); }
            double best = double.MaxValue;
            for (int r = 0; r < 15; r++)
            {
                var sw = Stopwatch.StartNew();
                var s = shuffle();
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
                GC.KeepAlive(s);
            }
            return raw.Length / (1024.0 * 1024.0) / (best / 1000.0);
        }

        double simdMiB = MeasureMiBps(() => StreamingStoreCodec.ShuffleForTest(raw, 4));
        double naiveMiB = MeasureMiBps(() => { var d = new byte[raw.Length]; NaiveShuffle(raw, d, 4); return d; });
        _out.WriteLine($"byte-shuffle ({(ssse3 ? "SSSE3" : "tiled-scalar")}) {simdMiB:F0} MiB/s vs naive {naiveMiB:F0} MiB/s " +
                       $"= {simdMiB / naiveMiB:F1}x");

        // The SSSE3 path is the SIMD claim: it must beat the naive whole-array pass by a clear
        // margin. A RATIO (both timed on this run's CPU) is runner-independent — unlike an
        // absolute MiB/s bar, which flakes on slow/shared CI VMs (this test once asserted
        // >1 GiB/s and failed at 614 MiB/s on a CI runner where naive was proportionally slower
        // too). The tiled-scalar fallback (no SSSE3) is correctness-only here.
        if (ssse3)
        {
            Assert.True(simdMiB > naiveMiB * 1.5,
                $"SSSE3 shuffle ({simdMiB:F0} MiB/s) should be >1.5x the naive whole-array pass ({naiveMiB:F0} MiB/s).");
        }
    }
#endif
}
