using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class StrategyDefaultTableTests
{
    [Fact]
    public void HardwareKey_Exposes_Simd_Vendor_CpuBucket()
    {
        var key = HardwareFingerprint.Key;
        Assert.False(string.IsNullOrEmpty(key.Simd));
        Assert.False(string.IsNullOrEmpty(key.Vendor));
        // Upper bound raised 2 -> 3 in PR #1034: band 2 used to mean "everything above 16
        // threads", which lumped the 32-thread Ryzen its routing was calibrated on together
        // with 64+-thread parts that measure differently. See BucketFor.
        Assert.True(key.CpuBucket >= 0 && key.CpuBucket <= 3);
    }

    [Fact]
    public void CpuBucket_SeparatesVeryWideMachinesFromTheRyzenBand()
    {
        Assert.Equal(0, HardwareFingerprint.BucketFor(4));
        Assert.Equal(1, HardwareFingerprint.BucketFor(16));
        Assert.Equal(2, HardwareFingerprint.BucketFor(32));
        Assert.Equal(2, HardwareFingerprint.BucketFor(64));
        Assert.Equal(3, HardwareFingerprint.BucketFor(65));
        Assert.Equal(3, HardwareFingerprint.BucketFor(128));
    }

    [Theory]
    // AVX-512 thin-K A^T products (the linear layers' weight gradients) go to PackBoth: its AVX-512 tile ran them 3.4-4.8x
    // faster than PackAOnly's AVX2 8x8 strided-B kernel on Zen 4 and Granite Rapids runners (3136x128x64: 0.16-0.29 against
    // 0.73-1.04 ms), and PackAOnly left the CNN's classifier weight gradient 2.5-6x behind PyTorch.
    [InlineData(3136, 128, 64, "amd")]
    [InlineData(784, 512, 128, "intel")]
    [InlineData(512, 256, 128, "amd")]
    public void Route_Avx512_ThinK_PrefersPackBoth(int m, int n, int k, string vendor)
    {
        Assert.Equal(StrategyDefaultTable.ShapeBucket.ThinK, StrategyDefaultTable.Bucket(m, n, k));
        var key = new HardwareFingerprint.HwKey("avx512", vendor, 0);
        Assert.Equal(PackingMode.ForcePackBoth, StrategyDefaultTable.Route(key, m, n, k, transA: true, transB: false));
    }

    [Theory]
    // Band 3 (>64T) avx2 with a TRANSPOSED B routes to Streaming. Measured on
    // x64-amd-avx2-cpu128: Streaming won all 17 transposed shapes swept, by 1.12x-3.22x.
    [InlineData(512, 512, 64)]
    [InlineData(128, 128, 128)]
    [InlineData(48, 1024, 256)]
    [InlineData(96, 1024, 512)]
    [InlineData(128, 768, 768)]
    [InlineData(49, 512, 512)]
    public void Route_VeryWideAvx2_TransposedB_PrefersStreaming(int m, int n, int k)
    {
        var key = new HardwareFingerprint.HwKey("avx2", "amd", 3);
        Assert.Equal(
            PackingMode.ForceStreaming,
            StrategyDefaultTable.Route(key, m, n, k, transA: false, transB: true));
    }

    [Theory]
    // Past the sweep's measured range (1.07G work) the band-3 transposed-B rule does not apply: the
    // LM-head products of a 49152-token vocabulary ran 5-6.5x faster on PackBoth than on Streaming on a
    // 128-thread 3990X (1331 vs 266 ms, 1507 vs 232 ms).
    [InlineData(512, 256, 49152)]   // fused linear-CE dHidden = dLogits . W^T
    [InlineData(512, 49152, 256)]   // logits = h . table^T
    [InlineData(510, 256, 49152)]
    public void Route_VeryWideAvx2_TransposedB_BeyondMeasuredWork_PacksBoth(int m, int n, int k)
    {
        Assert.True((long)m * n * k > StrategyDefaultTable.VeryWideTransBStreamingMaxWork);
        var key = new HardwareFingerprint.HwKey("avx2", "amd", 3);
        Assert.Equal(
            PackingMode.ForcePackBoth,
            StrategyDefaultTable.Route(key, m, n, k, transA: false, transB: true));
    }

    [Fact]
    public void Route_VeryWideAvx2_TransposedB_MeasuredWorkBoundary_StillStreams()
    {
        // 1024 x 1024 x 1024 = 2^30, exactly the cap: still inside the measured range.
        var key = new HardwareFingerprint.HwKey("avx2", "amd", 3);
        Assert.Equal(PackingMode.ForceStreaming, StrategyDefaultTable.Route(key, 1024, 1024, 1024, transA: false, transB: true));
    }

    [Theory]
    // The UNTRANSPOSED routing for band 3 is deliberately identical to band 2. The sweep
    // showed untransposed optima differ and sometimes oppose the transposed ones
    // (512x2048x512 untransposed wants PackBoth; Streaming is ~3.0x slower there).
    [InlineData(512, 2048, 512)]
    [InlineData(256, 256, 256)]
    [InlineData(128, 128, 128)]
    [InlineData(512, 512, 64)]
    public void Route_VeryWideAvx2_Untransposed_IsUnchangedFromLargeBand(int m, int n, int k)
    {
        var band2 = new HardwareFingerprint.HwKey("avx2", "amd", 2);
        var band3 = new HardwareFingerprint.HwKey("avx2", "amd", 3);
        Assert.Equal(
            StrategyDefaultTable.Route(band2, m, n, k),
            StrategyDefaultTable.Route(band3, m, n, k));
    }

    [Fact]
    public void CpuBucket_Bands_16_And_32_Differ()
    {
        // The motivating collision is amd-avx2-cpu16 vs amd-avx2-cpu32; the bucket
        // MUST separate them (G1) or the table can't resolve it.
        Assert.NotEqual(HardwareFingerprint.BucketFor(16), HardwareFingerprint.BucketFor(32));
    }

    [Theory]
    // amd-avx2-cpu16 (this box): k≤128 shapes won on Streaming in the A/B.
    [InlineData("avx2", "amd", 1, 128, 128, 128, PackingMode.ForceStreaming)]
    [InlineData("avx2", "amd", 1, 96, 128, 64, PackingMode.ForceStreaming)]
    // amd-avx2-cpu32 (Ryzen, #464): blocking won on the medium square.
    [InlineData("avx2", "amd", 2, 128, 128, 128, PackingMode.ForcePackBoth)]
    // Large compute-bound: PackBoth everywhere.
    [InlineData("avx2", "amd", 1, 1024, 3072, 768, PackingMode.ForcePackBoth)]
    public void Route_ReturnsExpectedStrategy(string simd, string vendor, int bucket,
        int m, int n, int k, PackingMode expected)
    {
        var key = new HardwareFingerprint.HwKey(simd, vendor, bucket);
        Assert.Equal(expected, StrategyDefaultTable.Route(key, m, n, k));
    }

    [Fact]
    public void Route_UnknownKey_FallsBackToConservativeDefault()
    {
        // Unknown vendor on a known simd → never throws, returns a valid strategy.
        var key = new HardwareFingerprint.HwKey("avx2", "totally-unknown", 1);
        var mode = StrategyDefaultTable.Route(key, 128, 128, 128);
        Assert.True(mode is PackingMode.ForceStreaming or PackingMode.ForcePackBoth
            or PackingMode.ForcePackAOnly);
    }
}
