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
