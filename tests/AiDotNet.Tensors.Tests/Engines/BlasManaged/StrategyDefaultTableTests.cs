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
        Assert.True(key.CpuBucket >= 0 && key.CpuBucket <= 2);
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
