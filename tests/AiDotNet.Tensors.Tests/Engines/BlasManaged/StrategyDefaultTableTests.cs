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
}
