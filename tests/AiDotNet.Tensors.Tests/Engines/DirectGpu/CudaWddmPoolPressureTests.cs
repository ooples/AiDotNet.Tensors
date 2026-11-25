using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class CudaWddmPoolPressureTests
{
    private const ulong GiB = 1UL << 30;
    private const ulong MiB = 1UL << 20;

    [Fact]
    public void ComputeDefaultMemPoolReleaseThreshold_TccOrLinux_KeepsUnboundedThreshold()
    {
        ulong threshold = CudaBackend.ComputeDefaultMemPoolReleaseThreshold(
            totalBytes: 24 * GiB,
            freeBytes: 8 * GiB,
            isWddm: false,
            envOverrideMb: null);

        Assert.Equal(ulong.MaxValue, threshold);
    }

    [Fact]
    public void ComputeDefaultMemPoolReleaseThreshold_WddmSharedDevice_CapsToHalfFreeVram()
    {
        // Shared 24 GiB RTX 3090 with 8 GiB free VRAM must cap async mempool retention at 4 GiB
        // so stream-ordered pool caching cannot cross Windows WDDM's shared-memory paging cliff.
        ulong threshold = CudaBackend.ComputeDefaultMemPoolReleaseThreshold(
            totalBytes: 24 * GiB,
            freeBytes: 8 * GiB,
            isWddm: true,
            envOverrideMb: null);

        Assert.Equal(4 * GiB, threshold);
    }

    [Fact]
    public void ComputeDefaultMemPoolReleaseThreshold_WddmLowHeadroom_Respects256MiBFloorCappedByTotal()
    {
        ulong threshold = CudaBackend.ComputeDefaultMemPoolReleaseThreshold(
            totalBytes: 12 * GiB,
            freeBytes: 128 * MiB,
            isWddm: true,
            envOverrideMb: null);

        Assert.Equal(256 * MiB, threshold);
    }

    [Theory]
    [InlineData("0", 0UL)]
    [InlineData("2048", 2048UL * MiB)]
    [InlineData("-1", ulong.MaxValue)]
    public void ComputeDefaultMemPoolReleaseThreshold_EnvOverride_HonoredOnAllDrivers(string envMb, ulong expected)
    {
        Assert.Equal(expected, CudaBackend.ComputeDefaultMemPoolReleaseThreshold(24 * GiB, 8 * GiB, isWddm: true, envMb));
        Assert.Equal(expected, CudaBackend.ComputeDefaultMemPoolReleaseThreshold(24 * GiB, 8 * GiB, isWddm: false, envMb));
    }

    [Fact]
    public void ComputeDefaultWddmHostPoolMaxElements_ScalesDownWhenDeviceIsShared()
    {
        long defaultMaxElements = (long)(24 * GiB / 6) / sizeof(float); // 4 GiB host pool on 24 GiB card
        long capped = CudaBackend.ComputeDefaultWddmHostPoolMaxElements(defaultMaxElements, freeBytes: 8 * GiB);

        // 8 GiB / 4 = 2 GiB cap (< 4 GiB default)
        Assert.Equal((long)(2 * GiB) / sizeof(float), capped);
    }
}
