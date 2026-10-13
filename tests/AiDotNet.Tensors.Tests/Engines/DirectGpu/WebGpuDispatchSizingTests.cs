#if NET7_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// WebGPU dispatch sizing (PR #1231 review): work that needs more workgroups than the device dispatches per dimension
/// must be refused, not clamped. Every WebGPU shader covers one element per invocation, so a clamped dispatch leaves the
/// tail of its output uncomputed. Uninitialized, the device reports WebGPU's default limit of 65,535 per dimension.
/// </summary>
public sealed class WebGpuDispatchSizingTests
{
    [Fact]
    public void OneDimensionalDispatch_FitsUpToTheLimit_AndRefusesBeyondIt()
    {
        var device = WebGpuDevice.Instance;
        int limit = device.MaxWorkgroupsPerDimension;
        Assert.Equal((limit, 256), device.CalculateWorkgroups1D(limit * 256));
        Assert.Equal((1, 256), device.CalculateWorkgroups1D(1));
        var ex = Assert.Throws<GpuCapabilityUnavailableException>(() => device.CalculateWorkgroups1D(limit * 256 + 1));
        Assert.Contains($"at most {limit}", ex.Message);
    }

    [Fact]
    public void TwoDimensionalDispatch_RefusesAnAxisBeyondTheLimit()
    {
        var device = WebGpuDevice.Instance;
        var (x, y, sx, sy) = device.CalculateWorkgroups2D(64, 64);
        Assert.True(x * sx >= 64 && y * sy >= 64);
        Assert.Throws<GpuCapabilityUnavailableException>(() => device.CalculateWorkgroups2D(int.MaxValue / 2, 1));
    }
}
#endif