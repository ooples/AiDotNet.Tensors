#if NET5_0_OR_GREATER
using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The order <see cref="GpuDriverCrossPlatformResolver"/> tries native library names in. The resolver runs before the
/// default search, so whatever it offers first wins: a CUDA 12 NVRTC on PATH (PyTorch's torch/lib) used to answer for
/// the NVRTC 13 binding, which then compiled without -default-device and took the CUDA backend down.
/// </summary>
public sealed class GpuDriverCrossPlatformResolverTests
{
    [Fact]
    public void Nvrtc_ExactVersionIsTriedBeforeAnyOtherVersion()
    {
        bool windows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        string newest = windows ? "nvrtc64_130_0" : "libnvrtc.so.13";
        string older = windows ? "nvrtc64_120_0" : "libnvrtc.so.12";

        var forNewest = GpuDriverCrossPlatformResolver.LoadOrder(newest);
        Assert.Equal(newest, forNewest[0]);
        Assert.Contains(older, forNewest);

        // The 12 binding still gets 12 first, and falls back to the other versions only when 12 is absent.
        var forOlder = GpuDriverCrossPlatformResolver.LoadOrder(older);
        Assert.Equal(older, forOlder[0]);
        Assert.Contains(newest, forOlder);
        Assert.Equal(forOlder.Length, new System.Collections.Generic.HashSet<string>(forOlder).Count);
    }

    [Fact]
    public void Nvrtc_CrossOsAliasOffersOnlyThisPlatformsNames()
    {
        bool windows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        // A Linux name asked for on Windows (or the reverse) is not a name this platform can load, so only aliases.
        string foreign = windows ? "libnvrtc.so.13" : "nvrtc64_130_0";
        var order = GpuDriverCrossPlatformResolver.LoadOrder(foreign);
        Assert.NotEmpty(order);
        Assert.DoesNotContain(foreign, order);
    }

    [Fact]
    public void OtherLibraries_LeaveTheRequestedNameToTheDefaultSearch()
    {
        bool windows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        string cublas = windows ? "cublas64_12" : "libcublas.so.12";
        Assert.DoesNotContain(cublas, GpuDriverCrossPlatformResolver.LoadOrder(cublas));
        Assert.Empty(GpuDriverCrossPlatformResolver.LoadOrder("not-a-gpu-library"));
    }
}
#endif
