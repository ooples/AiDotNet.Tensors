#if NET5_0_OR_GREATER
using System;
using System.Reflection;
using System.Runtime.InteropServices;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1027: HIP's DLL-search setup used to call SetDefaultDllDirectories, which is process-wide and
/// sticky. On a machine with ROCm installed, a plain-name load of the Vulkan loader then failed for
/// the rest of the process, so whichever ran first, a HIP test or a Vulkan test, decided whether
/// every Vulkan test ran or reported "unavailable".
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class HipDllSearchPathIsolationTests
{
    [SkippableFact]
    public void HipDllSearchSetup_LeavesTheVulkanLoaderLoadable()
    {
        Skip.IfNot(OperatingSystem.IsWindows(), "SetDefaultDllDirectories is a Windows concern.");
        Skip.IfNot(NativeLibrary.TryLoad("vulkan-1", out var before), "No Vulkan loader on this machine.");
        NativeLibrary.Free(before);

        var hip = typeof(AiDotNet.Tensors.Engines.CpuEngine).Assembly
            .GetType("AiDotNet.Tensors.Engines.DirectGpu.HIP.HipNativeBindings");
        var init = hip?.GetMethod("InitializeDllSearchPath", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(init);
        init.Invoke(null, null);

        bool after = NativeLibrary.TryLoad("vulkan-1", out var handle);
        if (after) NativeLibrary.Free(handle);
        Assert.True(after, "vulkan-1 stopped loading after HIP set up its DLL search path.");
    }
}
#endif
