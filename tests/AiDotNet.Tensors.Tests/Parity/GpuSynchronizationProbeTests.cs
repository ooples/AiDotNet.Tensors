using System;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Parity;

/// <summary>
/// Every host wait on a device is a residency-probe synchronization. The probe can only see the waits that go through a
/// counting wrapper, so the raw driver entry points must not be callable directly.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuSynchronizationProbeTests
{
    public static TheoryData<Type, string> BlockingWaits => new()
    {
        { typeof(CudaNativeBindings), "cuStreamSynchronize" },
        { typeof(CudaNativeBindings), "cuCtxSynchronize" },
        { typeof(CudaNativeBindings), "cuEventSynchronize" },
        { typeof(HipNativeBindings), "hipStreamSynchronize" },
        { typeof(HipNativeBindings), "hipDeviceSynchronize" },
        { typeof(HipNativeBindings), "hipEventSynchronize" },
        { typeof(OpenClNativeBindings), "clFinish" },
        { typeof(OpenClNativeBindings), "clWaitForEvents" },
        { typeof(VulkanNativeBindings), "vkDeviceWaitIdle" },
        { typeof(VulkanNativeBindings), "vkWaitForFences" },
    };

    [Theory]
    [MemberData(nameof(BlockingWaits))]
    public void BlockingDriverWait_IsReachableOnlyThroughItsCountingWrapper(Type bindings, string entryPoint)
    {
        var imports = bindings
            .GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(m => m.GetCustomAttribute<DllImportAttribute>() is { } import
                        && (import.EntryPoint ?? m.Name) == entryPoint)
            .ToList();

        Assert.NotEmpty(imports);
        Assert.All(imports, m => Assert.False(m.IsPublic || m.IsAssembly,
            $"{bindings.Name}.{m.Name} imports {entryPoint} directly and is callable, so its waits bypass the probe."));
    }

    [SkippableFact]
    public void OpenClSynchronize_CountsOneSynchronization()
    {
        using var backend = new OpenClBackend();
        Skip.If(!backend.IsAvailable, "needs an OpenCL device.");
        using var scope = GpuResidencyScope.Begin();
        backend.Synchronize();
        Assert.Equal(1, scope.Synchronizations);
        Assert.All(scope.Events, e => Assert.Equal(GpuBackendType.OpenCl, e.Backend));
    }

    [SkippableFact]
    public void VulkanSynchronize_CountsOneSynchronization()
    {
        var backend = VulkanBackend.Instance;
        Skip.If(!backend.Initialize(), "needs a Vulkan device.");
        using var scope = GpuResidencyScope.Begin();
        backend.Synchronize();
        Assert.Equal(1, scope.Synchronizations);
        Assert.All(scope.Events, e => Assert.Equal(GpuBackendType.Vulkan, e.Backend));
    }
}
