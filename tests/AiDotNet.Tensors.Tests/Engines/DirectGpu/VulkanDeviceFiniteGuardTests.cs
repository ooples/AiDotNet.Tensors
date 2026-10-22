using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>#1521: Vulkan's device-side finiteness guard (the checks are in <see cref="DeviceFiniteGuardTestBase"/>).</summary>
[Collection("DirectGpuSerial")]
public sealed class VulkanDeviceFiniteGuardTests : DeviceFiniteGuardTestBase
{
    protected override BackendLease Lease()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        return new BackendLease(backend, owned: false);
    }

    protected override void Finish(IDirectGpuBackend backend, IGpuBuffer accumulator)
        => ((VulkanBackend)backend).FinishSumOfSquaresDouble(accumulator);
}
