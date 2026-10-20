using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>#1450: the resident program on Vulkan, every stage in one command buffer and one submission.</summary>
[Collection("DirectGpuSerial")]
public sealed class VulkanResidentProgramTests : ResidentProgramTestBase<VulkanBackend>
{
    private protected override VulkanBackend RequireBackend()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable && backend.SupportsResidentProgram,
            "Vulkan with a GLSL compiler (and the resident program shader) is not available.");
        return backend;
    }

    private protected override int RunResident(VulkanBackend backend, ResidentStage[] stages) => backend.RunResidentProgram(stages);

    private protected override void RunSeparately(VulkanBackend backend, ResidentStage[] stages) => backend.RunStagesSeparately(stages);

    // The backend is a process-wide singleton: disposing it would break every later Vulkan test.
    private protected override void ReleaseBackend(VulkanBackend backend) { }
}
