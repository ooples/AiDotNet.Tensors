namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IGpuSharedAddressSpace
{
    // A VkBuffer and its memory belong to the VkDevice that created them; each VulkanBackend owns its own device.
    bool IGpuSharedAddressSpace.SharesAddressSpaceWith(IDirectGpuBackend other) => ReferenceEquals(this, other);
}
