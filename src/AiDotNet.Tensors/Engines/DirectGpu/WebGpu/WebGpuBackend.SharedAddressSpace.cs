namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IGpuSharedAddressSpace
{
    // A GPUBuffer belongs to the GPUDevice that created it; each WebGpuBackend owns its own device.
    bool IGpuSharedAddressSpace.SharesAddressSpaceWith(IDirectGpuBackend other) => ReferenceEquals(this, other);
}
