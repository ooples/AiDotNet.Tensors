namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : IGpuSharedAddressSpace
{
    // A cl_mem belongs to the cl_context that created it; each OpenClBackend owns its own context.
    bool IGpuSharedAddressSpace.SharesAddressSpaceWith(IDirectGpuBackend other) => ReferenceEquals(this, other);
}
