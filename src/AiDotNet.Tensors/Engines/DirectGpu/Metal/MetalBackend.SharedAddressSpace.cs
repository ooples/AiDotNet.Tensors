namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IGpuSharedAddressSpace
{
    // An MTLBuffer is usable only through the MTLDevice/command queue pair of the backend that created it.
    bool IGpuSharedAddressSpace.SharesAddressSpaceWith(IDirectGpuBackend other) => ReferenceEquals(this, other);
}
