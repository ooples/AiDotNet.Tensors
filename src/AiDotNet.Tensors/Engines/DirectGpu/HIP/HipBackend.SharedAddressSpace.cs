namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IGpuSharedAddressSpace
{
    // HIP buffers are per device too, but this backend records no device or context identity to prove two instances
    // share one; until it does, only the same instance qualifies and a buffer another HipBackend placed is migrated.
    bool IGpuSharedAddressSpace.SharesAddressSpaceWith(IDirectGpuBackend other) => ReferenceEquals(this, other);
}
