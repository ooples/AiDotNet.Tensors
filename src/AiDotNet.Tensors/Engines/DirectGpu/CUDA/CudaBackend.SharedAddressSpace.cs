namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IGpuSharedAddressSpace
{
    // Backends on one device share its primary context (and DrainPeerPools already treats them as peers): a device
    // pointer one allocated is a valid address in the other, as long as the context is the same live one.
    bool IGpuSharedAddressSpace.SharesAddressSpaceWith(IDirectGpuBackend other)
        => other is CudaBackend peer && !_disposed && !peer._disposed
           && _cudaContext != System.IntPtr.Zero && peer._cudaContext == _cudaContext && ContextIsLive && peer.ContextIsLive;
}
