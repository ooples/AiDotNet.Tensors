namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IGpuSharedAddressSpace
{
    // Backends on one device share its primary context (DrainPeerPools treats them as peers), so a device pointer one
    // allocated is a valid address in the other. Validity is not enough, though: each backend orders its work on its
    // own stream, and a buffer written asynchronously on one stream handed to kernels on another, with no event
    // between them, can be read before the write lands, or recycled by the owner's pool while still in use. Only a
    // backend on the same live context AND the same stream may take the buffer as-is; any other is migrated, which
    // reads back through the owner and so synchronizes with it.
    bool IGpuSharedAddressSpace.SharesAddressSpaceWith(IDirectGpuBackend other)
        => other is CudaBackend peer && !_disposed && !peer._disposed
           && _cudaContext != System.IntPtr.Zero && peer._cudaContext == _cudaContext && ContextIsLive && peer.ContextIsLive
           && _mainStream != System.IntPtr.Zero && peer._mainStream == _mainStream;
}
