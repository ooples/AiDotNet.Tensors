namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A backend whose device buffers are valid addresses for some OTHER backend instance too: two CUDA backends on the
/// same live primary context AND the same stream may pass buffers to each other's kernels with no synchronization.
/// Every backend implements it; one that cannot share (each OpenCL, Vulkan, Metal or WebGPU backend owns its own
/// context or device) answers true only for itself, and a buffer it owns is migrated or refused when another
/// backend's op meets it (CrossBackendTensors).
/// </summary>
internal interface IGpuSharedAddressSpace
{
    /// <summary>Whether a buffer this backend allocated may be read and written by <paramref name="other"/>'s kernels.</summary>
    bool SharesAddressSpaceWith(IDirectGpuBackend other);
}
