namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A backend whose device buffers are valid addresses for some OTHER backend instance too: two CUDA backends on the
/// same live primary context share one address space, so a buffer one allocated may be passed to the other's kernels.
/// Backends that cannot share (separate OpenCL, Vulkan, Metal or WebGPU contexts) do not implement it, and a buffer
/// they own is migrated or refused when another backend's op meets it (CrossBackendTensors).
/// </summary>
internal interface IGpuSharedAddressSpace
{
    /// <summary>Whether a buffer this backend allocated may be read and written by <paramref name="other"/>'s kernels.</summary>
    bool SharesAddressSpaceWith(IDirectGpuBackend other);
}
