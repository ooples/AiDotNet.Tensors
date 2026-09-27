namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A backend that can hand out a device buffer WITHOUT zero-filling it.
/// </summary>
/// <remarks>
/// <see cref="IDirectGpuBackend.AllocateBuffer(int)"/> zero-initialises every buffer. For an output the next kernel
/// overwrites completely that fill is pure waste, and on Windows/WDDM the memset costs about as much as the kernel
/// launch itself — so an allocate-then-launch op paid two launches. Callers may use this ONLY where the kernel that
/// follows provably writes every element (elementwise maps, GEMM with beta = 0); anything that accumulates into its
/// output (atomicAdd reductions, scatter-add, split-K) must keep the zeroed allocation.
/// </remarks>
internal interface IUninitializedGpuAllocation
{
    /// <summary>Allocates <paramref name="size"/> float elements with undefined contents.</summary>
    IGpuBuffer AllocateBufferUninitialized(int size);
}
