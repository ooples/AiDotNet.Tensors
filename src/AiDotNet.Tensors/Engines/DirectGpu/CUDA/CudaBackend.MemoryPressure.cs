// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// cuMemGetInfo, the backed-off sweep and the deferred-free drain live in CudaBackend.cs.
public sealed partial class CudaBackend : IDeviceMemoryPressure
{
    bool IDeviceMemoryPressure.TryGetDeviceMemory(out ulong free, out ulong total) => TryGetDeviceMemory(out free, out total);

    void IDeviceMemoryPressure.ReclaimUnderPressure() => ReclaimUnderPressure();

    void IDeviceMemoryPressure.ReclaimNow() => ReclaimNow();

    int IDeviceMemoryPressure.ReleaseCachedBuffers()
    {
        if (!IsAvailable) return 0;
        // Event-deferred frees whose work has completed go back first, so the pool drain below frees them too.
        DrainDeferredFrees(blockOldest: false);
        return _bufferPool.DrainAll();
    }

    /// <remarks>Allocation drains the buffer pool and retries once on out-of-memory.</remarks>
    bool IDeviceMemoryPressure.ReclaimsOnAllocationFailure => true;
}