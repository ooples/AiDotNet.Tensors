// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

// OpenCL has no standard free-memory query. Pressure-gated reclamation therefore never triggers; the unconditional sweep still runs.
public sealed partial class OpenClBackend : IDeviceMemoryPressure
{
    bool IDeviceMemoryPressure.TryGetDeviceMemory(out ulong free, out ulong total)
    {
        free = 0; total = 0;
        return false;
    }

    void IDeviceMemoryPressure.ReclaimUnderPressure()
    {
    }

    void IDeviceMemoryPressure.ReclaimNow() => DeviceMemorySweep.Run();

    int IDeviceMemoryPressure.ReleaseCachedBuffers()
    {
        DrainCollectedBuffers();
        int drained = _bufferPool.DrainAll();
        // A drained buffer still used by queued work is retired behind a completion marker, not freed: wait for those
        // markers so the memory is actually released when this returns.
        _context?.WaitAndReleaseRetiredMemory();
        return drained;
    }

    /// <remarks>AllocateReclaiming collects unreachable buffers, drains the pool and retries once on out-of-memory.</remarks>
    bool IDeviceMemoryPressure.ReclaimsOnAllocationFailure => true;
}
