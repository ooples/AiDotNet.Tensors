#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// WebGPU exposes no device-memory query. Pressure-gated reclamation therefore never triggers; the unconditional sweep still runs.
public sealed partial class WebGpuBackend : IDeviceMemoryPressure
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
        WebGpuBuffer.PendingFrees.Drain();
        return WebGpuBuffer.Pool.DrainAll();
    }

    /// <remarks>WebGpuBuffer collects unreachable buffers, drains the pool and retries once on allocation failure.</remarks>
    bool IDeviceMemoryPressure.ReclaimsOnAllocationFailure => true;
}
#endif
