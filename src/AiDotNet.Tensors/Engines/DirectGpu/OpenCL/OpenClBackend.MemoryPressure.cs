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

    bool IDeviceMemoryPressure.ReclaimsOnAllocationFailure => false;
}
