// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// Metal reports a working-set budget, not free device memory. Pressure-gated reclamation therefore never triggers; the unconditional sweep still runs.
public sealed partial class MetalBackend : IDeviceMemoryPressure
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
