// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// cuMemGetInfo, the backed-off sweep and the deferred-free drain live in CudaBackend.cs.
public sealed partial class CudaBackend : IDeviceMemoryPressure
{
    bool IDeviceMemoryPressure.TryGetDeviceMemory(out ulong free, out ulong total) => TryGetDeviceMemory(out free, out total);

    void IDeviceMemoryPressure.ReclaimUnderPressure() => ReclaimUnderPressure();

    void IDeviceMemoryPressure.ReclaimNow() => ReclaimNow();

    /// <remarks>Allocation drains the buffer pool and retries once on out-of-memory.</remarks>
    bool IDeviceMemoryPressure.ReclaimsOnAllocationFailure => true;
}