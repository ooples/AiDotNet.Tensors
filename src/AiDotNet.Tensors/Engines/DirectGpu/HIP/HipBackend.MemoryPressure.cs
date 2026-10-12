// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IDeviceMemoryPressure
{
    private int _reclaimCooldownRemaining;
    private int _reclaimBackoffSteps = 1;

    /// <inheritdoc/>
    bool IDeviceMemoryPressure.TryGetDeviceMemory(out ulong free, out ulong total)
    {
        free = 0; total = 0;
        if (!IsAvailable) return false;
        UIntPtr f = UIntPtr.Zero, t = UIntPtr.Zero;
        if (HipNativeBindings.hipMemGetInfo(ref f, ref t) != HipError.Success) return false;
        free = f.ToUInt64(); total = t.ToUInt64();
        return total > 0;
    }

    /// <inheritdoc/>
    /// <remarks>As CUDA's: a sweep that frees under 1% of the device doubles a cooldown (up to 256 calls), so a
    /// workload legitimately resident below the threshold does not pay two collections per step.</remarks>
    void IDeviceMemoryPressure.ReclaimUnderPressure()
    {
        var self = (IDeviceMemoryPressure)this;
        if (_reclaimCooldownRemaining > 0) { _reclaimCooldownRemaining--; return; }
        if (!self.TryGetDeviceMemory(out ulong free, out ulong total)) return;
        if ((double)free / total > 0.20) { _reclaimBackoffSteps = 1; return; }
        DeviceMemorySweep.Run();
        ulong freeAfter = self.TryGetDeviceMemory(out ulong f2, out _) ? f2 : free;
        bool effective = freeAfter > free && (freeAfter - free) >= total / 100;
        _reclaimBackoffSteps = effective ? 1 : Math.Min(_reclaimBackoffSteps * 2, 256);
        _reclaimCooldownRemaining = effective ? 0 : _reclaimBackoffSteps;
    }

    /// <inheritdoc/>
    void IDeviceMemoryPressure.ReclaimNow() => DeviceMemorySweep.Run();

    /// <inheritdoc/>
    bool IDeviceMemoryPressure.ReclaimsOnAllocationFailure => false;
}