// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Device-memory pressure (#1154): reading free memory, and releasing device buffers that only unreachable managed
/// objects still hold (their finalizers free them). A backend whose driver cannot report free memory answers
/// <see cref="TryGetDeviceMemory"/> false, so pressure-gated reclamation never triggers there.
/// </summary>
internal interface IDeviceMemoryPressure
{
    /// <summary>Free and total device memory in bytes, or false when the driver cannot report them.</summary>
    bool TryGetDeviceMemory(out ulong free, out ulong total);

    /// <summary>Runs the finalizer sweep when free memory is below a fifth of the device, with back-off.</summary>
    void ReclaimUnderPressure();

    /// <summary>Runs the finalizer sweep unconditionally.</summary>
    void ReclaimNow();

    /// <summary>True when the backend's allocator already reclaims and retries on an out-of-memory failure.</summary>
    bool ReclaimsOnAllocationFailure { get; }
}

/// <summary>The sweep every backend runs: finalizers release device buffers unreachable managed objects held.</summary>
internal static class DeviceMemorySweep
{
    public static void Run()
    {
        System.GC.Collect();
        System.GC.WaitForPendingFinalizers();
        System.GC.Collect();
    }
}