// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DevicePrimitives;

/// <summary>
/// Device memory of one GPU device (or of all of them) at one moment: the counterpart of
/// <c>torch.cuda.memory_allocated()</c>, <c>memory_reserved()</c> and their <c>max_*</c> forms.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="AllocatedBytes"/> is the memory in buffers a caller holds, by each buffer's physical capacity.
/// <see cref="ReservedBytes"/> is everything the library holds from the driver: allocated memory, plus buffers
/// parked in a reuse pool, plus, on CUDA with the stream-ordered allocator, freed blocks the driver pool keeps.
/// The difference is what the library could give back without freeing a tensor.
/// </para>
/// <para>
/// <see cref="HostShadowBytes"/> has no PyTorch counterpart: the host arrays kept beside GPU-resident data, such as
/// a downloaded copy of a resident tensor. PyTorch never keeps one; here it is host memory a GPU workload pays for.
/// It is process-wide, the same in every device's snapshot.
/// </para>
/// </remarks>
public readonly struct GpuMemorySnapshot
{
    internal GpuMemorySnapshot(string device, long allocated, long peakAllocated, long reserved, long peakReserved,
        long deviceAllocations, long deviceFrees, long hostShadow, long peakHostShadow)
    {
        Device = device;
        AllocatedBytes = allocated;
        PeakAllocatedBytes = peakAllocated;
        ReservedBytes = reserved;
        PeakReservedBytes = peakReserved;
        DeviceAllocations = deviceAllocations;
        DeviceFrees = deviceFrees;
        HostShadowBytes = hostShadow;
        PeakHostShadowBytes = peakHostShadow;
    }

    /// <summary>The device, <c>backend:index</c> such as <c>cuda:0</c>, or <c>all</c> for the process total.</summary>
    public string Device { get; }

    /// <summary>Bytes in buffers handed to a caller and not yet given back.</summary>
    public long AllocatedBytes { get; }

    /// <summary>The most <see cref="AllocatedBytes"/> since the last peak reset.</summary>
    public long PeakAllocatedBytes { get; }

    /// <summary>Bytes the library holds from the driver, in use or cached.</summary>
    public long ReservedBytes { get; }

    /// <summary>The most <see cref="ReservedBytes"/> since the last peak reset. With CUDA's stream-ordered allocator
    /// it adds the driver pool's own high-water mark to the library's.</summary>
    public long PeakReservedBytes { get; }

    /// <summary>Driver allocations the library made, over the process lifetime.</summary>
    public long DeviceAllocations { get; }

    /// <summary>Driver frees the library made, over the process lifetime.</summary>
    public long DeviceFrees { get; }

    /// <summary>Host bytes kept beside GPU-resident data (process-wide).</summary>
    public long HostShadowBytes { get; }

    /// <summary>The most <see cref="HostShadowBytes"/> since the last peak reset.</summary>
    public long PeakHostShadowBytes { get; }

    /// <inheritdoc/>
    public override string ToString() =>
        $"{Device}: allocated {AllocatedBytes:N0} B (peak {PeakAllocatedBytes:N0}), reserved {ReservedBytes:N0} B " +
        $"(peak {PeakReservedBytes:N0}), host shadow {HostShadowBytes:N0} B (peak {PeakHostShadowBytes:N0})";
}
