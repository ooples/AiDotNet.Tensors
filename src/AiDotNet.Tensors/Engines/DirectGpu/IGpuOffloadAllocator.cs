// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Per-backend allocator for the issue-#276 GPU-offload paths. Every
/// direct-GPU backend (CUDA / HIP / Metal / OpenCL / Vulkan / WebGPU)
/// implements this so a model author writing
/// <c>weight.Lifetime = WeightLifetime.GpuOffload</c> gets the same
/// pinned-host / managed-memory placement contract regardless of which
/// runtime is loaded.
///
/// <para>The allocator returns a host-visible pointer + a backend-specific
/// device handle. For the pinned scheme they're the same address (DMA-
/// mapped); for managed memory the driver handles migration so the
/// device handle is the same pointer too.</para>
/// </summary>
public interface IGpuOffloadAllocator : IDisposable
{
    /// <summary>True when this backend's offload allocator is loadable on
    /// the host. Mirrors the IsAvailable probe pattern from #267.</summary>
    bool IsAvailable { get; }

    /// <summary>Allocates <paramref name="bytes"/> bytes under
    /// <paramref name="scheme"/>. Returns a handle the caller frees via
    /// <see cref="Free"/>.</summary>
    GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme);

    /// <summary>Frees a handle returned by <see cref="Allocate"/>.</summary>
    void Free(GpuOffloadHandle handle);
}

/// <summary>Result of an offload allocation. <see cref="HostPointer"/> is
/// safe for host writes; <see cref="DevicePointer"/> is what kernels read
/// (often the same value for pinned/managed schemes).</summary>
public readonly struct GpuOffloadHandle
{
    public readonly IntPtr HostPointer;
    public readonly IntPtr DevicePointer;
    public readonly long Bytes;
    public readonly OffloadScheme Scheme;
    public readonly object? BackendOpaque;

    public GpuOffloadHandle(IntPtr host, IntPtr device, long bytes, OffloadScheme scheme, object? opaque = null)
    {
        HostPointer = host;
        DevicePointer = device;
        Bytes = bytes;
        Scheme = scheme;
        BackendOpaque = opaque;
    }
}
