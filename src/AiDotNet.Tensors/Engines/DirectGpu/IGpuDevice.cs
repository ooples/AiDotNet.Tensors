// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// What every GPU backend shares however it runs operations: the device it describes, and buffers it allocates
/// and fills from the host.
/// </summary>
/// <remarks>
/// <para>
/// Operations live on two sibling interfaces that both extend this one and neither extends the other.
/// <see cref="IDirectGpuBackend"/> runs them synchronously and is what the native backends (CUDA, HIP, OpenCL, Vulkan,
/// Metal) implement. <see cref="IAwaitableGpuBackend"/> returns a task for every operation that has to wait for the
/// device, and is what a backend implements when it cannot block its caller: WebGPU in a browser completes work
/// through the page's event loop, on the same thread that would be waiting.
/// </para>
/// <para>
/// Everything here is safe to call on any thread, including a browser's only thread: nothing waits for the device.
/// Writing host data into a buffer queues the copy, which the device orders before any later operation.
/// </para>
/// </remarks>
public interface IGpuDevice : IDisposable
{
    /// <summary>
    /// Gets whether this backend is available and initialized.
    /// </summary>
    bool IsAvailable { get; }

    /// <summary>
    /// Gets the backend name (e.g., "OpenCL", "CUDA").
    /// </summary>
    string BackendName { get; }

    /// <summary>
    /// Gets the tensor device type for this backend.
    /// </summary>
    TensorDevice DeviceType { get; }

    /// <summary>
    /// Gets the GPU device name.
    /// </summary>
    string DeviceName { get; }

    /// <summary>
    /// Gets the GPU vendor (AMD, NVIDIA, Intel).
    /// </summary>
    string DeviceVendor { get; }

    /// <summary>
    /// Gets the number of compute units on the GPU.
    /// </summary>
    int ComputeUnits { get; }

    /// <summary>
    /// Gets the total global memory in bytes.
    /// </summary>
    long GlobalMemoryBytes { get; }

    /// <summary>
    /// Gets the local (shared) memory per workgroup in bytes.
    /// </summary>
    long LocalMemoryBytes { get; }

    /// <summary>
    /// Gets the maximum size in bytes of a SINGLE buffer allocation on this
    /// device. This is independent of <see cref="GlobalMemoryBytes"/> — even
    /// when total VRAM is large, GPU drivers cap individual allocations
    /// (OpenCL's <c>CL_DEVICE_MAX_MEM_ALLOC_SIZE</c>, CUDA's per-allocation
    /// virtual address window, Metal's <c>maxBufferLength</c>, Vulkan's
    /// <c>maxStorageBufferRange</c>, WebGPU's <c>maxBufferSize</c>).
    /// </summary>
    /// <remarks>
    /// <para>Issue #285: blindly requesting an allocation above this cap
    /// produces an opaque driver error (e.g. <c>CL_INVALID_BUFFER_SIZE = -61</c>).
    /// Engine dispatch shims should validate against this value and fall
    /// through to a chunked or CPU path instead of crashing.</para>
    /// <para>Returns 0 when the cap is unknown / not queryable; callers
    /// should treat 0 as "skip the validation, let the native call fail
    /// with its own error if any."</para>
    /// </remarks>
    long MaxBufferAllocBytes { get; }

    /// <summary>
    /// Gets the estimated theoretical peak FP32 GFLOPS for this GPU.
    /// Computed from clock rate, compute units, and FP32 cores per compute unit.
    /// </summary>
    double TheoreticalGflops { get; }

    /// <summary>
    /// Allocates a GPU buffer and uploads data.
    /// </summary>
    /// <param name="data">CPU data to upload.</param>
    /// <returns>Handle to the GPU buffer.</returns>
    IGpuBuffer AllocateBuffer(float[] data);

    /// <summary>
    /// Allocates an empty GPU buffer.
    /// </summary>
    /// <param name="size">Number of float elements.</param>
    /// <returns>Handle to the GPU buffer.</returns>
    IGpuBuffer AllocateBuffer(int size);

    /// <summary>
    /// Whether this backend can hold buffers of <paramref name="type"/> on this device. It says nothing about which
    /// operations compute in that type: an operation without a kernel for it still reports its own fallback.
    /// </summary>
    bool SupportsElementType(GpuElementType type);

    /// <summary>Allocates a buffer of <paramref name="size"/> elements of <paramref name="type"/>.</summary>
    /// <param name="size">Number of elements.</param>
    /// <param name="type">The element type; the buffer reports it as <see cref="IGpuBuffer.ElementType"/>.</param>
    /// <exception cref="NotSupportedException">The backend cannot hold <paramref name="type"/> (<see cref="SupportsElementType"/>).</exception>
    IGpuBuffer AllocateBuffer(int size, GpuElementType type);
}
