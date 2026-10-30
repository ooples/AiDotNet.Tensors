using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: zero-copy DLPack exchange of device memory (<see cref="Interop.DLPack"/>). A backend with it
/// hands its device pointer to a DLPack consumer and wraps a producer's device pointer as a buffer. Every other
/// backend still exchanges its tensors through DLPack: the tensor's data comes back with
/// <see cref="IDirectGpuBackend.DownloadBuffer(IGpuBuffer)"/> and is exported as a host copy.
/// </summary>
/// <remarks>
/// DLPack's device types that carry plain device pointers are CUDA (kDLCUDA) and ROCm (kDLROCM); OpenCL, Vulkan,
/// Metal and WebGPU memory has no pointer a foreign library can address. CUDA implements this today. HIP does not yet
/// (issue #1621): a GPU-resident tensor on the HIP backend exports a host copy (kDLCPU; writes through the consumer's
/// copy do not reach the tensor), and a kDLROCM import throws <see cref="NotSupportedException"/>.
/// </remarks>
[BackendSpecificRoute("DLPack device-pointer exchange exists only for pointer-addressed device memory (CUDA); every backend exports through a host copy",
    nameof(IDirectGpuBackend.DownloadBuffer))]
internal interface IDLPackDeviceBackend
{
    /// <summary>The DLPack <c>DLDeviceType</c> of this backend's memory (kDLCUDA = 2).</summary>
    int DLPackDeviceType { get; }

    /// <summary>The device ordinal (<c>DLDevice.device_id</c>).</summary>
    int DeviceOrdinal { get; }

    /// <summary>The handle of the stream this backend orders its work on.</summary>
    IntPtr WorkStreamHandle { get; }

    /// <summary>Makes <paramref name="consumerStream"/> wait for every operation queued so far on this backend.</summary>
    void OrderStreamAfterPendingWork(IntPtr consumerStream);

    /// <summary>Wraps device memory another library owns; disposing the buffer waits for queued work, then runs
    /// <paramref name="release"/> once.</summary>
    IGpuBuffer WrapExternalDeviceMemory(IntPtr devicePointer, int elementCount, int elementByteSize, Action release);

    /// <summary>
    /// Runs <paramref name="release"/> once every operation queued on this backend so far has finished, without calling
    /// the driver from the calling thread: the finalizer path of an imported buffer that was never disposed. The
    /// release is queued and runs at this backend's next operation.
    /// </summary>
    void ReleaseAfterPendingWorkDeferred(Action release);

    /// <summary>
    /// Holds <paramref name="buffer"/>'s device memory for an export: until the result is disposed (by the consumer's
    /// deleter), releasing the buffer neither frees it nor lets another allocation reuse it. Disposing it orders this
    /// backend's work after the work queued on <paramref name="consumerStream"/> (zero = the consumer synchronizes
    /// itself), then runs the releases that waited.
    /// </summary>
    IDisposable HoldForExport(IGpuBuffer buffer, IntPtr consumerStream);
}
