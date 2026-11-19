using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // The plan alone owns the supplied buffer. Its input's current binding may have been detached when host
    // batch storage was rebound; upload the new values into that same allocation instead of abandoning it.
    internal bool TryRefreshPlanOwnedLeafInput<T>(Tensor<T> input, IDirectGpuBackend backend, IGpuBuffer owned)
    {
        using var gpuOp = EnterGpuOp();
        if (typeof(T) != typeof(float) || input.IsDisposed || input._gpuBuffer is not null
            || owned.Handle == IntPtr.Zero || owned.Size < input.Length || input.Length <= 0
            || InferenceCaptureActive || !TryGetBackend(out var current) || !ReferenceEquals(current, backend))
            return false;
        if (backend is IGpuGraphCapture capture && capture.IsStreamCapturing()) return false;
        // Mirror EnsureResidentInput's host-authoritative input contract. A pending download on a host array
        // can describe an earlier device-side view; it must not overwrite the batch the caller just wrote.
        if (input.GetBackingArrayForCacheLookupUnsafe() is { } host) Helpers.HostSync.Remove(host);
        using var values = input.LeaseArray();
        backend.UploadBufferInPlace((float[])(object)values.Array, owned);
        input._gpuBuffer = owned;
        input._gpuBackend = backend;
        input._gpuBufferVersion = input.GpuCacheVersion;
        return true;
    }
}
