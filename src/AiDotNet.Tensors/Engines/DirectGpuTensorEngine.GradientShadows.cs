using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    internal bool CanUseCompiledGradientShadow<T>(Tensor<T> tensor)
    {
        if (typeof(T) != typeof(float) || tensor.IsDisposed || !tensor.IsContiguous || tensor._storageOffset != 0
            || tensor._ownsGpuBuffer || tensor._storage.HasGpuBufferOwner
            || !tensor._storage.TryGetBackingArrayIdentity(out var array, out int offset)
            || offset != 0 || array is not float[] floats || floats.Length < tensor.Length) return false;
        lock (_persistentBufferLock)
            return !_persistentBufferCache.ContainsKey(array);
    }

    internal bool TryCopyCompiledGradientToHost<T>(Tensor<T> tensor, IGpuBuffer buffer, IDirectGpuBackend backend)
    {
        if (!CanUseCompiledGradientShadow(tensor) || !ReferenceEquals(tensor._gpuBuffer, buffer)
            || !ReferenceEquals(tensor._gpuBackend, backend)
            || !ReferenceEquals(GpuBufferViewRoot.Of(buffer), buffer) || buffer.Handle == IntPtr.Zero
            || !tensor._storage.TryGetBackingArrayIdentity(out var array, out int offset)
            || offset != 0 || array is not float[] destination || destination.Length < buffer.Size
            || tensor.Length != buffer.Size) return false;
        lock (_activationCacheLock)
        {
            if (_activationCache.TryGetValue(array, out var byArray)
                && !ReferenceEquals(GpuBufferViewRoot.Of(byArray.Buffer), buffer)) return false;
            if (_activationCache.TryGetValue(tensor.DataVector, out var byVector)
                && !ReferenceEquals(GpuBufferViewRoot.Of(byVector.Buffer), buffer)) return false;
        }
        // The backend writes the existing shadow. Keep residency intact if a backend declines the transfer.
        try { backend.DownloadBuffer(buffer, destination); }
        catch (NotSupportedException capabilityGap)
        { GpuLaunchProbe.OnFallback("CompiledGradientShadow", capabilityGap); return false; }
        return true;
    }

    // The caller has completed the native download and temporarily marked the backing pooled before eviction.
    // Unlike dead intermediates, these public gradients keep valid CPU values and must never receive Release notes.
    internal void DetachHostCurrentCompiledGradient<T>(Tensor<T> tensor)
    {
        var array = tensor.GetBackingArrayForCacheLookupUnsafe();
        HostSync.Remove(tensor.DataVector);
        if (array is not null) HostSync.Remove(array);
        if (tensor._gpuMaterializerKey is { } key) HostSync.Remove(key);
        tensor._gpuMaterializerKey = null;
        if (array is not null) InvalidateActivationCacheEntry(array);
        InvalidateActivationCacheEntry(tensor.DataVector);
        tensor._gpuBuffer = null;
        tensor._gpuBackend = null;
        tensor._gpuBufferVersion = -1;
    }
}
