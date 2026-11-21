using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <inheritdoc />
    public override Tensor<T> KvCacheAppend<T>(Tensor<T> cache, Tensor<T> position, Tensor<T> value)
    {
        if (GraphMode.IsActive) return base.KvCacheAppend(cache, position, value);
        var (batch, capacity, width) = ValidateKvCacheAppend(cache, position, value);
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IKvCacheAppendBackend append
            || IsPartialView(cache) || IsPartialView(position))
            return base.KvCacheAppend(cache, position, value);
        using var __gpuOp = EnterGpuOp();
        // The cache's and the position's own persistent buffers, holding their current values: written in place.
        var cacheBuffer = PinInputResident(cache) ?? throw new InvalidOperationException("The cache could not be made device-resident.");
        var positionBuffer = PinInputResident(position) ?? throw new InvalidOperationException("The position could not be made device-resident.");
        using var v = GetOrAllocateBuffer(backend, value);
        var mask = AllocateFullyWrittenOutputBuffer(backend, capacity);   // every slot written
        append.KvCacheAppend(cacheBuffer, positionBuffer, v.Buffer, mask.Buffer, batch, capacity, width);
        cache.IncrementVersion();
        BindResidentBuffer(cache, cacheBuffer, backend);
        position.IncrementVersion();
        BindResidentBuffer(position, positionBuffer, backend);
        return DeferTensorResult<T>(backend, mask.Buffer, capacity, new[] { capacity });
    }
}
