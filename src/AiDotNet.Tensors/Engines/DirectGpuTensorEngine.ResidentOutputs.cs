using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>
    /// Creates a fully written resident result without allocating an unused host payload. Compiled actions borrow
    /// their plan's stable scratch buffer when the caller binds it; outside an action the result owns its buffer,
    /// including the normal TensorScope release and deferred host-read contracts.
    /// </summary>
    private Tensor<T> CreateResidentOutput<T>(IDirectGpuBackend backend, int[] shape)
    {
        if (ScratchPoolingActive)
            return Tensor<T>.CreateGpuResident(shape, backend.DeviceType);

        int length = 1;
        foreach (int dimension in shape) length = checked(length * dimension);
        var buffer = AllocateDeviceBufferUninitialized(backend, length);
        try
        {
            return DeferTensorResult<T>(backend, buffer, length, shape);
        }
        catch
        {
            buffer.Dispose();
            throw;
        }
    }
}
