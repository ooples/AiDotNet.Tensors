// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IDeviceZeroFill
{
    /// <inheritdoc/>
    /// <remarks>Capturable: an asynchronous byte memset on the compute stream.</remarks>
    void IDeviceZeroFill.ZeroBuffer(IGpuBuffer buffer, int floatCount)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (floatCount <= 0) return;
        MemsetBuffer(buffer, 0, (long)floatCount * sizeof(float));
    }

    /// <inheritdoc/>
    void IDeviceZeroFill.ZeroBytes(IGpuBuffer buffer, long byteCount)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (byteCount <= 0) return;
        MemsetBuffer(buffer, 0, byteCount);
    }
}
