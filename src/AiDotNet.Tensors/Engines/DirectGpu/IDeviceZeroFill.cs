// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Zeroing a float buffer in a way a captured device graph can record (#1154). CUDA uses an asynchronous byte memset
/// on its compute stream (Fill's cuMemsetD32 would abort a capture); backends without graph capture fill with 0.
/// </summary>
internal interface IDeviceZeroFill
{
    /// <summary>Sets the first <paramref name="floatCount"/> floats of <paramref name="buffer"/> to +0.</summary>
    void ZeroBuffer(IGpuBuffer buffer, int floatCount);

    /// <summary>Sets the first <paramref name="byteCount"/> bytes to zero (any element size).</summary>
    void ZeroBytes(IGpuBuffer buffer, long byteCount);
}