#if NET7_0_OR_GREATER
// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IDeviceZeroFill
{
    /// <inheritdoc/>
    /// <remarks>No graph capture on this backend, so the float fill serves.</remarks>
    void IDeviceZeroFill.ZeroBuffer(IGpuBuffer buffer, int floatCount)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (floatCount <= 0) return;
        Fill(buffer, 0f, floatCount);
    }
}
#endif
