// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IDeviceZeroFill
{
    /// <inheritdoc/>
    /// <remarks>No graph capture on this backend, so the float fill serves.</remarks>
    void IDeviceZeroFill.ZeroBuffer(IGpuBuffer buffer, int floatCount)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (floatCount <= 0) return;
        Fill(buffer, 0f, floatCount);
    }

    /// <inheritdoc/>
    void IDeviceZeroFill.ZeroBytes(IGpuBuffer buffer, long byteCount)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (byteCount <= 0) return;
        // A float fill zeroes whole 4-byte words; a byte range ending mid-word cannot be zeroed exactly that way.
        if (byteCount % sizeof(float) != 0)
            throw new ArgumentException($"A {byteCount}-byte range is not whole 4-byte words.", nameof(byteCount));
        Fill(buffer, 0f, checked((int)(byteCount / sizeof(float))));
    }
}
