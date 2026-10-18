#if NET7_0_OR_GREATER
// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IGpuBufferViews
{
    /// <summary>Byte alignment a view's offset must meet: WebGPU's default minStorageBufferOffsetAlignment.</summary>
    internal const int ViewOffsetAlignment = 256;

    /// <inheritdoc/>
    // Offsets must meet ViewOffsetAlignment.
    bool IGpuBufferViews.ViewsAreFreeAtAnyOffset => false;

    /// <inheritdoc/>
    /// <remarks>
    /// The view shares the parent's device buffer: bind groups bind only its byte range and reads, writes and copies
    /// start at its offset, so kernels and transfers address only its elements. An offset that is not a multiple of
    /// <see cref="ViewOffsetAlignment"/> bytes yields null, as the contract allows.
    /// </remarks>
    IGpuBuffer? IGpuBufferViews.TryCreateView(IGpuBuffer parent, int elementOffset, int elementCount)
    {
        if (parent is not WebGpuBuffer owner || !owner.IsValid) return null;
        if (elementOffset < 0 || elementCount <= 0 || (long)elementOffset + elementCount > owner.Size) return null;
        if ((owner.OffsetBytes + (long)elementOffset * sizeof(float)) % ViewOffsetAlignment != 0) return null;
        return new WebGpuBuffer(owner, elementOffset, elementCount);
    }
}
#endif