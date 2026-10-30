// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IGpuBufferViews
{
    /// <summary>Byte alignment a view's offset must meet: the strictest setBuffer:offset: alignment Metal requires.</summary>
    internal const int ViewOffsetAlignment = 256;

    /// <inheritdoc/>
    // Offsets must meet ViewOffsetAlignment.
    bool IGpuBufferViews.ViewsAreFreeAtAnyOffset => false;

    /// <inheritdoc/>
    /// <remarks>
    /// The view binds the parent's MTLBuffer at a byte offset (every compute binding goes through
    /// <see cref="ScopedComputeEncoder.SetBuffer(MetalGpuBuffer, int)"/>, which applies it) and offsets its CPU-visible contents the
    /// same way, so kernels, copies and transfers address only its range. An offset that is not a multiple of
    /// <see cref="ViewOffsetAlignment"/> bytes yields null, as the contract allows.
    /// </remarks>
    IGpuBuffer? IGpuBufferViews.TryCreateView(IGpuBuffer parent, int elementOffset, int elementCount)
    {
        if (parent is not MetalGpuBuffer owner || owner.IsDisposed) return null;
        if (elementOffset < 0 || elementCount <= 0 || (long)elementOffset + elementCount > owner.Size) return null;
        if ((owner.ByteOffset + (long)elementOffset * sizeof(float)) % ViewOffsetAlignment != 0) return null;
        return new MetalGpuBuffer(owner, elementOffset, elementCount);
    }
}
