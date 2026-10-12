namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IGpuBufferViews
{
    /// <inheritdoc/>
    /// <remarks>
    /// A HIP buffer is a device pointer, so a view is the parent's pointer advanced by the offset. It is a
    /// <see cref="HipGpuBuffer"/> like any other, so every backend path accepts it, but it never frees or pools.
    /// </remarks>
    IGpuBuffer? IGpuBufferViews.TryCreateView(IGpuBuffer parent, int elementOffset, int elementCount)
    {
        if (parent is null || parent.Handle == IntPtr.Zero) return null;
        if (elementOffset < 0 || elementCount <= 0 || (long)elementOffset + elementCount > parent.Size) return null;
        return HipGpuBuffer.CreateView(parent, parent.Handle + (elementOffset * sizeof(float)), elementCount, this);
    }
}
