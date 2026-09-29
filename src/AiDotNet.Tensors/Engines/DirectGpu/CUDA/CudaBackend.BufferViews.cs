namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IGpuBufferViews
{
    /// <inheritdoc/>
    IGpuBuffer? IGpuBufferViews.TryCreateView(IGpuBuffer parent, int elementOffset, int elementCount)
    {
        if (parent is null || parent.Handle == IntPtr.Zero) return null;
        if (elementOffset < 0 || elementCount <= 0 || (long)elementOffset + elementCount > parent.Size) return null;
        return new CudaBufferView(parent, elementOffset, elementCount);
    }

    /// <summary>
    /// Non-owning float view into a parent device buffer. The device pointer is recomputed from the parent on each
    /// access, and the view holds the parent so it cannot be collected while the view is in use.
    /// </summary>
    private sealed class CudaBufferView : IGpuBuffer
    {
        private readonly IGpuBuffer _parent;
        private readonly int _elementOffset;

        public CudaBufferView(IGpuBuffer parent, int elementOffset, int elementCount)
        {
            _parent = parent;
            _elementOffset = elementOffset;
            Size = elementCount;
        }

        public int Size { get; }
        public long SizeInBytes => (long)Size * sizeof(float);
        public IntPtr Handle => _parent.Handle == IntPtr.Zero
            ? IntPtr.Zero
            : _parent.Handle + (_elementOffset * sizeof(float));

        public void Dispose() { }   // non-owning
    }
}
