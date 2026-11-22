namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IGpuBufferViews
{
    /// <inheritdoc/>
    // A view is the parent's device pointer plus the offset.
    bool IGpuBufferViews.ViewsAreFreeAtAnyOffset => true;

    /// <inheritdoc/>
    IGpuBuffer? IGpuBufferViews.TryCreateView(IGpuBuffer parent, int elementOffset, int elementCount)
    {
        if (parent is null || parent.Handle == IntPtr.Zero) return null;
        if (elementOffset < 0 || elementCount <= 0 || (long)elementOffset + elementCount > parent.Size) return null;
        return new CudaBufferView(parent, elementOffset, elementCount);
    }

    /// <summary>
    /// Non-owning view into a parent device buffer, in the parent's elements: its offset, extent and reported type
    /// are all the parent's (an untyped parent is addressed in float words, as before). The device pointer is
    /// recomputed from the parent on each access, and the view holds the parent so it cannot be collected while the
    /// view is in use.
    /// </summary>
    private sealed class CudaBufferView : IGpuBuffer, IGpuBufferViewSource
    {
        private readonly IGpuBuffer _parent;
        private readonly int _elementOffset;
        private readonly int _elementBytes;

        public CudaBufferView(IGpuBuffer parent, int elementOffset, int elementCount)
        {
            _parent = parent;
            _elementOffset = elementOffset;
            _elementBytes = parent.ElementType == GpuElementType.Unspecified
                ? sizeof(float)
                : GpuElementTypes.SizeInBytes(parent.ElementType);
            Size = elementCount;
        }

        public GpuElementType ElementType => _parent.ElementType;
        public int Size { get; }
        public long SizeInBytes => (long)Size * _elementBytes;
        IGpuBuffer IGpuBufferViewSource.ViewParent => _parent;
        public IntPtr Handle => _parent.Handle == IntPtr.Zero
            ? IntPtr.Zero
            : new IntPtr(_parent.Handle.ToInt64() + (long)_elementOffset * _elementBytes);

        public void Dispose() { }   // non-owning
    }
}
