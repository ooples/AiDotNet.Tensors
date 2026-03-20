using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// GPU-backed tensor workspace that uses a single contiguous GPU allocation for all
/// intermediate tensors. Sub-regions are addressed via offset+length, eliminating
/// per-operation GPU memory allocation during inference.
/// </summary>
/// <remarks>
/// <para>
/// This is the GPU equivalent of <see cref="TensorWorkspace{T}"/>, backed by a single
/// <see cref="IGpuBuffer"/> instead of a CPU array. Combined with batch execution via
/// <see cref="IDirectGpuBackend.BeginBatch"/>/<see cref="IDirectGpuBackend.EndBatch"/>,
/// this enables zero-allocation GPU graph execution.
/// </para>
/// </remarks>
public sealed class GpuWorkspace : IDisposable
{
    private readonly IDirectGpuBackend _backend;
    private readonly List<int[]> _shapes = new();
    private readonly List<int> _offsets = new();
    private IGpuBuffer? _buffer;
    private bool _isAllocated;
    private bool _disposed;

    /// <summary>Total number of float elements in the workspace.</summary>
    public int TotalElements { get; private set; }

    /// <summary>Number of registered tensor slots.</summary>
    public int SlotCount => _shapes.Count;

    /// <summary>Whether the workspace buffer has been allocated on the GPU.</summary>
    public bool IsAllocated => _isAllocated;

    /// <summary>The underlying GPU buffer (null until allocated).</summary>
    public IGpuBuffer? Buffer => _buffer;

    /// <summary>
    /// Creates a GPU workspace backed by the specified GPU backend.
    /// </summary>
    /// <param name="backend">The GPU backend to allocate memory from.</param>
    public GpuWorkspace(IDirectGpuBackend backend)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
    }

    /// <summary>
    /// Registers a tensor shape and returns a slot ID.
    /// Must be called before <see cref="Allocate"/>.
    /// </summary>
    public int Register(int[] shape)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuWorkspace));
        if (_isAllocated)
            throw new InvalidOperationException("Cannot register after allocation. Call Reset() first.");
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape cannot be null or empty.", nameof(shape));
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] <= 0)
                throw new ArgumentException($"All dimensions must be positive. Dimension {i} was {shape[i]}.", nameof(shape));
        }

        int id = _shapes.Count;
        _shapes.Add((int[])shape.Clone());
        return id;
    }

    /// <summary>
    /// Allocates a single contiguous GPU buffer for all registered tensor slots.
    /// Uses <see cref="IDirectGpuBackend.AllocateWorkspaceBuffer"/> for a single GPU allocation.
    /// </summary>
    public void Allocate()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuWorkspace));
        if (_isAllocated) return;

        _offsets.Clear();
        int offset = 0;
        for (int i = 0; i < _shapes.Count; i++)
        {
            _offsets.Add(offset);
            int slotSize = 1;
            foreach (int dim in _shapes[i])
                slotSize = checked(slotSize * dim);
            offset = checked(offset + slotSize);
        }

        TotalElements = offset;
        _buffer = _backend.AllocateWorkspaceBuffer(TotalElements);
        _isAllocated = true;
    }

    /// <summary>
    /// Gets the GPU buffer offset and length for a slot.
    /// Use these to create sub-buffer views for GPU operations.
    /// </summary>
    /// <param name="slotId">The slot ID returned by <see cref="Register"/>.</param>
    /// <returns>Offset and length within the workspace buffer.</returns>
    public (int offset, int length) GetSlotRegion(int slotId)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuWorkspace));
        if (!_isAllocated) throw new InvalidOperationException("Not allocated.");
        if (slotId < 0 || slotId >= _shapes.Count)
            throw new ArgumentOutOfRangeException(nameof(slotId));

        int length = 1;
        foreach (int dim in _shapes[slotId])
            length *= dim;

        return (_offsets[slotId], length);
    }

    /// <summary>
    /// Gets a copy of the shape for a slot.
    /// </summary>
    public int[] GetShape(int slotId)
    {
        if (slotId < 0 || slotId >= _shapes.Count)
            throw new ArgumentOutOfRangeException(nameof(slotId));
        return (int[])_shapes[slotId].Clone();
    }

    /// <summary>
    /// Resets the workspace for re-registration. Disposes the GPU buffer.
    /// </summary>
    public void Reset()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuWorkspace));
        _shapes.Clear();
        _offsets.Clear();
        _isAllocated = false;
        _buffer?.Dispose();
        _buffer = null;
    }

    /// <summary>
    /// Executes a batch of GPU operations using workspace slots.
    /// All operations are recorded into a single command buffer and submitted at once.
    /// </summary>
    /// <param name="operations">Actions that dispatch GPU operations using workspace buffer regions.</param>
    public void ExecuteBatch(Action<GpuWorkspace> operations)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuWorkspace));
        if (!_isAllocated || _buffer == null)
            throw new InvalidOperationException("Not allocated.");

        if (_backend.SupportsBatchExecution)
        {
            _backend.BeginBatch();
            try
            {
                operations(this);
            }
            finally
            {
                _backend.EndBatch();
            }
        }
        else
        {
            // Fallback: execute immediately (no batching)
            operations(this);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _buffer?.Dispose();
        _buffer = null;
    }
}
