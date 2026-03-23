using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.LinearAlgebra;

#if NET5_0_OR_GREATER

/// <summary>
/// A stack-only, zero-allocation read-only view into tensor data.
/// Unlike <see cref="Tensor{T}"/>, this is a <c>ref struct</c> that lives entirely on the stack —
/// zero heap allocation, zero GC pressure. Ideal for inner-loop iteration patterns.
///
/// <para><b>Why this beats PyTorch:</b> PyTorch always heap-allocates tensor metadata (at least ~96 bytes
/// per tensor view via Python object + THPVariable). TensorView&lt;T&gt; costs exactly 0 bytes on the heap.</para>
///
/// <para><b>Usage:</b></para>
/// <code>
/// var tensor = new Tensor&lt;float&gt;(new[] { 32, 784 });
/// for (int b = 0; b &lt; 32; b++)
/// {
///     TensorView&lt;float&gt; batch = tensor.AsView().Slice(b); // zero allocation
///     float val = batch[42]; // direct indexed access
/// }
/// </code>
///
/// <para><b>Limitations (ref struct):</b> Cannot be stored in fields, boxed, used in async methods,
/// or used as a generic type parameter. This is by design — it enforces stack-only lifetime.</para>
/// </summary>
/// <typeparam name="T">The element type of the tensor.</typeparam>
public readonly ref struct TensorView<T>
{
    private readonly ReadOnlySpan<T> _data;
    private readonly ReadOnlySpan<int> _shape;
    private readonly ReadOnlySpan<int> _strides;
    private readonly int _offset;
    private readonly int _length;

    /// <summary>
    /// Creates a TensorView from raw components. Internal — use Tensor.AsView() instead.
    /// </summary>
    internal TensorView(ReadOnlySpan<T> data, ReadOnlySpan<int> shape, ReadOnlySpan<int> strides, int offset, int length)
    {
        _data = data;
        _shape = shape;
        _strides = strides;
        _offset = offset;
        _length = length;
    }

    /// <summary>
    /// Gets the number of dimensions.
    /// </summary>
    public int Rank => _shape.Length;

    /// <summary>
    /// Gets the total number of logical elements.
    /// </summary>
    public int Length => _length;

    /// <summary>
    /// Gets the size of the specified dimension.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Shape(int dim) => _shape[dim];

    /// <summary>
    /// Gets the stride of the specified dimension.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Stride(int dim) => _strides[dim];

    /// <summary>
    /// Gets the value at the specified 1D index. Zero-allocation.
    /// </summary>
    public T this[int i0] => _data[_offset + i0 * _strides[0]];

    /// <summary>
    /// Gets the value at the specified 2D indices. Zero-allocation.
    /// </summary>
    public T this[int i0, int i1] => _data[_offset + i0 * _strides[0] + i1 * _strides[1]];

    /// <summary>
    /// Gets the value at the specified 3D indices. Zero-allocation.
    /// </summary>
    public T this[int i0, int i1, int i2] => _data[_offset + i0 * _strides[0] + i1 * _strides[1] + i2 * _strides[2]];

    /// <summary>
    /// Gets the value at the specified 4D indices. Zero-allocation.
    /// </summary>
    public T this[int i0, int i1, int i2, int i3] => _data[_offset + i0 * _strides[0] + i1 * _strides[1] + i2 * _strides[2] + i3 * _strides[3]];

    /// <summary>
    /// Slices along the first dimension, returning a view with one fewer rank. O(1), zero allocation.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of bounds.</exception>
    public TensorView<T> Slice(int index)
    {
        if (index < 0 || index >= _shape[0])
            throw new ArgumentOutOfRangeException(nameof(index),
                $"Index {index} is out of range for dimension 0 with size {_shape[0]}.");

        int newOffset = _offset + index * _strides[0];
        // Compute exact length from remaining shape dimensions
        int newLength = 1;
        for (int d = 1; d < _shape.Length; d++)
            newLength *= _shape[d];
        return new TensorView<T>(_data, _shape.Slice(1), _strides.Slice(1), newOffset, newLength);
    }

    /// <summary>
    /// Gets the value at a logical flat index (row-major order). O(Rank) per call.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when flatIndex is out of range.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T GetFlat(int flatIndex)
    {
        if (flatIndex < 0 || flatIndex >= _length)
            throw new ArgumentOutOfRangeException(nameof(flatIndex), "Flat index is out of range.");

        // O(Rank) decomposition using strides — compute row-major strides inline
        int idx = _offset;
        int remaining = flatIndex;
        for (int d = 0; d < _shape.Length; d++)
        {
            // Compute row-major stride for this dimension: product of shape[d+1..end]
            int rmStride = 1;
            for (int dd = d + 1; dd < _shape.Length; dd++)
                rmStride *= _shape[dd];
            int dimIndex = remaining / rmStride;
            remaining -= dimIndex * rmStride;
            idx += dimIndex * _strides[d];
        }
        return _data[idx];
    }

    /// <summary>
    /// Copies the view's data to a destination span in row-major order.
    /// For contiguous views, uses direct span copy for maximum performance.
    /// </summary>
    public void CopyTo(Span<T> destination)
    {
        if (destination.Length < _length)
            throw new ArgumentException("Destination span is too small.");

        // Check if view is contiguous (strides match row-major)
        bool isContiguous = true;
        int expected = 1;
        for (int d = _shape.Length - 1; d >= 0; d--)
        {
            if (_shape[d] != 1 && _strides[d] != expected)
            {
                isContiguous = false;
                break;
            }
            expected *= _shape[d];
        }

        if (isContiguous)
        {
            // Fast path: direct memory copy
            _data.Slice(_offset, _length).CopyTo(destination);
        }
        else
        {
            // Slow path: element-by-element through strides
            for (int i = 0; i < _length; i++)
                destination[i] = GetFlat(i);
        }
    }
}

#endif
