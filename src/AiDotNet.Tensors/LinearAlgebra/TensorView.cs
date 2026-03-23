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
    public TensorView<T> Slice(int index)
    {
        int newOffset = _offset + index * _strides[0];
        return new TensorView<T>(_data, _shape.Slice(1), _strides.Slice(1), newOffset, _length / _shape[0]);
    }

    /// <summary>
    /// Gets the value at a logical flat index (row-major order).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T GetFlat(int flatIndex)
    {
        int idx = _offset;
        int remaining = flatIndex;
        for (int d = 0; d < _shape.Length; d++)
        {
            int stride = 1;
            for (int dd = d + 1; dd < _shape.Length; dd++)
                stride *= _shape[dd];
            int dimIndex = remaining / stride;
            remaining -= dimIndex * stride;
            idx += dimIndex * _strides[d];
        }
        return _data[idx];
    }

    /// <summary>
    /// Copies the view's data to a destination span in row-major order.
    /// </summary>
    public void CopyTo(Span<T> destination)
    {
        if (destination.Length < _length)
            throw new ArgumentException("Destination span is too small.");

        for (int i = 0; i < _length; i++)
            destination[i] = GetFlat(i);
    }
}

#endif
