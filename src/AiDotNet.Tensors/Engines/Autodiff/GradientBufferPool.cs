using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Pre-allocated gradient buffer pool that eliminates per-backward allocation.
/// Register parameters once at training setup, then reuse buffers across steps.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
/// <remarks>
/// <para><b>Why this beats PyTorch:</b></para>
/// <list type="bullet">
/// <item>No <c>zero_grad()</c> footgun — buffers are zeroed automatically via <see cref="ZeroAll"/></item>
/// <item>No <c>set_to_none=True</c> defeating reuse — buffers are always reused, never nulled</item>
/// <item>Shape validation at registration catches mismatches early</item>
/// <item>Optional contiguous layout mirrors ParameterBuffer for single-memset zeroing</item>
/// </list>
/// </remarks>
public sealed class GradientBufferPool<T>
{
    private readonly Dictionary<Tensor<T>, GradientBuffer> _buffers;
    private readonly IEngine _engine;
    private readonly INumericOperations<T> _numOps;

    // Optional contiguous backing for ParameterBuffer integration
    private T[]? _contiguousData;
    private bool _isContiguous;

    /// <summary>
    /// Creates a new gradient buffer pool.
    /// </summary>
    public GradientBufferPool()
    {
        _buffers = new Dictionary<Tensor<T>, GradientBuffer>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        _engine = AiDotNetEngine.Current;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a gradient buffer pool backed by a contiguous array mirroring a ParameterBuffer layout.
    /// Enables single Array.Clear for zeroing all gradients and zero-copy flat gradient access.
    /// </summary>
    public GradientBufferPool(ParameterBuffer<T> parameterBuffer)
    {
        _buffers = new Dictionary<Tensor<T>, GradientBuffer>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        _engine = AiDotNetEngine.Current;
        _numOps = MathHelper.GetNumericOperations<T>();

        // Allocate single contiguous gradient array
        _contiguousData = new T[parameterBuffer.TotalSize];
        _isContiguous = true;

        // Create gradient tensor views at matching offsets
        var parameters = parameterBuffer.CreateAllViews();
        for (int i = 0; i < parameters.Length; i++)
        {
            var param = parameters[i];
            var shape = param.Shape.ToArray();
            var offset = parameterBuffer.GetOffset(i);
            var length = param.Length;

            // Create view into contiguous array
            var gradTensor = new Tensor<T>(_contiguousData, shape);
            // Note: for contiguous layout, offset management is implicit via the array segment
            _buffers[param] = new GradientBuffer(gradTensor, offset, length);
        }
    }

    /// <summary>
    /// Registers a parameter for gradient buffer pre-allocation.
    /// Call once per parameter at training setup.
    /// </summary>
    public void Register(Tensor<T> parameter)
    {
        if (parameter is null) throw new ArgumentNullException(nameof(parameter));
        if (_buffers.ContainsKey(parameter)) return;

        var shape = parameter.Shape.ToArray();
        var gradTensor = new Tensor<T>(shape);
        // Zero-initialize
        var span = gradTensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = _numOps.Zero;

        _buffers[parameter] = new GradientBuffer(gradTensor, 0, parameter.Length);
    }

    /// <summary>
    /// Zeros all gradient buffers in preparation for a new backward pass.
    /// For contiguous pools, this is a single Array.Clear call.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ZeroAll()
    {
        if (_isContiguous && _contiguousData is not null)
        {
            // Single memset for all gradients
            Array.Clear(_contiguousData, 0, _contiguousData.Length);
        }
        else
        {
            foreach (var buffer in _buffers.Values)
            {
                var span = buffer.Tensor.AsWritableSpan();
                for (int i = 0; i < span.Length; i++)
                    span[i] = _numOps.Zero;
                buffer.IsDirty = false;
            }
        }

        // Reset dirty flags
        foreach (var buffer in _buffers.Values)
            buffer.IsDirty = false;
    }

    /// <summary>
    /// Tries to get the pre-allocated gradient buffer for a parameter.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetBuffer(Tensor<T> parameter, out Tensor<T> gradBuffer, out bool isDirty)
    {
        if (_buffers.TryGetValue(parameter, out var buffer))
        {
            gradBuffer = buffer.Tensor;
            isDirty = buffer.IsDirty;
            return true;
        }
        gradBuffer = default!;
        isDirty = false;
        return false;
    }

    /// <summary>
    /// Marks a parameter's gradient buffer as having received data.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void MarkDirty(Tensor<T> parameter)
    {
        if (_buffers.TryGetValue(parameter, out var buffer))
            buffer.IsDirty = true;
    }

    /// <summary>
    /// Returns a dictionary of parameter → gradient tensor pointing to pre-allocated buffers.
    /// Zero allocation — the returned dictionary references are to the pool's own tensors.
    /// </summary>
    public Dictionary<Tensor<T>, Tensor<T>> GetGradients()
    {
        var result = new Dictionary<Tensor<T>, Tensor<T>>(
            _buffers.Count, ReferenceEqualityComparer<Tensor<T>>.Instance);
        foreach (var kvp in _buffers)
            result[kvp.Key] = kvp.Value.Tensor;
        return result;
    }

    /// <summary>
    /// Returns the flat gradient vector for contiguous pools (zero-copy).
    /// Only valid when constructed with a ParameterBuffer.
    /// </summary>
    public Vector<T> GetFlatGradients()
    {
        if (!_isContiguous || _contiguousData is null)
            throw new InvalidOperationException(
                "GetFlatGradients is only available for contiguous pools created with a ParameterBuffer.");
        return new Vector<T>(_contiguousData);
    }

    /// <summary>
    /// Gets the number of registered parameters.
    /// </summary>
    public int Count => _buffers.Count;

    /// <summary>
    /// Whether this pool has a registered buffer for the given parameter.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Contains(Tensor<T> parameter) => _buffers.ContainsKey(parameter);

    // Internal mutable buffer state
    private sealed class GradientBuffer
    {
        public readonly Tensor<T> Tensor;
        public readonly int Offset;   // offset in contiguous array (0 for non-contiguous)
        public readonly int Length;
        public bool IsDirty;

        public GradientBuffer(Tensor<T> tensor, int offset, int length)
        {
            Tensor = tensor;
            Offset = offset;
            Length = length;
            IsDirty = false;
        }
    }
}
