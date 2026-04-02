using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// A contiguous memory buffer that backs all trainable parameter tensors in a model.
/// Parameter tensors are views into this buffer, enabling zero-copy access to the full
/// parameter vector for second-order optimizers, serialization, and distributed training.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
/// <remarks>
/// <para><b>Design:</b> Follows the PyTorch parameter flattening pattern used by
/// FSDP (Fully Sharded Data Parallel) and <c>torch.nn.utils.parameters_to_vector</c>.
/// A single contiguous allocation eliminates the flatten/unflatten overhead that
/// second-order optimizers (BFGS, L-BFGS, Trust Region) otherwise pay per step.</para>
///
/// <para><b>Performance advantages:</b>
/// <list type="bullet">
/// <item><b>Zero-copy for second-order optimizers:</b> The flat parameter vector IS the
/// buffer — no allocation, no copying. <c>AsVector()</c> returns the backing vector directly.</item>
/// <item><b>Single allocation:</b> One large array instead of hundreds of small ones.
/// Reduces GC pressure and improves cache locality for sequential parameter scans.</item>
/// <item><b>GPU-friendly:</b> A single contiguous buffer can be transferred to/from GPU
/// in one DMA operation instead of per-tensor transfers.</item>
/// <item><b>View-based parameters:</b> Layer weight tensors are views (shared storage +
/// offset) into the buffer. In-place updates by first-order optimizers automatically
/// reflect in the flat vector, and vice versa.</item>
/// </list>
/// </para>
///
/// <para><b>Usage:</b>
/// <code>
/// // Collect shapes from layers
/// var shapes = layers.SelectMany(l => l.GetTrainableParameters().Select(p => p.Shape.ToArray()));
/// var buffer = new ParameterBuffer&lt;float&gt;(shapes);
///
/// // Replace layer parameter tensors with views into the buffer
/// // Create parameter views backed by the buffer
/// var views = buffer.CreateAllViews();
///
/// // Initialize buffer from existing layer weights
/// buffer.CopyFrom(existingParams);
///
/// // Second-order optimizer operates directly on the buffer — zero copy
/// Vector&lt;float&gt; flatParams = buffer.AsVector();
/// Vector&lt;float&gt; flatGrads = buffer.FlattenGradients(views, gradients);
/// Vector&lt;float&gt; updated = bfgsOptimizer.UpdateParameters(flatParams, flatGrads);
/// buffer.CopyFrom(updated);  // writes back through views to all layers
/// </code>
/// </para>
///
/// <para><b>For Beginners:</b> Think of this as a single long shelf that holds all the
/// weights of every layer in your neural network side by side. Instead of each layer
/// having its own separate box of weights, they all share this shelf — each layer just
/// knows which section of the shelf belongs to it. This makes it very fast to look at
/// all weights at once (which advanced optimizers need to do).</para>
/// </remarks>
public sealed class ParameterBuffer<T>
{
    private readonly TensorStorage<T> _storage;
    private readonly Vector<T> _data;
    private readonly int[] _offsets;
    private readonly int[][] _shapes;
    private readonly int _totalSize;

    /// <summary>
    /// Creates a new parameter buffer sized to hold all parameter tensors with the given shapes.
    /// </summary>
    /// <param name="parameterShapes">The shapes of each parameter tensor, in order.</param>
    public ParameterBuffer(IReadOnlyList<int[]> parameterShapes)
    {
        _shapes = new int[parameterShapes.Count][];
        _offsets = new int[parameterShapes.Count];

        int offset = 0;
        for (int i = 0; i < parameterShapes.Count; i++)
        {
            _shapes[i] = (int[])parameterShapes[i].Clone();
            _offsets[i] = offset;
            long size = 1;
            foreach (int dim in _shapes[i])
            {
                if (dim < 0)
                    throw new ArgumentException($"Shape dimension must be non-negative, got {dim} in parameter {i}.");
                size *= dim;
                if (size > int.MaxValue)
                    throw new OverflowException(
                        $"Parameter {i} shape produces more than {int.MaxValue} elements. " +
                        "ParameterBuffer uses int indexing; reduce parameter sizes or use multiple buffers.");
            }
            offset = checked(offset + (int)size);
        }

        _totalSize = offset;
        _data = new Vector<T>(_totalSize);
        _storage = new TensorStorage<T>(_data);
    }

    /// <summary>
    /// Gets the total number of parameters across all tensors in this buffer.
    /// </summary>
    public int TotalSize => _totalSize;

    /// <summary>
    /// Gets the number of parameter tensors backed by this buffer.
    /// </summary>
    public int Count => _shapes.Length;

    /// <summary>
    /// Returns the flat parameter vector. This is the actual backing data — zero copy.
    /// Second-order optimizers can operate directly on this vector.
    /// </summary>
    public Vector<T> AsVector() => _data;

    /// <summary>
    /// Returns the underlying storage for creating tensor views.
    /// </summary>
    internal TensorStorage<T> Storage => _storage;

    /// <summary>
    /// Gets the element offset (index) into the buffer where the i-th parameter tensor begins.
    /// </summary>
    /// <param name="index">The parameter index.</param>
    /// <returns>The element offset.</returns>
    public int GetOffset(int index) => _offsets[index];

    /// <summary>
    /// Gets the shape of the i-th parameter tensor.
    /// </summary>
    /// <param name="index">The parameter index.</param>
    /// <returns>A copy of the shape array.</returns>
    public int[] GetShape(int index) => (int[])_shapes[index].Clone();

    /// <summary>
    /// Creates a tensor view into this buffer at the specified parameter index.
    /// The returned tensor shares storage with the buffer — mutations are bidirectional.
    /// </summary>
    /// <param name="index">The parameter index (0-based, in the order shapes were provided).</param>
    /// <returns>A tensor view backed by this buffer's storage.</returns>
    public Tensor<T> CreateView(int index)
    {
        var shape = _shapes[index];
        var strides = ComputeRowMajorStrides(shape);
        return new Tensor<T>(_data, shape, strides, _offsets[index], _storage);
    }

    /// <summary>
    /// Creates tensor views for all parameters in this buffer.
    /// </summary>
    /// <returns>An array of tensor views, one per parameter.</returns>
    public Tensor<T>[] CreateAllViews()
    {
        var views = new Tensor<T>[_shapes.Length];
        for (int i = 0; i < _shapes.Length; i++)
            views[i] = CreateView(i);
        return views;
    }

    /// <summary>
    /// Copies data from an existing set of parameter tensors into this buffer.
    /// Use this to initialize the buffer from a model's current weights.
    /// </summary>
    /// <param name="parameters">The parameter tensors to copy from, in order.</param>
    public void CopyFrom(IReadOnlyList<Tensor<T>> parameters)
    {
        if (parameters.Count != _shapes.Length)
            throw new ArgumentException(
                $"Expected {_shapes.Length} parameter tensors, got {parameters.Count}.",
                nameof(parameters));

        var bufferSpan = _data.AsWritableSpan();
        for (int i = 0; i < parameters.Count; i++)
        {
            var param = parameters[i];
            // Ensure contiguous layout before copying raw storage
            var contiguous = param.IsContiguous ? param : param.Contiguous();
            var srcData = contiguous.DataVector;
            var src = srcData.AsSpan().Slice(contiguous._storageOffset, contiguous.Length);
            int expectedSize = 1;
            foreach (int d in _shapes[i]) expectedSize *= d;
            if (src.Length != expectedSize)
                throw new ArgumentException(
                    $"Parameter {i} length ({src.Length}) does not match expected shape size. " +
                    "Ensure parameter tensors match the shapes provided at buffer construction.");
            var dst = bufferSpan.Slice(_offsets[i], contiguous.Length);
            src.CopyTo(dst);
        }
    }

    /// <summary>
    /// Copies data from a flat vector into this buffer.
    /// Use this after a second-order optimizer returns an updated parameter vector.
    /// </summary>
    /// <param name="source">The flat parameter vector to copy from.</param>
    public void CopyFrom(Vector<T> source)
    {
        if (source.Length != _totalSize)
            throw new ArgumentException(
                $"Source vector length ({source.Length}) must match buffer size ({_totalSize}).",
                nameof(source));

        source.AsSpan().CopyTo(_data.AsWritableSpan());
    }

    /// <summary>
    /// Creates a flat gradient vector with the same layout as this buffer.
    /// Use this to accumulate gradients in a contiguous vector that matches
    /// the parameter layout for second-order optimizers.
    /// </summary>
    /// <returns>A zero-initialized vector of the same size as this buffer.</returns>
    public Vector<T> CreateGradientVector() => new Vector<T>(_totalSize);

    /// <summary>
    /// Flattens per-parameter gradient tensors into a contiguous vector matching
    /// this buffer's layout. Zero-fills gaps for parameters without gradients.
    /// </summary>
    /// <param name="parameters">The parameter tensors (same order as buffer).</param>
    /// <param name="gradients">Gradient dictionary keyed by parameter tensor identity.</param>
    /// <returns>A flat gradient vector aligned with the buffer layout.</returns>
    public Vector<T> FlattenGradients(IReadOnlyList<Tensor<T>> parameters, Dictionary<Tensor<T>, Tensor<T>> gradients)
    {
        var flatGrad = new Vector<T>(_totalSize);
        var gradSpan = flatGrad.AsWritableSpan();

        for (int i = 0; i < parameters.Count; i++)
        {
            if (gradients.TryGetValue(parameters[i], out var grad))
            {
                var contiguous = grad.IsContiguous ? grad : grad.Contiguous();
                var srcData = contiguous.DataVector;
                var src = srcData.AsSpan().Slice(contiguous._storageOffset, contiguous.Length);
                int copyLen = Math.Min(src.Length, parameters[i].Length);
                var dst = gradSpan.Slice(_offsets[i], copyLen);
                src.Slice(0, copyLen).CopyTo(dst);
            }
        }

        return flatGrad;
    }

    private static int[] ComputeRowMajorStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        if (shape.Length == 0) return strides;
        strides[shape.Length - 1] = 1;
        for (int i = shape.Length - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
        return strides;
    }
}
