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
    private readonly SparsityLayout?[] _sparseLayouts;
    private readonly int _totalSize;

    // Held by callers that need to atomically swap, observe, and restore
    // the buffer's contents — most importantly TensorFunc.FunctionalCall.
    // Internal so TensorFunc can `lock (buffer.SwapLock)` for the full
    // swap/call/restore sequence without exposing the lock object on
    // the public surface; not meant for hot-path coordination.
    private readonly object _swapLock = new();

    /// <summary>
    /// Synchronisation root for atomic multi-step operations against
    /// this buffer's contents (swap, call, restore). Held by
    /// <see cref="Transforms.TensorFunc{T}.FunctionalCall"/> for the
    /// full call so concurrent invocations on the same buffer cannot
    /// observe each other's intermediate state.
    /// </summary>
    internal object SwapLock => _swapLock;

    /// <summary>
    /// Creates a new parameter buffer sized to hold all parameter tensors with the given shapes.
    /// </summary>
    /// <param name="parameterShapes">The shapes of each parameter tensor, in order.</param>
    public ParameterBuffer(IReadOnlyList<int[]> parameterShapes)
        : this(WrapAsDenseLayouts(parameterShapes))
    {
    }

    /// <summary>
    /// Creates a parameter buffer that supports BOTH dense and sparse
    /// trainable parameters. Each <see cref="ParameterLayout"/> describes
    /// either a dense leaf (full-shape slab) or a sparse leaf (only the
    /// pattern's non-zero values are stored).
    /// </summary>
    /// <remarks>
    /// <para>Sparse leaves are the production-ready path for training
    /// <c>SparseLinearLayer&lt;T&gt;</c> and similar pattern-fixed layers
    /// without paying the memory cost of a dense shadow. The buffer slot
    /// for a sparse leaf is sized at <c>NonZeroCount</c> (the pattern's
    /// non-zero count) instead of <c>rows × columns</c>; views over the
    /// slot reconstruct a <see cref="SparseTensor{T}"/> using the
    /// pattern's row/column indices and the sliced values vector.</para>
    ///
    /// <para>The dense ctor (<see cref="ParameterBuffer(IReadOnlyList{int[]})"/>)
    /// remains for backward compatibility and delegates here with
    /// all-dense layouts.</para>
    /// </remarks>
    /// <param name="layouts">Per-parameter layout descriptors.</param>
    public ParameterBuffer(IReadOnlyList<ParameterLayout> layouts)
    {
        if (layouts is null) throw new ArgumentNullException(nameof(layouts));
        _shapes = new int[layouts.Count][];
        _offsets = new int[layouts.Count];
        _sparseLayouts = new SparsityLayout?[layouts.Count];

        int offset = 0;
        for (int i = 0; i < layouts.Count; i++)
        {
            var layout = layouts[i] ?? throw new ArgumentNullException(
                nameof(layouts), $"layouts[{i}] is null.");
            _shapes[i] = (int[])layout.DenseShape.Clone();
            _sparseLayouts[i] = layout.Sparse;
            _offsets[i] = offset;

            // Dense slot size = product of dense shape; sparse slot size =
            // pattern's NonZeroCount. The dense-shape validation still runs
            // for sparse leaves (we want a sane semantic shape recorded for
            // shape-inference and serialization paths) but the buffer
            // allocation uses the cheaper sparse count.
            long size = 1;
            foreach (int dim in _shapes[i])
            {
                if (dim < 0)
                    throw new ArgumentException($"Shape dimension must be non-negative, got {dim} in parameter {i}.");
                size *= dim;
                if (size > int.MaxValue)
                    throw new OverflowException(
                        $"Parameter {i} dense shape produces more than {int.MaxValue} elements. " +
                        "ParameterBuffer uses int indexing; reduce parameter sizes or use multiple buffers.");
            }

            long bufferElements = layout.BufferElementCount;
            if (bufferElements > int.MaxValue)
                throw new OverflowException(
                    $"Parameter {i} buffer size ({bufferElements}) exceeds int.MaxValue.");
            offset = checked(offset + (int)bufferElements);
        }

        _totalSize = offset;
        _data = new Vector<T>(_totalSize);
        _storage = new TensorStorage<T>(_data);
    }

    private static IReadOnlyList<ParameterLayout> WrapAsDenseLayouts(IReadOnlyList<int[]> shapes)
    {
        if (shapes is null) throw new ArgumentNullException(nameof(shapes));
        var wrapped = new ParameterLayout[shapes.Count];
        for (int i = 0; i < shapes.Count; i++)
            wrapped[i] = new ParameterLayout(shapes[i]);
        return wrapped;
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
    /// Returns the sparsity layout for the i-th parameter when it is
    /// sparse; <c>null</c> otherwise. Use <see cref="IsSparse"/> for a
    /// quick boolean test before reading.
    /// </summary>
    public SparsityLayout? GetSparseLayout(int index) => _sparseLayouts[index];

    /// <summary>
    /// Whether the i-th parameter is stored as a sparse-pattern values vector.
    /// </summary>
    public bool IsSparse(int index) => _sparseLayouts[index] is not null;

    /// <summary>
    /// Creates a tensor view into this buffer at the specified parameter index.
    /// Dense parameters return a <see cref="Tensor{T}"/> view sharing storage
    /// with the buffer. Sparse parameters return a <see cref="SparseTensor{T}"/>
    /// whose <c>Values</c> share storage with the buffer slice and whose
    /// row/column indices come from the layout — mutations to the buffer
    /// flow into the sparse tensor's values automatically.
    /// </summary>
    /// <param name="index">The parameter index (0-based, in the order layouts were provided).</param>
    /// <returns>A tensor view backed by this buffer's storage.</returns>
    public Tensor<T> CreateView(int index)
    {
        var shape = _shapes[index];
        var sparse = _sparseLayouts[index];
        if (sparse is null)
        {
            var strides = ComputeRowMajorStrides(shape);
            return new Tensor<T>(_data, shape, strides, _offsets[index], _storage);
        }

        // Sparse leaf: build a SparseTensor whose Values array is a
        // dedicated copy of the buffer slice. The SparseTensor ctor wraps
        // the values via Vector<T>.Wrap, which takes the array reference —
        // we deliberately copy here so the SparseTensor's storage isn't
        // aliased to the buffer's TensorStorage (which would let two
        // different storage objects view the same memory and break
        // tape-side identity comparisons). Updates flow back to the buffer
        // through CopyFrom(IReadOnlyList<Tensor<T>>) at end-of-step or
        // through the explicit values-vector accessors below.
        var values = new T[sparse.NonZeroCount];
        var bufferSpan = _data.AsSpan().Slice(_offsets[index], sparse.NonZeroCount);
        bufferSpan.CopyTo(values);
        return new SparseTensor<T>(sparse.Rows, sparse.Columns,
            sparse.RowIndices, sparse.ColumnIndices, values);
    }

    /// <summary>
    /// Returns a writable span over the i-th sparse parameter's non-zero
    /// values directly inside the buffer. Mutations flow into the buffer
    /// in-place. Used by sparse-aware autograd ops and optimizers to
    /// push values updates back without rebuilding the SparseTensor.
    /// Throws if the i-th parameter is dense.
    /// </summary>
    public Span<T> GetSparseValuesSpan(int index)
    {
        var sparse = _sparseLayouts[index]
            ?? throw new InvalidOperationException(
                $"Parameter {index} is dense; use CreateView instead.");
        return _data.AsWritableSpan().Slice(_offsets[index], sparse.NonZeroCount);
    }

    /// <summary>
    /// Read-only counterpart of <see cref="GetSparseValuesSpan"/> for
    /// inspecting a sparse leaf's current values without copying.
    /// </summary>
    public ReadOnlySpan<T> GetSparseValuesReadOnlySpan(int index)
    {
        var sparse = _sparseLayouts[index]
            ?? throw new InvalidOperationException(
                $"Parameter {index} is dense; use CreateView instead.");
        return _data.AsSpan().Slice(_offsets[index], sparse.NonZeroCount);
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
            var sparseLayout = _sparseLayouts[i];

            if (sparseLayout is not null)
            {
                // Sparse leaf: source must be a SparseTensor<T> with the
                // same pattern. Copy only the Values vector into the slot
                // — the buffer slot was sized at NonZeroCount, not the
                // full dense shape.
                if (param is not SparseTensor<T> sparseParam)
                    throw new ArgumentException(
                        $"Parameter {i} is registered as sparse but the source tensor is dense. " +
                        "Convert to SparseTensor with the matching COO pattern before copying.",
                        nameof(parameters));
                if (sparseParam.Rows != sparseLayout.Rows || sparseParam.Columns != sparseLayout.Columns)
                    throw new ArgumentException(
                        $"Parameter {i} sparse shape [{sparseParam.Rows}, {sparseParam.Columns}] " +
                        $"does not match layout [{sparseLayout.Rows}, {sparseLayout.Columns}].",
                        nameof(parameters));
                // Validate the source's COO pattern matches our layout.
                // Pattern is fixed at layer init; a mismatch here means the
                // caller passed a different sparse tensor than the layer
                // registered — fail loudly rather than silently copying
                // values into wrong-pattern positions.
                var coo = sparseParam.ToCoo();
                if (coo.RowIndices.Length != sparseLayout.NonZeroCount)
                    throw new ArgumentException(
                        $"Parameter {i} non-zero count ({coo.RowIndices.Length}) does not match " +
                        $"layout NonZeroCount ({sparseLayout.NonZeroCount}). The sparsity pattern " +
                        "is fixed at construction time.",
                        nameof(parameters));
                for (int k = 0; k < sparseLayout.NonZeroCount; k++)
                {
                    if (coo.RowIndices[k] != sparseLayout.RowIndices[k]
                        || coo.ColumnIndices[k] != sparseLayout.ColumnIndices[k])
                    {
                        throw new ArgumentException(
                            $"Parameter {i} COO pattern at index {k} differs from layout " +
                            $"(source [{coo.RowIndices[k]}, {coo.ColumnIndices[k]}] vs layout " +
                            $"[{sparseLayout.RowIndices[k]}, {sparseLayout.ColumnIndices[k]}]). " +
                            "Sparsity pattern is fixed.",
                            nameof(parameters));
                    }
                }

                // Copy Values into the buffer slot.
                var valuesSrc = coo.DataVector.AsSpan()
                    .Slice(coo._storageOffset, sparseLayout.NonZeroCount);
                var dst = bufferSpan.Slice(_offsets[i], sparseLayout.NonZeroCount);
                valuesSrc.CopyTo(dst);
                continue;
            }

            // Dense leaf — original path.
            Tensor<T> contiguous;
            if (param.IsSparse)
            {
                // Caller registered this leaf as dense but is passing a
                // sparse tensor. Densify so the dense slot fills correctly.
                contiguous = ((SparseTensor<T>)param).ToDense();
            }
            else
            {
                contiguous = param.IsContiguous ? param : param.Contiguous();
            }
            var srcData = contiguous.DataVector;
            var src = srcData.AsSpan().Slice(contiguous._storageOffset, contiguous.Length);
            int expectedSize = 1;
            foreach (int d in _shapes[i]) expectedSize *= d;
            if (src.Length != expectedSize)
                throw new ArgumentException(
                    $"Parameter {i} length ({src.Length}) does not match expected shape size. " +
                    "Ensure parameter tensors match the shapes provided at buffer construction.");
            var denseDst = bufferSpan.Slice(_offsets[i], contiguous.Length);
            src.CopyTo(denseDst);
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
            if (!gradients.TryGetValue(parameters[i], out var grad)) continue;

            var sparseLayout = _sparseLayouts[i];
            if (sparseLayout is not null)
            {
                // Sparse leaf — the gradient slot is sized at
                // NonZeroCount. Two source forms to handle:
                //   (a) sparse gradient with matching pattern: copy Values
                //       directly (zero allocation).
                //   (b) dense gradient (the standard PyTorch return form):
                //       gather only the values at the pattern's positions.
                //       Densifying then copying would overflow the slot.
                var dst = gradSpan.Slice(_offsets[i], sparseLayout.NonZeroCount);

                if (grad is SparseTensor<T> sparseGrad
                    && sparseGrad.Rows == sparseLayout.Rows
                    && sparseGrad.Columns == sparseLayout.Columns
                    && PatternsMatch(sparseGrad.ToCoo(), sparseLayout))
                {
                    var coo = sparseGrad.ToCoo();
                    var valuesSrc = coo.DataVector.AsSpan()
                        .Slice(coo._storageOffset, sparseLayout.NonZeroCount);
                    valuesSrc.CopyTo(dst);
                }
                else
                {
                    // Dense gradient: gather at pattern positions. Use the
                    // multi-dim indexer so strided / view tensors work.
                    Tensor<T> denseGrad = grad.IsSparse
                        ? ((SparseTensor<T>)grad).ToDense()
                        : grad;
                    if (denseGrad.Rank != 2)
                        throw new ArgumentException(
                            $"Sparse leaf {i} expects a rank-2 dense gradient or a matching sparse " +
                            $"gradient; got rank {denseGrad.Rank}.");
                    if (denseGrad.Shape[0] != sparseLayout.Rows
                        || denseGrad.Shape[1] != sparseLayout.Columns)
                        throw new ArgumentException(
                            $"Sparse leaf {i} dense gradient shape " +
                            $"[{denseGrad.Shape[0]}, {denseGrad.Shape[1]}] does not match layout " +
                            $"[{sparseLayout.Rows}, {sparseLayout.Columns}].");
                    for (int k = 0; k < sparseLayout.NonZeroCount; k++)
                    {
                        dst[k] = denseGrad[sparseLayout.RowIndices[k], sparseLayout.ColumnIndices[k]];
                    }
                }
                continue;
            }

            // Dense leaf — original path.
            Tensor<T> contiguous;
            if (grad.IsSparse)
                contiguous = ((SparseTensor<T>)grad).ToDense();
            else
                contiguous = grad.IsContiguous ? grad : grad.Contiguous();
            var denseSrcData = contiguous.DataVector;
            var denseSrc = denseSrcData.AsSpan().Slice(contiguous._storageOffset, contiguous.Length);
            int copyLen = Math.Min(denseSrc.Length, parameters[i].Length);
            var denseDst = gradSpan.Slice(_offsets[i], copyLen);
            denseSrc.Slice(0, copyLen).CopyTo(denseDst);
        }

        return flatGrad;
    }

    private static bool PatternsMatch(SparseTensor<T> coo, SparsityLayout layout)
    {
        if (coo.RowIndices.Length != layout.NonZeroCount) return false;
        for (int k = 0; k < layout.NonZeroCount; k++)
        {
            if (coo.RowIndices[k] != layout.RowIndices[k]
                || coo.ColumnIndices[k] != layout.ColumnIndices[k])
                return false;
        }
        return true;
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
