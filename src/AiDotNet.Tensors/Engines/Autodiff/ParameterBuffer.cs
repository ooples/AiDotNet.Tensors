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
    // Most fields are immutable post-construction in the original API.
    // ResizeSparseLeaf is the one operation that can replace _data /
    // _storage / _totalSize (when a sparse leaf's NonZeroCount changes
    // and the buffer must grow/shrink). All other code paths treat
    // these as readonly-after-ctor.
    private TensorStorage<T> _storage;
    private Vector<T> _data;
    private readonly int[] _offsets;
    private readonly int[][] _shapes;
    private readonly SparsityLayout?[] _sparseLayouts;
    private int _totalSize;

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

            // Dense leaves: validate shape dimensions are non-negative AND
            // the dense product fits in int (since the buffer slot is
            // sized by that product).
            // Sparse leaves: the buffer slot is sized by NonZeroCount, NOT
            // the dense product. Only validate non-negativity of each dim
            // (so callers can declare e.g. a 1M × 1M sparse parameter with
            // 100 non-zeros without tripping the dense int.MaxValue
            // overflow check that would otherwise defeat the entire
            // memory-efficiency promise of sparse leaves).
            foreach (int dim in _shapes[i])
            {
                if (dim < 0)
                    throw new ArgumentException($"Shape dimension must be non-negative, got {dim} in parameter {i}.");
            }
            if (!layout.IsSparse)
            {
                long denseSize = 1;
                foreach (int dim in _shapes[i])
                {
                    denseSize *= dim;
                    if (denseSize > int.MaxValue)
                        throw new OverflowException(
                            $"Parameter {i} dense shape produces more than {int.MaxValue} elements. " +
                            "ParameterBuffer uses int indexing; reduce parameter sizes or use multiple buffers.");
                }
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

        // Sparse leaf: zero-copy view. Vector<T>.CreateSlice wraps a
        // window of the buffer's underlying memory (no allocation, no
        // copy), and the SparseTensor(Vector<T>) ctor takes that vector
        // directly without re-wrapping. Mutations to the buffer flow
        // into the returned SparseTensor's Values, and vice versa —
        // matches the dense-leaf contract where CreateView returns a
        // live view of the buffer.
        var valuesVector = _data.CreateSlice(_offsets[index], sparse.NonZeroCount);
        // RowIndicesArray / ColumnIndicesArray are internal accessors that
        // hand SparseTensor the layout's own backing int[] without an
        // extra defensive copy. The layout's immutability is preserved
        // because SparseTensor only reads the index arrays.
        return new SparseTensor<T>(sparse.Rows, sparse.Columns,
            sparse.RowIndicesArray, sparse.ColumnIndicesArray, valuesVector);
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
    /// Replaces a sparse leaf's pattern in place when the new pattern
    /// has the SAME <c>NonZeroCount</c> as the current one. Use this for
    /// dynamic-sparsity training where the sparsity ratio is fixed but
    /// the active positions move between optimizer steps (rigid pruning,
    /// SET / RigL / threshold-based dynamic-sparse training).
    /// </summary>
    /// <remarks>
    /// <para>The buffer slot is unchanged in size and offset, so other
    /// leaves' offsets are unaffected. <paramref name="newValues"/> is
    /// optional — if omitted the existing values stay in place at their
    /// new pattern positions, which the caller must understand (the
    /// position–value association is now relative to the new indices).
    /// </para>
    /// <para>
    /// <b>View invalidation:</b> any <see cref="SparseTensor{T}"/>
    /// instances obtained via <see cref="CreateView"/> before this call
    /// were constructed with the old pattern's RowIndices /
    /// ColumnIndices baked in — they will continue to interpret the
    /// (now-shared) values slot using the OLD coordinates. Callers
    /// MUST re-fetch views via <see cref="CreateView"/> after rebuilding
    /// the pattern. (The same contract applies to
    /// <see cref="ResizeSparseLeaf"/>, where the buffer reallocation
    /// makes view invalidation even more obvious.)
    /// </para>
    /// <para>For pattern changes that also alter the count, use
    /// <see cref="ResizeSparseLeaf"/> which reallocates the buffer.</para>
    /// </remarks>
    public void RebuildSparsePattern(int index, SparsityLayout newLayout, ReadOnlySpan<T> newValues = default)
    {
        if (newLayout is null) throw new ArgumentNullException(nameof(newLayout));
        var current = _sparseLayouts[index]
            ?? throw new InvalidOperationException(
                $"Parameter {index} is dense; cannot replace its sparsity pattern.");

        if (newLayout.Rows != current.Rows || newLayout.Columns != current.Columns)
            throw new ArgumentException(
                $"New pattern dense shape [{newLayout.Rows}, {newLayout.Columns}] must match " +
                $"existing [{current.Rows}, {current.Columns}]; use ResizeSparseLeaf for shape changes.",
                nameof(newLayout));
        if (newLayout.NonZeroCount != current.NonZeroCount)
            throw new ArgumentException(
                $"RebuildSparsePattern requires the new NonZeroCount ({newLayout.NonZeroCount}) " +
                $"to match the existing ({current.NonZeroCount}). Use ResizeSparseLeaf for size changes.",
                nameof(newLayout));

        _sparseLayouts[index] = newLayout;

        if (!newValues.IsEmpty)
        {
            if (newValues.Length != newLayout.NonZeroCount)
                throw new ArgumentException(
                    $"newValues length ({newValues.Length}) must equal NonZeroCount " +
                    $"({newLayout.NonZeroCount}).",
                    nameof(newValues));
            var dst = _data.AsWritableSpan().Slice(_offsets[index], newLayout.NonZeroCount);
            newValues.CopyTo(dst);
        }
    }

    /// <summary>
    /// Replaces a sparse leaf's layout AND reallocates the buffer to
    /// accommodate the new <c>NonZeroCount</c>. Other leaves' values
    /// are preserved; their offsets shift to reflect the new sparse
    /// slot size. Use this for fully variable-pattern training (positions
    /// AND count both change) — e.g. when a model dynamically prunes /
    /// regrows connections.
    /// </summary>
    /// <remarks>
    /// <para>Cost: <see cref="O(TotalSize)"/> for the buffer copy and
    /// <see cref="O(Count)"/> for the offset rebuild. Tensor views
    /// returned by previous <see cref="CreateView"/> calls are
    /// invalidated — re-create them after this method returns.</para>
    /// </remarks>
    public void ResizeSparseLeaf(int index, SparsityLayout newLayout, ReadOnlySpan<T> newValues = default)
    {
        if (newLayout is null) throw new ArgumentNullException(nameof(newLayout));
        var current = _sparseLayouts[index]
            ?? throw new InvalidOperationException(
                $"Parameter {index} is dense; cannot resize a dense slot via this API.");

        if (newLayout.Rows != current.Rows || newLayout.Columns != current.Columns)
            throw new ArgumentException(
                $"New pattern dense shape [{newLayout.Rows}, {newLayout.Columns}] must match " +
                $"existing [{current.Rows}, {current.Columns}].",
                nameof(newLayout));

        if (!newValues.IsEmpty && newValues.Length != newLayout.NonZeroCount)
            throw new ArgumentException(
                $"newValues length ({newValues.Length}) must equal NonZeroCount " +
                $"({newLayout.NonZeroCount}).",
                nameof(newValues));

        int oldNnz = current.NonZeroCount;
        int newNnz = newLayout.NonZeroCount;
        int delta = newNnz - oldNnz;

        if (delta == 0)
        {
            // No size change — fall through to the in-place rebuild path.
            RebuildSparsePattern(index, newLayout, newValues);
            return;
        }

        // Reallocate the underlying vector with the new total size.
        // Preserve every other leaf's contents at their (possibly shifted)
        // new offsets. The new sparse leaf's slot lives at the SAME
        // offset as before; subsequent leaves shift by `delta`.
        int newTotal = checked(_totalSize + delta);
        var newData = new Vector<T>(newTotal);
        var oldSpan = _data.AsSpan();
        var newSpan = newData.AsWritableSpan();

        // Copy everything before the resized slot (unchanged).
        int prefixLen = _offsets[index];
        if (prefixLen > 0)
            oldSpan.Slice(0, prefixLen).CopyTo(newSpan.Slice(0, prefixLen));

        // Fill the resized slot with newValues if provided, else zeros.
        if (!newValues.IsEmpty)
        {
            newValues.CopyTo(newSpan.Slice(_offsets[index], newNnz));
        }
        // (Default case: leave zeros — Vector<T> allocates zero-initialized.)

        // Copy the suffix (everything after the resized slot) to its new
        // offset. The sparse leaf's slot grew/shrunk by `delta`, so the
        // suffix starts `delta` positions later in the new buffer.
        int oldSuffixStart = _offsets[index] + oldNnz;
        int newSuffixStart = _offsets[index] + newNnz;
        int suffixLen = _totalSize - oldSuffixStart;
        if (suffixLen > 0)
            oldSpan.Slice(oldSuffixStart, suffixLen).CopyTo(newSpan.Slice(newSuffixStart, suffixLen));

        // Replace the storage. Existing tensor views obtained via
        // CreateView before this Resize point at the OLD vector and are
        // now stale — callers must re-create views after a Resize. The
        // buffer's own read paths (CreateView, GetSparseValuesSpan, ...)
        // use _data which we update here.
        _data = newData;
        _storage = new TensorStorage<T>(newData);
        _totalSize = newTotal;

        _sparseLayouts[index] = newLayout;
        // Shift subsequent offsets by delta.
        for (int i = index + 1; i < _offsets.Length; i++)
        {
            _offsets[i] += delta;
        }
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
                // Hoist the ReadOnlySpan views once so the property
                // dispatch and ReadOnlyMemory→Span conversion don't run
                // on every k.
                var layoutRows = sparseLayout.RowIndices.Span;
                var layoutCols = sparseLayout.ColumnIndices.Span;
                for (int k = 0; k < sparseLayout.NonZeroCount; k++)
                {
                    if (coo.RowIndices[k] != layoutRows[k]
                        || coo.ColumnIndices[k] != layoutCols[k])
                    {
                        throw new ArgumentException(
                            $"Parameter {i} COO pattern at index {k} differs from layout " +
                            $"(source [{coo.RowIndices[k]}, {coo.ColumnIndices[k]}] vs layout " +
                            $"[{layoutRows[k]}, {layoutCols[k]}]). " +
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
                    && sparseGrad.Columns == sparseLayout.Columns)
                {
                    // ToCoo can allocate / canonicalize ordering, so call
                    // it ONCE and reuse the result for both pattern check
                    // and value copy. Hot path on every backward step.
                    var coo = sparseGrad.ToCoo();
                    if (PatternsMatch(coo, sparseLayout))
                    {
                        var valuesSrc = coo.DataVector.AsSpan()
                            .Slice(coo._storageOffset, sparseLayout.NonZeroCount);
                        valuesSrc.CopyTo(dst);
                        continue;
                    }
                    // Pattern mismatch — fall through to dense projection.
                }
                {
                    // Dense gradient (or pattern-mismatched sparse that
                    // fell through above): gather at pattern positions.
                    // Use the multi-dim indexer so strided / view tensors
                    // work.
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
                    var layoutRows = sparseLayout.RowIndices.Span;
                    var layoutCols = sparseLayout.ColumnIndices.Span;
                    for (int k = 0; k < sparseLayout.NonZeroCount; k++)
                    {
                        dst[k] = denseGrad[layoutRows[k], layoutCols[k]];
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
        // Take spans once outside the loop so the property dispatch and
        // ReadOnlyMemory→ReadOnlySpan conversion don't run on every k.
        var layoutRows = layout.RowIndices.Span;
        var layoutCols = layout.ColumnIndices.Span;
        for (int k = 0; k < layout.NonZeroCount; k++)
        {
            if (coo.RowIndices[k] != layoutRows[k]
                || coo.ColumnIndices[k] != layoutCols[k])
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
