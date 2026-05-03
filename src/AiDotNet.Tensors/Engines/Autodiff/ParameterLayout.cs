using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Describes the storage layout of a single trainable parameter inside a
/// <see cref="ParameterBuffer{T}"/>. A layout is either dense (the
/// historical case — the entire <see cref="DenseShape"/> participates in
/// training) or sparse, in which case only the non-zero positions of a
/// <see cref="LinearAlgebra.SparseTensor{T}"/> are stored and trained.
/// Sparsity pattern is fixed at construction; only values move during
/// training (this matches PyTorch's <c>torch.sparse.Parameter</c>
/// contract — pattern is metadata, not a learnable quantity).
/// </summary>
/// <remarks>
/// <para><b>Why a per-leaf layout instead of just a shape:</b>
/// Sparse leaves like <c>SparseLinearLayer&lt;T&gt;</c>'s weight matrix
/// have a dense semantic shape (e.g. <c>[outFeatures, inFeatures]</c>)
/// but only allocate <c>NonZeroCount</c> trainable slots. The older
/// <c>IReadOnlyList&lt;int[]&gt;</c> ctor sized the buffer slot off the
/// dense shape and would have wasted <c>O(rows × columns)</c> memory
/// per sparse leaf — defeating the entire memory-efficiency promise of
/// sparse layers. <see cref="ParameterLayout"/> lets callers describe
/// which leaves should occupy only their pattern's worth of buffer space
/// while still preserving the dense shape for shape inference and
/// serialization.</para>
///
/// <para><b>Immutability:</b> Both the dense shape and the sparsity
/// pattern are cloned at construction so layouts are immutable by
/// ownership: subsequent mutations to caller-supplied arrays cannot
/// desynchronize <see cref="ParameterBuffer{T}.CreateView"/> from the
/// actual storage layout. The public <c>DenseShape</c> getter exposes
/// the layout's own copy; sparse pattern indices are exposed only via
/// <see cref="System.ReadOnlyMemory{T}"/> views to prevent external
/// mutation of the layout's storage.</para>
/// </remarks>
public sealed class ParameterLayout
{
    private readonly int[] _denseShape;

    /// <summary>
    /// The dense semantic shape of the parameter. For sparse leaves this
    /// is the shape of the underlying dense matrix the sparse pattern
    /// projects onto (e.g. <c>[rows, columns]</c>); for dense leaves it
    /// is the actual storage shape. Returns a <em>fresh clone</em> on
    /// every access so callers cannot mutate the layout's backing
    /// storage through this property; <see cref="BufferElementCount"/>
    /// computes against the private backing field, so even if a caller
    /// retained the array reference and wrote to it, derived values
    /// stay correct.
    /// </summary>
    public int[] DenseShape => (int[])_denseShape.Clone();

    /// <summary>
    /// Returns the layout's <c>DenseShape</c> as a non-allocating
    /// <see cref="ReadOnlySpan{T}"/> over the backing storage. Internal
    /// callers (e.g. <see cref="ParameterBuffer{T}"/>) prefer this over
    /// the public <see cref="DenseShape"/> getter to avoid the per-call
    /// clone allocation. NEVER expose externally — the span aliases the
    /// layout's own storage and breaking the immutability contract is
    /// only safe inside the assembly.
    /// </summary>
    internal ReadOnlySpan<int> DenseShapeSpan => _denseShape;

    /// <summary>
    /// Length of the dense shape. Cheap when callers only need the rank
    /// — avoids the per-call clone allocation of <see cref="DenseShape"/>.
    /// </summary>
    public int Rank => _denseShape.Length;

    /// <summary>
    /// Sparsity layout when the parameter is sparse, or <c>null</c> for
    /// a dense parameter.
    /// </summary>
    public SparsityLayout? Sparse { get; }

    /// <summary>
    /// True when this parameter is sparse (only its pattern's non-zero
    /// values are stored in the buffer).
    /// </summary>
    public bool IsSparse => Sparse is not null;

    /// <summary>
    /// Creates a dense layout from a shape array.
    /// </summary>
    public ParameterLayout(int[] denseShape)
    {
        if (denseShape is null) throw new ArgumentNullException(nameof(denseShape));
        // Clone so the layout's shape can't be mutated via the caller's
        // array after construction. The fixed-shape contract is enforced
        // by ownership rather than caller discipline.
        _denseShape = (int[])denseShape.Clone();
        Sparse = null;
    }

    /// <summary>
    /// Creates a sparse layout. <paramref name="denseShape"/> must match
    /// <c>[sparse.Rows, sparse.Columns]</c>.
    /// </summary>
    public ParameterLayout(int[] denseShape, SparsityLayout sparse)
    {
        if (denseShape is null) throw new ArgumentNullException(nameof(denseShape));
        Sparse = sparse ?? throw new ArgumentNullException(nameof(sparse));
        if (denseShape.Length != 2)
            throw new ArgumentException(
                $"Sparse parameter dense shape must be rank-2 [rows, columns]; got rank {denseShape.Length}.",
                nameof(denseShape));
        if (denseShape[0] != sparse.Rows || denseShape[1] != sparse.Columns)
            throw new ArgumentException(
                $"Dense shape [{denseShape[0]}, {denseShape[1]}] does not match sparse pattern " +
                $"[{sparse.Rows}, {sparse.Columns}].",
                nameof(denseShape));
        // Clone after validation so the layout retains an immutable copy
        // even if the caller mutates their array afterwards.
        _denseShape = (int[])denseShape.Clone();
    }

    /// <summary>
    /// Number of trainable scalars in this parameter's buffer slot.
    /// Dense: product of <see cref="DenseShape"/>. Sparse: pattern's
    /// non-zero count.
    /// </summary>
    internal long BufferElementCount
    {
        get
        {
            if (Sparse is { } s) return s.NonZeroCount;
            long product = 1;
            // Iterate the private backing field directly so this
            // computation never depends on the public DenseShape getter
            // — even a hostile caller mutating a previously-returned
            // clone cannot perturb BufferElementCount.
            foreach (int d in _denseShape) product *= d;
            return product;
        }
    }
}

/// <summary>
/// Captures a sparse parameter's fixed pattern: COO row/column indices.
/// The constructor takes its OWN COPIES of the supplied index arrays
/// (via <see cref="System.Array.Clone"/>), so subsequent mutations to
/// the caller's arrays do not affect this layout. The fixed-pattern
/// contract is enforced by ownership: the public
/// <see cref="RowIndices"/> / <see cref="ColumnIndices"/> properties
/// return a <em>fresh copy on every access</em>, so even
/// <see cref="System.Runtime.InteropServices.MemoryMarshal.TryGetArray{T}(System.ReadOnlyMemory{T},out System.ArraySegment{T})"/>
/// can only recover the per-call snapshot — never the layout's own
/// backing arrays.
/// </summary>
public sealed class SparsityLayout
{
    private readonly int[] _rowIndices;
    private readonly int[] _columnIndices;

    /// <summary>Dense matrix row count.</summary>
    public int Rows { get; }

    /// <summary>Dense matrix column count.</summary>
    public int Columns { get; }

    /// <summary>
    /// Row indices of the non-zero positions, in COO order. Length =
    /// <see cref="NonZeroCount"/>. Returns a <em>fresh
    /// <see cref="System.ReadOnlyMemory{T}"/> over a per-call array
    /// copy</em>, so even
    /// <see cref="System.Runtime.InteropServices.MemoryMarshal.TryGetArray{T}(System.ReadOnlyMemory{T},out System.ArraySegment{T})"/>
    /// cannot reach the layout's backing storage. If you need
    /// allocation-free indexed access from inside the assembly, use
    /// <see cref="RowIndicesSpan"/>.
    /// </summary>
    public ReadOnlyMemory<int> RowIndices => new ReadOnlyMemory<int>((int[])_rowIndices.Clone());

    /// <summary>
    /// Column indices of the non-zero positions, in COO order. Length =
    /// <see cref="NonZeroCount"/>. Same copy-on-access semantics as
    /// <see cref="RowIndices"/>; pair with
    /// <see cref="ColumnIndicesSpan"/> when calling from inside the
    /// assembly to avoid the per-call allocation.
    /// </summary>
    public ReadOnlyMemory<int> ColumnIndices => new ReadOnlyMemory<int>((int[])_columnIndices.Clone());

    /// <summary>
    /// Allocation-free <see cref="ReadOnlySpan{T}"/> over the underlying
    /// row-index storage. Internal callers iterating the pattern in a
    /// hot loop should use this instead of <see cref="RowIndices"/> to
    /// avoid the per-call clone. NEVER expose externally — the span
    /// aliases the layout's own storage and is only safe inside the
    /// assembly.
    /// </summary>
    internal ReadOnlySpan<int> RowIndicesSpan => _rowIndices;

    /// <summary>
    /// Allocation-free <see cref="ReadOnlySpan{T}"/> over the underlying
    /// column-index storage. Same caveats as <see cref="RowIndicesSpan"/>.
    /// </summary>
    internal ReadOnlySpan<int> ColumnIndicesSpan => _columnIndices;

    /// <summary>Number of non-zero values stored.</summary>
    public int NonZeroCount => _rowIndices.Length;

    /// <summary>
    /// Builds a layout from explicit row/column index arrays. The arrays
    /// must have equal length; out-of-range indices throw
    /// <see cref="ArgumentOutOfRangeException"/>.
    /// </summary>
    public SparsityLayout(int rows, int columns, int[] rowIndices, int[] columnIndices)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns < 0) throw new ArgumentOutOfRangeException(nameof(columns));
        if (rowIndices is null) throw new ArgumentNullException(nameof(rowIndices));
        if (columnIndices is null) throw new ArgumentNullException(nameof(columnIndices));
        if (rowIndices.Length != columnIndices.Length)
            throw new ArgumentException(
                $"Row index length ({rowIndices.Length}) must equal column index length " +
                $"({columnIndices.Length}).");
        for (int i = 0; i < rowIndices.Length; i++)
        {
            if (rowIndices[i] < 0 || rowIndices[i] >= rows)
                throw new ArgumentOutOfRangeException(
                    nameof(rowIndices),
                    $"rowIndices[{i}] = {rowIndices[i]} out of range [0, {rows}).");
            if (columnIndices[i] < 0 || columnIndices[i] >= columns)
                throw new ArgumentOutOfRangeException(
                    nameof(columnIndices),
                    $"columnIndices[{i}] = {columnIndices[i]} out of range [0, {columns}).");
        }

        Rows = rows;
        Columns = columns;
        // Clone caller-owned arrays so the fixed-pattern contract is
        // enforced even if the caller mutates their copy after
        // construction. Without the clones, an external mutation would
        // silently desynchronise SparsityLayout from the buffer slots
        // that depend on these indices being immutable.
        _rowIndices = (int[])rowIndices.Clone();
        _columnIndices = (int[])columnIndices.Clone();
    }

    /// <summary>
    /// Builds a layout from an existing <see cref="SparseTensor{T}"/>'s
    /// COO pattern. The layout takes its own COPY of the index arrays
    /// (via the regular ctor) so subsequent mutations to the source
    /// tensor's pattern don't desynchronise this layout.
    /// </summary>
    public static SparsityLayout FromSparseTensor<T>(SparseTensor<T> source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        // ToCoo() guarantees RowIndices / ColumnIndices are populated
        // regardless of the source's underlying storage format. The
        // SparsityLayout ctor will clone them for us.
        var coo = source.ToCoo();
        return new SparsityLayout(coo.Rows, coo.Columns, coo.RowIndices, coo.ColumnIndices);
    }
}
