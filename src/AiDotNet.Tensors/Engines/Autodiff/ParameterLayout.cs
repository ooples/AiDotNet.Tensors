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
/// <para><b>Sparsity pattern immutability:</b> The pattern's row and
/// column index arrays are captured by reference at layout creation and
/// must not be mutated afterwards. Mutating them while the buffer is
/// live would desynchronize <see cref="ParameterBuffer{T}.CreateView"/>
/// from the actual storage layout.</para>
/// </remarks>
public sealed class ParameterLayout
{
    /// <summary>
    /// The dense semantic shape of the parameter. For sparse leaves this
    /// is the shape of the underlying dense matrix the sparse pattern
    /// projects onto (e.g. <c>[rows, columns]</c>); for dense leaves it
    /// is the actual storage shape.
    /// </summary>
    public int[] DenseShape { get; }

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
        DenseShape = denseShape ?? throw new ArgumentNullException(nameof(denseShape));
        Sparse = null;
    }

    /// <summary>
    /// Creates a sparse layout. <paramref name="denseShape"/> must match
    /// <c>[sparse.Rows, sparse.Columns]</c>.
    /// </summary>
    public ParameterLayout(int[] denseShape, SparsityLayout sparse)
    {
        DenseShape = denseShape ?? throw new ArgumentNullException(nameof(denseShape));
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
            foreach (int d in DenseShape) product *= d;
            return product;
        }
    }
}

/// <summary>
/// Captures a sparse parameter's fixed pattern: COO row/column indices.
/// The constructor takes its OWN COPIES of the supplied index arrays
/// (via <see cref="System.Array.Clone"/>), so subsequent mutations to
/// the caller's arrays do not affect this layout. The fixed-pattern
/// contract is enforced by ownership, not by caller discipline.
/// </summary>
public sealed class SparsityLayout
{
    /// <summary>Dense matrix row count.</summary>
    public int Rows { get; }

    /// <summary>Dense matrix column count.</summary>
    public int Columns { get; }

    /// <summary>
    /// Row indices of the non-zero positions, in COO order. Length =
    /// <see cref="NonZeroCount"/>. This is the layout's own copy, taken
    /// at construction; mutating the caller's source array after
    /// construction has no effect here.
    /// </summary>
    public int[] RowIndices { get; }

    /// <summary>
    /// Column indices of the non-zero positions, in COO order. Length =
    /// <see cref="NonZeroCount"/>. This is the layout's own copy, taken
    /// at construction; mutating the caller's source array after
    /// construction has no effect here.
    /// </summary>
    public int[] ColumnIndices { get; }

    /// <summary>Number of non-zero values stored.</summary>
    public int NonZeroCount => RowIndices.Length;

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
        RowIndices = (int[])rowIndices.Clone();
        ColumnIndices = (int[])columnIndices.Clone();
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
