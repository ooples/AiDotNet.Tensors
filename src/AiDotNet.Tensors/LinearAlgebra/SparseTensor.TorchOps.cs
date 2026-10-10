using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// torch's sparse COO operations on <see cref="SparseTensor{T}"/>: indices, sparse_mask, the sparse-dense products
/// (smm, hsmm, hspmm, saddmm, sspaddmm) and the in-place resizes (sparse_resize_, sparse_resize_and_clear_,
/// resize_as_sparse_).
/// </summary>
public partial class SparseTensor<T>
{
    /// <summary>
    /// The COO indices as a [2, nnz] tensor, row indices then column indices, in coalesced (row-major, duplicates
    /// summed) order (torch <c>Tensor.indices()</c>, which requires a coalesced tensor).
    /// </summary>
    public Tensor<int> Indices()
    {
        var c = Coalesce();
        int nnz = c.NonZeroCount;
        var data = new int[2 * nnz];
        Array.Copy(c.RowIndices, 0, data, 0, nnz);
        Array.Copy(c.ColumnIndices, 0, data, nnz, nnz);
        return new Tensor<int>(data, new[] { 2, nnz });
    }

    /// <summary>
    /// A sparse tensor with <paramref name="mask"/>'s sparsity pattern and <paramref name="source"/>'s values at those
    /// positions (torch <c>source.sparse_mask(mask)</c>). A sparse source contributes zero where it stores nothing.
    /// </summary>
    public static SparseTensor<T> SparseMask(Tensor<T> source, SparseTensor<T> mask)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (mask is null) throw new ArgumentNullException(nameof(mask));
        if (source.Rank != 2 || source.Shape[0] != mask.Rows || source.Shape[1] != mask.Columns)
            throw new ArgumentException($"The source must be [{mask.Rows}, {mask.Columns}] like the mask.", nameof(source));
        var dense = source is SparseTensor<T> s ? s.ToDense() : source;
        var pattern = mask.Coalesce();
        int nnz = pattern.NonZeroCount;
        var values = new T[nnz];
        for (int i = 0; i < nnz; i++) values[i] = dense[pattern.RowIndices[i], pattern.ColumnIndices[i]];
        return new SparseTensor<T>(mask.Rows, mask.Columns, (int[])pattern.RowIndices.Clone(), (int[])pattern.ColumnIndices.Clone(), values);
    }

    /// <summary>
    /// <paramref name="sparse"/> @ <paramref name="dense"/> as a sparse tensor (torch.smm): every entry of each row
    /// that holds a stored element, zeros included, as torch stores them.
    /// </summary>
    public static SparseTensor<T> Smm(SparseTensor<T> sparse, Tensor<T> dense)
    {
        var (rows, product) = RowProduct(sparse, dense);
        int n = dense.Shape[1];
        var ri = new int[rows.Count * n]; var ci = new int[rows.Count * n]; var v = new T[rows.Count * n];
        int k = 0;
        foreach (int r in rows)
            for (int c = 0; c < n; c++) { ri[k] = r; ci[k] = c; v[k] = product[r * n + c]; k++; }
        return new SparseTensor<T>(sparse.Rows, n, ri, ci, v);
    }

    /// <summary>torch.hsmm: the same product as <see cref="Smm"/>.</summary>
    public static SparseTensor<T> Hsmm(SparseTensor<T> sparse, Tensor<T> dense) => Smm(sparse, dense);

    /// <summary>
    /// torch.hspmm: <paramref name="sparse"/> @ <paramref name="dense"/> with the rows that hold a stored element kept
    /// whole. torch returns it as a hybrid tensor (sparse rows of dense rows); this 2-D sparse tensor stores the same
    /// rows and values.
    /// </summary>
    public static SparseTensor<T> Hspmm(SparseTensor<T> sparse, Tensor<T> dense) => Smm(sparse, dense);

    /// <summary>
    /// torch.saddmm: <paramref name="beta"/> * <paramref name="input"/> + <paramref name="alpha"/> *
    /// (<paramref name="mat1"/> @ <paramref name="mat2"/>), sparse and coalesced.
    /// </summary>
    public static SparseTensor<T> Saddmm(SparseTensor<T> input, SparseTensor<T> mat1, Tensor<T> mat2, T beta, T alpha)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var product = Smm(mat1, mat2);
        if (input.Rows != product.Rows || input.Columns != product.Columns)
            throw new ArgumentException($"input is [{input.Rows}, {input.Columns}] but mat1 @ mat2 is [{product.Rows}, {product.Columns}].", nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        var a = input.ToCoo(); var b = product;
        int na = a.NonZeroCount, nb = b.NonZeroCount;
        var ri = new int[na + nb]; var ci = new int[na + nb]; var v = new T[na + nb];
        var av = a.DataVector.AsSpan(); var bv = b.DataVector.AsSpan();
        for (int i = 0; i < na; i++) { ri[i] = a.RowIndices[i]; ci[i] = a.ColumnIndices[i]; v[i] = ops.Multiply(beta, av[i]); }
        for (int i = 0; i < nb; i++) { ri[na + i] = b.RowIndices[i]; ci[na + i] = b.ColumnIndices[i]; v[na + i] = ops.Multiply(alpha, bv[i]); }
        return new SparseTensor<T>(input.Rows, input.Columns, ri, ci, v).Coalesce();
    }

    /// <summary>torch.sspaddmm: the same sum as <see cref="Saddmm"/> (torch leaves it uncoalesced; the values are equal).</summary>
    public static SparseTensor<T> Sspaddmm(SparseTensor<T> input, SparseTensor<T> mat1, Tensor<T> mat2, T beta, T alpha)
        => Saddmm(input, mat1, mat2, beta, alpha);

    // The dense product restricted to the rows of `sparse` that store something: (those rows, row-major product).
    private static (List<int> Rows, T[] Product) RowProduct(SparseTensor<T> sparse, Tensor<T> dense)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (dense.Rank != 2 || dense.Shape[0] != sparse.Columns)
            throw new ArgumentException($"Expected a dense [{sparse.Columns}, N] operand.", nameof(dense));
        var ops = MathHelper.GetNumericOperations<T>();
        var c = sparse.Coalesce();
        int n = dense.Shape[1];
        var d = dense.IsContiguous ? dense : dense.Contiguous();
        var ds = d.AsSpan();
        var product = new T[sparse.Rows * n];
        for (int i = 0; i < product.Length; i++) product[i] = ops.Zero;
        var rows = new List<int>();
        var vals = c.DataVector.AsSpan();
        for (int e = 0; e < c.NonZeroCount; e++)
        {
            int r = c.RowIndices[e], k = c.ColumnIndices[e];
            if (rows.Count == 0 || rows[rows.Count - 1] != r) rows.Add(r);   // coalesced: rows ascend
            T a = vals[e];
            for (int j = 0; j < n; j++) product[r * n + j] = ops.Add(product[r * n + j], ops.Multiply(a, ds[k * n + j]));
        }
        return (rows, product);
    }

    /// <summary>
    /// Resizes this sparse tensor in place to [<paramref name="rows"/>, <paramref name="columns"/>] (torch
    /// <c>sparse_resize_</c>), keeping its elements. As in torch, a tensor that stores elements can only grow.
    /// </summary>
    public void SparseResize(int rows, int columns)
    {
        if (rows < 0 || columns < 0) throw new ArgumentOutOfRangeException(rows < 0 ? nameof(rows) : nameof(columns), "A dimension cannot be negative.");
        if (NonZeroCount > 0 && (rows < Rows || columns < Columns))
            throw new InvalidOperationException(
                $"Shrinking the size of the sparse dimensions (from [{Rows}, {Columns}] to [{rows}, {columns}]) of a non-empty sparse " +
                "tensor is not supported; clear it first (SparseResizeAndClear) or build a new tensor.");
        if (BlockRowSize > 0 && (rows % BlockRowSize != 0 || columns % BlockColSize != 0))
            throw new ArgumentException($"A block-sparse tensor's size must stay a multiple of its {BlockRowSize}x{BlockColSize} blocks.");
        if (Format == SparseStorageFormat.Csr || Format == SparseStorageFormat.Bsr)
            RowPointers = ExtendPointers(RowPointers, BlockRowSize > 0 ? rows / BlockRowSize : rows);
        if (Format == SparseStorageFormat.Csc || Format == SparseStorageFormat.Bsc)
            ColumnPointers = ExtendPointers(ColumnPointers, BlockColSize > 0 ? columns / BlockColSize : columns);
        Rows = rows;
        Columns = columns;
        ReplaceSparseLayout(new[] { rows, columns }, null);
    }

    /// <summary>
    /// Resizes this sparse tensor in place to [<paramref name="rows"/>, <paramref name="columns"/>] and drops every
    /// stored element (torch <c>sparse_resize_and_clear_</c>).
    /// </summary>
    public void SparseResizeAndClear(int rows, int columns)
    {
        if (rows < 0 || columns < 0) throw new ArgumentOutOfRangeException(rows < 0 ? nameof(rows) : nameof(columns), "A dimension cannot be negative.");
        if (BlockRowSize > 0 && (rows % BlockRowSize != 0 || columns % BlockColSize != 0))
            throw new ArgumentException($"A block-sparse tensor's size must stay a multiple of its {BlockRowSize}x{BlockColSize} blocks.");
        RowIndices = RowIndices.Length == 0 ? RowIndices : Array.Empty<int>();
        ColumnIndices = ColumnIndices.Length == 0 ? ColumnIndices : Array.Empty<int>();
        if (Format == SparseStorageFormat.Csr || Format == SparseStorageFormat.Bsr)
            RowPointers = new int[(BlockRowSize > 0 ? rows / BlockRowSize : rows) + 1];
        if (Format == SparseStorageFormat.Csc || Format == SparseStorageFormat.Bsc)
            ColumnPointers = new int[(BlockColSize > 0 ? columns / BlockColSize : columns) + 1];
        Rows = rows;
        Columns = columns;
        ReplaceSparseLayout(new[] { rows, columns }, Vector<T>.Wrap(Array.Empty<T>()));
    }

    /// <summary>Resizes this sparse tensor in place to <paramref name="other"/>'s size (torch <c>resize_as_sparse_</c>).</summary>
    public void ResizeAsSparse(SparseTensor<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        if (other.Rows != Rows || other.Columns != Columns) SparseResize(other.Rows, other.Columns);
    }

    // CSR/CSC pointers for more rows (columns): the new ones are empty, so they repeat the last pointer.
    private static int[] ExtendPointers(int[] pointers, int count)
    {
        if (pointers.Length == count + 1) return pointers;
        var p = new int[count + 1];
        int last = pointers.Length == 0 ? 0 : pointers[pointers.Length - 1];
        for (int i = 0; i <= count; i++) p[i] = i < pointers.Length ? pointers[i] : last;
        return p;
    }
}
