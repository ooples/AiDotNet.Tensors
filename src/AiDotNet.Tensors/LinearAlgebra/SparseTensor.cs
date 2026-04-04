using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Represents a 2D sparse tensor with COO/CSR/CSC storage, inheriting from <see cref="Tensor{T}"/>
/// to get GPU residency, gradient tape tracking, parameter buffer, and serialization for free.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// SparseTensor inherits from Tensor, following the PyTorch model where sparse and dense tensors
/// share the same base type. The backing <see cref="Tensor{T}.DataVector"/> contains only non-zero
/// values (length = <see cref="NonZeroCount"/>), while <see cref="TensorBase{T}.Shape"/> reports
/// the full logical dimensions (e.g., [1000, 1000]).
/// </para>
/// <para><b>Inherited capabilities (zero additional code):</b>
/// <list type="bullet">
/// <item>GPU residency via <c>_gpuBuffer</c> and <c>IsGpuResident</c></item>
/// <item>Gradient tape tracking via <c>_version</c> counter</item>
/// <item>RegisterTrainableParameter compatibility in LayerBase</item>
/// <item>Engine persistent tensor registration for GPU memory management</item>
/// </list>
/// </para>
/// <para><b>Sparse-specific:</b> Index arrays (COO/CSR/CSC) are stored alongside the
/// values. Format conversion methods (ToCoo, ToCsr, ToCsc) create new SparseTensors.
/// <see cref="ToDense"/> materializes the full dense tensor.</para>
/// <para><b>For Beginners:</b> A sparse tensor is like a regular tensor but it only
/// stores the non-zero values to save memory. A 1000×1000 matrix with only 100 non-zero
/// values stores 100 values instead of 1,000,000. It works with all the same training
/// tools (optimizers, gradient tape) as a regular tensor.</para>
/// </remarks>
public class SparseTensor<T> : Tensor<T>
{
    /// <summary>
    /// Gets the number of rows in the logical matrix.
    /// </summary>
    public int Rows { get; }

    /// <summary>
    /// Gets the number of columns in the logical matrix.
    /// </summary>
    public int Columns { get; }

    /// <summary>
    /// Gets the sparse storage format (COO, CSR, or CSC).
    /// </summary>
    public SparseStorageFormat Format { get; }

    /// <summary>
    /// Row indices for COO format, or row indices for CSC format.
    /// Empty for CSR format.
    /// </summary>
    public int[] RowIndices { get; }

    /// <summary>
    /// Column indices for COO format, or column indices for CSR format.
    /// Empty for CSC format.
    /// </summary>
    public int[] ColumnIndices { get; }

    /// <summary>
    /// Row pointers for CSR format. Length = Rows + 1.
    /// Empty for COO and CSC formats.
    /// </summary>
    public int[] RowPointers { get; }

    /// <summary>
    /// Column pointers for CSC format. Length = Columns + 1.
    /// Empty for COO and CSR formats.
    /// </summary>
    public int[] ColumnPointers { get; }

    /// <summary>
    /// Gets the number of non-zero elements stored.
    /// This is the actual length of the backing data, not the logical element count.
    /// </summary>
    public int NonZeroCount => DataVector.Length;

    /// <summary>
    /// Gets the non-zero values as an array. Backward-compatible accessor.
    /// For zero-copy span access, use <c>DataVector.AsSpan()</c> instead.
    /// </summary>
    public T[] Values => DataVector.ToArray();

    /// <summary>
    /// Creates a COO-format sparse tensor.
    /// </summary>
    public SparseTensor(int rows, int columns, int[] rowIndices, int[] columnIndices, T[] values)
        : base(Vector<T>.Wrap(values ?? throw new ArgumentNullException(nameof(values))),
               new[] { rows, columns }, isSparse: true)
    {
        // Note: rows/columns non-negative validation is handled by TensorBase.ValidateShape
        // in the base constructor call above, so no redundant checks here.
        if (rowIndices is null)
            throw new ArgumentNullException(nameof(rowIndices));
        if (columnIndices is null)
            throw new ArgumentNullException(nameof(columnIndices));
        if (rowIndices.Length != columnIndices.Length || rowIndices.Length != values.Length)
            throw new ArgumentException("COO indices and values must have the same length.");

        Rows = rows;
        Columns = columns;
        RowIndices = rowIndices;
        ColumnIndices = columnIndices;
        RowPointers = Array.Empty<int>();
        ColumnPointers = Array.Empty<int>();
        Format = SparseStorageFormat.Coo;
    }

    private SparseTensor(int rows, int columns, SparseStorageFormat format,
        int[] rowIndices, int[] columnIndices, int[] rowPointers, int[] columnPointers, T[] values)
        : base(Vector<T>.Wrap(values), new[] { rows, columns }, isSparse: true)
    {
        Rows = rows;
        Columns = columns;
        Format = format;
        RowIndices = rowIndices;
        ColumnIndices = columnIndices;
        RowPointers = rowPointers;
        ColumnPointers = columnPointers;
    }

    /// <summary>
    /// Creates a CSR-format sparse tensor.
    /// </summary>
    public static SparseTensor<T> FromCsr(int rows, int columns, int[] rowPointers, int[] columnIndices, T[] values)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns < 0) throw new ArgumentOutOfRangeException(nameof(columns));
        if (rowPointers is null) throw new ArgumentNullException(nameof(rowPointers));
        if (columnIndices is null) throw new ArgumentNullException(nameof(columnIndices));
        if (values is null) throw new ArgumentNullException(nameof(values));
        if (rowPointers.Length != rows + 1)
            throw new ArgumentException("RowPointers length must be rows + 1.", nameof(rowPointers));
        if (columnIndices.Length != values.Length)
            throw new ArgumentException("ColumnIndices and Values must have the same length.", nameof(columnIndices));

        return new SparseTensor<T>(rows, columns, SparseStorageFormat.Csr,
            Array.Empty<int>(), columnIndices, rowPointers, Array.Empty<int>(), values);
    }

    /// <summary>
    /// Creates a CSC-format sparse tensor.
    /// </summary>
    public static SparseTensor<T> FromCsc(int rows, int columns, int[] columnPointers, int[] rowIndices, T[] values)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns < 0) throw new ArgumentOutOfRangeException(nameof(columns));
        if (columnPointers is null) throw new ArgumentNullException(nameof(columnPointers));
        if (rowIndices is null) throw new ArgumentNullException(nameof(rowIndices));
        if (values is null) throw new ArgumentNullException(nameof(values));
        if (columnPointers.Length != columns + 1)
            throw new ArgumentException("ColumnPointers length must be columns + 1.", nameof(columnPointers));
        if (rowIndices.Length != values.Length)
            throw new ArgumentException("RowIndices and Values must have the same length.", nameof(rowIndices));

        return new SparseTensor<T>(rows, columns, SparseStorageFormat.Csc,
            rowIndices, Array.Empty<int>(), Array.Empty<int>(), columnPointers, values);
    }

    /// <summary>
    /// Creates a sparse tensor from a dense tensor, keeping only non-zero values.
    /// </summary>
    public static SparseTensor<T> FromDense(Tensor<T> dense)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return FromDense(dense, ops.Zero);
    }

    /// <summary>
    /// Creates a sparse tensor from a dense tensor, keeping only values above the tolerance.
    /// </summary>
    public static SparseTensor<T> FromDense(Tensor<T> dense, T tolerance)
    {
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (dense.Rank != 2)
            throw new ArgumentException("SparseTensor only supports rank-2 tensors.", nameof(dense));

        var ops = MathHelper.GetNumericOperations<T>();
        var rowIndices = new List<int>();
        var colIndices = new List<int>();
        var values = new List<T>();

        int rows = dense._shape[0];
        int cols = dense._shape[1];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                T value = dense[i, j];
                if (ops.LessThanOrEquals(ops.Abs(value), tolerance))
                    continue;

                rowIndices.Add(i);
                colIndices.Add(j);
                values.Add(value);
            }
        }

        return new SparseTensor<T>(rows, cols, rowIndices.ToArray(), colIndices.ToArray(), values.ToArray());
    }

    /// <summary>
    /// Converts to COO format. Returns this if already COO.
    /// </summary>
    public SparseTensor<T> ToCoo()
    {
        if (Format == SparseStorageFormat.Coo)
            return this;

        var valuesArray = DataVector.ToArray();

        if (Format == SparseStorageFormat.Csr)
        {
            var rowIdx = new int[valuesArray.Length];
            var colIdx = new int[valuesArray.Length];
            int index = 0;
            for (int row = 0; row < Rows; row++)
            {
                for (int i = RowPointers[row]; i < RowPointers[row + 1]; i++)
                {
                    rowIdx[index] = row;
                    colIdx[index] = ColumnIndices[i];
                    index++;
                }
            }
            return new SparseTensor<T>(Rows, Columns, rowIdx, colIdx, valuesArray);
        }

        // CSC → COO
        var rowIdxCsc = new int[valuesArray.Length];
        var colIdxCsc = new int[valuesArray.Length];
        int idx = 0;
        for (int col = 0; col < Columns; col++)
        {
            for (int i = ColumnPointers[col]; i < ColumnPointers[col + 1]; i++)
            {
                rowIdxCsc[idx] = RowIndices[i];
                colIdxCsc[idx] = col;
                idx++;
            }
        }
        return new SparseTensor<T>(Rows, Columns, rowIdxCsc, colIdxCsc, valuesArray);
    }

    /// <summary>
    /// Converts to CSR format. Returns this if already CSR.
    /// </summary>
    public SparseTensor<T> ToCsr()
    {
        if (Format == SparseStorageFormat.Csr)
            return this;

        var coo = ToCoo().Coalesce();
        var cooValues = coo.DataVector.ToArray();
        int nnz = cooValues.Length;
        var rowPointers = new int[Rows + 1];
        for (int i = 0; i < nnz; i++)
            rowPointers[coo.RowIndices[i] + 1]++;
        for (int i = 0; i < Rows; i++)
            rowPointers[i + 1] += rowPointers[i];

        var columnIndices = new int[nnz];
        var values = new T[nnz];
        var offsets = (int[])rowPointers.Clone();
        for (int i = 0; i < nnz; i++)
        {
            int row = coo.RowIndices[i];
            int dest = offsets[row]++;
            columnIndices[dest] = coo.ColumnIndices[i];
            values[dest] = cooValues[i];
        }

        return FromCsr(Rows, Columns, rowPointers, columnIndices, values);
    }

    /// <summary>
    /// Converts to CSC format. Returns this if already CSC.
    /// </summary>
    public SparseTensor<T> ToCsc()
    {
        if (Format == SparseStorageFormat.Csc)
            return this;

        var coo = ToCoo().Coalesce();
        var cooValues = coo.DataVector.ToArray();
        int nnz = cooValues.Length;
        var columnPointers = new int[Columns + 1];
        for (int i = 0; i < nnz; i++)
            columnPointers[coo.ColumnIndices[i] + 1]++;
        for (int i = 0; i < Columns; i++)
            columnPointers[i + 1] += columnPointers[i];

        var rowIndices = new int[nnz];
        var values = new T[nnz];
        var offsets = (int[])columnPointers.Clone();
        for (int i = 0; i < nnz; i++)
        {
            int col = coo.ColumnIndices[i];
            int dest = offsets[col]++;
            rowIndices[dest] = coo.RowIndices[i];
            values[dest] = cooValues[i];
        }

        return FromCsc(Rows, Columns, columnPointers, rowIndices, values);
    }

    /// <summary>
    /// Merges duplicate entries and removes zeros. Returns a new coalesced COO tensor.
    /// </summary>
    public SparseTensor<T> Coalesce()
    {
        var coo = ToCoo();
        var cooValues = coo.DataVector.ToArray();
        int nnz = cooValues.Length;
        if (nnz == 0) return coo;

        var ops = MathHelper.GetNumericOperations<T>();
        var entries = new List<(int Row, int Col, T Value)>(nnz);
        for (int i = 0; i < nnz; i++)
            entries.Add((coo.RowIndices[i], coo.ColumnIndices[i], cooValues[i]));

        entries.Sort((a, b) =>
        {
            int rowCompare = a.Row.CompareTo(b.Row);
            return rowCompare != 0 ? rowCompare : a.Col.CompareTo(b.Col);
        });

        var rowIdx = new List<int>();
        var colIdx = new List<int>();
        var vals = new List<T>();

        int currentRow = entries[0].Row;
        int currentCol = entries[0].Col;
        T currentValue = entries[0].Value;

        for (int i = 1; i < entries.Count; i++)
        {
            var entry = entries[i];
            if (entry.Row == currentRow && entry.Col == currentCol)
            {
                currentValue = ops.Add(currentValue, entry.Value);
            }
            else
            {
                if (!ops.Equals(currentValue, ops.Zero))
                {
                    rowIdx.Add(currentRow);
                    colIdx.Add(currentCol);
                    vals.Add(currentValue);
                }
                currentRow = entry.Row;
                currentCol = entry.Col;
                currentValue = entry.Value;
            }
        }

        if (!ops.Equals(currentValue, ops.Zero))
        {
            rowIdx.Add(currentRow);
            colIdx.Add(currentCol);
            vals.Add(currentValue);
        }

        return new SparseTensor<T>(Rows, Columns, rowIdx.ToArray(), colIdx.ToArray(), vals.ToArray());
    }

    /// <summary>
    /// Returns a transposed sparse tensor. Format is preserved where possible.
    /// Overrides the base class Transpose to preserve sparse format.
    /// </summary>
    public override Tensor<T> Transpose()
    {
        var valuesArray = DataVector.ToArray();

        if (Format == SparseStorageFormat.Coo)
            return new SparseTensor<T>(Columns, Rows, (int[])ColumnIndices.Clone(), (int[])RowIndices.Clone(), valuesArray);

        if (Format == SparseStorageFormat.Csr)
            return FromCsc(Columns, Rows, (int[])RowPointers.Clone(), (int[])ColumnIndices.Clone(), valuesArray);

        return FromCsr(Columns, Rows, (int[])ColumnPointers.Clone(), (int[])RowIndices.Clone(), valuesArray);
    }

    /// <summary>
    /// Materializes the full dense tensor from the sparse representation.
    /// </summary>
    public Tensor<T> ToDense()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var dense = new Tensor<T>(new[] { Rows, Columns });
        if (NonZeroCount == 0) return dense;

        var coo = ToCoo();
        var cooValues = coo.DataVector.ToArray();
        for (int i = 0; i < cooValues.Length; i++)
        {
            int r = coo.RowIndices[i];
            int c = coo.ColumnIndices[i];
            // Accumulate duplicates instead of overwriting (handles uncoalesced COO)
            dense[r, c] = ops.Add(dense[r, c], cooValues[i]);
        }

        return dense;
    }

    /// <summary>
    /// Gets or sets the value at the specified (row, col) indices via sparse lookup.
    /// For COO: linear scan. For CSR: binary search within row. Returns zero for missing entries.
    /// </summary>
    public override T this[params int[] indices]
    {
        get
        {
            if (indices.Length != 2)
                throw new ArgumentException("SparseTensor indexing requires exactly 2 indices [row, col].");
            int row = indices[0], col = indices[1];
            var ops = MathHelper.GetNumericOperations<T>();

            if (Format == SparseStorageFormat.Csr)
            {
                int start = RowPointers[row], end = RowPointers[row + 1];
                for (int i = start; i < end; i++)
                {
                    if (ColumnIndices[i] == col)
                        return DataVector[i];
                }
                return ops.Zero;
            }

            // COO: linear scan
            var coo = (Format == SparseStorageFormat.Coo) ? this : ToCoo();
            for (int i = 0; i < coo.NonZeroCount; i++)
            {
                if (coo.RowIndices[i] == row && coo.ColumnIndices[i] == col)
                    return coo.DataVector[i];
            }
            return ops.Zero;
        }
        set => throw new NotSupportedException(
            "SparseTensor does not support setting individual elements. " +
            "Reconstruct the sparse tensor with updated values or use ToDense().");
    }

    /// <summary>
    /// Creates a deep copy of this sparse tensor, preserving the sparse format.
    /// </summary>
    public SparseTensor<T> CloneSparse()
    {
        var values = DataVector.ToArray();
        return Format switch
        {
            SparseStorageFormat.Coo => new SparseTensor<T>(Rows, Columns,
                (int[])RowIndices.Clone(), (int[])ColumnIndices.Clone(), (T[])values.Clone()),
            SparseStorageFormat.Csr => FromCsr(Rows, Columns,
                (int[])RowPointers.Clone(), (int[])ColumnIndices.Clone(), (T[])values.Clone()),
            _ => ToCsr().CloneSparse() // CSC → convert to CSR then clone
        };
    }

    /// <inheritdoc />
    protected override TensorBase<T> CreateInstance(int[] shape) =>
        new Tensor<T>(shape); // Sparse operations that need a new tensor create dense

    /// <inheritdoc />
    protected override TensorBase<T> CreateInstance(T[] data, int[] shape) =>
        new Tensor<T>(data, shape);

    /// <inheritdoc />
    protected override TensorBase<TResult> CreateInstance<TResult>(params int[] shape) =>
        new Tensor<TResult>(shape);
}
