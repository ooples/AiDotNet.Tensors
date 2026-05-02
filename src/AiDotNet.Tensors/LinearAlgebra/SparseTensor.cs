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
    /// Block-row dimension for BSR/BSC formats; 0 for COO/CSR/CSC. The
    /// rectangular block shape is <c>BlockRowSize × BlockColSize</c>; both
    /// must divide <see cref="Rows"/> and <see cref="Columns"/> respectively.
    /// Common values for LLM weight pruning: 2, 4, 8, 16.
    /// </summary>
    public int BlockRowSize { get; }

    /// <summary>Block-column dimension for BSR/BSC formats; 0 for non-block layouts.</summary>
    public int BlockColSize { get; }

    /// <summary>
    /// Gets the number of non-zero elements stored. For BSR/BSC this counts
    /// every value in every stored block (so a 4×4-block sparse tensor with
    /// 10 stored blocks reports <c>NonZeroCount = 160</c>); use
    /// <see cref="NonZeroBlockCount"/> for the block count.
    /// </summary>
    public int NonZeroCount => DataVector.Length;

    /// <summary>Number of stored blocks for BSR/BSC; equal to <c>NonZeroCount</c>
    /// divided by <c>BlockRowSize · BlockColSize</c>. Returns 0 for non-block
    /// layouts.</summary>
    public int NonZeroBlockCount =>
        (BlockRowSize > 0 && BlockColSize > 0) ? NonZeroCount / (BlockRowSize * BlockColSize) : 0;

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
        BlockRowSize = 0;
        BlockColSize = 0;
    }

    /// <summary>
    /// Creates a COO-format sparse tensor whose <c>Values</c> backing
    /// vector is a caller-supplied <see cref="Vector{T}"/>. Use this
    /// (paired with <see cref="Vector{T}.CreateSlice"/>) to construct
    /// a sparse tensor that shares value storage with another vector —
    /// the contract <see cref="ParameterBuffer{T}.CreateView"/> relies
    /// on for live, zero-copy sparse views over the parameter buffer.
    /// </summary>
    /// <remarks>
    /// Unlike the <c>T[]</c> overload, this ctor does NOT wrap-then-copy:
    /// mutations to the supplied vector are reflected in the sparse
    /// tensor and vice versa. <paramref name="valuesVector"/> must have
    /// length equal to <paramref name="rowIndices"/>.Length.
    /// </remarks>
    public SparseTensor(int rows, int columns, int[] rowIndices, int[] columnIndices, Vector<T> valuesVector)
        : base(valuesVector ?? throw new ArgumentNullException(nameof(valuesVector)),
               new[] { rows, columns }, isSparse: true)
    {
        if (rowIndices is null)
            throw new ArgumentNullException(nameof(rowIndices));
        if (columnIndices is null)
            throw new ArgumentNullException(nameof(columnIndices));
        if (rowIndices.Length != columnIndices.Length || rowIndices.Length != valuesVector.Length)
            throw new ArgumentException(
                "COO indices and values vector must have the same length: " +
                $"rowIndices.Length={rowIndices.Length}, columnIndices.Length={columnIndices.Length}, " +
                $"valuesVector.Length={valuesVector.Length}.");

        Rows = rows;
        Columns = columns;
        RowIndices = rowIndices;
        ColumnIndices = columnIndices;
        RowPointers = Array.Empty<int>();
        ColumnPointers = Array.Empty<int>();
        Format = SparseStorageFormat.Coo;
        BlockRowSize = 0;
        BlockColSize = 0;
    }

    private SparseTensor(int rows, int columns, SparseStorageFormat format,
        int[] rowIndices, int[] columnIndices, int[] rowPointers, int[] columnPointers, T[] values,
        int blockRowSize = 0, int blockColSize = 0)
        : base(Vector<T>.Wrap(values), new[] { rows, columns }, isSparse: true)
    {
        Rows = rows;
        Columns = columns;
        Format = format;
        RowIndices = rowIndices;
        ColumnIndices = columnIndices;
        RowPointers = rowPointers;
        ColumnPointers = columnPointers;
        BlockRowSize = blockRowSize;
        BlockColSize = blockColSize;
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
    /// Creates a BSR-format sparse tensor with rectangular blocks of
    /// shape <paramref name="blockRowSize"/>×<paramref name="blockColSize"/>.
    /// <paramref name="blockRowPointers"/> indexes the block-row stripes
    /// (length = <c>rows / blockRowSize + 1</c>). <paramref name="blockColumnIndices"/>
    /// gives the column-block index for each stored block. <paramref name="values"/>
    /// is flat with length <c>nnzBlocks · blockRowSize · blockColSize</c>;
    /// each block is row-major within itself.
    /// </summary>
    public static SparseTensor<T> FromBsr(int rows, int columns, int blockRowSize, int blockColSize,
        int[] blockRowPointers, int[] blockColumnIndices, T[] values)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns < 0) throw new ArgumentOutOfRangeException(nameof(columns));
        if (blockRowSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockRowSize));
        if (blockColSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockColSize));
        if (rows % blockRowSize != 0)
            throw new ArgumentException($"Rows {rows} must be divisible by blockRowSize {blockRowSize}.", nameof(rows));
        if (columns % blockColSize != 0)
            throw new ArgumentException($"Columns {columns} must be divisible by blockColSize {blockColSize}.", nameof(columns));
        if (blockRowPointers is null) throw new ArgumentNullException(nameof(blockRowPointers));
        if (blockColumnIndices is null) throw new ArgumentNullException(nameof(blockColumnIndices));
        if (values is null) throw new ArgumentNullException(nameof(values));

        int blockRows = rows / blockRowSize;
        int blockCols = columns / blockColSize;
        int blockSize = blockRowSize * blockColSize;
        if (blockRowPointers.Length != blockRows + 1)
            throw new ArgumentException($"blockRowPointers length must be {blockRows + 1} (= rows/blockRowSize + 1).",
                nameof(blockRowPointers));
        if (values.Length != blockColumnIndices.Length * blockSize)
            throw new ArgumentException(
                $"values length {values.Length} must equal blockColumnIndices.Length ({blockColumnIndices.Length}) · " +
                $"blockRowSize · blockColSize ({blockSize}).", nameof(values));

        // Pointer-array invariants — without these, ToCoo/ToDense walks
        // past the buffer or writes outside the logical matrix.
        // 1. blockRowPointers[0] must be 0.
        // 2. monotonic non-decreasing.
        // 3. blockRowPointers[^1] must equal blockColumnIndices.Length.
        // 4. every block-column index must be in [0, blockCols).
        if (blockRowPointers[0] != 0)
            throw new ArgumentException(
                $"blockRowPointers[0] must be 0; got {blockRowPointers[0]}.", nameof(blockRowPointers));
        for (int i = 0; i < blockRows; i++)
        {
            if (blockRowPointers[i] > blockRowPointers[i + 1])
                throw new ArgumentException(
                    $"blockRowPointers must be non-decreasing; entry {i}={blockRowPointers[i]} > {i + 1}={blockRowPointers[i + 1]}.",
                    nameof(blockRowPointers));
        }
        if (blockRowPointers[blockRows] != blockColumnIndices.Length)
            throw new ArgumentException(
                $"blockRowPointers terminal entry must equal blockColumnIndices.Length ({blockColumnIndices.Length}); got {blockRowPointers[blockRows]}.",
                nameof(blockRowPointers));
        for (int i = 0; i < blockColumnIndices.Length; i++)
        {
            if (blockColumnIndices[i] < 0 || blockColumnIndices[i] >= blockCols)
                throw new ArgumentOutOfRangeException(nameof(blockColumnIndices),
                    $"blockColumnIndices[{i}]={blockColumnIndices[i]} is outside [0, {blockCols}).");
        }

        return new SparseTensor<T>(rows, columns, SparseStorageFormat.Bsr,
            Array.Empty<int>(), blockColumnIndices, blockRowPointers, Array.Empty<int>(),
            values, blockRowSize, blockColSize);
    }

    /// <summary>
    /// Creates a BSC-format sparse tensor — column-block analog of
    /// <see cref="FromBsr"/>. <paramref name="blockColumnPointers"/> indexes
    /// the block-column stripes (length = <c>columns / blockColSize + 1</c>);
    /// <paramref name="blockRowIndices"/> gives the row-block index for each
    /// stored block.
    /// </summary>
    public static SparseTensor<T> FromBsc(int rows, int columns, int blockRowSize, int blockColSize,
        int[] blockColumnPointers, int[] blockRowIndices, T[] values)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns < 0) throw new ArgumentOutOfRangeException(nameof(columns));
        if (blockRowSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockRowSize));
        if (blockColSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockColSize));
        if (rows % blockRowSize != 0)
            throw new ArgumentException($"Rows {rows} must be divisible by blockRowSize {blockRowSize}.", nameof(rows));
        if (columns % blockColSize != 0)
            throw new ArgumentException($"Columns {columns} must be divisible by blockColSize {blockColSize}.", nameof(columns));
        if (blockColumnPointers is null) throw new ArgumentNullException(nameof(blockColumnPointers));
        if (blockRowIndices is null) throw new ArgumentNullException(nameof(blockRowIndices));
        if (values is null) throw new ArgumentNullException(nameof(values));

        int blockRows = rows / blockRowSize;
        int blockCols = columns / blockColSize;
        int blockSize = blockRowSize * blockColSize;
        if (blockColumnPointers.Length != blockCols + 1)
            throw new ArgumentException($"blockColumnPointers length must be {blockCols + 1} (= columns/blockColSize + 1).",
                nameof(blockColumnPointers));
        if (values.Length != blockRowIndices.Length * blockSize)
            throw new ArgumentException(
                $"values length {values.Length} must equal blockRowIndices.Length ({blockRowIndices.Length}) · " +
                $"blockRowSize · blockColSize ({blockSize}).", nameof(values));

        // Same pointer/index invariants as the BSR factory above —
        // see that comment for the why.
        if (blockColumnPointers[0] != 0)
            throw new ArgumentException(
                $"blockColumnPointers[0] must be 0; got {blockColumnPointers[0]}.", nameof(blockColumnPointers));
        for (int i = 0; i < blockCols; i++)
        {
            if (blockColumnPointers[i] > blockColumnPointers[i + 1])
                throw new ArgumentException(
                    $"blockColumnPointers must be non-decreasing; entry {i}={blockColumnPointers[i]} > {i + 1}={blockColumnPointers[i + 1]}.",
                    nameof(blockColumnPointers));
        }
        if (blockColumnPointers[blockCols] != blockRowIndices.Length)
            throw new ArgumentException(
                $"blockColumnPointers terminal entry must equal blockRowIndices.Length ({blockRowIndices.Length}); got {blockColumnPointers[blockCols]}.",
                nameof(blockColumnPointers));
        for (int i = 0; i < blockRowIndices.Length; i++)
        {
            if (blockRowIndices[i] < 0 || blockRowIndices[i] >= blockRows)
                throw new ArgumentOutOfRangeException(nameof(blockRowIndices),
                    $"blockRowIndices[{i}]={blockRowIndices[i]} is outside [0, {blockRows}).");
        }

        return new SparseTensor<T>(rows, columns, SparseStorageFormat.Bsc,
            blockRowIndices, Array.Empty<int>(), Array.Empty<int>(), blockColumnPointers,
            values, blockRowSize, blockColSize);
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

        if (Format == SparseStorageFormat.Bsr || Format == SparseStorageFormat.Bsc)
        {
            // Block formats expand block-by-block into per-element COO.
            return ExpandBlocksToCoo();
        }

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

    private SparseTensor<T> ExpandBlocksToCoo()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int br = BlockRowSize, bc = BlockColSize;
        int blockSize = br * bc;
        var src = DataVector.ToArray();
        var rowIdx = new List<int>();
        var colIdx = new List<int>();
        var vals = new List<T>();

        if (Format == SparseStorageFormat.Bsr)
        {
            int blockRows = Rows / br;
            for (int br_i = 0; br_i < blockRows; br_i++)
            {
                for (int p = RowPointers[br_i]; p < RowPointers[br_i + 1]; p++)
                {
                    int blockCol = ColumnIndices[p];
                    int valOff = p * blockSize;
                    for (int i = 0; i < br; i++)
                    {
                        for (int j = 0; j < bc; j++)
                        {
                            T v = src[valOff + i * bc + j];
                            if (!ops.Equals(v, ops.Zero))
                            {
                                rowIdx.Add(br_i * br + i);
                                colIdx.Add(blockCol * bc + j);
                                vals.Add(v);
                            }
                        }
                    }
                }
            }
        }
        else // Bsc
        {
            int blockCols = Columns / bc;
            for (int bc_j = 0; bc_j < blockCols; bc_j++)
            {
                for (int p = ColumnPointers[bc_j]; p < ColumnPointers[bc_j + 1]; p++)
                {
                    int blockRow = RowIndices[p];
                    int valOff = p * blockSize;
                    for (int i = 0; i < br; i++)
                    {
                        for (int j = 0; j < bc; j++)
                        {
                            T v = src[valOff + i * bc + j];
                            if (!ops.Equals(v, ops.Zero))
                            {
                                rowIdx.Add(blockRow * br + i);
                                colIdx.Add(bc_j * bc + j);
                                vals.Add(v);
                            }
                        }
                    }
                }
            }
        }

        return new SparseTensor<T>(Rows, Columns, rowIdx.ToArray(), colIdx.ToArray(), vals.ToArray());
    }

    /// <summary>
    /// Converts to BSR format with blocks of shape <paramref name="blockRowSize"/>×
    /// <paramref name="blockColSize"/>. Any block that contains at least one
    /// non-zero is materialized; the rest of the block holds zeros.
    /// </summary>
    public SparseTensor<T> ToBsr(int blockRowSize, int blockColSize)
    {
        if (blockRowSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockRowSize));
        if (blockColSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockColSize));
        if (Rows % blockRowSize != 0)
            throw new ArgumentException($"Rows {Rows} must be divisible by blockRowSize {blockRowSize}.");
        if (Columns % blockColSize != 0)
            throw new ArgumentException($"Columns {Columns} must be divisible by blockColSize {blockColSize}.");
        if (Format == SparseStorageFormat.Bsr && BlockRowSize == blockRowSize && BlockColSize == blockColSize)
            return this;

        int blockRows = Rows / blockRowSize;
        int blockCols = Columns / blockColSize;
        int blockSize = blockRowSize * blockColSize;

        // Collect non-zeros into a (blockRow, blockCol) → block-buffer dictionary.
        var blocks = new Dictionary<(int, int), T[]>();
        var ops = MathHelper.GetNumericOperations<T>();
        var coo = ToCoo();
        var cooVals = coo.DataVector.ToArray();
        for (int k = 0; k < cooVals.Length; k++)
        {
            int r = coo.RowIndices[k], c = coo.ColumnIndices[k];
            int br_i = r / blockRowSize, bc_j = c / blockColSize;
            int li = r % blockRowSize, lj = c % blockColSize;
            if (!blocks.TryGetValue((br_i, bc_j), out var buf))
            {
                buf = new T[blockSize];
                for (int i = 0; i < blockSize; i++) buf[i] = ops.Zero;
                blocks[(br_i, bc_j)] = buf;
            }
            // Accumulate so duplicates from un-coalesced COO survive the round-trip.
            buf[li * blockColSize + lj] = ops.Add(buf[li * blockColSize + lj], cooVals[k]);
        }

        // Sort blocks in (block-row, block-col) order so each row's column
        // indices come out monotone.
        var sortedKeys = new List<(int br_i, int bc_j)>(blocks.Keys);
        sortedKeys.Sort((a, b) => a.br_i != b.br_i ? a.br_i.CompareTo(b.br_i) : a.bc_j.CompareTo(b.bc_j));

        var blockRowPtr = new int[blockRows + 1];
        var blockColIdx = new int[sortedKeys.Count];
        var values = new T[sortedKeys.Count * blockSize];
        for (int p = 0; p < sortedKeys.Count; p++)
        {
            var (br_i, bc_j) = sortedKeys[p];
            blockColIdx[p] = bc_j;
            blockRowPtr[br_i + 1]++;
            Array.Copy(blocks[sortedKeys[p]], 0, values, p * blockSize, blockSize);
        }
        for (int i = 0; i < blockRows; i++) blockRowPtr[i + 1] += blockRowPtr[i];
        _ = blockCols;

        return FromBsr(Rows, Columns, blockRowSize, blockColSize, blockRowPtr, blockColIdx, values);
    }

    /// <summary>Converts to BSC format — column-block analog of <see cref="ToBsr"/>.</summary>
    public SparseTensor<T> ToBsc(int blockRowSize, int blockColSize)
    {
        if (blockRowSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockRowSize));
        if (blockColSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockColSize));
        if (Rows % blockRowSize != 0)
            throw new ArgumentException($"Rows {Rows} must be divisible by blockRowSize {blockRowSize}.");
        if (Columns % blockColSize != 0)
            throw new ArgumentException($"Columns {Columns} must be divisible by blockColSize {blockColSize}.");
        if (Format == SparseStorageFormat.Bsc && BlockRowSize == blockRowSize && BlockColSize == blockColSize)
            return this;

        int blockRows = Rows / blockRowSize;
        int blockCols = Columns / blockColSize;
        int blockSize = blockRowSize * blockColSize;

        var blocks = new Dictionary<(int, int), T[]>();
        var ops = MathHelper.GetNumericOperations<T>();
        var coo = ToCoo();
        var cooVals = coo.DataVector.ToArray();
        for (int k = 0; k < cooVals.Length; k++)
        {
            int r = coo.RowIndices[k], c = coo.ColumnIndices[k];
            int br_i = r / blockRowSize, bc_j = c / blockColSize;
            int li = r % blockRowSize, lj = c % blockColSize;
            if (!blocks.TryGetValue((br_i, bc_j), out var buf))
            {
                buf = new T[blockSize];
                for (int i = 0; i < blockSize; i++) buf[i] = ops.Zero;
                blocks[(br_i, bc_j)] = buf;
            }
            buf[li * blockColSize + lj] = ops.Add(buf[li * blockColSize + lj], cooVals[k]);
        }

        // Sort by (block-col, block-row) so each column's row-indices come out monotone.
        var sortedKeys = new List<(int br_i, int bc_j)>(blocks.Keys);
        sortedKeys.Sort((a, b) => a.bc_j != b.bc_j ? a.bc_j.CompareTo(b.bc_j) : a.br_i.CompareTo(b.br_i));

        var blockColPtr = new int[blockCols + 1];
        var blockRowIdx = new int[sortedKeys.Count];
        var values = new T[sortedKeys.Count * blockSize];
        for (int p = 0; p < sortedKeys.Count; p++)
        {
            var (br_i, bc_j) = sortedKeys[p];
            blockRowIdx[p] = br_i;
            blockColPtr[bc_j + 1]++;
            Array.Copy(blocks[sortedKeys[p]], 0, values, p * blockSize, blockSize);
        }
        for (int j = 0; j < blockCols; j++) blockColPtr[j + 1] += blockColPtr[j];
        _ = blockRows;

        return FromBsc(Rows, Columns, blockRowSize, blockColSize, blockColPtr, blockRowIdx, values);
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

        if (Format == SparseStorageFormat.Csc)
            return FromCsr(Columns, Rows, (int[])ColumnPointers.Clone(), (int[])RowIndices.Clone(), valuesArray);

        // Block formats: route through dense for the per-block transpose
        // since each block itself transposes too. Cheap given block formats
        // are typically loaded once and held for many forward passes.
        return FromDense(ToDense().Transpose()).ToCsr();
    }

    /// <summary>
    /// Materializes the full dense tensor from the sparse representation.
    /// Handles every supported format (COO/CSR/CSC/BSR/BSC).
    /// </summary>
    public Tensor<T> ToDense()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var dense = new Tensor<T>(new[] { Rows, Columns });
        if (NonZeroCount == 0) return dense;

        if (Format == SparseStorageFormat.Bsr)
        {
            int br = BlockRowSize, bc = BlockColSize, blockSize = br * bc;
            var src = DataVector.ToArray();
            int blockRows = Rows / br;
            for (int br_i = 0; br_i < blockRows; br_i++)
            {
                for (int p = RowPointers[br_i]; p < RowPointers[br_i + 1]; p++)
                {
                    int blockCol = ColumnIndices[p];
                    int valOff = p * blockSize;
                    for (int i = 0; i < br; i++)
                        for (int j = 0; j < bc; j++)
                        {
                            // Accumulate rather than assign — duplicate
                            // (block-row, block-col) pairs from FromBsr()
                            // would otherwise produce order-dependent
                            // dense materialisation. Matches the COO
                            // path's ops.Add accumulator below.
                            int rIdx = br_i * br + i, cIdx = blockCol * bc + j;
                            dense[rIdx, cIdx] = ops.Add(dense[rIdx, cIdx], src[valOff + i * bc + j]);
                        }
                }
            }
            return dense;
        }
        if (Format == SparseStorageFormat.Bsc)
        {
            int br = BlockRowSize, bc = BlockColSize, blockSize = br * bc;
            var src = DataVector.ToArray();
            int blockCols = Columns / bc;
            for (int bc_j = 0; bc_j < blockCols; bc_j++)
            {
                for (int p = ColumnPointers[bc_j]; p < ColumnPointers[bc_j + 1]; p++)
                {
                    int blockRow = RowIndices[p];
                    int valOff = p * blockSize;
                    for (int i = 0; i < br; i++)
                        for (int j = 0; j < bc; j++)
                        {
                            // Accumulate — same rationale as the BSR
                            // branch above. Transpose() routes through
                            // here for block formats so this also fixes
                            // duplicate-block transpose correctness.
                            int rIdx = blockRow * br + i, cIdx = bc_j * bc + j;
                            dense[rIdx, cIdx] = ops.Add(dense[rIdx, cIdx], src[valOff + i * bc + j]);
                        }
                }
            }
            return dense;
        }

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
    /// For CSR: binary search within row's column index range. For COO: linear scan.
    /// Returns zero for missing entries.
    /// </summary>
    public override T this[params int[] indices]
    {
        get
        {
            if (indices.Length != 2)
                throw new ArgumentException("SparseTensor indexing requires exactly 2 indices [row, col].");
            int row = indices[0], col = indices[1];
            if ((uint)row >= (uint)Rows)
                throw new ArgumentOutOfRangeException(nameof(indices), $"Row index {row} out of range [0, {Rows}).");
            if ((uint)col >= (uint)Columns)
                throw new ArgumentOutOfRangeException(nameof(indices), $"Column index {col} out of range [0, {Columns}).");
            var ops = MathHelper.GetNumericOperations<T>();

            if (Format == SparseStorageFormat.Csr)
            {
                int start = RowPointers[row], end = RowPointers[row + 1];
                // Binary search: CSR column indices are sorted within each row
                int idx = Array.BinarySearch(ColumnIndices, start, end - start, col);
                return idx >= 0 ? DataVector[idx] : ops.Zero;
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
        // DataVector.ToArray() already returns a new copy — no need to clone again
        var values = DataVector.ToArray();
        return Format switch
        {
            SparseStorageFormat.Coo => new SparseTensor<T>(Rows, Columns,
                (int[])RowIndices.Clone(), (int[])ColumnIndices.Clone(), values),
            SparseStorageFormat.Csr => FromCsr(Rows, Columns,
                (int[])RowPointers.Clone(), (int[])ColumnIndices.Clone(), values),
            SparseStorageFormat.Csc => FromCsc(Rows, Columns,
                (int[])ColumnPointers.Clone(), (int[])RowIndices.Clone(), values),
            SparseStorageFormat.Bsr => FromBsr(Rows, Columns, BlockRowSize, BlockColSize,
                (int[])RowPointers.Clone(), (int[])ColumnIndices.Clone(), values),
            SparseStorageFormat.Bsc => FromBsc(Rows, Columns, BlockRowSize, BlockColSize,
                (int[])ColumnPointers.Clone(), (int[])RowIndices.Clone(), values),
            _ => ToCsr().CloneSparse()
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
