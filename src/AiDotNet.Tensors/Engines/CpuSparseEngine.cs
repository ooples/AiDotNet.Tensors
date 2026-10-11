using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// CPU implementation of sparse tensor operations.
/// </summary>
/// <remarks>
/// <para>
/// CpuSparseEngine provides efficient sparse matrix operations using standard algorithms
/// optimized for CPU execution. All operations work with the SparseTensor type which
/// supports COO, CSR, and CSC storage formats.
/// </para>
/// <para><b>For Beginners:</b> This class does the actual work for sparse operations on the CPU.
/// It's used when you don't have a GPU or when working with custom numeric types.
/// </para>
/// </remarks>
public sealed class CpuSparseEngine : ISparseEngine
{
    /// <summary>
    /// Singleton instance for convenience.
    /// </summary>
    public static CpuSparseEngine Instance { get; } = new CpuSparseEngine();

    #region Sparse Matrix-Vector Operations

    /// <inheritdoc/>
    public Vector<T> SpMV<T>(SparseTensor<T> sparse, Vector<T> dense)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (sparse.Columns != dense.Length)
        {
            throw new ArgumentException(
                $"Sparse matrix columns ({sparse.Columns}) must match vector length ({dense.Length}).");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new T[sparse.Rows];

        // Initialize result to zero
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = ops.Zero;
        }

        // Convert to CSR format for efficient row-major access
        var csr = sparse.ToCsr();
        var rowPtrs = csr.RowPointers;
        var colIndices = csr.ColumnIndices;
        var values = csr.Values;

        // SpMV using CSR format: y[i] = sum_j A[i,j] * x[j]
        for (int row = 0; row < sparse.Rows; row++)
        {
            T sum = ops.Zero;
            int start = rowPtrs[row];
            int end = rowPtrs[row + 1];

            for (int idx = start; idx < end; idx++)
            {
                int col = colIndices[idx];
                T val = values[idx];
                sum = ops.Add(sum, ops.Multiply(val, dense[col]));
            }

            result[row] = sum;
        }

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public Vector<T> SpMVTranspose<T>(SparseTensor<T> sparse, Vector<T> dense)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (sparse.Rows != dense.Length)
        {
            throw new ArgumentException(
                $"Sparse matrix rows ({sparse.Rows}) must match vector length ({dense.Length}).");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new T[sparse.Columns];

        // Initialize result to zero
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = ops.Zero;
        }

        // Convert to CSR format
        var csr = sparse.ToCsr();
        var rowPtrs = csr.RowPointers;
        var colIndices = csr.ColumnIndices;
        var values = csr.Values;

        // SpMV transpose: y[j] = sum_i A[i,j] * x[i] = sum_i A^T[j,i] * x[i]
        for (int row = 0; row < sparse.Rows; row++)
        {
            int start = rowPtrs[row];
            int end = rowPtrs[row + 1];
            T xVal = dense[row];

            for (int idx = start; idx < end; idx++)
            {
                int col = colIndices[idx];
                T val = values[idx];
                result[col] = ops.Add(result[col], ops.Multiply(val, xVal));
            }
        }

        return new Vector<T>(result);
    }

    #endregion

    #region Sparse Matrix-Matrix Operations

    /// <inheritdoc/>
    public Matrix<T> SpMM<T>(SparseTensor<T> sparse, Matrix<T> dense)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (sparse.Columns != dense.Rows)
        {
            throw new ArgumentException(
                $"Sparse matrix columns ({sparse.Columns}) must match dense matrix rows ({dense.Rows}).");
        }

        // The public API is unconstrained over T (no struct/unmanaged constraint) so
        // unconstrained-T consumers — e.g. the neural-network layers, generic over an
        // open T backed by INumericOperations<T> — can call it. Route the blittable
        // element types through the SIMD/parallel BLAS SpMM kernel (#379); fall back to
        // a generic CSR scalar path for any other T. float/double satisfy `unmanaged`,
        // so the (object) cast to the concrete type is sound at runtime.
        if (typeof(T) == typeof(float))
            return (Matrix<T>)(object)SpMMBlas((SparseTensor<float>)(object)sparse, (Matrix<float>)(object)dense);
        if (typeof(T) == typeof(double))
            return (Matrix<T>)(object)SpMMBlas((SparseTensor<double>)(object)sparse, (Matrix<double>)(object)dense);
        return SpMMGenericScalar(sparse, dense);
    }

    // #379 SIMD/parallel managed BLAS SpMM kernel (C = 1·A·B + 0·C): CSR row-parallel,
    // cache-friendly (B rows read contiguously), bit-deterministic. Constrained to
    // unmanaged because the kernel reinterprets element spans (MemoryMarshal.Cast).
    private static Matrix<TU> SpMMBlas<TU>(SparseTensor<TU> sparse, Matrix<TU> dense) where TU : unmanaged
    {
        var ops = MathHelper.GetNumericOperations<TU>();
        var result = new Matrix<TU>(sparse.Rows, dense.Columns);

        var csr = sparse.ToCsr();
        var layout = new BlasManaged.SparseLayout<TU>
        {
            Rows = sparse.Rows,
            Cols = sparse.Columns,
            Pointers = csr.RowPointers,
            Indices = csr.ColumnIndices,
            Values = csr.Values,
            Format = BlasManaged.SparseLayoutFormat.Csr,
        };
        int n = dense.Columns;
        BlasManaged.BlasManaged.SpMM<TU>(
            ops.One, layout,
            dense.AsSpan(), n, n,
            ops.Zero, result.AsWritableSpan(), n);

        return result;
    }

    // Generic CSR scalar SpMM for arbitrary T (mirrors SpMV's fully generic
    // INumericOperations<T> path): C[i, :] += A[i, j] · B[j, :] over the non-zeros of
    // each sparse row, in the same row-major-over-CSR order the BLAS kernel uses, so
    // results agree to floating-point rounding.
    private static Matrix<T> SpMMGenericScalar<T>(SparseTensor<T> sparse, Matrix<T> dense)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int rows = sparse.Rows;
        int n = dense.Columns;
        var result = new Matrix<T>(rows, n);   // T[] backing is zero-initialized

        var csr = sparse.ToCsr();
        var rowPtrs = csr.RowPointers;
        var colIndices = csr.ColumnIndices;
        var values = csr.Values;

        for (int row = 0; row < rows; row++)
        {
            int start = rowPtrs[row];
            int end = rowPtrs[row + 1];
            for (int idx = start; idx < end; idx++)
            {
                int col = colIndices[idx];
                T val = values[idx];
                for (int k = 0; k < n; k++)
                {
                    result[row, k] = ops.Add(result[row, k], ops.Multiply(val, dense[col, k]));
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public SparseTensor<T> SpSpMM<T>(SparseTensor<T> a, SparseTensor<T> b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException(
                $"Matrix A columns ({a.Columns}) must match matrix B rows ({b.Rows}).");
        }

        var ops = MathHelper.GetNumericOperations<T>();

        // Convert both to CSR format for efficient row access
        var aCsr = a.ToCsr();
        var bCsr = b.ToCsr();

        var aRowPtrs = aCsr.RowPointers;
        var aColIndices = aCsr.ColumnIndices;
        var aValues = aCsr.Values;
        var bRowPtrs = bCsr.RowPointers;
        var bColIndices = bCsr.ColumnIndices;
        var bValues = bCsr.Values;

        // Use hash map to accumulate results
        var resultEntries = new List<(int row, int col, T value)>();

        for (int i = 0; i < a.Rows; i++)
        {
            // Hash map for row i of result
            var rowAccum = new Dictionary<int, T>();

            int aStart = aRowPtrs[i];
            int aEnd = aRowPtrs[i + 1];

            for (int aIdx = aStart; aIdx < aEnd; aIdx++)
            {
                int k = aColIndices[aIdx];
                T aVal = aValues[aIdx];

                // Multiply row k of B
                int bStart = bRowPtrs[k];
                int bEnd = bRowPtrs[k + 1];

                for (int bIdx = bStart; bIdx < bEnd; bIdx++)
                {
                    int j = bColIndices[bIdx];
                    T bVal = bValues[bIdx];
                    T product = ops.Multiply(aVal, bVal);

                    if (rowAccum.TryGetValue(j, out T? existing) && existing is not null)
                    {
                        rowAccum[j] = ops.Add(existing, product);
                    }
                    else
                    {
                        rowAccum[j] = product;
                    }
                }
            }

            // Add non-zero entries to result
            foreach (var kvp in rowAccum)
            {
                if (!ops.Equals(kvp.Value, ops.Zero))
                {
                    resultEntries.Add((i, kvp.Key, kvp.Value));
                }
            }
        }

        // Build result sparse tensor
        var rows = new int[resultEntries.Count];
        var cols = new int[resultEntries.Count];
        var vals = new T[resultEntries.Count];

        for (int i = 0; i < resultEntries.Count; i++)
        {
            rows[i] = resultEntries[i].row;
            cols[i] = resultEntries[i].col;
            vals[i] = resultEntries[i].value;
        }

        return new SparseTensor<T>(a.Rows, b.Columns, rows, cols, vals);
    }

    #endregion

    #region Sparse-Dense Element-wise Operations

    /// <inheritdoc/>
    public Matrix<T> AddSparseDense<T>(SparseTensor<T> sparse, Matrix<T> dense)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (sparse.Rows != dense.Rows || sparse.Columns != dense.Columns)
        {
            throw new ArgumentException("Sparse and dense matrix dimensions must match.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(dense.Rows, dense.Columns);

        // Copy dense matrix
        for (int i = 0; i < dense.Rows; i++)
        {
            for (int j = 0; j < dense.Columns; j++)
            {
                result[i, j] = dense[i, j];
            }
        }

        // Add sparse entries - use COO format
        var coo = sparse.ToCoo();
        var rowIndices = coo.RowIndices;
        var colIndices = coo.ColumnIndices;
        var values = coo.Values;

        for (int idx = 0; idx < values.Length; idx++)
        {
            int row = rowIndices[idx];
            int col = colIndices[idx];
            result[row, col] = ops.Add(result[row, col], values[idx]);
        }

        return result;
    }

    /// <inheritdoc/>
    public SparseTensor<T> MultiplySparseDense<T>(SparseTensor<T> sparse, Matrix<T> dense)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (sparse.Rows != dense.Rows || sparse.Columns != dense.Columns)
        {
            throw new ArgumentException("Sparse and dense matrix dimensions must match.");
        }

        var ops = MathHelper.GetNumericOperations<T>();

        // Use COO format for element-wise access
        var coo = sparse.ToCoo();
        var rowIndices = coo.RowIndices;
        var colIndices = coo.ColumnIndices;
        var oldValues = coo.Values;

        var newValues = new T[oldValues.Length];

        for (int idx = 0; idx < oldValues.Length; idx++)
        {
            int row = rowIndices[idx];
            int col = colIndices[idx];
            newValues[idx] = ops.Multiply(oldValues[idx], dense[row, col]);
        }

        // Create new sparse tensor with same structure but new values
        var newRowIndices = new int[rowIndices.Length];
        var newColIndices = new int[colIndices.Length];
        Array.Copy(rowIndices, newRowIndices, rowIndices.Length);
        Array.Copy(colIndices, newColIndices, colIndices.Length);

        return new SparseTensor<T>(sparse.Rows, sparse.Columns, newRowIndices, newColIndices, newValues);
    }

    #endregion

    #region Gather and Scatter Operations

    /// <inheritdoc/>
    public Vector<T> SparseGather<T>(Matrix<T> source, SparseTensor<T> indices)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (indices is null) throw new ArgumentNullException(nameof(indices));

        // Use COO format for index access
        var coo = indices.ToCoo();
        var rowIndices = coo.RowIndices;
        var colIndices = coo.ColumnIndices;

        var result = new T[rowIndices.Length];

        for (int i = 0; i < rowIndices.Length; i++)
        {
            int row = rowIndices[i];
            int col = colIndices[i];

            if (row < 0 || row >= source.Rows || col < 0 || col >= source.Columns)
            {
                throw new ArgumentOutOfRangeException($"Index ({row}, {col}) is out of bounds for matrix of size ({source.Rows}, {source.Columns}).");
            }

            result[i] = source[row, col];
        }

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public Matrix<T> SparseScatter<T>(Vector<T> values, SparseTensor<T> indices, int rows, int cols)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        if (indices is null) throw new ArgumentNullException(nameof(indices));

        // Use COO format for index access
        var coo = indices.ToCoo();
        var rowIndices = coo.RowIndices;
        var colIndices = coo.ColumnIndices;

        if (values.Length != rowIndices.Length)
        {
            throw new ArgumentException($"Values length ({values.Length}) must match number of indices ({rowIndices.Length}).");
        }

        var result = new Matrix<T>(rows, cols);

        for (int i = 0; i < values.Length; i++)
        {
            int row = rowIndices[i];
            int col = colIndices[i];

            if (row < 0 || row >= rows || col < 0 || col >= cols)
            {
                throw new ArgumentOutOfRangeException($"Index ({row}, {col}) is out of bounds for matrix of size ({rows}, {cols}).");
            }

            result[row, col] = values[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public void SparseScatterAdd<T>(Vector<T> values, (int[] rows, int[] cols) indices, Matrix<T> target)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        if (indices.rows is null) throw new ArgumentNullException(nameof(indices));
        if (indices.cols is null) throw new ArgumentNullException(nameof(indices));
        if (target is null) throw new ArgumentNullException(nameof(target));

        if (values.Length != indices.rows.Length || values.Length != indices.cols.Length)
        {
            throw new ArgumentException("Values length must match indices length.");
        }

        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < values.Length; i++)
        {
            int row = indices.rows[i];
            int col = indices.cols[i];

            if (row < 0 || row >= target.Rows || col < 0 || col >= target.Columns)
            {
                throw new ArgumentOutOfRangeException($"Index ({row}, {col}) is out of bounds.");
            }

            target[row, col] = ops.Add(target[row, col], values[i]);
        }
    }

    #endregion

    #region Sparse Tensor Utilities

    /// <inheritdoc/>
    public Matrix<T> SparseToDense<T>(SparseTensor<T> sparse)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));

        var result = new Matrix<T>(sparse.Rows, sparse.Columns);

        // Use COO format for iteration
        var coo = sparse.ToCoo();
        var rowIndices = coo.RowIndices;
        var colIndices = coo.ColumnIndices;
        var values = coo.Values;

        for (int idx = 0; idx < values.Length; idx++)
        {
            result[rowIndices[idx], colIndices[idx]] = values[idx];
        }

        return result;
    }

    /// <inheritdoc/>
    public SparseTensor<T> DenseToSparse<T>(Matrix<T> dense, T threshold)
    {
        if (dense is null) throw new ArgumentNullException(nameof(dense));

        var ops = MathHelper.GetNumericOperations<T>();
        var entries = new List<(int row, int col, T value)>();

        for (int i = 0; i < dense.Rows; i++)
        {
            for (int j = 0; j < dense.Columns; j++)
            {
                T val = dense[i, j];
                T absVal = ops.Abs(val);

                if (ops.GreaterThan(absVal, threshold))
                {
                    entries.Add((i, j, val));
                }
            }
        }

        var rows = new int[entries.Count];
        var cols = new int[entries.Count];
        var vals = new T[entries.Count];

        for (int i = 0; i < entries.Count; i++)
        {
            rows[i] = entries[i].row;
            cols[i] = entries[i].col;
            vals[i] = entries[i].value;
        }

        return new SparseTensor<T>(dense.Rows, dense.Columns, rows, cols, vals);
    }

    /// <inheritdoc/>
    public SparseTensor<T> Coalesce<T>(SparseTensor<T> sparse)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));

        // SparseTensor already has a Coalesce method - delegate to it
        return sparse.Coalesce();
    }

    /// <inheritdoc/>
    public SparseTensor<T> SparseTranspose<T>(SparseTensor<T> sparse)
    {
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));

        // SparseTensor.Transpose() overrides Tensor<T>.Transpose() and returns a SparseTensor
        return (SparseTensor<T>)sparse.Transpose();
    }

    #endregion

    #region Tape-Aware Differentiable Operations

    // Each delegates to the SparseAutograd Record helper, which performs the forward via
    // SparseOps (GPU-dispatched when available) and records the tape edge + backward when a
    // tape is active. The sparse operand is passed as its own gradient target so a registered
    // trainable SparseTensor parameter receives its gradient directly.

    /// <inheritdoc/>
    // Uses the dense-gradient record so the gradient for A is accumulated against A on the
    // standard autodiff tape (retrievable via ComputeGradients like any other operand). The
    // pattern-preserving variant keeps A's gradient in a sparse-aware ParameterBuffer slot
    // instead, which the plain tape API cannot read — see SparseMatMulPatternPreserving for
    // that opt-in path when the sparse operand is backed by such a buffer.
    public Tensor<T> SparseMatMul<T>(SparseTensor<T> a, Tensor<T> b)
        => SparseAutograd.SparseMatMulRecord(a, a, b);

    /// <inheritdoc/>
    public Tensor<T> SparseMatMulPatternPreserving<T>(SparseTensor<T> a, Tensor<T> b)
        => SparseAutograd.SparsePatternPreservingMatMulRecord(a, b);

    /// <inheritdoc/>
    public Tensor<T> SparseAddMM<T>(Tensor<T> c, SparseTensor<T> a, Tensor<T> b, T alpha, T beta)
        => SparseAutograd.SparseAddMMRecord(c, a, a, b, alpha, beta);

    /// <inheritdoc/>
    public Tensor<T> SparseSampledAddMM<T>(SparseTensor<T> pattern, Tensor<T> a, Tensor<T> b, Tensor<T> c, T alpha, T beta)
        => SparseAutograd.SparseSampledAddMMRecord(pattern, a, b, c, alpha, beta);

    /// <inheritdoc/>
    public Tensor<T> SparseSpGeMM<T>(SparseTensor<T> a, SparseTensor<T> b)
        => SparseAutograd.SparsePatternPreservingSpGeMMRecord(a, b);

    /// <inheritdoc/>
    public Tensor<T> SparseSum<T>(SparseTensor<T> a, int? axis = null)
        => SparseAutograd.SparseSumRecord(a, a, axis);

    /// <inheritdoc/>
    public Tensor<T> SparseMean<T>(SparseTensor<T> a, int? axis = null)
        => SparseAutograd.SparseMeanRecord(a, a, axis);

    /// <inheritdoc/>
    public Tensor<T> SparseSoftmax<T>(SparseTensor<T> a)
        => SparseAutograd.SparseSoftmaxRecord(a, a);

    /// <inheritdoc/>
    public Tensor<T> SparseLogSoftmax<T>(SparseTensor<T> a)
        => SparseAutograd.SparseLogSoftmaxRecord(a, a);

    #endregion
}
