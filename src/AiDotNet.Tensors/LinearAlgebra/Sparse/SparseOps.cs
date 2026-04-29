// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra.Sparse;

/// <summary>
/// Sparse-tensor op surface — mirrors PyTorch's <c>torch.sparse</c>
/// namespace. Every op below dispatches on <see cref="SparseTensor{T}.Format"/>
/// and reaches for the format's natural traversal pattern (CSR for
/// row-major SpMM, CSC for transpose multiplies, block formats for
/// LLM-pruned weights).
///
/// <para>The <see cref="ISparseDeviceOps"/> wrapper from #219 owns the
/// CUDA / HIP dispatch; this file hosts the managed CPU reference path.
/// Both produce numerically-equivalent answers — autotune-driven
/// dispatch from #221's "How we beat PyTorch" point #1 picks the
/// faster tier per shape × sparsity × device.</para>
/// </summary>
public static class SparseOps
{
    /// <summary>Sparse · dense matmul. Output shape: <c>[A.Rows, B.Columns]</c>.
    /// Routes through CSR for the natural row-major traversal; non-CSR
    /// inputs convert once on entry.</summary>
    public static Tensor<T> SparseMatMul<T>(SparseTensor<T> a, Tensor<T> b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (b.Rank != 2) throw new ArgumentException("SparseMatMul expects a 2-D dense right operand.", nameof(b));
        if (a.Columns != b._shape[0])
            throw new ArgumentException($"Inner dim mismatch: A cols {a.Columns} vs B rows {b._shape[0]}.");

        var ops = MathHelper.GetNumericOperations<T>();
        int rows = a.Rows, k = a.Columns, n = b._shape[1];
        var output = new Tensor<T>(new[] { rows, n });

        if (a.Format == SparseStorageFormat.Bsr) return BsrTimesDense(a, b, output, ops);

        var csr = a.Format == SparseStorageFormat.Csr ? a : a.ToCsr();
        var rowPtr = csr.RowPointers;
        var colIdx = csr.ColumnIndices;
        var vals = csr.DataVector;
        var bSpan = b.AsSpan();
        var outSpan = output.AsWritableSpan();

        for (int r = 0; r < rows; r++)
        {
            int rs = rowPtr[r], re = rowPtr[r + 1];
            for (int j = 0; j < n; j++)
            {
                T acc = ops.Zero;
                for (int p = rs; p < re; p++)
                    acc = ops.Add(acc, ops.Multiply(vals[p], bSpan[colIdx[p] * n + j]));
                outSpan[r * n + j] = acc;
            }
        }
        _ = k;
        return output;
    }

    /// <summary>Sparse · dense matvec. Mirrors <c>torch.sparse.mv</c>.</summary>
    public static Tensor<T> SparseSpMV<T>(SparseTensor<T> a, Tensor<T> x)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (x.Length != a.Columns)
            throw new ArgumentException($"Vector length {x.Length} doesn't match A.Columns {a.Columns}.");

        var ops = MathHelper.GetNumericOperations<T>();
        var csr = a.Format == SparseStorageFormat.Csr ? a : a.ToCsr();
        var rowPtr = csr.RowPointers;
        var colIdx = csr.ColumnIndices;
        var vals = csr.DataVector;
        var xSpan = x.AsSpan();
        var output = new Tensor<T>(new[] { a.Rows });
        var outSpan = output.AsWritableSpan();

        for (int r = 0; r < a.Rows; r++)
        {
            int rs = rowPtr[r], re = rowPtr[r + 1];
            T acc = ops.Zero;
            for (int p = rs; p < re; p++)
                acc = ops.Add(acc, ops.Multiply(vals[p], xSpan[colIdx[p]]));
            outSpan[r] = acc;
        }
        return output;
    }

    /// <summary>
    /// <c>α · (A · B) + β · C</c> with sparse <paramref name="a"/> and dense
    /// <paramref name="b"/>, <paramref name="c"/>. Mirrors <c>torch.sparse.addmm</c>.
    /// </summary>
    public static Tensor<T> SparseAddMM<T>(Tensor<T> c, SparseTensor<T> a, Tensor<T> b, T alpha, T beta)
    {
        var product = SparseMatMul(a, b);
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])product._shape.Clone());
        var pSpan = product.AsSpan();
        var cSpan = c.AsSpan();
        var oSpan = output.AsWritableSpan();
        if (cSpan.Length != pSpan.Length)
            throw new ArgumentException($"C length {cSpan.Length} doesn't match A·B output length {pSpan.Length}.");
        for (int i = 0; i < oSpan.Length; i++)
            oSpan[i] = ops.Add(ops.Multiply(alpha, pSpan[i]), ops.Multiply(beta, cSpan[i]));
        return output;
    }

    /// <summary>Batched sparse · dense matmul. <paramref name="batchSparse"/>
    /// is treated as a list of per-batch sparse tensors with the same shape.</summary>
    public static Tensor<T> SparseBmm<T>(SparseTensor<T>[] batchSparse, Tensor<T> denseBatch)
    {
        if (batchSparse is null) throw new ArgumentNullException(nameof(batchSparse));
        if (denseBatch is null) throw new ArgumentNullException(nameof(denseBatch));
        if (denseBatch.Rank != 3)
            throw new ArgumentException("denseBatch must be 3-D [batch, k, n].", nameof(denseBatch));
        if (batchSparse.Length != denseBatch._shape[0])
            throw new ArgumentException("Batch dim mismatch between sparse list and dense batch.");

        int batch = batchSparse.Length;
        int outRows = batchSparse[0].Rows;
        int outCols = denseBatch._shape[2];
        int k = batchSparse[0].Columns;
        var output = new Tensor<T>(new[] { batch, outRows, outCols });
        var outSpan = output.AsWritableSpan();

        for (int b = 0; b < batch; b++)
        {
            // Slice the batch's dense slab into its own 2-D tensor.
            var slab = new Tensor<T>(new[] { k, outCols });
            denseBatch.AsSpan().Slice(b * k * outCols, k * outCols).CopyTo(slab.AsWritableSpan());
            var prod = SparseMatMul(batchSparse[b], slab);
            prod.AsSpan().CopyTo(outSpan.Slice(b * outRows * outCols, outRows * outCols));
        }
        return output;
    }

    /// <summary>Sums all stored non-zeros (or every entry along an axis if
    /// supplied). Zero entries don't contribute regardless. Mirrors
    /// <c>torch.sparse.sum</c>.</summary>
    public static Tensor<T> SparseSum<T>(SparseTensor<T> a, int? axis = null)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (axis is null)
        {
            T acc = ops.Zero;
            var span = a.DataVector.AsSpan();
            for (int i = 0; i < span.Length; i++) acc = ops.Add(acc, span[i]);
            var result = new Tensor<T>(new[] { 1 });
            result.AsWritableSpan()[0] = acc;
            return result;
        }
        // Axis-aware sum: dense the answer (PyTorch returns dense for axis-reduced sparse sums).
        var coo = a.ToCoo();
        var cooVals = coo.DataVector.ToArray();
        if (axis.Value == 0)
        {
            var sums = new Tensor<T>(new[] { a.Columns });
            var s = sums.AsWritableSpan();
            for (int i = 0; i < s.Length; i++) s[i] = ops.Zero;
            for (int k = 0; k < cooVals.Length; k++)
                s[coo.ColumnIndices[k]] = ops.Add(s[coo.ColumnIndices[k]], cooVals[k]);
            return sums;
        }
        if (axis.Value == 1)
        {
            var sums = new Tensor<T>(new[] { a.Rows });
            var s = sums.AsWritableSpan();
            for (int i = 0; i < s.Length; i++) s[i] = ops.Zero;
            for (int k = 0; k < cooVals.Length; k++)
                s[coo.RowIndices[k]] = ops.Add(s[coo.RowIndices[k]], cooVals[k]);
            return sums;
        }
        throw new ArgumentOutOfRangeException(nameof(axis), $"Axis must be null, 0, or 1; got {axis}.");
    }

    /// <summary>Mean across the same axis surface as <see cref="SparseSum{T}"/>.
    /// PyTorch divides by the dense element count along the axis (zero entries
    /// included), matching <c>torch.sparse.mean</c>.</summary>
    public static Tensor<T> SparseMean<T>(SparseTensor<T> a, int? axis = null)
    {
        var sum = SparseSum(a, axis);
        var ops = MathHelper.GetNumericOperations<T>();
        int divisor = axis is null
            ? a.Rows * a.Columns
            : axis.Value == 0 ? a.Rows
            : axis.Value == 1 ? a.Columns
            : throw new ArgumentOutOfRangeException(nameof(axis));
        var span = sum.AsWritableSpan();
        T denom = ops.FromDouble(divisor);
        for (int i = 0; i < span.Length; i++) span[i] = ops.Divide(span[i], denom);
        return sum;
    }

    /// <summary>Per-row softmax over the stored non-zeros. Zero entries are
    /// excluded from the softmax (matches <c>torch.sparse.softmax</c>'s
    /// "ignore implicit zeros" semantics — this differs from doing softmax
    /// on the dense tensor which would treat zeros as <c>exp(0) = 1</c>).</summary>
    public static SparseTensor<T> SparseSoftmax<T>(SparseTensor<T> a)
        => SparseSoftmaxCore(a, takeLog: false);

    /// <summary>Log-softmax companion of <see cref="SparseSoftmax{T}"/>.</summary>
    public static SparseTensor<T> SparseLogSoftmax<T>(SparseTensor<T> a)
        => SparseSoftmaxCore(a, takeLog: true);

    private static SparseTensor<T> SparseSoftmaxCore<T>(SparseTensor<T> a, bool takeLog)
    {
        var csr = a.Format == SparseStorageFormat.Csr ? a : a.ToCsr();
        var rowPtr = csr.RowPointers;
        var colIdx = csr.ColumnIndices;
        var src = csr.DataVector.ToArray();
        var ops = MathHelper.GetNumericOperations<T>();

        var outVals = new T[src.Length];
        for (int r = 0; r < a.Rows; r++)
        {
            int rs = rowPtr[r], re = rowPtr[r + 1];
            if (re == rs) continue;
            // Subtract the per-row max for numerical stability before exp.
            T max = src[rs];
            for (int p = rs + 1; p < re; p++)
                if (ops.GreaterThan(src[p], max)) max = src[p];

            double sumExp = 0;
            var expBuf = new double[re - rs];
            for (int p = rs; p < re; p++)
            {
                double e = Math.Exp(ops.ToDouble(ops.Subtract(src[p], max)));
                expBuf[p - rs] = e;
                sumExp += e;
            }
            double logSum = Math.Log(sumExp);
            for (int p = rs; p < re; p++)
            {
                if (takeLog)
                    outVals[p] = ops.FromDouble(ops.ToDouble(ops.Subtract(src[p], max)) - logSum);
                else
                    outVals[p] = ops.FromDouble(expBuf[p - rs] / sumExp);
            }
        }
        return SparseTensor<T>.FromCsr(a.Rows, a.Columns, (int[])rowPtr.Clone(), (int[])colIdx.Clone(), outVals);
    }

    /// <summary>Filters <paramref name="dense"/> down to the non-zero
    /// pattern of <paramref name="mask"/>. Returns a sparse tensor that
    /// holds <c>dense[i, j]</c> at every non-zero position of the mask.
    /// Mirrors <c>torch.sparse_mask</c>.</summary>
    public static SparseTensor<T> SparseMask<T>(Tensor<T> dense, SparseTensor<T> mask)
    {
        if (dense.Rank != 2)
            throw new ArgumentException("SparseMask expects a 2-D dense input.", nameof(dense));
        if (dense._shape[0] != mask.Rows || dense._shape[1] != mask.Columns)
            throw new ArgumentException("dense and mask shape mismatch.");

        var coo = mask.ToCoo();
        int nnz = coo.NonZeroCount;
        var rows = (int[])coo.RowIndices.Clone();
        var cols = (int[])coo.ColumnIndices.Clone();
        var vals = new T[nnz];
        var dSpan = dense.AsSpan();
        int n = dense._shape[1];
        for (int k = 0; k < nnz; k++)
            vals[k] = dSpan[rows[k] * n + cols[k]];
        return new SparseTensor<T>(mask.Rows, mask.Columns, rows, cols, vals);
    }

    /// <summary>
    /// <c>α · (A · B) + β · C</c> sampled at the non-zero pattern of
    /// <paramref name="pattern"/>. Output is sparse with the same pattern.
    /// Mirrors <c>torch.sparse.sampled_addmm</c> — the building block for
    /// sparse-masked attention.
    /// </summary>
    public static SparseTensor<T> SparseSampledAddMM<T>(SparseTensor<T> pattern,
        Tensor<T> a, Tensor<T> b, Tensor<T> c, T alpha, T beta)
    {
        if (a.Rank != 2 || b.Rank != 2 || c.Rank != 2)
            throw new ArgumentException("a, b, c must be 2-D.");
        int m = pattern.Rows, n = pattern.Columns, k = a._shape[1];
        if (a._shape[0] != m || b._shape[0] != k || b._shape[1] != n || c._shape[0] != m || c._shape[1] != n)
            throw new ArgumentException("Shape mismatch for sampled_addmm.");

        var ops = MathHelper.GetNumericOperations<T>();
        var coo = pattern.ToCoo();
        int nnz = coo.NonZeroCount;
        var rows = (int[])coo.RowIndices.Clone();
        var cols = (int[])coo.ColumnIndices.Clone();
        var outVals = new T[nnz];
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        var cSpan = c.AsSpan();

        for (int p = 0; p < nnz; p++)
        {
            int r = rows[p], col = cols[p];
            T acc = ops.Zero;
            for (int i = 0; i < k; i++)
                acc = ops.Add(acc, ops.Multiply(aSpan[r * k + i], bSpan[i * n + col]));
            T scaled = ops.Multiply(alpha, acc);
            T cVal = cSpan[r * n + col];
            outVals[p] = ops.Add(scaled, ops.Multiply(beta, cVal));
        }
        return new SparseTensor<T>(m, n, rows, cols, outVals);
    }

    /// <summary>
    /// Constructs a sparse tensor with the supplied diagonals.
    /// <paramref name="diagonals"/> is a 2-D dense tensor where row <c>i</c>
    /// is the values for offset <c>offsets[i]</c>; positive offsets are
    /// above the main diagonal, negative below. Mirrors <c>torch.sparse.spdiags</c>.
    /// </summary>
    public static SparseTensor<T> SparseSpDiags<T>(Tensor<T> diagonals, int[] offsets, int rows, int cols)
    {
        if (diagonals.Rank != 2)
            throw new ArgumentException("diagonals must be 2-D.", nameof(diagonals));
        if (offsets.Length != diagonals._shape[0])
            throw new ArgumentException("offsets length must match diagonals row count.", nameof(offsets));

        var ops = MathHelper.GetNumericOperations<T>();
        var rowList = new System.Collections.Generic.List<int>();
        var colList = new System.Collections.Generic.List<int>();
        var valList = new System.Collections.Generic.List<T>();
        var dSpan = diagonals.AsSpan();
        int diagLen = diagonals._shape[1];

        for (int dRow = 0; dRow < offsets.Length; dRow++)
        {
            int offset = offsets[dRow];
            for (int j = 0; j < diagLen; j++)
            {
                int r, c;
                if (offset >= 0) { r = j; c = j + offset; }
                else { r = j - offset; c = j; }
                if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
                T v = dSpan[dRow * diagLen + j];
                if (ops.Equals(v, ops.Zero)) continue;
                rowList.Add(r);
                colList.Add(c);
                valList.Add(v);
            }
        }
        return new SparseTensor<T>(rows, cols, rowList.ToArray(), colList.ToArray(), valList.ToArray());
    }

    /// <summary>Sparse × sparse matmul producing CSR output (cuSPARSE
    /// SpGEMM semantics). Both inputs auto-convert to CSR on entry.</summary>
    public static SparseTensor<T> SparseSpGeMM<T>(SparseTensor<T> a, SparseTensor<T> b)
    {
        if (a.Columns != b.Rows)
            throw new ArgumentException($"Inner dim mismatch: A cols {a.Columns} vs B rows {b.Rows}.");
        var ops = MathHelper.GetNumericOperations<T>();
        var aCsr = a.Format == SparseStorageFormat.Csr ? a : a.ToCsr();
        var bCsr = b.Format == SparseStorageFormat.Csr ? b : b.ToCsr();
        var aRp = aCsr.RowPointers; var aCi = aCsr.ColumnIndices; var aVals = aCsr.DataVector;
        var bRp = bCsr.RowPointers; var bCi = bCsr.ColumnIndices; var bVals = bCsr.DataVector;

        var rowPtr = new int[a.Rows + 1];
        var rowAcc = new System.Collections.Generic.Dictionary<int, T>();
        var values = new System.Collections.Generic.List<T>();
        var colIdx = new System.Collections.Generic.List<int>();

        for (int r = 0; r < a.Rows; r++)
        {
            rowAcc.Clear();
            for (int p = aRp[r]; p < aRp[r + 1]; p++)
            {
                int aCol = aCi[p];
                T aV = aVals[p];
                for (int q = bRp[aCol]; q < bRp[aCol + 1]; q++)
                {
                    int bCol = bCi[q];
                    T contribution = ops.Multiply(aV, bVals[q]);
                    rowAcc[bCol] = rowAcc.TryGetValue(bCol, out var existing)
                        ? ops.Add(existing, contribution)
                        : contribution;
                }
            }
            var keys = new int[rowAcc.Count];
            rowAcc.Keys.CopyTo(keys, 0);
            Array.Sort(keys);
            foreach (var c in keys) { values.Add(rowAcc[c]); colIdx.Add(c); }
            rowPtr[r + 1] = values.Count;
        }
        return SparseTensor<T>.FromCsr(a.Rows, b.Columns, rowPtr, colIdx.ToArray(), values.ToArray());
    }

    private static Tensor<T> BsrTimesDense<T>(SparseTensor<T> a, Tensor<T> b, Tensor<T> output, Interfaces.INumericOperations<T> ops)
    {
        // Dense BSR×dense: walk by block-row, multiply each non-zero block into
        // the right strip of B, accumulate into the corresponding strip of out.
        int br = a.BlockRowSize, bc = a.BlockColSize;
        int blockSize = br * bc;
        int n = b._shape[1];
        int blockRows = a.Rows / br;
        var src = a.DataVector;
        var bSpan = b.AsSpan();
        var outSpan = output.AsWritableSpan();

        for (int br_i = 0; br_i < blockRows; br_i++)
        {
            int outRowOff = br_i * br;
            for (int p = a.RowPointers[br_i]; p < a.RowPointers[br_i + 1]; p++)
            {
                int blockCol = a.ColumnIndices[p];
                int valOff = p * blockSize;
                int bRowOff = blockCol * bc;
                for (int i = 0; i < br; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        T acc = outSpan[(outRowOff + i) * n + j];
                        for (int kk = 0; kk < bc; kk++)
                        {
                            acc = ops.Add(acc, ops.Multiply(
                                src[valOff + i * bc + kk],
                                bSpan[(bRowOff + kk) * n + j]));
                        }
                        outSpan[(outRowOff + i) * n + j] = acc;
                    }
                }
            }
        }
        return output;
    }
}
