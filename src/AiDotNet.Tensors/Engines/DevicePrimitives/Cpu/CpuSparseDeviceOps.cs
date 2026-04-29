// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;

/// <summary>
/// CPU CSR-format sparse-matrix kernels. SpMM / SpMV walk the CSR
/// triple sequentially; SpGEMM uses the symbolic + numeric two-pass
/// pattern that cuSPARSE's <c>cusparseSpGEMM</c> exposes (cheaper than
/// a triangle-counting approach for moderate nnz). This is the
/// fallback path for hosts without a native sparse runtime — the
/// numerical answer must match what cuSPARSE produces.
/// </summary>
public sealed class CpuSparseDeviceOps : ISparseDeviceOps
{
    /// <summary>
    /// Validates the structural invariants of a CSR triple before the
    /// kernel loops dereference into <c>rowPtr[r+1]</c> /
    /// <c>colIdx[p]</c>. Without this guard, a malformed input crashes
    /// on a low-level IndexOutOfRangeException with no diagnostic
    /// context — common when CSR triples come from user code or a
    /// sparse-matrix loader.
    /// </summary>
    private static void ValidateCsr(
        Tensor<int> rowPtr, Tensor<int> colIdx, Tensor<int>? values,
        int rows, int cols, string component)
    {
        if (rowPtr.Length != rows + 1)
            throw new ArgumentException(
                $"{component}: rowPtr length must be rows+1 ({rows + 1}); got {rowPtr.Length}.", nameof(rowPtr));
        var rp = rowPtr.AsSpan();
        if (rp[0] != 0)
            throw new ArgumentException($"{component}: rowPtr[0] must be 0; got {rp[0]}.", nameof(rowPtr));
        for (int r = 0; r < rows; r++)
        {
            if (rp[r] > rp[r + 1])
                throw new ArgumentException(
                    $"{component}: rowPtr must be non-decreasing; rowPtr[{r}]={rp[r]} > rowPtr[{r + 1}]={rp[r + 1]}.",
                    nameof(rowPtr));
        }
        int nnz = rp[rows];
        if (colIdx.Length != nnz)
            throw new ArgumentException(
                $"{component}: colIdx length must equal nnz ({nnz}); got {colIdx.Length}.", nameof(colIdx));
        var ci = colIdx.AsSpan();
        for (int i = 0; i < ci.Length; i++)
        {
            if (ci[i] < 0 || ci[i] >= cols)
                throw new ArgumentOutOfRangeException(
                    nameof(colIdx),
                    $"{component}: colIdx[{i}]={ci[i]} is outside [0, {cols}).");
        }
        if (values is not null && values.Length != nnz)
            throw new ArgumentException(
                $"{component}: values length must equal nnz ({nnz}); got {values.Length}.", nameof(values));
    }

    private static void ValidateCsr<T>(
        Tensor<T> values, Tensor<int> rowPtr, Tensor<int> colIdx,
        int rows, int cols, string component)
    {
        if (rowPtr.Length != rows + 1)
            throw new ArgumentException(
                $"{component}: rowPtr length must be rows+1 ({rows + 1}); got {rowPtr.Length}.", nameof(rowPtr));
        var rp = rowPtr.AsSpan();
        if (rp[0] != 0)
            throw new ArgumentException($"{component}: rowPtr[0] must be 0; got {rp[0]}.", nameof(rowPtr));
        for (int r = 0; r < rows; r++)
        {
            if (rp[r] > rp[r + 1])
                throw new ArgumentException(
                    $"{component}: rowPtr must be non-decreasing; rowPtr[{r}]={rp[r]} > rowPtr[{r + 1}]={rp[r + 1]}.",
                    nameof(rowPtr));
        }
        int nnz = rp[rows];
        if (colIdx.Length != nnz)
            throw new ArgumentException(
                $"{component}: colIdx length must equal nnz ({nnz}); got {colIdx.Length}.", nameof(colIdx));
        if (values.Length != nnz)
            throw new ArgumentException(
                $"{component}: values length must equal nnz ({nnz}); got {values.Length}.", nameof(values));
        var ci = colIdx.AsSpan();
        for (int i = 0; i < ci.Length; i++)
        {
            if (ci[i] < 0 || ci[i] >= cols)
                throw new ArgumentOutOfRangeException(
                    nameof(colIdx),
                    $"{component}: colIdx[{i}]={ci[i]} is outside [0, {cols}).");
        }
    }

    /// <inheritdoc/>
    public Tensor<T> SpMM<T>(
        Tensor<T> csrValues, Tensor<int> csrRowPtr, Tensor<int> csrColIdx,
        int rows, int cols, Tensor<T> dense)
    {
        if (dense.Rank != 2) throw new ArgumentException("dense must be a 2-D matrix.", nameof(dense));
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (cols < 0) throw new ArgumentOutOfRangeException(nameof(cols));
        ValidateCsr(csrValues, csrRowPtr, csrColIdx, rows, cols, "SpMM");
        var ops = MathHelper.GetNumericOperations<T>();
        int k = dense._shape[0]; // matches `cols`
        int n = dense._shape[1];
        if (k != cols) throw new ArgumentException($"Inner dimension mismatch: sparse cols {cols} vs dense rows {k}.");

        var output = new Tensor<T>(new[] { rows, n });
        var outSpan = output.AsWritableSpan();
        var vals = csrValues.AsSpan();
        var rowPtr = csrRowPtr.AsSpan();
        var colIdx = csrColIdx.AsSpan();
        var denseSpan = dense.AsSpan();

        for (int r = 0; r < rows; r++)
        {
            int rs = rowPtr[r], re = rowPtr[r + 1];
            for (int j = 0; j < n; j++)
            {
                T acc = ops.Zero;
                for (int p = rs; p < re; p++)
                    acc = ops.Add(acc, ops.Multiply(vals[p], denseSpan[colIdx[p] * n + j]));
                outSpan[r * n + j] = acc;
            }
        }
        return output;
    }

    /// <inheritdoc/>
    public Tensor<T> SpMV<T>(
        Tensor<T> csrValues, Tensor<int> csrRowPtr, Tensor<int> csrColIdx,
        int rows, int cols, Tensor<T> denseVec)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (cols < 0) throw new ArgumentOutOfRangeException(nameof(cols));
        if (denseVec.Length != cols)
            throw new ArgumentException($"Vector length {denseVec.Length} doesn't match sparse cols {cols}.");
        ValidateCsr(csrValues, csrRowPtr, csrColIdx, rows, cols, "SpMV");
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>(new[] { rows });
        var outSpan = output.AsWritableSpan();
        var vals = csrValues.AsSpan();
        var rowPtr = csrRowPtr.AsSpan();
        var colIdx = csrColIdx.AsSpan();
        var x = denseVec.AsSpan();

        for (int r = 0; r < rows; r++)
        {
            int rs = rowPtr[r], re = rowPtr[r + 1];
            T acc = ops.Zero;
            for (int p = rs; p < re; p++)
                acc = ops.Add(acc, ops.Multiply(vals[p], x[colIdx[p]]));
            outSpan[r] = acc;
        }
        return output;
    }

    /// <inheritdoc/>
    public (Tensor<T> Values, Tensor<int> RowPtr, Tensor<int> ColIdx) SpGEMM<T>(
        Tensor<T> aValues, Tensor<int> aRowPtr, Tensor<int> aColIdx, int aRows, int aCols,
        Tensor<T> bValues, Tensor<int> bRowPtr, Tensor<int> bColIdx, int bRows, int bCols)
    {
        if (aRows < 0) throw new ArgumentOutOfRangeException(nameof(aRows));
        if (aCols < 0) throw new ArgumentOutOfRangeException(nameof(aCols));
        if (bRows < 0) throw new ArgumentOutOfRangeException(nameof(bRows));
        if (bCols < 0) throw new ArgumentOutOfRangeException(nameof(bCols));
        if (aCols != bRows)
            throw new ArgumentException($"Inner dim mismatch: A cols {aCols} vs B rows {bRows}.");
        ValidateCsr(aValues, aRowPtr, aColIdx, aRows, aCols, "SpGEMM.A");
        ValidateCsr(bValues, bRowPtr, bColIdx, bRows, bCols, "SpGEMM.B");
        var ops = MathHelper.GetNumericOperations<T>();
        var aVals = aValues.AsSpan();
        var aRp = aRowPtr.AsSpan();
        var aCi = aColIdx.AsSpan();
        var bVals = bValues.AsSpan();
        var bRp = bRowPtr.AsSpan();
        var bCi = bColIdx.AsSpan();

        var rowPtr = new int[aRows + 1];
        var rowAcc = new Dictionary<int, T>();
        var values = new List<T>();
        var colIdx = new List<int>();

        for (int r = 0; r < aRows; r++)
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
            // Emit this row sorted by column for canonical CSR layout.
            var keys = new int[rowAcc.Count];
            rowAcc.Keys.CopyTo(keys, 0);
            Array.Sort(keys);
            foreach (var c in keys) { values.Add(rowAcc[c]); colIdx.Add(c); }
            rowPtr[r + 1] = values.Count;
        }

        var vTensor = new Tensor<T>(new[] { values.Count });
        var ciTensor = new Tensor<int>(new[] { colIdx.Count });
        var rpTensor = new Tensor<int>(new[] { rowPtr.Length });
        var vSpan = vTensor.AsWritableSpan();
        var ciSpan = ciTensor.AsWritableSpan();
        var rpSpan = rpTensor.AsWritableSpan();
        for (int i = 0; i < values.Count; i++) { vSpan[i] = values[i]; ciSpan[i] = colIdx[i]; }
        for (int i = 0; i < rowPtr.Length; i++) rpSpan[i] = rowPtr[i];
        _ = aRows; _ = bCols;
        return (vTensor, rpTensor, ciTensor);
    }
}
