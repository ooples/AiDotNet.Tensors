using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// Row gather / in-place row scatter at 64-bit row ids, skipping ids outside the table: the sparse-embedding update's
// gather -> dense engine math -> scatter, in O(k * rowSize) rather than the O(vocab * rowSize) of a full-table copy.
public partial class CpuEngine
{
    /// <summary>Rows of <paramref name="table"/> at <paramref name="rows"/>; see
    /// <see cref="TensorDataEngineExtensions.TensorGatherRows{T}"/>.</summary>
    public virtual Tensor<T> TensorGatherRows<T>(Tensor<T> table, Tensor<long> rows)
    {
        var (vocab, rowSize, outShape) = GatherRowsShape(table, rows);
        var ids = ReadAll(rows);
        var result = new Tensor<T>(outShape);
        if (ids.Length > 0 && rowSize > 0)
        {
            using var source = table.Lease();
            var src = source.Span;
            var dst = result.GetDataArray().AsSpan();
            for (int k = 0; k < ids.Length; k++)
            {
                long row = ids[k];
                if (row < 0 || row >= vocab) continue;   // the result is zero-initialised
                src.Slice((int)row * rowSize, rowSize).CopyTo(dst.Slice(k * rowSize, rowSize));
            }
        }
        RecordGatherRows(result, table, ids);
        return result;
    }

    /// <summary>Records the gather's tape node (the table; the ids are saved state).</summary>
    private protected static void RecordGatherRows<T>(Tensor<T> result, Tensor<T> table, long[] ids) =>
        DifferentiableOps.RecordIfActive("GatherRows", result, new[] { table }, GatherRowsBackward<T>, new object[] { ids });

    /// <summary>Writes <paramref name="source"/>'s rows into <paramref name="table"/> at <paramref name="rows"/>; see
    /// <see cref="TensorDataEngineExtensions.TensorScatterRowsInPlace{T}"/>.</summary>
    public virtual void TensorScatterRowsInPlace<T>(Tensor<T> table, Tensor<long> rows, Tensor<T> source)
    {
        var (vocab, rowSize, expected) = GatherRowsShape(table, rows);
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (!SameShape(source._shape, expected))
            throw new ArgumentException($"source must have shape [{string.Join(", ", expected)}] (rows' shape, then the table's row shape); got [{string.Join(", ", source._shape)}].", nameof(source));
        if (table.IsSparse) throw new InvalidOperationException("Row scatter into a SparseTensor is not supported.");
        // A broadcast (zero-stride) or otherwise self-overlapping view would write unselected rows too.
        table.EnsureNonOverlappingWritableLayout();
        var ids = ReadAll(rows);
        if (ids.Length == 0 || rowSize == 0) return;
        // A source sharing the table's storage (the table itself, or a view of it) would read rows this scatter has
        // already overwritten: copy it first, so every row written is the source's value before the call.
        source = SnapshotIfAliased(table, source);
        using var sourceLease = source.Lease();
        var src = sourceLease.Span;
        if (table.IsContiguous)
        {
            using var lease = table.LeaseWritable();
            var dst = lease.Span;
            for (int k = 0; k < ids.Length; k++)
            {
                long row = ids[k];
                if (row < 0 || row >= vocab) continue;
                src.Slice(k * rowSize, rowSize).CopyTo(dst.Slice((int)row * rowSize, rowSize));
            }
            // The lease does not bump the version: an in-place write must, or a cached device copy (and a tape's
            // saved-tensor check) would keep the old rows.
            table.IncrementVersion();
            return;
        }
        // A strided view: each element goes to its storage slot.
        var storage = table.RawWritableStorageSpan;
        for (int k = 0; k < ids.Length; k++)
        {
            long row = ids[k];
            if (row < 0 || row >= vocab) continue;
            int at = (int)row * rowSize;
            for (int j = 0; j < rowSize; j++) storage[table.LogicalToStorageIndex(at + j)] = src[k * rowSize + j];
        }
        table.IncrementVersion();
    }

    /// <summary><paramref name="source"/>, or a private copy of it when it shares <paramref name="table"/>'s storage.</summary>
    private protected static Tensor<T> SnapshotIfAliased<T>(Tensor<T> table, Tensor<T> source) =>
        ReferenceEquals(source.StorageIdentity, table.StorageIdentity) ? source.Clone() : source;

    /// <summary>(vocabulary, elements per row, gathered shape = rows' shape then the table's row shape).</summary>
    internal static (int Vocab, int RowSize, int[] GatheredShape) GatherRowsShape<T>(Tensor<T> table, Tensor<long> rows)
    {
        if (table is null) throw new ArgumentNullException(nameof(table));
        if (rows is null) throw new ArgumentNullException(nameof(rows));
        if (table.Rank < 1) throw new ArgumentException("The table needs at least one dimension [vocab, ...].", nameof(table));
        int vocab = table._shape[0];
        int rowSize = vocab == 0 ? 1 : table.Length / vocab;
        if (vocab == 0)
            for (int d = 1; d < table.Rank; d++) rowSize *= table._shape[d];
        var shape = new int[rows.Rank + table.Rank - 1];
        Array.Copy(rows._shape, shape, rows.Rank);
        Array.Copy(table._shape, 1, shape, rows.Rank, table.Rank - 1);
        if ((long)rows.Length * rowSize > int.MaxValue)
            throw new ArgumentException("The gathered rows hold more than int.MaxValue elements.", nameof(rows));
        return (vocab, rowSize, shape);
    }

    /// <summary>
    /// Backward of <see cref="TensorGatherRows{T}"/>: the table's gradient is zero except at the gathered rows, which
    /// accumulate their output rows' gradients (a row gathered twice gets both): an index-add of the in-range rows.
    /// </summary>
    internal static void GatherRowsBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var table = inputs[0];
        if (!DifferentiableOps.IsGradientRequired(table)) return;
        var ids = (long[])savedState[0];
        int vocab = table._shape[0];
        int rowSize = vocab == 0 ? 0 : table.Length / vocab;
        var keptPositions = new List<int>(ids.Length);
        var keptRows = new List<int>(ids.Length);
        for (int k = 0; k < ids.Length; k++)
        {
            if (ids[k] < 0 || ids[k] >= vocab) continue;
            keptPositions.Add(k);
            keptRows.Add((int)ids[k]);
        }
        var zeros = new Tensor<T>(new[] { vocab, rowSize });
        Tensor<T> grad = zeros;
        if (keptRows.Count > 0 && rowSize > 0)
        {
            var flat = engine.Reshape(gradOutput, new[] { ids.Length, rowSize });
            var picked = engine.TensorIndexSelect(flat, new Tensor<int>(keptPositions.ToArray(), new[] { keptPositions.Count }), 0);
            grad = engine.TensorIndexAdd(zeros, 0, new Tensor<int>(keptRows.ToArray(), new[] { keptRows.Count }), picked);
        }
        DifferentiableOps.AccumulateGrad(grads, table, engine.Reshape(grad, (int[])table._shape.Clone()), engine);
    }
}
