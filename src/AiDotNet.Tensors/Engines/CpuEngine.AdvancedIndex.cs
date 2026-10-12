using System;
using AiDotNet.Tensors.Engines.Compilation;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Advanced (NumPy-style) indexing, torch's <c>aten::index</c> / <c>x[idx0, :, idx2]</c>.
/// </summary>
/// <remarks>
/// <para>The index tensors broadcast to one shape B. When they sit on consecutive dimensions, B replaces those
/// dimensions in place; when a full slice (a null entry) separates them, B comes first and the sliced dimensions
/// follow in order. Negative indices count from the end. A boolean mask over k leading dimensions selects its true
/// positions, in row-major order.</para>
/// <para>The flat source position of every output element is computed once, and the gather is
/// <see cref="IEngine.TensorTake{T}"/>: differentiable (its backward scatters the gradient back, summing repeats) and
/// run by whatever engine executes it.</para>
/// </remarks>
public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> TensorIndex<T>(Tensor<T> tensor, Tensor<int>?[] indices)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousInput, "TensorIndex");
        if (indices.Length > tensor.Rank)
            throw new ArgumentException($"{indices.Length} indices for a tensor of rank {tensor.Rank}.", nameof(indices));
        int rank = tensor.Rank;
        var shape = tensor._shape;

        // Advanced positions and their broadcast shape.
        var advanced = new List<int>();
        for (int d = 0; d < indices.Length; d++) if (indices[d] is not null) advanced.Add(d);
        if (advanced.Count == 0) return tensor;   // all full slices: the tensor itself, as torch returns a view
        int[] b = Array.Empty<int>();
        foreach (int d in advanced) b = BroadcastIndexShapes(b, indices[d]!._shape);
        bool adjacent = advanced[advanced.Count - 1] - advanced[0] == advanced.Count - 1;

        // Output layout: (dims before the block) B (dims after), or B (all sliced dims) when separated.
        var sliced = new List<int>();
        for (int d = 0; d < rank; d++) if (d >= indices.Length || indices[d] is null) sliced.Add(d);
        var outShape = new List<int>();
        int firstAdv = advanced[0];
        if (adjacent)
        {
            foreach (int d in sliced) if (d < firstAdv) outShape.Add(shape[d]);
            outShape.AddRange(b);
            foreach (int d in sliced) if (d > firstAdv) outShape.Add(shape[d]);
        }
        else
        {
            outShape.AddRange(b);
            foreach (int d in sliced) outShape.Add(shape[d]);
        }

        // Row-major strides of the source and the broadcast index values.
        var stride = new long[rank];
        for (int d = rank - 1, s = 1; d >= 0; d--) { stride[d] = s; s *= shape[d]; }
        int bCount = 1; foreach (int v in b) bCount *= v;
        var advValues = new int[advanced.Count][];
        for (int a = 0; a < advanced.Count; a++)
        {
            int d = advanced[a];
            var idx = indices[d]!;
            var src = (idx.IsContiguous ? idx : idx.Contiguous()).AsSpan();
            var vals = new int[bCount];
            var bIndex = new int[b.Length];
            for (int i = 0; i < bCount; i++)
            {
                // position in the index tensor under broadcasting (its dims are right-aligned to B)
                int off = 0, mul = 1;
                for (int k = idx.Rank - 1, kb = b.Length - 1; k >= 0; k--, kb--)
                {
                    int coord = idx._shape[k] == 1 ? 0 : bIndex[kb];
                    off += coord * mul; mul *= idx._shape[k];
                }
                int v = src[off];
                if (v < -shape[d] || v >= shape[d])
                    throw new IndexOutOfRangeException($"Index {v} is out of bounds for dimension {d} of size {shape[d]}.");
                vals[i] = v < 0 ? v + shape[d] : v;
                for (int k = b.Length - 1; k >= 0; k--) { if (++bIndex[k] < b[k]) break; bIndex[k] = 0; }
            }
            advValues[a] = vals;
        }

        // Flat source index of every output element.
        int outCount = 1; foreach (int v in outShape) outCount *= v;
        var flat = new int[outCount];
        int beforeCount = 1, afterCount = 1;
        var beforeDims = new List<int>(); var afterDims = new List<int>();
        foreach (int d in sliced)
        {
            if (adjacent && d < firstAdv) { beforeDims.Add(d); beforeCount *= shape[d]; }
            else { afterDims.Add(d); afterCount *= shape[d]; }
        }
        int o = 0;
        var before = new int[beforeDims.Count];
        for (int bi = 0; bi < beforeCount; bi++)
        {
            long baseBefore = 0;
            for (int k = 0; k < beforeDims.Count; k++) baseBefore += before[k] * stride[beforeDims[k]];
            for (int i = 0; i < bCount; i++)
            {
                long baseAdv = baseBefore;
                for (int a = 0; a < advanced.Count; a++) baseAdv += advValues[a][i] * stride[advanced[a]];
                var after = new int[afterDims.Count];
                for (int ai = 0; ai < afterCount; ai++)
                {
                    long pos = baseAdv;
                    for (int k = 0; k < afterDims.Count; k++) pos += after[k] * stride[afterDims[k]];
                    flat[o++] = checked((int)pos);
                    for (int k = afterDims.Count - 1; k >= 0; k--) { if (++after[k] < shape[afterDims[k]]) break; after[k] = 0; }
                }
            }
            for (int k = beforeDims.Count - 1; k >= 0; k--) { if (++before[k] < shape[beforeDims[k]]) break; before[k] = 0; }
        }

        IEngine e = this;
        return e.TensorTake(tensor, new Tensor<int>(flat, outShape.ToArray()));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIndex<T>(Tensor<T> tensor, Tensor<bool> mask)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (mask is null) throw new ArgumentNullException(nameof(mask));
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousInput, "TensorIndex");
        int k = mask.Rank;
        if (k > tensor.Rank) throw new ArgumentException("The mask has more dimensions than the tensor.", nameof(mask));
        for (int d = 0; d < k; d++)
            if (mask._shape[d] != tensor._shape[d])
                throw new ArgumentException($"Mask dimension {d} is {mask._shape[d]} but the tensor's is {tensor._shape[d]}.", nameof(mask));
        // A mask over k dimensions is its true positions as k index tensors (torch's nonzero), in row-major order.
        var m = (mask.IsContiguous ? mask : mask.Contiguous()).AsSpan();
        var coords = new List<int>[k];
        for (int d = 0; d < k; d++) coords[d] = new List<int>();
        var at = new int[k];
        for (int i = 0; i < m.Length; i++)
        {
            if (m[i]) for (int d = 0; d < k; d++) coords[d].Add(at[d]);
            for (int d = k - 1; d >= 0; d--) { if (++at[d] < mask._shape[d]) break; at[d] = 0; }
        }
        var idx = new Tensor<int>?[k];
        for (int d = 0; d < k; d++) idx[d] = new Tensor<int>(coords[d].ToArray(), new[] { coords[d].Count });
        return TensorIndex(tensor, idx);
    }

    private static int[] BroadcastIndexShapes(int[] a, int[] b)
    {
        int n = Math.Max(a.Length, b.Length);
        var r = new int[n];
        for (int i = 0; i < n; i++)
        {
            int da = i < n - a.Length ? 1 : a[i - (n - a.Length)];
            int db = i < n - b.Length ? 1 : b[i - (n - b.Length)];
            if (da != db && da != 1 && db != 1)
                throw new ArgumentException($"Index shapes [{string.Join(", ", a)}] and [{string.Join(", ", b)}] do not broadcast.");
            r[i] = Math.Max(da, db);
        }
        return r;
    }
}
