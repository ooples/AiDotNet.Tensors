// Copyright (c) AiDotNet. All rights reserved.
// Advanced indexing on Tensor<T> — boolean-mask indexing, tensor-of-indices
// indexing, unsqueeze/squeeze aliases, and None-axis insertion. Landed as a
// partial class alongside Tensor.cs so the existing numeric / shape API
// stays the same.

using System;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tensors.LinearAlgebra;

public partial class Tensor<T>
{
    /// <summary>
    /// Memory layout of this tensor's elements. Defaults to
    /// <see cref="MemoryFormat.Contiguous"/> (NCHW). Ops that detect
    /// <see cref="MemoryFormat.ChannelsLast"/> route through layout-
    /// preserving kernels so gather / scatter / pool / norm don't
    /// degrade to the slow generic stride walker the way PyTorch's
    /// CPU path does for non-contiguous channels-last tensors.
    /// </summary>
    public MemoryFormat MemoryFormat { get; set; } = MemoryFormat.Contiguous;

    /// <summary>
    /// Copies the <see cref="MemoryFormat"/> of <paramref name="source"/>
    /// onto this tensor. Used by movement / indexing ops whose output
    /// retains the input's channels-last / channels-first layout.
    /// </summary>
    public Tensor<T> PreserveLayoutFrom(Tensor<T> source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        MemoryFormat = source.MemoryFormat;
        return this;
    }

    /// <summary>
    /// Boolean-mask indexing: returns the 1-D tensor of elements where
    /// <paramref name="mask"/> is true.  Shape of <paramref name="mask"/>
    /// must match the tensor's shape exactly.  Equivalent to
    /// <c>tensor[mask]</c> in NumPy / PyTorch.
    /// </summary>
    public Tensor<T> this[Tensor<Bit> mask]
    {
        get
        {
            if (mask == null) throw new ArgumentNullException(nameof(mask));
            var engine = new CpuEngine();
            return engine.TensorMaskedSelect(this, mask);
        }
    }

    /// <summary>
    /// Tensor-of-indices indexing: returns elements along axis 0 picked by
    /// <paramref name="indices"/>. The result has shape
    /// <c>indices.Shape + this.Shape[1..]</c>.  Equivalent to
    /// <c>tensor[index_tensor]</c> in PyTorch when the index tensor is 1-D.
    /// </summary>
    public Tensor<T> this[Tensor<int> indices]
    {
        get
        {
            if (indices == null) throw new ArgumentNullException(nameof(indices));
            var engine = new CpuEngine();
            return engine.TensorIndexSelect(this, indices, 0);
        }
    }

    /// <summary>
    /// torch.Tensor.unsqueeze alias — inserts a new axis of length 1 at the
    /// specified position.  Equivalent to <c>None</c> (or <c>np.newaxis</c>)
    /// in NumPy slicing.  Negative positions count from the end; rank+1 is
    /// valid (append a trailing axis).
    /// </summary>
    public Tensor<T> Unsqueeze(int axis)
    {
        int rank = Rank;
        if (axis < 0) axis += rank + 1;
        if (axis < 0 || axis > rank)
            throw new ArgumentOutOfRangeException(nameof(axis));
        return ExpandDims(axis);
    }

    /// <summary>
    /// Inserts a new axis of length 1 at the specified position — NumPy /
    /// PyTorch <c>None</c> or <c>np.newaxis</c> spelling.  Identical to
    /// <see cref="Unsqueeze"/>; provided for readability when the intent is
    /// "insert axis" rather than "squeeze away".
    /// </summary>
    public Tensor<T> InsertAxis(int position) => Unsqueeze(position);

    /// <summary>
    /// Negative-index normalisation helper — mirrors PyTorch's convention of
    /// treating <c>-1</c> as "last axis", <c>-2</c> as "second-to-last", etc.
    /// Callers pass the raw index and the tensor's rank; returns the
    /// positive equivalent or throws <see cref="ArgumentOutOfRangeException"/>.
    /// </summary>
    public static int NormalizeAxis(int axis, int rank)
    {
        int result = axis < 0 ? axis + rank : axis;
        if (result < 0 || result >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis),
                $"Axis {axis} out of range for rank {rank}");
        return result;
    }

    /// <summary>
    /// Select a single element along <paramref name="axis"/> at position
    /// <paramref name="index"/> — rank decreases by one.  Matches
    /// <c>torch.Tensor.select(dim, index)</c> / <c>t[..., i, ...]</c>.
    /// </summary>
    public Tensor<T> SelectAlong(int axis, int index)
    {
        int rank = Rank;
        int ax = axis < 0 ? axis + rank : axis;
        if (ax < 0 || ax >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis));
        int len = Shape[ax];
        if (index < 0) index += len;
        if (index < 0 || index >= len)
            throw new ArgumentOutOfRangeException(nameof(index));
        // Single-element select reduces to IndexSelect with a 1-element
        // index tensor, then Squeeze of the axis.
        var idx = new Tensor<int>(new[] { index }, new[] { 1 });
        var engine = new CpuEngine();
        var picked = engine.TensorIndexSelect(this, idx, ax);
        return picked.Squeeze(ax);
    }
}
