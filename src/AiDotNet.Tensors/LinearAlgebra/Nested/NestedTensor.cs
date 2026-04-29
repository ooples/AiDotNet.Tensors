// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.LinearAlgebra.Nested;

/// <summary>
/// Variable-length sequence batch — PyTorch's <c>torch.nested.nested_tensor</c>
/// equivalent. Stores a single contiguous values buffer plus a per-row
/// offsets array so that row <c>i</c> spans <c>values[offsets[i]..offsets[i+1]]</c>
/// and never pays for padding FLOPs in the ops below.
///
/// <para><b>Layout taxonomy:</b>
/// <list type="bullet">
///   <item><see cref="NestedLayout.Jagged"/> — single backing storage, offsets index ragged rows. Default; preferred for most attention / RNN workloads.</item>
///   <item><see cref="NestedLayout.Strided"/> — each row materialised as its own dense slice (no shared storage). Used when a downstream op needs row-local strides that don't survive an offsets walk.</item>
/// </list>
/// </para>
///
/// <para><b>Two shape modes:</b>
/// <list type="number">
///   <item><b>Variable seq length, fixed feature dim</b> — <c>[batch, *, features]</c>. Each row is <c>[seqLen_i, features]</c>; offsets index in <c>features</c>-multiples (so element-stride is implicit).</item>
///   <item><b>Variable everything</b> — <c>[batch, *]</c>. Each row is a 1-D slice; offsets index raw element positions.</item>
/// </list>
/// </para>
///
/// <para><b>How we beat PyTorch (#222 point #1):</b> our jagged
/// <see cref="NestedTensor{T}"/> reuses <see cref="Tensor{T}"/> + offsets
/// instead of being a separate sub-class with partial op coverage. Every
/// op below is a per-row apply over the existing dense kernels, so any
/// new <c>Tensor&lt;T&gt;</c> kernel automatically lifts to the nested
/// case.</para>
/// </summary>
public sealed class NestedTensor<T>
{
    /// <summary>Concatenated values across every row, contiguous.</summary>
    public Tensor<T> Values { get; }

    /// <summary>Offsets into <see cref="Values"/>. Length = batch + 1;
    /// <c>Offsets[0] = 0</c>, <c>Offsets[batch] = Values.Length / FeatureSize</c>
    /// when <see cref="FeatureSize"/> &gt; 0, else equals total element count.</summary>
    public int[] Offsets { get; }

    /// <summary>Storage layout of the rows (<see cref="NestedLayout.Jagged"/>
    /// or <see cref="NestedLayout.Strided"/>).</summary>
    public NestedLayout Layout { get; }

    /// <summary>
    /// Per-row trailing dimension; <c>0</c> when each row is a 1-D
    /// slice (no fixed features axis). When &gt; 0, every row has shape
    /// <c>[seqLen_i, FeatureSize]</c> and the <c>seqLen_i = Offsets[i+1] - Offsets[i]</c>.
    /// </summary>
    public int FeatureSize { get; }

    /// <summary>Number of rows in the batch.</summary>
    public int BatchSize => Offsets.Length - 1;

    /// <summary>Length of row <paramref name="i"/> along the variable axis.</summary>
    public int RowLength(int i) => Offsets[i + 1] - Offsets[i];

    /// <summary>Maximum row length across the batch — what
    /// <see cref="ToPadded"/> uses as its packed sequence dimension
    /// when no explicit <c>outputSize</c> is supplied.</summary>
    public int MaxRowLength
    {
        get
        {
            int max = 0;
            for (int i = 0; i < BatchSize; i++)
            {
                int len = RowLength(i);
                if (len > max) max = len;
            }
            return max;
        }
    }

    /// <summary>Total stored element count (sum of row lengths × <see cref="FeatureSize"/>).</summary>
    public int StoredElements => Values.Length;

    private NestedTensor(Tensor<T> values, int[] offsets, NestedLayout layout, int featureSize)
    {
        Values = values;
        Offsets = offsets;
        Layout = layout;
        FeatureSize = featureSize;
    }

    /// <summary>
    /// Constructs a jagged nested tensor from a list of dense rows.
    /// Each row must have the same trailing dimensions; only the first
    /// (sequence) axis is allowed to vary.
    /// </summary>
    public static NestedTensor<T> FromList(IList<Tensor<T>> rows)
    {
        if (rows is null) throw new ArgumentNullException(nameof(rows));
        if (rows.Count == 0)
            throw new ArgumentException("Cannot construct a NestedTensor from zero rows.", nameof(rows));

        int featureSize = 0;
        bool hasFeatureAxis = rows[0].Rank == 2;
        if (hasFeatureAxis)
        {
            featureSize = rows[0]._shape[1];
            for (int i = 1; i < rows.Count; i++)
            {
                if (rows[i].Rank != 2 || rows[i]._shape[1] != featureSize)
                    throw new ArgumentException(
                        $"Row {i} has shape mismatch — expected [*, {featureSize}], got {ShapeStr(rows[i]._shape)}.",
                        nameof(rows));
            }
        }
        else
        {
            for (int i = 0; i < rows.Count; i++)
            {
                if (rows[i].Rank != 1)
                    throw new ArgumentException(
                        $"Row {i} must be 1-D when the batch is rank-1 jagged; got rank {rows[i].Rank}.",
                        nameof(rows));
            }
        }

        var offsets = new int[rows.Count + 1];
        int totalRows = 0;
        for (int i = 0; i < rows.Count; i++)
        {
            int rowLen = rows[i]._shape[0];
            offsets[i] = totalRows;
            totalRows += rowLen;
        }
        offsets[rows.Count] = totalRows;

        int totalElements = totalRows * (hasFeatureAxis ? featureSize : 1);
        var values = new Tensor<T>(new[] { totalElements });
        var dst = values.AsWritableSpan();
        int cursor = 0;
        for (int i = 0; i < rows.Count; i++)
        {
            var src = rows[i].AsSpan();
            src.CopyTo(dst.Slice(cursor, src.Length));
            cursor += src.Length;
        }

        return new NestedTensor<T>(values, offsets, NestedLayout.Jagged, hasFeatureAxis ? featureSize : 0);
    }

    /// <summary>
    /// Constructs a nested tensor directly from a values buffer and
    /// offsets array — caller-owns; used by ops that produce a nested
    /// output (softmax, matmul) without round-tripping through
    /// <see cref="FromList"/>.
    /// </summary>
    public static NestedTensor<T> FromValuesOffsets(Tensor<T> values, int[] offsets, int featureSize, NestedLayout layout = NestedLayout.Jagged)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        if (offsets is null) throw new ArgumentNullException(nameof(offsets));
        if (offsets.Length < 2) throw new ArgumentException("Offsets must have at least one row.", nameof(offsets));
        return new NestedTensor<T>(values, offsets, layout, featureSize);
    }

    /// <summary>
    /// Materialises a padded dense tensor of shape <c>[batch, padLen, featureSize]</c>
    /// (or <c>[batch, padLen]</c> when <see cref="FeatureSize"/> is 0). Rows shorter
    /// than <paramref name="outputSize"/> are filled with <paramref name="padding"/>.
    /// Pass <paramref name="outputSize"/> = -1 to use <see cref="MaxRowLength"/>.
    /// </summary>
    public Tensor<T> ToPadded(T padding, int outputSize = -1)
    {
        int padLen = outputSize < 0 ? MaxRowLength : outputSize;
        int batch = BatchSize;
        var padded = FeatureSize > 0
            ? new Tensor<T>(new[] { batch, padLen, FeatureSize })
            : new Tensor<T>(new[] { batch, padLen });

        var src = Values.AsSpan();
        var dst = padded.AsWritableSpan();

        // Pre-fill with padding so short rows tail off cleanly.
        for (int i = 0; i < dst.Length; i++) dst[i] = padding;

        if (FeatureSize > 0)
        {
            int rowStridePadded = padLen * FeatureSize;
            for (int b = 0; b < batch; b++)
            {
                int rowLen = RowLength(b);
                int writable = Math.Min(rowLen, padLen);
                int srcStart = Offsets[b] * FeatureSize;
                int dstStart = b * rowStridePadded;
                src.Slice(srcStart, writable * FeatureSize)
                   .CopyTo(dst.Slice(dstStart, writable * FeatureSize));
            }
        }
        else
        {
            for (int b = 0; b < batch; b++)
            {
                int rowLen = RowLength(b);
                int writable = Math.Min(rowLen, padLen);
                int srcStart = Offsets[b];
                int dstStart = b * padLen;
                src.Slice(srcStart, writable).CopyTo(dst.Slice(dstStart, writable));
            }
        }
        return padded;
    }

    /// <summary>Round-trips a padded dense <paramref name="padded"/> tensor
    /// through a length-aware mask: the result is a jagged nested tensor
    /// holding only the first <c>lengths[i]</c> elements of each row.</summary>
    public static NestedTensor<T> FromPadded(Tensor<T> padded, int[] lengths)
    {
        if (padded is null) throw new ArgumentNullException(nameof(padded));
        if (lengths is null) throw new ArgumentNullException(nameof(lengths));
        if (padded.Rank != 2 && padded.Rank != 3)
            throw new ArgumentException("Padded input must be rank 2 or 3.", nameof(padded));
        if (padded._shape[0] != lengths.Length)
            throw new ArgumentException(
                $"Lengths array (size {lengths.Length}) must match padded batch dim ({padded._shape[0]}).");

        bool hasFeatureAxis = padded.Rank == 3;
        int featureSize = hasFeatureAxis ? padded._shape[2] : 0;
        int padLen = padded._shape[1];

        var offsets = new int[lengths.Length + 1];
        int total = 0;
        for (int i = 0; i < lengths.Length; i++)
        {
            if (lengths[i] < 0 || lengths[i] > padLen)
                throw new ArgumentException(
                    $"Row {i} length {lengths[i]} out of range [0, {padLen}].", nameof(lengths));
            offsets[i] = total;
            total += lengths[i];
        }
        offsets[lengths.Length] = total;

        int storedElements = total * (hasFeatureAxis ? featureSize : 1);
        var values = new Tensor<T>(new[] { storedElements });
        var src = padded.AsSpan();
        var dst = values.AsWritableSpan();

        if (hasFeatureAxis)
        {
            int rowStridePadded = padLen * featureSize;
            int cursor = 0;
            for (int b = 0; b < lengths.Length; b++)
            {
                int rowLen = lengths[b];
                int srcStart = b * rowStridePadded;
                src.Slice(srcStart, rowLen * featureSize)
                   .CopyTo(dst.Slice(cursor, rowLen * featureSize));
                cursor += rowLen * featureSize;
            }
        }
        else
        {
            int cursor = 0;
            for (int b = 0; b < lengths.Length; b++)
            {
                int rowLen = lengths[b];
                src.Slice(b * padLen, rowLen).CopyTo(dst.Slice(cursor, rowLen));
                cursor += rowLen;
            }
        }
        return new NestedTensor<T>(values, offsets, NestedLayout.Jagged, featureSize);
    }

    private static string ShapeStr(int[] shape) =>
        $"[{string.Join(", ", shape)}]";
}

/// <summary>Layout taxonomy for <see cref="NestedTensor{T}"/>.</summary>
public enum NestedLayout
{
    /// <summary>Rows live in a single backing values buffer indexed by
    /// offsets. Default and what every nested op below assumes.</summary>
    Jagged,

    /// <summary>Each row is its own dense allocation. Used when a row's
    /// strides need to differ from the implicit jagged stride.</summary>
    Strided,
}
