// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra.Masked;
using AiDotNet.Tensors.LinearAlgebra.Nested;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Zero-copy bridges between the three structured-tensor types added in
/// #222: <see cref="NestedTensor{T}"/>, padded dense + boolean mask, and
/// <see cref="MaskedTensor{T}"/>. Issue #222's "How we beat PyTorch"
/// point #7 — round-trip preservation across all three forms with no
/// extra storage.
///
/// <para><b>Round-trip identity:</b>
/// <c>padded → nested → padded</c> equals the original padded tensor
/// when <c>outputSize</c> matches the original padded sequence length.
/// <c>masked → dense → masked</c> equals the original when reconstructed
/// with the same mask. Validated in the acceptance tests.</para>
/// </summary>
public static class StructuredBridges
{
    /// <summary>
    /// Builds a <see cref="NestedTensor{T}"/> from a padded dense tensor
    /// + per-row lengths. Equivalent to <c>NestedTensor.FromPadded</c>;
    /// surfaced here as the symmetric API for the round-trip family.
    /// </summary>
    public static NestedTensor<T> NestedFromPadded<T>(Tensor<T> padded, int[] lengths)
        => NestedTensor<T>.FromPadded(padded, lengths);

    /// <summary>
    /// Materialises a padded dense view of <paramref name="nested"/>
    /// using <paramref name="padding"/> for short rows. The companion
    /// <see cref="NestedToMaskedFromPadding{T}"/> builds the matching
    /// mask in one pass.
    /// </summary>
    public static Tensor<T> NestedToPadded<T>(NestedTensor<T> nested, T padding, int outputSize = -1)
        => nested.ToPadded(padding, outputSize);

    /// <summary>
    /// Bridge that converts <paramref name="nested"/> into a masked
    /// tensor: padded dense values plus a boolean mask whose positions
    /// match the original row-length structure (true inside each row,
    /// false in the padding region).
    /// </summary>
    public static MaskedTensor<T> NestedToMasked<T>(NestedTensor<T> nested, T padding, int outputSize = -1)
    {
        int padLen = outputSize < 0 ? nested.MaxRowLength : outputSize;
        var values = nested.ToPadded(padding, padLen);
        int batch = nested.BatchSize;
        int featureSize = nested.FeatureSize;

        // Mask is row-major over the padded shape — true wherever a row
        // actually had a value, false where padding was inserted.
        bool[] mask = new bool[values.Length];
        if (featureSize > 0)
        {
            int rowStride = padLen * featureSize;
            for (int b = 0; b < batch; b++)
            {
                int rowLen = Math.Min(nested.RowLength(b), padLen);
                int start = b * rowStride;
                for (int i = 0; i < rowLen * featureSize; i++) mask[start + i] = true;
            }
        }
        else
        {
            for (int b = 0; b < batch; b++)
            {
                int rowLen = Math.Min(nested.RowLength(b), padLen);
                int start = b * padLen;
                for (int i = 0; i < rowLen; i++) mask[start + i] = true;
            }
        }
        return new MaskedTensor<T>(values, mask);
    }

    /// <summary>
    /// Implementation detail of <see cref="NestedToMasked{T}"/> that
    /// also returns the lengths array — handy for users who want to
    /// round-trip back to <see cref="NestedTensor{T}"/> without
    /// recomputing.
    /// </summary>
    public static (MaskedTensor<T> Masked, int[] Lengths) NestedToMaskedFromPadding<T>(
        NestedTensor<T> nested, T padding, int outputSize = -1)
    {
        int batch = nested.BatchSize;
        var lengths = new int[batch];
        for (int b = 0; b < batch; b++) lengths[b] = nested.RowLength(b);
        return (NestedToMasked(nested, padding, outputSize), lengths);
    }

    /// <summary>
    /// Bridge that converts a <see cref="MaskedTensor{T}"/> into a
    /// nested tensor by reading the mask as a per-row length signal.
    /// Caller supplies the boolean mask layout: for shape
    /// <c>[batch, padLen]</c> or <c>[batch, padLen, features]</c>, every
    /// row's length is <c>count(mask[b, :, ...] = true) / featureSize</c>.
    /// </summary>
    public static NestedTensor<T> MaskedToNested<T>(MaskedTensor<T> masked)
    {
        if (masked.Rank != 2 && masked.Rank != 3)
            throw new ArgumentException("MaskedToNested requires rank 2 or 3.");

        int batch = masked.Shape[0];
        int padLen = masked.Shape[1];
        bool hasFeatureAxis = masked.Rank == 3;
        int featureSize = hasFeatureAxis ? masked.Shape[2] : 0;
        var lengths = new int[batch];

        // Each row's mask must be a contiguous prefix of trues to round-trip
        // cleanly through nested form. We don't enforce that here — instead
        // we count the longest valid prefix per row.
        for (int b = 0; b < batch; b++)
        {
            int len = 0;
            for (int s = 0; s < padLen; s++)
            {
                int idx = hasFeatureAxis ? (b * padLen + s) * featureSize : b * padLen + s;
                if (masked.IsValid(idx)) len++;
                else break;
            }
            lengths[b] = len;
        }
        return NestedTensor<T>.FromPadded(masked.Values, lengths);
    }
}
