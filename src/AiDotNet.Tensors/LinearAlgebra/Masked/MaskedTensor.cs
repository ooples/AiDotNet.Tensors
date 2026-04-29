// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.LinearAlgebra.Masked;

/// <summary>
/// Tensor with explicit per-element validity. Mirrors PyTorch's
/// <c>torch.masked.masked_tensor</c>: a dense values buffer plus a
/// boolean mask where <c>true</c> = valid and <c>false</c> = masked-out.
/// Reductions, elementwise ops, and autograd ignore masked-out
/// positions, so the long-standing <c>nan</c>/<c>inf</c>-pollutes-mean
/// problem PyTorch has documented disappears.
///
/// <para><b>How we beat PyTorch (#222 point #4):</b> our mask is bit-
/// packed — <see cref="MaskedTensor{T}._packedMask"/> stores 8 lanes per
/// byte, so the per-element memory cost is 1 bit (vs PyTorch's 1 byte
/// per element when stored as <c>torch.bool</c>). For a 1B-element
/// tensor that's 1 GB → 128 MB.</para>
///
/// <para><b>How we beat PyTorch (#222 point #5):</b> the mask survives
/// op dispatch — <see cref="MaskedOps"/> reductions consume the bit-
/// packed mask directly without unpacking, so a masked sum is the same
/// number of FMAs as a dense sum minus the masked-out terms (no extra
/// pass).</para>
/// </summary>
public sealed class MaskedTensor<T>
{
    /// <summary>Underlying dense values. Masked-out lanes still have
    /// storage (caller may have left arbitrary content there); ops
    /// must consult the mask to ignore them.</summary>
    public Tensor<T> Values { get; }

    /// <summary>Bit-packed mask. Bit <c>i mod 8</c> in
    /// <c>_packedMask[i / 8]</c> is <c>1</c> when lane <c>i</c> is
    /// valid, <c>0</c> when masked out. Length:
    /// <c>(Values.Length + 7) / 8</c>.</summary>
    private readonly byte[] _packedMask;

    /// <summary>Read-only access to the packed mask bytes — internal
    /// to <see cref="MaskedOps"/>; ops consume the packed form
    /// directly to avoid unpacking on the hot path.</summary>
    internal ReadOnlySpan<byte> PackedMask => _packedMask;

    /// <summary>Element count (= Values.Length).</summary>
    public int Length => Values.Length;

    /// <summary>Shape of the underlying values tensor.</summary>
    public int[] Shape => (int[])Values._shape.Clone();

    /// <summary>Rank of the underlying values tensor.</summary>
    public int Rank => Values.Rank;

    /// <summary>Number of valid (unmasked) elements. Computed eagerly
    /// at construction so reductions don't pay popcount per call.</summary>
    public int ValidCount { get; }

    /// <summary>Wraps an existing values tensor with a parallel boolean
    /// mask. The mask is bit-packed on the way in.</summary>
    public MaskedTensor(Tensor<T> values, bool[] mask)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        if (mask is null) throw new ArgumentNullException(nameof(mask));
        if (mask.Length != values.Length)
            throw new ArgumentException(
                $"Mask length {mask.Length} doesn't match values length {values.Length}.", nameof(mask));

        Values = values;
        _packedMask = new byte[(values.Length + 7) >> 3];
        int valid = 0;
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i])
            {
                _packedMask[i >> 3] |= (byte)(1 << (i & 7));
                valid++;
            }
        }
        ValidCount = valid;
    }

    /// <summary>Constructs from a values tensor and an already-packed
    /// mask. Used by ops that produce a masked output without
    /// unpacking — keeps the hot path allocation-free.</summary>
    internal MaskedTensor(Tensor<T> values, byte[] packedMask, int validCount)
    {
        Values = values;
        _packedMask = packedMask;
        ValidCount = validCount;
    }

    /// <summary>Returns whether lane <paramref name="i"/> is valid.</summary>
    public bool IsValid(int i) => (_packedMask[i >> 3] & (1 << (i & 7))) != 0;

    /// <summary>Materializes a dense tensor where masked-out lanes are
    /// replaced with <paramref name="fill"/>. Useful for handing a
    /// masked tensor to a non-mask-aware op.</summary>
    public Tensor<T> ToDense(T fill)
    {
        var result = new Tensor<T>(Shape);
        var src = Values.AsSpan();
        var dst = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
            dst[i] = IsValid(i) ? src[i] : fill;
        return result;
    }

    /// <summary>Returns the mask as a managed bool array. Allocates;
    /// hot-path callers should consume the packed form via
    /// <see cref="PackedMask"/>.</summary>
    public bool[] GetMask()
    {
        var mask = new bool[Length];
        for (int i = 0; i < Length; i++) mask[i] = IsValid(i);
        return mask;
    }

    /// <summary>Constructs a masked tensor from a dense input where
    /// every lane equal to <paramref name="maskValue"/> is treated as
    /// masked-out. Common pattern: <c>FromDenseWithSentinel(t, T.NaN)</c>
    /// for "ignore the NaNs in this float tensor".
    ///
    /// <para>NaN handling: IEEE 754 says <c>NaN.Equals(NaN) == false</c>,
    /// so a plain <c>Equals</c> never matches a NaN sentinel. We
    /// special-case <c>float.NaN</c> / <c>double.NaN</c> using
    /// <c>IsNaN</c>; other sentinel values (including ±Inf, 0, any
    /// normal float) compare via <c>Equals</c> as before.</para></summary>
    public static MaskedTensor<T> FromDenseWithSentinel(Tensor<T> values, T maskValue)
    {
        var mask = new bool[values.Length];
        var src = values.AsSpan();

        bool isFloatNan = maskValue is float fSentinel && float.IsNaN(fSentinel);
        bool isDoubleNan = maskValue is double dSentinel && double.IsNaN(dSentinel);

        for (int i = 0; i < src.Length; i++)
        {
            if (isFloatNan && src[i] is float fv)
                mask[i] = !float.IsNaN(fv);
            else if (isDoubleNan && src[i] is double dv)
                mask[i] = !double.IsNaN(dv);
            else
                mask[i] = !src[i]!.Equals(maskValue);
        }
        return new MaskedTensor<T>(values, mask);
    }
}
