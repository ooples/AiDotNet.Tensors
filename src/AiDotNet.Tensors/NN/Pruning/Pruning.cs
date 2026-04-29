// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.NN.Pruning;

/// <summary>
/// Pruning APIs — <c>torch.nn.utils.prune</c> equivalent. Each entry
/// returns a <see cref="PrunedTensor{T}"/> bundling the original raw
/// values + a binary mask. The masked product
/// <c>raw ⊙ mask</c> is what downstream ops should consume; calling
/// <see cref="Pruning.Remove{T}"/> consolidates the mask into the raw
/// values.
/// </summary>
public static class Pruning
{
    /// <summary>L1-magnitude unstructured pruning — zeros out the
    /// <paramref name="amount"/> fraction of weights with the smallest
    /// absolute value. Mirrors <c>prune.l1_unstructured</c>.</summary>
    public static PrunedTensor<T> L1Unstructured<T>(Tensor<T> raw, double amount)
    {
        if (raw is null) throw new ArgumentNullException(nameof(raw));
        if (amount < 0 || amount > 1) throw new ArgumentOutOfRangeException(nameof(amount));
        var ops = MathHelper.GetNumericOperations<T>();
        int total = raw.Length;
        int toPrune = (int)Math.Round(amount * total);
        var mask = new bool[total];
        if (toPrune == 0)
        {
            for (int i = 0; i < total; i++) mask[i] = true;
            return new PrunedTensor<T>(raw, mask);
        }
        if (toPrune >= total)
            return new PrunedTensor<T>(raw, mask);

        // Threshold-only pruning under-counts when values tie at the
        // cutoff. Switch to the (score, index) sort: take the first
        // `toPrune` indices in ascending magnitude as the prune set.
        // Exact-count regardless of ties; matches PyTorch's
        // prune.l1_unstructured semantics.
        var src = raw.AsSpan();
        var indexed = new (double Mag, int Idx)[total];
        for (int i = 0; i < total; i++) indexed[i] = (Math.Abs(ops.ToDouble(src[i])), i);
        Array.Sort(indexed, (a, b) => a.Mag.CompareTo(b.Mag));
        for (int i = 0; i < total; i++) mask[i] = true;
        for (int k = 0; k < toPrune; k++) mask[indexed[k].Idx] = false;
        return new PrunedTensor<T>(raw, mask);
    }

    /// <summary>Random unstructured pruning. <paramref name="seed"/>
    /// chooses the random subset.</summary>
    public static PrunedTensor<T> RandomUnstructured<T>(Tensor<T> raw, double amount, int seed = 0)
    {
        if (amount < 0 || amount > 1) throw new ArgumentOutOfRangeException(nameof(amount));
        int total = raw.Length;
        int toPrune = (int)Math.Round(amount * total);
        var mask = new bool[total];
        for (int i = 0; i < total; i++) mask[i] = true;
        if (toPrune == 0) return new PrunedTensor<T>(raw, mask);

        var rng = new Random(seed);
        var indices = new int[total];
        for (int i = 0; i < total; i++) indices[i] = i;
        // Fisher-Yates shuffle, then pick the first `toPrune` to mask out.
        for (int i = total - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
        for (int k = 0; k < toPrune; k++) mask[indices[k]] = false;
        return new PrunedTensor<T>(raw, mask);
    }

    /// <summary>Ln-norm structured pruning along the dimension
    /// <paramref name="dim"/>. Slices with the smallest L<paramref name="n"/>
    /// norm get the entire slice masked. Mirrors <c>prune.ln_structured</c>.</summary>
    public static PrunedTensor<T> LnStructured<T>(Tensor<T> raw, double amount, int n, int dim)
    {
        if (amount < 0 || amount > 1) throw new ArgumentOutOfRangeException(nameof(amount));
        if (dim < 0 || dim >= raw.Rank) throw new ArgumentOutOfRangeException(nameof(dim));
        var ops = MathHelper.GetNumericOperations<T>();

        int outer = 1, axisLen = raw._shape[dim], inner = 1;
        for (int i = 0; i < dim; i++) outer *= raw._shape[i];
        for (int i = dim + 1; i < raw.Rank; i++) inner *= raw._shape[i];

        var sliceNorms = new double[axisLen];
        var src = raw.AsSpan();
        for (int a = 0; a < axisLen; a++)
        {
            double normN = 0;
            for (int o = 0; o < outer; o++)
            {
                for (int i = 0; i < inner; i++)
                {
                    double v = ops.ToDouble(src[(o * axisLen + a) * inner + i]);
                    normN += Math.Pow(Math.Abs(v), n);
                }
            }
            sliceNorms[a] = Math.Pow(normN, 1.0 / n);
        }

        int toPrune = (int)Math.Round(amount * axisLen);
        // (norm, slice-index) sort + take-k for exact-count pruning
        // even when norms tie at the cutoff.
        var indexedNorms = new (double Norm, int Idx)[axisLen];
        for (int a = 0; a < axisLen; a++) indexedNorms[a] = (sliceNorms[a], a);
        Array.Sort(indexedNorms, (a, b) => a.Norm.CompareTo(b.Norm));
        var pruneSlice = new bool[axisLen];
        for (int k = 0; k < toPrune; k++) pruneSlice[indexedNorms[k].Idx] = true;

        var mask = new bool[raw.Length];
        for (int a = 0; a < axisLen; a++)
        {
            bool keep = !pruneSlice[a];
            for (int o = 0; o < outer; o++)
                for (int i = 0; i < inner; i++)
                    mask[(o * axisLen + a) * inner + i] = keep;
        }
        return new PrunedTensor<T>(raw, mask);
    }

    /// <summary>Random structured pruning along
    /// <paramref name="dim"/>. Same shape as
    /// <see cref="LnStructured{T}"/> but the slice selection is random.</summary>
    public static PrunedTensor<T> RandomStructured<T>(Tensor<T> raw, double amount, int dim, int seed = 0)
    {
        if (amount < 0 || amount > 1) throw new ArgumentOutOfRangeException(nameof(amount));
        if (dim < 0 || dim >= raw.Rank) throw new ArgumentOutOfRangeException(nameof(dim));

        int outer = 1, axisLen = raw._shape[dim], inner = 1;
        for (int i = 0; i < dim; i++) outer *= raw._shape[i];
        for (int i = dim + 1; i < raw.Rank; i++) inner *= raw._shape[i];

        int toPrune = (int)Math.Round(amount * axisLen);
        var rng = new Random(seed);
        var sliceIndices = new int[axisLen];
        for (int i = 0; i < axisLen; i++) sliceIndices[i] = i;
        for (int i = axisLen - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (sliceIndices[i], sliceIndices[j]) = (sliceIndices[j], sliceIndices[i]);
        }
        var prune = new bool[axisLen];
        for (int k = 0; k < toPrune; k++) prune[sliceIndices[k]] = true;

        var mask = new bool[raw.Length];
        for (int a = 0; a < axisLen; a++)
        {
            bool keep = !prune[a];
            for (int o = 0; o < outer; o++)
                for (int i = 0; i < inner; i++)
                    mask[(o * axisLen + a) * inner + i] = keep;
        }
        return new PrunedTensor<T>(raw, mask);
    }

    /// <summary>Global unstructured L1 pruning across multiple
    /// tensors — picks the <paramref name="amount"/> fraction of the
    /// smallest-magnitude weights from the combined pool. Mirrors
    /// <c>prune.global_unstructured</c>.</summary>
    public static PrunedTensor<T>[] GlobalUnstructured<T>(Tensor<T>[] raws, double amount)
    {
        if (raws is null) throw new ArgumentNullException(nameof(raws));
        if (amount < 0 || amount > 1) throw new ArgumentOutOfRangeException(nameof(amount));
        var ops = MathHelper.GetNumericOperations<T>();
        int total = 0;
        for (int t = 0; t < raws.Length; t++) total += raws[t].Length;
        int toPrune = (int)Math.Round(amount * total);

        var pool = new double[total];
        int cursor = 0;
        for (int t = 0; t < raws.Length; t++)
        {
            var span = raws[t].AsSpan();
            for (int i = 0; i < span.Length; i++) pool[cursor++] = Math.Abs(ops.ToDouble(span[i]));
        }
        // Exact-count global pruning via (score, global-index) sort
        // + take-k. Ties at the cutoff don't bias the prune count.
        var indexed = new (double Mag, int GlobalIdx)[total];
        for (int i = 0; i < total; i++) indexed[i] = (pool[i], i);
        Array.Sort(indexed, (a, b) => a.Mag.CompareTo(b.Mag));
        var pruneSet = new bool[total];
        for (int k = 0; k < toPrune; k++) pruneSet[indexed[k].GlobalIdx] = true;

        var result = new PrunedTensor<T>[raws.Length];
        cursor = 0;
        for (int t = 0; t < raws.Length; t++)
        {
            int len = raws[t].Length;
            var mask = new bool[len];
            for (int i = 0; i < len; i++) mask[i] = !pruneSet[cursor + i];
            result[t] = new PrunedTensor<T>(raws[t], mask);
            cursor += len;
        }
        return result;
    }

    /// <summary>Apply a custom user-supplied mask. Mirrors
    /// <c>prune.custom_from_mask</c>.</summary>
    public static PrunedTensor<T> CustomFromMask<T>(Tensor<T> raw, bool[] mask)
    {
        if (mask is null) throw new ArgumentNullException(nameof(mask));
        if (mask.Length != raw.Length)
            throw new ArgumentException(
                $"Mask length {mask.Length} doesn't match tensor length {raw.Length}.", nameof(mask));
        return new PrunedTensor<T>(raw, (bool[])mask.Clone());
    }

    /// <summary>Consolidates the mask into the raw values — the
    /// resulting tensor has zeros wherever the mask was false. Mirrors
    /// <c>prune.remove</c>.</summary>
    public static Tensor<T> Remove<T>(PrunedTensor<T> pruned)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])pruned.Raw._shape.Clone());
        var src = pruned.Raw.AsSpan();
        var dst = output.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
            dst[i] = pruned.IsKept(i) ? src[i] : ops.Zero;
        return output;
    }
}

/// <summary>Bundle of a raw parameter + a per-element keep/prune mask.
/// <see cref="EffectiveValue"/> returns <c>raw ⊙ mask</c>; the mask
/// can be inspected via <see cref="IsKept"/>.</summary>
public sealed class PrunedTensor<T>
{
    /// <summary>The underlying raw values — unaffected by pruning until
    /// <see cref="Pruning.Remove{T}"/> consolidates.</summary>
    public Tensor<T> Raw { get; }

    private readonly bool[] _mask;

    /// <summary>Number of un-pruned (kept) lanes.</summary>
    public int KeptCount { get; }

    /// <summary>Constructs a pruned tensor with an explicit mask.
    /// <paramref name="mask"/> must have the same length as
    /// <paramref name="raw"/>; mismatched masks would surface as
    /// IndexOutOfRangeException in <see cref="IsKept"/> /
    /// <see cref="Pruning.Remove{T}"/>.</summary>
    public PrunedTensor(Tensor<T> raw, bool[] mask)
    {
        if (raw is null) throw new ArgumentNullException(nameof(raw));
        if (mask is null) throw new ArgumentNullException(nameof(mask));
        if (mask.Length != raw.Length)
            throw new ArgumentException(
                $"Mask length {mask.Length} doesn't match tensor length {raw.Length}.", nameof(mask));
        Raw = raw;
        _mask = mask;
        int kept = 0;
        for (int i = 0; i < mask.Length; i++) if (mask[i]) kept++;
        KeptCount = kept;
    }

    /// <summary>True when lane <paramref name="i"/> survived pruning.</summary>
    public bool IsKept(int i) => _mask[i];

    /// <summary>The raw tensor with masked-out lanes zeroed —
    /// equivalent to calling <see cref="Pruning.Remove{T}"/>.</summary>
    public Tensor<T> EffectiveValue() => Pruning.Remove(this);

    /// <summary>Read-only view of the mask. Useful for serializing
    /// or for a custom forward path.</summary>
    public IReadOnlyList<bool> Mask => _mask;
}
