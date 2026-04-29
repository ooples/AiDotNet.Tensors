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

        var src = raw.AsSpan();
        // Compute |raw_i|, find the k-th smallest as the threshold.
        var abs = new double[total];
        for (int i = 0; i < total; i++) abs[i] = Math.Abs(ops.ToDouble(src[i]));
        var sorted = (double[])abs.Clone();
        Array.Sort(sorted);
        double threshold = sorted[toPrune];
        int actual = 0;
        for (int i = 0; i < total; i++)
        {
            mask[i] = abs[i] >= threshold;
            if (!mask[i]) actual++;
            if (actual >= toPrune && abs[i] == threshold) mask[i] = true;
        }
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
        var sortedNorms = (double[])sliceNorms.Clone();
        Array.Sort(sortedNorms);
        double threshold = toPrune > 0 ? sortedNorms[Math.Min(toPrune, axisLen - 1)] : double.MinValue;

        var mask = new bool[raw.Length];
        for (int a = 0; a < axisLen; a++)
        {
            bool keep = sliceNorms[a] >= threshold;
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
        var sortedPool = (double[])pool.Clone();
        Array.Sort(sortedPool);
        double threshold = toPrune > 0 ? sortedPool[Math.Min(toPrune, total - 1)] : double.MinValue;

        var result = new PrunedTensor<T>[raws.Length];
        cursor = 0;
        for (int t = 0; t < raws.Length; t++)
        {
            var mask = new bool[raws[t].Length];
            var span = raws[t].AsSpan();
            for (int i = 0; i < span.Length; i++)
                mask[i] = Math.Abs(ops.ToDouble(span[i])) >= threshold;
            result[t] = new PrunedTensor<T>(raws[t], mask);
            cursor += raws[t].Length;
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

    /// <summary>Constructs a pruned tensor with an explicit mask.</summary>
    public PrunedTensor(Tensor<T> raw, bool[] mask)
    {
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
