// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Data;

/// <summary>
/// Function that turns a list of samples into a single batch. The
/// <see cref="DataLoader{TSample, TBatch}"/> calls this with one batch worth
/// of samples and yields the result.
/// </summary>
public delegate TBatch CollateFn<TSample, TBatch>(IReadOnlyList<TSample> samples);

/// <summary>
/// Default collation strategies — handle the common cases users would
/// otherwise write by hand.
/// </summary>
public static class Collators
{
    /// <summary>
    /// Stacks a list of <see cref="Tensor{T}"/> samples along a new leading
    /// axis. Equivalent to <c>torch.stack</c> over the sample list. All
    /// samples must share their shape; mismatches throw early.
    /// </summary>
    public static Tensor<T> Stack<T>(IReadOnlyList<Tensor<T>> samples)
    {
        if (samples is null) throw new ArgumentNullException(nameof(samples));
        if (samples.Count == 0) throw new ArgumentException("Cannot stack zero samples.");

        var first = samples[0];
        var sampleShape = first._shape;

        // Validate matching shapes — variadic-shape stacking would silently
        // produce malformed batches.
        for (int s = 1; s < samples.Count; s++)
        {
            var t = samples[s];
            if (t._shape.Length != sampleShape.Length)
                throw new ArgumentException(
                    $"All samples must share rank. Sample 0 rank={sampleShape.Length}, sample {s} rank={t._shape.Length}.");
            for (int d = 0; d < sampleShape.Length; d++)
                if (t._shape[d] != sampleShape[d])
                    throw new ArgumentException(
                        $"Shape mismatch at sample {s} dim {d}: got {t._shape[d]}, expected {sampleShape[d]}.");
        }

        var batchShape = new int[sampleShape.Length + 1];
        batchShape[0] = samples.Count;
        for (int d = 0; d < sampleShape.Length; d++) batchShape[d + 1] = sampleShape[d];

        var batch = new Tensor<T>(batchShape);
        var dst = batch.AsWritableSpan();
        int sampleLen = first.Length;
        for (int s = 0; s < samples.Count; s++)
        {
            samples[s].AsSpan().CopyTo(dst.Slice(s * sampleLen, sampleLen));
        }
        return batch;
    }

    /// <summary>
    /// Pads variable-length 1D sequences to the longest length, then stacks
    /// along a new leading axis. The pad cell is filled with
    /// <paramref name="padValue"/>. Useful for batching tokenized text where
    /// each sample's sequence length differs.
    /// </summary>
    public static Tensor<T> PadAndStack1D<T>(IReadOnlyList<Tensor<T>> samples, T padValue)
    {
        if (samples is null) throw new ArgumentNullException(nameof(samples));
        if (samples.Count == 0) throw new ArgumentException("Cannot stack zero samples.");

        int maxLen = 0;
        for (int s = 0; s < samples.Count; s++)
        {
            if (samples[s]._shape.Length != 1)
                throw new ArgumentException(
                    $"PadAndStack1D requires rank-1 samples. Sample {s} has rank {samples[s]._shape.Length}.");
            if (samples[s]._shape[0] > maxLen) maxLen = samples[s]._shape[0];
        }

        var batch = new Tensor<T>(new[] { samples.Count, maxLen });
        var dst = batch.AsWritableSpan();
        for (int i = 0; i < dst.Length; i++) dst[i] = padValue;

        for (int s = 0; s < samples.Count; s++)
        {
            int len = samples[s]._shape[0];
            samples[s].AsSpan().CopyTo(dst.Slice(s * maxLen, len));
        }
        return batch;
    }

    /// <summary>
    /// Default collator for <see cref="Tensor{T}"/>[] samples (the shape
    /// emitted by <see cref="TensorDataset{T}"/>). Stacks each component
    /// independently and returns the resulting array of stacked batches.
    /// </summary>
    public static Tensor<T>[] StackTuple<T>(IReadOnlyList<Tensor<T>[]> samples)
    {
        if (samples is null) throw new ArgumentNullException(nameof(samples));
        if (samples.Count == 0) throw new ArgumentException("Cannot stack zero samples.");
        int components = samples[0].Length;
        var result = new Tensor<T>[components];
        var componentLists = new List<Tensor<T>>[components];
        for (int c = 0; c < components; c++) componentLists[c] = new List<Tensor<T>>(samples.Count);
        for (int s = 0; s < samples.Count; s++)
        {
            var sample = samples[s];
            if (sample.Length != components)
                throw new ArgumentException(
                    $"All samples must have {components} components; sample {s} has {sample.Length}.");
            for (int c = 0; c < components; c++) componentLists[c].Add(sample[c]);
        }
        for (int c = 0; c < components; c++) result[c] = Stack(componentLists[c]);
        return result;
    }
}
