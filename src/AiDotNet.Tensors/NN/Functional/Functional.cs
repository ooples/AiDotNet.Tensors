// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.NN.Functional;

/// <summary>
/// <c>torch.nn.functional</c> utility surface added by #223. Engine
/// already exposes <c>TensorPDist</c>, <c>TensorCDist</c>, and
/// <c>TensorCosineSimilarity</c>; this file fills in the remaining
/// utility ops PyTorch ships under <c>F.</c>.
/// </summary>
public static class Functional
{
    /// <summary>
    /// L<paramref name="p"/>-normalize <paramref name="input"/> along
    /// <paramref name="dim"/>. Each slice is divided by its L<paramref name="p"/>
    /// norm (clamped at <paramref name="eps"/> to avoid division by zero).
    /// Mirrors <c>F.normalize</c>.
    /// </summary>
    public static Tensor<T> Normalize<T>(Tensor<T> input, double p = 2.0, int dim = 1, double eps = 1e-12)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        int actualDim = dim < 0 ? input.Rank + dim : dim;
        if (actualDim < 0 || actualDim >= input.Rank)
            throw new ArgumentOutOfRangeException(nameof(dim));
        var ops = MathHelper.GetNumericOperations<T>();

        int outer = 1, axisLen = input._shape[actualDim], inner = 1;
        for (int i = 0; i < actualDim; i++) outer *= input._shape[i];
        for (int i = actualDim + 1; i < input.Rank; i++) inner *= input._shape[i];

        var output = new Tensor<T>((int[])input._shape.Clone());
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();

        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                double normP = 0;
                for (int a = 0; a < axisLen; a++)
                {
                    double v = ops.ToDouble(src[(o * axisLen + a) * inner + i]);
                    normP += Math.Pow(Math.Abs(v), p);
                }
                double norm = Math.Max(eps, Math.Pow(normP, 1.0 / p));
                for (int a = 0; a < axisLen; a++)
                {
                    int idx = (o * axisLen + a) * inner + i;
                    dst[idx] = ops.FromDouble(ops.ToDouble(src[idx]) / norm);
                }
            }
        }
        return output;
    }

    /// <summary>
    /// One-hot encode integer indices. Output shape adds a final
    /// <paramref name="numClasses"/> axis. Mirrors <c>F.one_hot</c>.
    /// </summary>
    public static Tensor<int> OneHot(Tensor<int> input, int numClasses)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (numClasses <= 0) throw new ArgumentOutOfRangeException(nameof(numClasses));
        var newShape = new int[input.Rank + 1];
        Array.Copy(input._shape, newShape, input.Rank);
        newShape[input.Rank] = numClasses;
        var output = new Tensor<int>(newShape);
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
        {
            int cls = src[i];
            if ((uint)cls >= (uint)numClasses)
                throw new ArgumentException($"Index {cls} out of range [0, {numClasses}).", nameof(input));
            dst[i * numClasses + cls] = 1;
        }
        return output;
    }

    /// <summary>
    /// Pairwise L<paramref name="p"/> distance between rows of
    /// <paramref name="x1"/> and <paramref name="x2"/>. Both must be
    /// rank-2 with the same shape; output is rank-1 of length
    /// <c>batch</c>. Mirrors <c>F.pairwise_distance</c>.
    /// </summary>
    public static Tensor<T> PairwiseDistance<T>(Tensor<T> x1, Tensor<T> x2, double p = 2.0, double eps = 1e-6)
    {
        if (x1.Rank != 2 || x2.Rank != 2)
            throw new ArgumentException("PairwiseDistance expects rank-2 inputs.", nameof(x1));
        if (x1._shape[0] != x2._shape[0] || x1._shape[1] != x2._shape[1])
            throw new ArgumentException("x1 and x2 must have the same shape.");
        var ops = MathHelper.GetNumericOperations<T>();
        int batch = x1._shape[0], features = x1._shape[1];
        var output = new Tensor<T>(new[] { batch });
        var s1 = x1.AsSpan();
        var s2 = x2.AsSpan();
        var dst = output.AsWritableSpan();
        for (int b = 0; b < batch; b++)
        {
            double accum = 0;
            for (int f = 0; f < features; f++)
            {
                double diff = ops.ToDouble(s1[b * features + f]) - ops.ToDouble(s2[b * features + f]) + eps;
                accum += Math.Pow(Math.Abs(diff), p);
            }
            dst[b] = ops.FromDouble(Math.Pow(accum, 1.0 / p));
        }
        return output;
    }

    /// <summary>
    /// Embedding lookup. Returns <c>weight[input[i]]</c> for every index
    /// in <paramref name="input"/>. Output shape is
    /// <c>input.shape + [embedding_dim]</c>. Optional max-norm rescaling
    /// + scale-grad-by-freq tracking mirror PyTorch's
    /// <c>F.embedding</c>.
    /// </summary>
    public static Tensor<T> Embedding<T>(Tensor<int> input, Tensor<T> weight,
        int? paddingIdx = null, double? maxNorm = null, double normType = 2.0,
        bool scaleGradByFreq = false)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        if (weight.Rank != 2)
            throw new ArgumentException("weight must be rank-2 [num_embeddings, embedding_dim].", nameof(weight));
        var ops = MathHelper.GetNumericOperations<T>();
        int numEmb = weight._shape[0];
        int embDim = weight._shape[1];

        // Apply max-norm clipping in-place on the weight rows that are
        // referenced. Matches PyTorch behavior — the clipped rows
        // become the "official" representation that backward sees.
        if (maxNorm is not null)
        {
            var wSpan = weight.AsWritableSpan();
            var visited = new bool[numEmb];
            var inputSpan = input.AsSpan();
            for (int i = 0; i < inputSpan.Length; i++)
            {
                int row = inputSpan[i];
                if ((uint)row >= (uint)numEmb || visited[row]) continue;
                visited[row] = true;
                double normP = 0;
                for (int e = 0; e < embDim; e++)
                {
                    double v = ops.ToDouble(wSpan[row * embDim + e]);
                    normP += Math.Pow(Math.Abs(v), normType);
                }
                double norm = Math.Pow(normP, 1.0 / normType);
                if (norm > maxNorm.Value)
                {
                    double scale = maxNorm.Value / norm;
                    for (int e = 0; e < embDim; e++)
                    {
                        var wv = wSpan[row * embDim + e];
                        wSpan[row * embDim + e] = ops.FromDouble(ops.ToDouble(wv) * scale);
                    }
                }
            }
        }

        var newShape = new int[input.Rank + 1];
        Array.Copy(input._shape, newShape, input.Rank);
        newShape[input.Rank] = embDim;
        var output = new Tensor<T>(newShape);
        var inSpan = input.AsSpan();
        var outSpan = output.AsWritableSpan();
        var weightSpan = weight.AsSpan();
        for (int i = 0; i < inSpan.Length; i++)
        {
            int row = inSpan[i];
            if (paddingIdx.HasValue && row == paddingIdx.Value)
            {
                // Zero-fill the embedding for the padding index.
                for (int e = 0; e < embDim; e++) outSpan[i * embDim + e] = ops.Zero;
                continue;
            }
            if ((uint)row >= (uint)numEmb)
                throw new ArgumentException($"Index {row} out of range [0, {numEmb}).", nameof(input));
            for (int e = 0; e < embDim; e++)
                outSpan[i * embDim + e] = weightSpan[row * embDim + e];
        }
        _ = scaleGradByFreq; // takes effect during backward; forward path is unchanged.
        return output;
    }

    /// <summary>Bag mode for <see cref="EmbeddingBag{T}"/>.</summary>
    public enum EmbeddingBagMode
    {
        /// <summary>Sum the embeddings in each bag.</summary>
        Sum,
        /// <summary>Average the embeddings in each bag.</summary>
        Mean,
        /// <summary>Element-wise max across the bag.</summary>
        Max,
    }

    /// <summary>
    /// Embedding bag — fetches embeddings, then reduces (sum / mean /
    /// max) within each bag delimited by <paramref name="offsets"/>.
    /// Output shape is <c>[bags, embedding_dim]</c>. Mirrors
    /// <c>F.embedding_bag</c>; sub-byte storage support is a follow-up
    /// gated on the existing <c>PackedInt1/2/3/4</c> codecs.
    /// </summary>
    public static Tensor<T> EmbeddingBag<T>(Tensor<int> input, Tensor<T> weight, int[] offsets,
        EmbeddingBagMode mode = EmbeddingBagMode.Mean)
    {
        if (offsets is null) throw new ArgumentNullException(nameof(offsets));
        if (weight.Rank != 2)
            throw new ArgumentException("weight must be rank-2 [num_embeddings, embedding_dim].", nameof(weight));
        if (input.Rank != 1)
            throw new ArgumentException("input must be rank-1 (flattened indices).", nameof(input));

        var ops = MathHelper.GetNumericOperations<T>();
        int bagCount = offsets.Length;
        int embDim = weight._shape[1];
        var output = new Tensor<T>(new[] { bagCount, embDim });
        var inSpan = input.AsSpan();
        var weightSpan = weight.AsSpan();
        var dst = output.AsWritableSpan();

        for (int b = 0; b < bagCount; b++)
        {
            int start = offsets[b];
            int end = b + 1 < bagCount ? offsets[b + 1] : inSpan.Length;
            int count = end - start;

            for (int e = 0; e < embDim; e++) dst[b * embDim + e] = ops.Zero;
            if (count == 0) continue;

            if (mode == EmbeddingBagMode.Max)
            {
                for (int e = 0; e < embDim; e++)
                {
                    double best = double.NegativeInfinity;
                    for (int k = start; k < end; k++)
                    {
                        int row = inSpan[k];
                        double v = ops.ToDouble(weightSpan[row * embDim + e]);
                        if (v > best) best = v;
                    }
                    dst[b * embDim + e] = ops.FromDouble(best);
                }
            }
            else
            {
                for (int k = start; k < end; k++)
                {
                    int row = inSpan[k];
                    for (int e = 0; e < embDim; e++)
                        dst[b * embDim + e] = ops.Add(dst[b * embDim + e], weightSpan[row * embDim + e]);
                }
                if (mode == EmbeddingBagMode.Mean)
                {
                    T inv = ops.Divide(ops.One, ops.FromDouble(count));
                    for (int e = 0; e < embDim; e++)
                        dst[b * embDim + e] = ops.Multiply(dst[b * embDim + e], inv);
                }
            }
        }
        return output;
    }
}
