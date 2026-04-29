// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra.Nested;

/// <summary>
/// Op surface for <see cref="NestedTensor{T}"/> — each entry is a per-row
/// apply over the existing dense kernels. Same numerical behaviour as
/// padding + masking afterward, but without the padding FLOPs.
///
/// <para>Mirrors <c>torch.nested</c>'s op coverage:
/// <c>linear / matmul / add / softmax / log_softmax / layer_norm /
/// gelu / silu / scaled_dot_product_attention / multi_head_attention</c>.</para>
/// </summary>
public static class NestedOps
{
    private static readonly IEngine Engine = new CpuEngine();

    /// <summary>Adds <paramref name="bias"/> (shape <c>[features]</c>) to
    /// every row in <paramref name="x"/>. Returns a new nested tensor
    /// sharing the same offsets.</summary>
    public static NestedTensor<T> Add<T>(NestedTensor<T> x, Tensor<T> bias)
    {
        if (x.FeatureSize == 0)
            throw new InvalidOperationException("Add(bias) requires a feature axis on the nested input.");
        if (bias.Length != x.FeatureSize)
            throw new ArgumentException($"Bias length {bias.Length} doesn't match feature size {x.FeatureSize}.");

        var ops = MathHelper.GetNumericOperations<T>();
        var values = new Tensor<T>(new[] { x.Values.Length });
        var src = x.Values.AsSpan();
        var biasSpan = bias.AsSpan();
        var dst = values.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
            dst[i] = ops.Add(src[i], biasSpan[i % x.FeatureSize]);

        return NestedTensor<T>.FromValuesOffsets(values, (int[])x.Offsets.Clone(), x.FeatureSize, x.Layout);
    }

    /// <summary>
    /// Linear transform: <c>y = x · W^T + b</c>. <paramref name="weight"/>
    /// is <c>[outFeatures, inFeatures]</c> (PyTorch <c>nn.Linear</c> layout);
    /// <paramref name="bias"/> is optional and length <c>outFeatures</c>.
    /// Output rows have the same lengths as input rows but trailing
    /// dimension <c>outFeatures</c>.
    /// </summary>
    public static NestedTensor<T> Linear<T>(NestedTensor<T> x, Tensor<T> weight, Tensor<T>? bias = null)
    {
        if (x.FeatureSize == 0)
            throw new InvalidOperationException("Linear requires a feature axis on the nested input.");
        if (weight.Rank != 2 || weight._shape[1] != x.FeatureSize)
            throw new ArgumentException(
                $"Weight must have shape [outFeatures, {x.FeatureSize}]; got {ShapeStr(weight._shape)}.",
                nameof(weight));

        int outFeatures = weight._shape[0];
        // Stack rows into a single [totalSeqLen, inFeatures] tensor and
        // run a single dense matmul against W^T — same FLOPs as per-row
        // but the dense kernel is the SIMD fast path.
        int totalSeqLen = x.Values.Length / x.FeatureSize;
        var stacked = ReshapeAsRows(x, totalSeqLen, x.FeatureSize);
        var wT = Engine.TensorTranspose(weight);
        var product = Engine.TensorMatMul(stacked, wT);
        var values = product.Reshape(new[] { totalSeqLen * outFeatures });

        if (bias is not null)
        {
            if (bias.Length != outFeatures)
                throw new ArgumentException($"Bias length {bias.Length} != outFeatures {outFeatures}.", nameof(bias));
            var ops = MathHelper.GetNumericOperations<T>();
            var biasSpan = bias.AsSpan();
            var span = values.AsWritableSpan();
            for (int i = 0; i < span.Length; i++)
                span[i] = ops.Add(span[i], biasSpan[i % outFeatures]);
        }
        return NestedTensor<T>.FromValuesOffsets(values, (int[])x.Offsets.Clone(), outFeatures, x.Layout);
    }

    /// <summary>
    /// Per-row matmul: <c>x[i] · b</c> with <paramref name="b"/>
    /// shape <c>[features, outFeatures]</c>. Equivalent to a Linear
    /// without transpose; used by attention's <c>Q · K^T</c> style
    /// chains where the right operand is already transposed.
    /// </summary>
    public static NestedTensor<T> MatMul<T>(NestedTensor<T> x, Tensor<T> b)
    {
        if (x.FeatureSize == 0)
            throw new InvalidOperationException("MatMul requires a feature axis on the nested input.");
        if (b.Rank != 2 || b._shape[0] != x.FeatureSize)
            throw new ArgumentException(
                $"B must have shape [{x.FeatureSize}, *]; got {ShapeStr(b._shape)}.", nameof(b));

        int outFeatures = b._shape[1];
        int totalSeqLen = x.Values.Length / x.FeatureSize;
        var stacked = ReshapeAsRows(x, totalSeqLen, x.FeatureSize);
        var product = Engine.TensorMatMul(stacked, b);
        var values = product.Reshape(new[] { totalSeqLen * outFeatures });
        return NestedTensor<T>.FromValuesOffsets(values, (int[])x.Offsets.Clone(), outFeatures, x.Layout);
    }

    /// <summary>Per-row softmax along the feature axis.</summary>
    public static NestedTensor<T> Softmax<T>(NestedTensor<T> x)
    {
        if (x.FeatureSize == 0)
            throw new InvalidOperationException("Softmax requires a feature axis on the nested input.");
        int totalSeqLen = x.Values.Length / x.FeatureSize;
        var stacked = ReshapeAsRows(x, totalSeqLen, x.FeatureSize);
        var sm = Engine.Softmax(stacked, axis: 1);
        var values = sm.Reshape(new[] { totalSeqLen * x.FeatureSize });
        return NestedTensor<T>.FromValuesOffsets(values, (int[])x.Offsets.Clone(), x.FeatureSize, x.Layout);
    }

    /// <summary>Per-row log-softmax along the feature axis.</summary>
    public static NestedTensor<T> LogSoftmax<T>(NestedTensor<T> x)
    {
        if (x.FeatureSize == 0)
            throw new InvalidOperationException("LogSoftmax requires a feature axis on the nested input.");
        var ops = MathHelper.GetNumericOperations<T>();
        var sm = Softmax(x);
        var values = new Tensor<T>(new[] { sm.Values.Length });
        var src = sm.Values.AsSpan();
        var dst = values.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
            dst[i] = ops.FromDouble(Math.Log(Math.Max(1e-30, ops.ToDouble(src[i]))));
        return NestedTensor<T>.FromValuesOffsets(values, (int[])x.Offsets.Clone(), x.FeatureSize, x.Layout);
    }

    /// <summary>LayerNorm along the feature axis. Identical to dense
    /// LayerNorm applied per-token.</summary>
    public static NestedTensor<T> LayerNorm<T>(NestedTensor<T> x, Tensor<T> gamma, Tensor<T> beta, double epsilon = 1e-5)
    {
        if (x.FeatureSize == 0)
            throw new InvalidOperationException("LayerNorm requires a feature axis on the nested input.");
        if (gamma.Length != x.FeatureSize || beta.Length != x.FeatureSize)
            throw new ArgumentException("gamma and beta must have length = FeatureSize.");

        int totalSeqLen = x.Values.Length / x.FeatureSize;
        var stacked = ReshapeAsRows(x, totalSeqLen, x.FeatureSize);
        var normed = Engine.LayerNorm(stacked, gamma, beta, epsilon, out _, out _);
        var values = normed.Reshape(new[] { totalSeqLen * x.FeatureSize });
        return NestedTensor<T>.FromValuesOffsets(values, (int[])x.Offsets.Clone(), x.FeatureSize, x.Layout);
    }

    /// <summary>Per-element GeLU.</summary>
    public static NestedTensor<T> GeLU<T>(NestedTensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var values = new Tensor<T>(new[] { x.Values.Length });
        var src = x.Values.AsSpan();
        var dst = values.AsWritableSpan();
        const double C = 0.7978845608028654; // sqrt(2/pi)
        for (int i = 0; i < src.Length; i++)
        {
            double v = ops.ToDouble(src[i]);
            double inner = C * (v + 0.044715 * v * v * v);
            dst[i] = ops.FromDouble(0.5 * v * (1.0 + Math.Tanh(inner)));
        }
        return NestedTensor<T>.FromValuesOffsets(values, (int[])x.Offsets.Clone(), x.FeatureSize, x.Layout);
    }

    /// <summary>Per-element SiLU (= x · sigmoid(x)).</summary>
    public static NestedTensor<T> SiLU<T>(NestedTensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var values = new Tensor<T>(new[] { x.Values.Length });
        var src = x.Values.AsSpan();
        var dst = values.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
        {
            double v = ops.ToDouble(src[i]);
            dst[i] = ops.FromDouble(v / (1.0 + Math.Exp(-v)));
        }
        return NestedTensor<T>.FromValuesOffsets(values, (int[])x.Offsets.Clone(), x.FeatureSize, x.Layout);
    }

    /// <summary>
    /// Variable-length scaled dot-product attention. Each row's Q/K/V
    /// has shape <c>[seqLen_i, headDim]</c>; output has the same shape.
    /// Applies <c>softmax(Q·K^T / sqrt(d)) · V</c> per row, no padding
    /// FLOPs across the batch.
    /// </summary>
    public static NestedTensor<T> ScaledDotProductAttention<T>(
        NestedTensor<T> q, NestedTensor<T> k, NestedTensor<T> v)
    {
        if (q.BatchSize != k.BatchSize || q.BatchSize != v.BatchSize)
            throw new ArgumentException("Q, K, V must share the same batch size.");
        if (q.FeatureSize != k.FeatureSize || q.FeatureSize != v.FeatureSize)
            throw new ArgumentException("Q, K, V must share the same head dimension.");
        for (int i = 0; i < q.BatchSize; i++)
        {
            if (q.RowLength(i) != k.RowLength(i) || q.RowLength(i) != v.RowLength(i))
                throw new ArgumentException($"Row {i} length mismatch between Q/K/V.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        int headDim = q.FeatureSize;
        T invSqrtD = ops.FromDouble(1.0 / Math.Sqrt(headDim));

        var outValues = new Tensor<T>(new[] { q.Values.Length });
        var outSpan = outValues.AsWritableSpan();
        var qSpan = q.Values.AsSpan();
        var kSpan = k.Values.AsSpan();
        var vSpan = v.Values.AsSpan();

        for (int b = 0; b < q.BatchSize; b++)
        {
            int rowLen = q.RowLength(b);
            int qOff = q.Offsets[b] * headDim;
            int kOff = k.Offsets[b] * headDim;
            int vOff = v.Offsets[b] * headDim;
            // Per-row attention: build the seqLen×seqLen score matrix in place.
            var qRow = new Tensor<T>(new[] { rowLen, headDim });
            var kRow = new Tensor<T>(new[] { rowLen, headDim });
            var vRow = new Tensor<T>(new[] { rowLen, headDim });
            qSpan.Slice(qOff, rowLen * headDim).CopyTo(qRow.AsWritableSpan());
            kSpan.Slice(kOff, rowLen * headDim).CopyTo(kRow.AsWritableSpan());
            vSpan.Slice(vOff, rowLen * headDim).CopyTo(vRow.AsWritableSpan());

            var kT = Engine.TensorTranspose(kRow);
            var scores = Engine.TensorMatMul(qRow, kT);
            var scaledScores = Engine.TensorMultiplyScalar(scores, invSqrtD);
            var probs = Engine.Softmax(scaledScores, axis: 1);
            var attended = Engine.TensorMatMul(probs, vRow);
            attended.AsSpan().CopyTo(outSpan.Slice(qOff, rowLen * headDim));
        }
        return NestedTensor<T>.FromValuesOffsets(outValues, (int[])q.Offsets.Clone(), headDim, q.Layout);
    }

    /// <summary>
    /// Multi-head attention with a single Q/K/V projection per head.
    /// <paramref name="numHeads"/> divides <c>FeatureSize</c>; each head
    /// runs its own <see cref="ScaledDotProductAttention{T}"/> over a
    /// per-row reshape <c>[seqLen, headDim]</c>.
    /// </summary>
    public static NestedTensor<T> MultiHeadAttention<T>(
        NestedTensor<T> q, NestedTensor<T> k, NestedTensor<T> v, int numHeads)
    {
        if (q is null) throw new ArgumentNullException(nameof(q));
        if (k is null) throw new ArgumentNullException(nameof(k));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        // Same Q/K/V compatibility checks SDPA does — without these,
        // mismatched batch / feature / row lengths would silently
        // index past spans and produce wrong attention.
        if (q.BatchSize != k.BatchSize || q.BatchSize != v.BatchSize)
            throw new ArgumentException("Q, K, V must share the same batch size.");
        if (q.FeatureSize == 0 || k.FeatureSize == 0 || v.FeatureSize == 0)
            throw new InvalidOperationException("MultiHeadAttention requires a non-zero feature axis.");
        if (q.FeatureSize != k.FeatureSize || q.FeatureSize != v.FeatureSize)
            throw new ArgumentException("Q, K, V must share the same feature size.");
        if (q.FeatureSize % numHeads != 0)
            throw new ArgumentException($"FeatureSize {q.FeatureSize} must be divisible by numHeads {numHeads}.");
        for (int i = 0; i < q.BatchSize; i++)
        {
            if (q.RowLength(i) != k.RowLength(i) || q.RowLength(i) != v.RowLength(i))
                throw new ArgumentException(
                    $"Row {i} length mismatch between Q/K/V (Q={q.RowLength(i)}, K={k.RowLength(i)}, V={v.RowLength(i)}).");
        }

        // Walk per row: reshape each row's [seqLen, embed] into
        // [numHeads, seqLen, headDim] then run SDPA on the head axis.
        var ops = MathHelper.GetNumericOperations<T>();
        int headDim = q.FeatureSize / numHeads;
        var outValues = new Tensor<T>(new[] { q.Values.Length });
        var outSpan = outValues.AsWritableSpan();
        var qSpan = q.Values.AsSpan();
        var kSpan = k.Values.AsSpan();
        var vSpan = v.Values.AsSpan();
        T invSqrtD = ops.FromDouble(1.0 / Math.Sqrt(headDim));

        for (int b = 0; b < q.BatchSize; b++)
        {
            int rowLen = q.RowLength(b);
            int qOff = q.Offsets[b] * q.FeatureSize;
            int kOff = k.Offsets[b] * k.FeatureSize;
            int vOff = v.Offsets[b] * v.FeatureSize;

            for (int h = 0; h < numHeads; h++)
            {
                var qRow = new Tensor<T>(new[] { rowLen, headDim });
                var kRow = new Tensor<T>(new[] { rowLen, headDim });
                var vRow = new Tensor<T>(new[] { rowLen, headDim });
                var qDst = qRow.AsWritableSpan();
                var kDst = kRow.AsWritableSpan();
                var vDst = vRow.AsWritableSpan();
                for (int s = 0; s < rowLen; s++)
                {
                    qSpan.Slice(qOff + s * q.FeatureSize + h * headDim, headDim)
                        .CopyTo(qDst.Slice(s * headDim, headDim));
                    kSpan.Slice(kOff + s * k.FeatureSize + h * headDim, headDim)
                        .CopyTo(kDst.Slice(s * headDim, headDim));
                    vSpan.Slice(vOff + s * v.FeatureSize + h * headDim, headDim)
                        .CopyTo(vDst.Slice(s * headDim, headDim));
                }

                var kT = Engine.TensorTranspose(kRow);
                var scores = Engine.TensorMatMul(qRow, kT);
                var scaledScores = Engine.TensorMultiplyScalar(scores, invSqrtD);
                var probs = Engine.Softmax(scaledScores, axis: 1);
                var attended = Engine.TensorMatMul(probs, vRow);
                var attSpan = attended.AsSpan();

                for (int s = 0; s < rowLen; s++)
                {
                    attSpan.Slice(s * headDim, headDim)
                        .CopyTo(outSpan.Slice(qOff + s * q.FeatureSize + h * headDim, headDim));
                }
            }
        }
        return NestedTensor<T>.FromValuesOffsets(outValues, (int[])q.Offsets.Clone(), q.FeatureSize, q.Layout);
    }

    private static Tensor<T> ReshapeAsRows<T>(NestedTensor<T> x, int totalRows, int features)
    {
        // Materialize a [totalRows, features] dense tensor. This IS a
        // copy of the contiguous values buffer — the dense matmul
        // kernels need a tensor with its own storage. Once Tensor<T>
        // grows a zero-copy reshape API for the contiguous case we'll
        // route through it.
        var t = new Tensor<T>(new[] { totalRows, features });
        x.Values.AsSpan().CopyTo(t.AsWritableSpan());
        return t;
    }

    private static string ShapeStr(int[] shape) => $"[{string.Join(", ", shape)}]";
}
