// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Nested;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra.Nested;

/// <summary>
/// Acceptance tests for <see cref="NestedTensor{T}"/> + <see cref="NestedOps"/>.
/// Covers round-trip construction (FromList → values/offsets), padding
/// equivalence, and per-row op correctness vs a dense + length-mask
/// reference.
/// </summary>
public class NestedTensorTests
{
    private readonly CpuEngine _engine = new();

    private static Tensor<float> Row(int seqLen, int features, float baseVal)
    {
        var t = new Tensor<float>(new[] { seqLen, features });
        for (int s = 0; s < seqLen; s++)
            for (int f = 0; f < features; f++)
                t[s, f] = baseVal + s * 10 + f;
        return t;
    }

    [Fact]
    public void FromList_BuildsOffsetsAndPreservesRowContent()
    {
        var rows = new[]
        {
            Row(2, 3, 1f),  // 2 elements, features 3
            Row(3, 3, 100f),
            Row(1, 3, 1000f),
        };
        var nested = NestedTensor<float>.FromList(rows);

        Assert.Equal(3, nested.BatchSize);
        Assert.Equal(3, nested.FeatureSize);
        Assert.Equal(new[] { 0, 2, 5, 6 }, nested.Offsets);
        Assert.Equal(2, nested.RowLength(0));
        Assert.Equal(3, nested.RowLength(1));
        Assert.Equal(1, nested.RowLength(2));
        Assert.Equal(3, nested.MaxRowLength);
        Assert.Equal(6 * 3, nested.StoredElements);
    }

    [Fact]
    public void ToPadded_FromPadded_RoundTripPreservesValues()
    {
        var rows = new[]
        {
            Row(2, 4, 1f),
            Row(3, 4, 100f),
            Row(1, 4, 1000f),
        };
        var original = NestedTensor<float>.FromList(rows);
        var padded = original.ToPadded(padding: -999f);
        Assert.Equal(new[] { 3, 3, 4 }, padded._shape);
        var lengths = new[] { 2, 3, 1 };
        var roundtrip = NestedTensor<float>.FromPadded(padded, lengths);

        Assert.Equal(original.Values.Length, roundtrip.Values.Length);
        for (int i = 0; i < original.Values.Length; i++)
            Assert.Equal(original.Values.AsSpan()[i], roundtrip.Values.AsSpan()[i]);
    }

    [Fact]
    public void ToPadded_PaddingFillsTheRest()
    {
        var rows = new[] { Row(1, 2, 1f), Row(2, 2, 10f) };
        var nested = NestedTensor<float>.FromList(rows);
        var padded = nested.ToPadded(padding: -1f, outputSize: 3);
        // Row 0 (len=1): valid at [0,*]; pad at [1,*] and [2,*].
        Assert.Equal(1f, padded[0, 0, 0]);
        Assert.Equal(2f, padded[0, 0, 1]);
        Assert.Equal(-1f, padded[0, 1, 0]);
        Assert.Equal(-1f, padded[0, 2, 1]);
        // Row 1 (len=2): valid at [0..1,*]; pad at [2,*].
        Assert.Equal(10f, padded[1, 0, 0]);
        Assert.Equal(20f, padded[1, 1, 0]);
        Assert.Equal(-1f, padded[1, 2, 0]);
    }

    [Fact]
    public void Add_BiasIsBroadcastPerToken()
    {
        var rows = new[] { Row(2, 3, 0f), Row(1, 3, 0f) };
        var nested = NestedTensor<float>.FromList(rows);
        var bias = new Tensor<float>(new[] { 3 });
        bias[0] = 100; bias[1] = 200; bias[2] = 300;

        var result = NestedOps.Add(nested, bias);
        Assert.Equal(nested.Values.Length, result.Values.Length);
        var src = nested.Values.AsSpan();
        var dst = result.Values.AsSpan();
        for (int i = 0; i < src.Length; i++)
            Assert.Equal(src[i] + bias.AsSpan()[i % 3], dst[i], 4);
    }

    [Fact]
    public void Linear_MatchesPaddedReferenceOnAllRows()
    {
        var rows = new[] { Row(2, 4, 1f), Row(3, 4, 100f) };
        var nested = NestedTensor<float>.FromList(rows);
        var weight = new Tensor<float>(new[] { 5, 4 }); // 5 outFeatures
        for (int i = 0; i < weight.Length; i++) weight.AsWritableSpan()[i] = (i % 7) * 0.1f;

        var nestedOut = NestedOps.Linear(nested, weight);
        Assert.Equal(5, nestedOut.FeatureSize);
        Assert.Equal(2, nestedOut.BatchSize);

        // Reference: per-row dense linear.
        for (int b = 0; b < 2; b++)
        {
            var wT = _engine.TensorTranspose(weight);
            var refOut = _engine.TensorMatMul(rows[b], wT);
            int rowOff = nested.Offsets[b] * 5;
            for (int s = 0; s < rows[b]._shape[0]; s++)
                for (int f = 0; f < 5; f++)
                    Assert.Equal(refOut[s, f], nestedOut.Values.AsSpan()[rowOff + s * 5 + f], 3);
        }
    }

    [Fact]
    public void Softmax_RowsSumToOnePerToken()
    {
        var rows = new[] { Row(2, 4, 0f), Row(1, 4, 0f) };
        var nested = NestedTensor<float>.FromList(rows);
        var sm = NestedOps.Softmax(nested);
        // Each (row, token) softmax should sum to 1 across the feature axis.
        var span = sm.Values.AsSpan();
        for (int t = 0; t < sm.Values.Length / 4; t++)
        {
            float sum = 0;
            for (int f = 0; f < 4; f++) sum += span[t * 4 + f];
            Assert.Equal(1f, sum, 3);
        }
    }

    [Fact]
    public void GeLU_AppliesPerElement()
    {
        var t = new Tensor<float>(new[] { 1, 4 });
        t[0, 0] = -1; t[0, 1] = 0; t[0, 2] = 1; t[0, 3] = 2;
        var nested = NestedTensor<float>.FromList(new[] { t });
        var result = NestedOps.GeLU(nested);
        // GeLU(0) = 0; GeLU(positive) > 0; GeLU(-1) ≈ -0.158.
        var span = result.Values.AsSpan();
        Assert.True(Math.Abs(span[1]) < 1e-3);
        Assert.True(span[2] > 0.5f);
        Assert.True(span[0] < 0);
    }

    [Fact]
    public void ScaledDotProductAttention_PerRowMatchesDenseReference()
    {
        // Two rows of length 2 and 3, head dim 4.
        var q1 = Row(2, 4, 1f);
        var q2 = Row(3, 4, 100f);
        var k1 = Row(2, 4, 0.5f);
        var k2 = Row(3, 4, 50f);
        var v1 = Row(2, 4, 0.1f);
        var v2 = Row(3, 4, 10f);
        var qN = NestedTensor<float>.FromList(new[] { q1, q2 });
        var kN = NestedTensor<float>.FromList(new[] { k1, k2 });
        var vN = NestedTensor<float>.FromList(new[] { v1, v2 });

        var nestedOut = NestedOps.ScaledDotProductAttention(qN, kN, vN);
        Assert.Equal(qN.Values.Length, nestedOut.Values.Length);

        // Reference: per-row SDPA.
        var qs = new[] { q1, q2 };
        var ks = new[] { k1, k2 };
        var vs = new[] { v1, v2 };
        for (int b = 0; b < 2; b++)
        {
            float invSqrtD = 1f / MathF.Sqrt(4);
            var kT = _engine.TensorTranspose(ks[b]);
            var scores = _engine.TensorMatMul(qs[b], kT);
            var scaled = _engine.TensorMultiplyScalar(scores, invSqrtD);
            var probs = _engine.Softmax(scaled, axis: 1);
            var refOut = _engine.TensorMatMul(probs, vs[b]);
            int rowOff = qN.Offsets[b] * 4;
            for (int s = 0; s < qs[b]._shape[0]; s++)
                for (int f = 0; f < 4; f++)
                    Assert.Equal(refOut[s, f], nestedOut.Values.AsSpan()[rowOff + s * 4 + f], 3);
        }
    }
}
