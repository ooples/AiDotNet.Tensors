// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Grouped-query attention under a tape: the key and value heads are shared by several query heads, and their gradient
/// must sum over those query heads. The CPU path repeated the shared heads with a copy that recorded nothing, so the
/// attention node recorded against the copies and key and value received no gradient.
/// </summary>
public sealed class GqaAttentionTapeGradientTests
{
    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    [Theory]
    [InlineData(4, 2)]
    [InlineData(6, 1)]
    public void Taped_grouped_query_attention_gives_key_and_value_the_summed_gradient(int qHeads, int kvHeads)
    {
        var e = new CpuEngine();
        const int batch = 2, seq = 3, headDim = 4;
        var q = Rand(new[] { batch, qHeads, seq, headDim }, 1);
        var k = Rand(new[] { batch, kvHeads, seq, headDim }, 2);
        var v = Rand(new[] { batch, kvHeads, seq, headDim }, 3);
        var r = Rand(new[] { batch, qHeads, seq, headDim }, 4);
        double scale = 1.0 / Math.Sqrt(headDim);
        int[] all = { 0, 1, 2, 3 };

        Tensor<float>[] gqa;
        using (var tape = new GradientTape<float>())
        {
            var o = e.ScaledDotProductAttentionGqa(q, k, v, scale, isCausal: false);
            var loss = e.ReduceSum(e.TensorMultiply(o, r), all, keepDims: false);
            var g = tape.ComputeGradients(loss, new[] { q, k, v });
            gqa = new[] { q, k, v }.Select(t => g.TryGetValue(t, out var gt) ? gt : null!).ToArray();
        }

        // Reference: the heads repeated by a recorded op, then ordinary attention.
        Tensor<float>[] reference;
        using (var tape = new GradientTape<float>())
        {
            int group = qHeads / kvHeads;
            var o = e.ScaledDotProductAttention(q, e.TensorRepeatInterleave(k, group, 1), e.TensorRepeatInterleave(v, group, 1),
                mask: null, scale: scale, isCausal: false);
            var loss = e.ReduceSum(e.TensorMultiply(o, r), all, keepDims: false);
            var g = tape.ComputeGradients(loss, new[] { q, k, v });
            reference = new[] { g[q], g[k], g[v] };
        }

        string[] names = { "dQ", "dK", "dV" };
        for (int i = 0; i < 3; i++)
        {
            Assert.True(gqa[i] is not null, $"{names[i]} is missing: no gradient reached it");
            var a = gqa[i].ToArray();
            var b = reference[i].ToArray();
            Assert.Equal(b.Length, a.Length);
            Assert.True(b.Any(x => x != 0f), $"reference {names[i]} is all zero");
            for (int j = 0; j < a.Length; j++)
                Assert.True(Math.Abs(a[j] - b[j]) <= 1e-5f, $"{names[i]}[{j}]: {a[j]} vs {b[j]}");
        }
    }
}
