// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Cross-entropy on [N, C, d] logits follows torch: the class axis is 1, targets are class indices [N, d] or
/// probabilities [N, C, d], and the mean runs over all N*d positions. That equals the [N*d, C] cross-entropy of the
/// same logits with the class axis moved last, and the gradient is that one's, moved back.
/// </summary>
public sealed class CrossEntropyNdTests
{
    private const int N = 2, C = 4, D = 3;

    private static Tensor<double> Logits() =>
        new Tensor<double>(Enumerable.Range(0, N * C * D).Select(i => Math.Sin(i * 0.7) * 2).ToArray(), new[] { N, C, D });

    /// <summary>[N, C, D] -> [N*D, C] (class axis last).</summary>
    private static Tensor<double> Rows(Tensor<double> x) =>
        new Tensor<double>(x.Transpose(new[] { 0, 2, 1 }).ToArray(), new[] { N * D, C });

    [Fact]
    public void Class_index_targets_match_the_row_layout()
    {
        IEngine e = new CpuEngine();
        var logits = Logits();
        var idx = new Tensor<double>(new double[] { 0, 3, 1, 2, 2, 0 }, new[] { N, D });
        var idxRows = new Tensor<double>(idx.ToArray(), new[] { N * D });

        Assert.Equal(e.CrossEntropyLoss(Rows(logits), idxRows), e.CrossEntropyLoss(logits, idx), 12);
        var g = e.CrossEntropyBackward(logits, idx);
        Assert.Equal(new[] { N, C, D }, g.Shape.ToArray());
        var expected = e.CrossEntropyBackward(Rows(logits), idxRows).ToArray();
        var actual = Rows(g).ToArray();
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 12);
    }

    [Fact]
    public void Probability_targets_match_the_row_layout()
    {
        IEngine e = new CpuEngine();
        var logits = Logits();
        var probs = e.Softmax(new Tensor<double>(Enumerable.Range(0, N * C * D).Select(i => Math.Cos(i * 1.3)).ToArray(), new[] { N, C, D }), 1);

        Assert.Equal(e.CrossEntropyLoss(Rows(logits), Rows(probs)), e.CrossEntropyLoss(logits, probs), 12);
        var expected = e.CrossEntropyBackward(Rows(logits), Rows(probs)).ToArray();
        var actual = Rows(e.CrossEntropyBackward(logits, probs)).ToArray();
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 12);
    }

    [Fact]
    public void Wrong_target_layouts_and_out_of_range_classes_throw()
    {
        IEngine e = new CpuEngine();
        var logits = Logits();
        Assert.Throws<ArgumentException>(() => e.CrossEntropyBackward(logits, new Tensor<double>(new[] { N })));
        Assert.Throws<ArgumentException>(() => e.CrossEntropyLoss(logits, new Tensor<double>(new[] { N, C })));
        var bad = new Tensor<double>(new double[] { 0, 4, 1, 2, 2, 0 }, new[] { N, D });
        Assert.ThrowsAny<ArgumentException>(() => e.CrossEntropyBackward(logits, bad));
        Assert.Throws<ArgumentException>(() => e.TensorCrossEntropyLoss(logits, new Tensor<double>(new[] { N, D })));
    }
}
