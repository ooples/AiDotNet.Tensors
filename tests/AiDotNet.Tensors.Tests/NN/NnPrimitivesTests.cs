// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NN.Activations;
using AiDotNet.Tensors.NN.Functional;
using AiDotNet.Tensors.NN.Losses;
using AiDotNet.Tensors.NN.Parametrizations;
using AiDotNet.Tensors.NN.Pruning;
using Xunit;

namespace AiDotNet.Tensors.Tests.NN;

/// <summary>
/// Acceptance tests for #223 (torch.nn primitives gap). Covers the
/// new losses, functional utilities, activations, parametrizations,
/// and pruning APIs.
/// </summary>
public class NnPrimitivesTests
{
    // ======================  LOSSES  ======================

    [Fact]
    public void SmoothL1Loss_BetaOneMatchesHuberAtAbsDiffAboveBeta()
    {
        var x = MakeFloats(new[] { 1f, 5f, -3f });
        var y = MakeFloats(new[] { 0f, 0f, 0f });
        // |diff| = [1, 5, 3]; with beta=1:
        //   diff=1   → 0.5*1²/1 = 0.5
        //   diff=5   → 5 - 0.5 = 4.5
        //   diff=3   → 3 - 0.5 = 2.5
        // sum = 7.5; mean = 2.5.
        var loss = Losses.SmoothL1Loss(x, y, beta: 1.0, reduction: LossReduction.Mean);
        Assert.Equal(2.5f, loss.AsSpan()[0], 4);
    }

    [Fact]
    public void PoissonNllLoss_NonLogInputComputesXminusTLogX()
    {
        var x = MakeFloats(new[] { 2f, 3f });
        var y = MakeFloats(new[] { 1f, 2f });
        // logInput=false: x − t·log(x+eps)
        // [2 − 1·log(2), 3 − 2·log(3)]
        var loss = Losses.PoissonNllLoss(x, y, logInput: false, reduction: LossReduction.None);
        Assert.Equal(2f - MathF.Log(2f), loss.AsSpan()[0], 3);
        Assert.Equal(3f - 2f * MathF.Log(3f), loss.AsSpan()[1], 3);
    }

    [Fact]
    public void GaussianNllLoss_PositivePredictedVarianceIsClampedAtEps()
    {
        var x = MakeFloats(new[] { 1f });
        var y = MakeFloats(new[] { 0f });
        var v = MakeFloats(new[] { -1f }); // negative — must be clamped to eps.
        var loss = Losses.GaussianNllLoss(x, y, v, eps: 1e-3, reduction: LossReduction.None);
        // 0.5 * (log(1e-3) + (0 − 1)² / 1e-3) = 0.5 * (-6.908 + 1000) ≈ 496.5
        Assert.True(loss.AsSpan()[0] > 100f);
    }

    [Fact]
    public void MultiMarginLoss_PositiveTargetMatchingHasZeroLoss()
    {
        // Class 0 is the target and is the largest score → loss = 0.
        var x = new Tensor<float>(new[] { 1, 3 });
        x[0, 0] = 5; x[0, 1] = 0; x[0, 2] = 0;
        var t = new Tensor<int>(new[] { 1 });
        t.AsWritableSpan()[0] = 0;
        var loss = Losses.MultiMarginLoss(x, t, reduction: LossReduction.None);
        // For each j ≠ 0: max(0, 1 - 5 + x[j]) — both negative → 0.
        Assert.Equal(0f, loss.AsSpan()[0], 4);
    }

    [Fact]
    public void TripletMarginLoss_AnchorEqualPositiveYieldsZeroBeforeMargin()
    {
        var a = MakeMatrix(new[,] { { 1f, 2f, 3f } });
        var p = MakeMatrix(new[,] { { 1f, 2f, 3f } });   // distance(a,p) = 0
        var n = MakeMatrix(new[,] { { 5f, 5f, 5f } });   // distance(a,n) > 0
        // Loss = max(0, 0 - dNeg + margin); margin=1, dNeg ≈ 5 → loss = 0.
        var loss = Losses.TripletMarginLoss(a, p, n, margin: 1.0, reduction: LossReduction.None);
        Assert.Equal(0f, loss.AsSpan()[0], 3);
    }

    [Fact]
    public void MarginRankingLoss_HigherRankedFirstWithPositiveLabelHasZeroLoss()
    {
        var x1 = MakeFloats(new[] { 5f });
        var x2 = MakeFloats(new[] { 1f });
        var y = MakeFloats(new[] { 1f });
        var loss = Losses.MarginRankingLoss(x1, x2, y, reduction: LossReduction.None);
        // -1 · (5 - 1) + 0 = -4 → max(0, -4) = 0.
        Assert.Equal(0f, loss.AsSpan()[0], 4);
    }

    [Fact]
    public void CosineEmbeddingLoss_IdenticalVectorsWithLabel1HaveZeroLoss()
    {
        var x = new Tensor<float>(new[] { 1, 3 });
        x[0, 0] = 1; x[0, 1] = 2; x[0, 2] = 3;
        var y = new Tensor<int>(new[] { 1 });
        y.AsWritableSpan()[0] = 1;
        var loss = Losses.CosineEmbeddingLoss(x, x, y, reduction: LossReduction.None);
        Assert.Equal(0f, loss.AsSpan()[0], 3);
    }

    [Fact]
    public void HingeEmbeddingLoss_PositiveLabelPassesInputThrough()
    {
        var x = MakeFloats(new[] { 0.5f, -0.2f });
        var y = new Tensor<int>(new[] { 2 });
        y.AsWritableSpan()[0] = 1;
        y.AsWritableSpan()[1] = 1;
        var loss = Losses.HingeEmbeddingLoss(x, y, reduction: LossReduction.None);
        Assert.Equal(0.5f, loss.AsSpan()[0], 4);
        Assert.Equal(-0.2f, loss.AsSpan()[1], 4);
    }

    [Fact]
    public void KLDiv_LogTargetTrueMatchesExpFormula()
    {
        var input = MakeFloats(new[] { -1f }); // log p
        var target = MakeFloats(new[] { -2f }); // log q (since log_target=true)
        var loss = Losses.KLDiv(input, target, logTarget: true, reduction: LossReduction.None);
        // exp(-2) * (-2 - (-1)) = exp(-2) * -1 ≈ -0.1353
        Assert.Equal(-MathF.Exp(-2f), loss.AsSpan()[0], 4);
    }

    // ======================  FUNCTIONAL  ======================

    [Fact]
    public void Normalize_RowsHaveUnitL2Norm()
    {
        var x = MakeMatrix(new[,] { { 3f, 4f }, { 1f, 0f } });
        var n = Functional.Normalize(x, p: 2.0, dim: 1);
        // Row 0: ||(3,4)|| = 5 → (0.6, 0.8). Row 1: ||(1,0)|| = 1 → (1, 0).
        Assert.Equal(0.6f, n[0, 0], 3);
        Assert.Equal(0.8f, n[0, 1], 3);
        Assert.Equal(1f, n[1, 0], 3);
    }

    [Fact]
    public void OneHot_PlacesOneAtTheCorrectClass()
    {
        var idx = new Tensor<int>(new[] { 3 });
        idx.AsWritableSpan()[0] = 0;
        idx.AsWritableSpan()[1] = 2;
        idx.AsWritableSpan()[2] = 1;
        var oh = Functional.OneHot(idx, 4);
        Assert.Equal(new[] { 3, 4 }, oh._shape);
        Assert.Equal(1, oh[0, 0]);
        Assert.Equal(1, oh[1, 2]);
        Assert.Equal(1, oh[2, 1]);
        Assert.Equal(0, oh[0, 1]);
    }

    [Fact]
    public void PairwiseDistance_L2MatchesEuclidean()
    {
        var x1 = MakeMatrix(new[,] { { 0f, 0f }, { 3f, 4f } });
        var x2 = MakeMatrix(new[,] { { 3f, 4f }, { 0f, 0f } });
        var d = Functional.PairwiseDistance(x1, x2, p: 2.0, eps: 0.0);
        Assert.Equal(5f, d.AsSpan()[0], 3);
        Assert.Equal(5f, d.AsSpan()[1], 3);
    }

    [Fact]
    public void Embedding_LooksUpRowsByIndex()
    {
        var idx = new Tensor<int>(new[] { 2 });
        idx.AsWritableSpan()[0] = 1;
        idx.AsWritableSpan()[1] = 0;
        var w = MakeMatrix(new[,] { { 1f, 2f, 3f }, { 10f, 20f, 30f } });
        var emb = Functional.Embedding(idx, w);
        Assert.Equal(new[] { 2, 3 }, emb._shape);
        Assert.Equal(10f, emb[0, 0]);
        Assert.Equal(20f, emb[0, 1]);
        Assert.Equal(1f, emb[1, 0]);
    }

    [Fact]
    public void EmbeddingBag_MeanModeAveragesWithinBags()
    {
        // Bag 0: indices [0, 1] → mean of weights[0] + weights[1] = (5.5, 11)
        // Bag 1: indices [1] → weights[1] = (10, 20)
        var idx = new Tensor<int>(new[] { 3 });
        idx.AsWritableSpan()[0] = 0;
        idx.AsWritableSpan()[1] = 1;
        idx.AsWritableSpan()[2] = 1;
        var w = MakeMatrix(new[,] { { 1f, 2f }, { 10f, 20f } });
        var bag = Functional.EmbeddingBag(idx, w, new[] { 0, 2 }, Functional.EmbeddingBagMode.Mean);
        Assert.Equal(5.5f, bag[0, 0], 3);
        Assert.Equal(11f, bag[0, 1], 3);
        Assert.Equal(10f, bag[1, 0], 3);
    }

    // ======================  ACTIVATIONS  ======================

    [Fact]
    public void Hardshrink_ZeroesValuesBelowLambda()
    {
        var x = MakeFloats(new[] { 0.1f, 0.6f, -0.3f, -0.7f });
        var r = Activations.Hardshrink(x, lambda: 0.5);
        Assert.Equal(0f, r.AsSpan()[0]);
        Assert.Equal(0.6f, r.AsSpan()[1]);
        Assert.Equal(0f, r.AsSpan()[2]);
        Assert.Equal(-0.7f, r.AsSpan()[3]);
    }

    [Fact]
    public void Softshrink_ShrinksTowardZero()
    {
        var x = MakeFloats(new[] { 0.1f, 0.6f, -0.7f });
        var r = Activations.Softshrink(x, lambda: 0.5);
        Assert.Equal(0f, r.AsSpan()[0], 4);
        Assert.Equal(0.1f, r.AsSpan()[1], 4);
        Assert.Equal(-0.2f, r.AsSpan()[2], 4);
    }

    [Fact]
    public void Tanhshrink_EqualsXMinusTanhX()
    {
        var x = MakeFloats(new[] { 1f });
        var r = Activations.Tanhshrink(x);
        Assert.Equal(1f - MathF.Tanh(1f), r.AsSpan()[0], 4);
    }

    [Fact]
    public void Threshold_ReplacesValuesBelowThreshold()
    {
        var r = Activations.Threshold(MakeFloats(new[] { -1f, 5f, -100f }), threshold: 0.0, value: 0.0);
        Assert.Equal(0f, r.AsSpan()[0]);   // -1 ≤ 0 → replaced with 0
        Assert.Equal(5f, r.AsSpan()[1]);   // 5 > 0 → kept
        Assert.Equal(0f, r.AsSpan()[2]);   // -100 ≤ 0 → replaced
    }

    [Fact]
    public void QuickGelu_ApproximatesGeluCloselyAtZero()
    {
        var x = MakeFloats(new[] { 0f });
        var r = Activations.QuickGelu(x);
        Assert.Equal(0f, r.AsSpan()[0], 4);
    }

    [Fact]
    public void GeluTanh_AtZeroIsZero()
    {
        var x = MakeFloats(new[] { 0f });
        var r = Activations.GeluTanh(x);
        Assert.Equal(0f, r.AsSpan()[0], 5);
    }

    [Fact]
    public void SwiGLU_HalvesLastDim()
    {
        var x = MakeMatrix(new[,] { { 1f, 2f, 3f, 4f } });
        var r = Activations.SwiGLU(x);
        Assert.Equal(new[] { 1, 2 }, r._shape);
        // a = (1, 2); b = (3, 4); SiLU(1) ≈ 0.7311; SiLU(2) ≈ 1.7616.
        // r = (silu(1)*3, silu(2)*4)
        Assert.Equal(0.7311f * 3f, r.AsSpan()[0], 2);
        Assert.Equal(1.7616f * 4f, r.AsSpan()[1], 2);
    }

    [Fact]
    public void ReGLU_HalvesLastDimAndAppliesReLU()
    {
        var x = MakeMatrix(new[,] { { 1f, -1f, 5f, 6f } });
        var r = Activations.ReGLU(x);
        Assert.Equal(new[] { 1, 2 }, r._shape);
        // a=(1,-1); ReLU(a)=(1,0); b=(5,6); r=(5, 0).
        Assert.Equal(5f, r.AsSpan()[0]);
        Assert.Equal(0f, r.AsSpan()[1]);
    }

    // ======================  PARAMETRIZATIONS  ======================

    [Fact]
    public void WeightNorm_ForwardScalesRowsBackToOriginalNorms()
    {
        var w = MakeMatrix(new[,] { { 3f, 4f }, { 1f, 0f } });
        var wn = new WeightNorm<float>(w, dim: 1);
        var out_ = wn.Forward(w);
        // Row 0 should still have ||row|| ≈ 5; row 1 still ≈ 1.
        // Norm is preserved exactly when the magnitude g is initialised
        // from the input's own per-row norms.
        double n0 = Math.Sqrt(out_[0, 0] * out_[0, 0] + out_[0, 1] * out_[0, 1]);
        double n1 = Math.Sqrt(out_[1, 0] * out_[1, 0] + out_[1, 1] * out_[1, 1]);
        Assert.Equal(5.0, n0, 3);
        Assert.Equal(1.0, n1, 3);
    }

    [Fact]
    public void SpectralNorm_BoundsTopSingularValueToOne()
    {
        // 2x2 matrix with known singular values 4 and 2.
        var w = MakeMatrix(new[,] { { 4f, 0f }, { 0f, 2f } });
        var sn = new SpectralNorm<float>(outFeatures: 2, iters: 50);
        var normed = sn.Forward(w);
        // After spectral normalization the matrix's largest singular
        // value should be approximately 1.
        double n0 = Math.Sqrt(normed[0, 0] * normed[0, 0] + normed[0, 1] * normed[0, 1]);
        double n1 = Math.Sqrt(normed[1, 0] * normed[1, 0] + normed[1, 1] * normed[1, 1]);
        Assert.True(Math.Max(n0, n1) <= 1.0 + 1e-2);
    }

    [Fact]
    public void Orthogonal_ProducesOrthogonalMatrixCayleyTransform()
    {
        // Skew-symmetric input → Cayley(skew) is orthogonal.
        var w = MakeMatrix(new[,] { { 0f, 1f }, { -1f, 0f } });
        var op = new OrthogonalParametrization<float>();
        var Q = op.Forward(w);
        // QQᵀ = I.
        double q00 = Q[0, 0] * Q[0, 0] + Q[0, 1] * Q[0, 1];
        double q11 = Q[1, 0] * Q[1, 0] + Q[1, 1] * Q[1, 1];
        double q01 = Q[0, 0] * Q[1, 0] + Q[0, 1] * Q[1, 1];
        Assert.Equal(1.0, q00, 3);
        Assert.Equal(1.0, q11, 3);
        Assert.Equal(0.0, q01, 3);
    }

    // ======================  PRUNING  ======================

    [Fact]
    public void L1Unstructured_RemovesSmallestMagnitudeFraction()
    {
        var w = MakeFloats(new[] { 0.1f, 0.5f, -0.05f, 1f, 2f });
        var pruned = Pruning.L1Unstructured(w, amount: 0.4); // prune 2 of 5.
        Assert.Equal(3, pruned.KeptCount);
        // The two smallest magnitudes are 0.05 and 0.1 — both should be pruned.
        Assert.False(pruned.IsKept(0));
        Assert.False(pruned.IsKept(2));
        Assert.True(pruned.IsKept(1));
        Assert.True(pruned.IsKept(3));
        Assert.True(pruned.IsKept(4));
    }

    [Fact]
    public void RandomUnstructured_ProducesDeterministicMaskFromSeed()
    {
        var w = MakeFloats(new[] { 1f, 2f, 3f, 4f, 5f });
        var p1 = Pruning.RandomUnstructured(w, amount: 0.4, seed: 42);
        var p2 = Pruning.RandomUnstructured(w, amount: 0.4, seed: 42);
        Assert.Equal(p1.KeptCount, p2.KeptCount);
        for (int i = 0; i < 5; i++)
            Assert.Equal(p1.IsKept(i), p2.IsKept(i));
    }

    [Fact]
    public void GlobalUnstructured_PrunesSmallestAcrossPool()
    {
        var w1 = MakeFloats(new[] { 0.1f, 5f });
        var w2 = MakeFloats(new[] { 0.2f, 0.05f });
        var pruned = Pruning.GlobalUnstructured(new[] { w1, w2 }, amount: 0.5);
        // Pool: [0.1, 5, 0.2, 0.05]; smallest 2 are 0.05 and 0.1 → mask both.
        Assert.False(pruned[0].IsKept(0));   // 0.1
        Assert.True(pruned[0].IsKept(1));    // 5
        Assert.True(pruned[1].IsKept(0));    // 0.2 — kept (3rd smallest)
        Assert.False(pruned[1].IsKept(1));   // 0.05
    }

    [Fact]
    public void Remove_ZeroesOutPrunedLanes()
    {
        var w = MakeFloats(new[] { 1f, 2f, 3f, 4f });
        var mask = new[] { true, false, true, false };
        var pruned = Pruning.CustomFromMask(w, mask);
        var consolidated = Pruning.Remove(pruned);
        Assert.Equal(1f, consolidated.AsSpan()[0]);
        Assert.Equal(0f, consolidated.AsSpan()[1]);
        Assert.Equal(3f, consolidated.AsSpan()[2]);
        Assert.Equal(0f, consolidated.AsSpan()[3]);
    }

    private static Tensor<float> MakeFloats(float[] data)
    {
        var t = new Tensor<float>(new[] { data.Length });
        var s = t.AsWritableSpan();
        for (int i = 0; i < data.Length; i++) s[i] = data[i];
        return t;
    }

    private static Tensor<float> MakeMatrix(float[,] data)
    {
        int r = data.GetLength(0), c = data.GetLength(1);
        var t = new Tensor<float>(new[] { r, c });
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                t[i, j] = data[i, j];
        return t;
    }
}
