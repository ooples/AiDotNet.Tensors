using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Regression coverage for the IEngine.ReduceSum / ReduceMean axis-
/// misinterpretation bug surfaced by
/// FusedLinearBiasGradIntegrationTests.TwoLayerFFL_Rank3Float_TrainsCleanly:
/// the explicit-interface fast path on DirectGpuTensorEngine called
/// backend.SumAxis(buffer, output, totalOuter, reduceSize) for ANY
/// single-axis reduction, but backend.SumAxis treats its buffer as
/// [N, reduceSize] rows — that math only matches the requested
/// reduction when the reduce axis is the innermost (axis == rank - 1).
/// For middle/outer axes the row-major strides scatter the reduce
/// elements across the buffer, so SumAxis silently summed the WRONG
/// axis — gradients for FusedLinear's [B,T,F]→[1,F] bias had the right
/// total but the wrong per-feature partition, and SGD diverged.
/// Tests cover the IEngine dispatch path (what user code hits when
/// _engine is typed as IEngine) for all single-axis and multi-axis
/// reductions on a rank-3 tensor, plus rank-4 to exercise the more
/// complex permute path that the public override falls through to.
/// </summary>
public class ReduceSumAxisCorrectnessTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> MakeAscending(int[] shape)
    {
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = i + 1;
        return t;
    }

    [Fact]
    public void ReduceSum_Rank3_Axis0_NotInnermost_GivesCorrectResult()
    {
        // gradOutput [2, 4, 2]: ascending values 1..16
        // Sum over axis 0: result[j, k] = grad[0, j, k] + grad[1, j, k]
        //   result[0, 0] = 1 + 9 = 10 (NOT 1 + 2 = 3 — that would be axis-2)
        var grad = MakeAscending(new[] { 2, 4, 2 });
        var sum = _engine.ReduceSum(grad, new[] { 0 }, keepDims: false);
        Assert.Equal(new[] { 4, 2 }, sum._shape);
        Assert.Equal(new[] { 10f, 12, 14, 16, 18, 20, 22, 24 }, sum.AsSpan().ToArray());
    }

    [Fact]
    public void ReduceSum_Rank3_Axis1_NotInnermost_GivesCorrectResult()
    {
        var grad = MakeAscending(new[] { 2, 4, 2 });
        // Sum over axis 1: result[i, k] = sum over j of grad[i, j, k]
        //   result[0, 0] = 1 + 3 + 5 + 7 = 16
        //   result[0, 1] = 2 + 4 + 6 + 8 = 20
        //   result[1, 0] = 9 + 11 + 13 + 15 = 48
        //   result[1, 1] = 10 + 12 + 14 + 16 = 52
        var sum = _engine.ReduceSum(grad, new[] { 1 }, keepDims: false);
        Assert.Equal(new[] { 2, 2 }, sum._shape);
        Assert.Equal(new[] { 16f, 20, 48, 52 }, sum.AsSpan().ToArray());
    }

    [Fact]
    public void ReduceSum_Rank3_Axis2_Innermost_GivesCorrectResult()
    {
        var grad = MakeAscending(new[] { 2, 4, 2 });
        // Sum over axis 2 (innermost — the fast path that was always correct):
        //   result[i, j] = grad[i, j, 0] + grad[i, j, 1]
        var sum = _engine.ReduceSum(grad, new[] { 2 }, keepDims: false);
        Assert.Equal(new[] { 2, 4 }, sum._shape);
        Assert.Equal(new[] { 3f, 7, 11, 15, 19, 23, 27, 31 }, sum.AsSpan().ToArray());
    }

    [Fact]
    public void ReduceSum_Rank3_MultiAxis_GivesCorrectResult()
    {
        var grad = MakeAscending(new[] { 2, 4, 2 });
        // Sum over axes [0, 1] keepDims=false:
        //   result[k] = sum over (i, j) of grad[i, j, k]
        //   result[0] = 1+3+5+7+9+11+13+15 = 64
        //   result[1] = 2+4+6+8+10+12+14+16 = 72
        var sum = _engine.ReduceSum(grad, new[] { 0, 1 }, keepDims: false);
        Assert.Equal(new[] { 2 }, sum._shape);
        Assert.Equal(new[] { 64f, 72 }, sum.AsSpan().ToArray());
    }

    [Fact]
    public void ReduceSum_Rank3_Axis0_KeepDims_GivesCorrectShape()
    {
        var grad = MakeAscending(new[] { 2, 4, 2 });
        var sum = _engine.ReduceSum(grad, new[] { 0 }, keepDims: true);
        Assert.Equal(new[] { 1, 4, 2 }, sum._shape);
        Assert.Equal(new[] { 10f, 12, 14, 16, 18, 20, 22, 24 }, sum.AsSpan().ToArray());
    }

    [Fact]
    public void ReduceSum_Rank4_MiddleAxis_GivesCorrectResult()
    {
        // Rank-4 [2, 3, 2, 2] = 24 elements 1..24.
        // Sum over axis 1: result[i, k, l] = sum_j of grad[i, j, k, l]
        // For i=0, k=0, l=0: grad[0,0,0,0] + grad[0,1,0,0] + grad[0,2,0,0]
        //   = data[0] + data[4] + data[8] = 1 + 5 + 9 = 15
        var grad = MakeAscending(new[] { 2, 3, 2, 2 });
        var sum = _engine.ReduceSum(grad, new[] { 1 }, keepDims: false);
        Assert.Equal(new[] { 2, 2, 2 }, sum._shape);
        Assert.Equal(15f, sum.AsSpan()[0]);  // [0, 0, 0]
        Assert.Equal(18f, sum.AsSpan()[1]);  // [0, 0, 1] = 2+6+10
        Assert.Equal(21f, sum.AsSpan()[2]);  // [0, 1, 0] = 3+7+11
        Assert.Equal(24f, sum.AsSpan()[3]);  // [0, 1, 1] = 4+8+12
    }

    [Fact]
    public void ReduceMean_Rank3_Axis0_NotInnermost_GivesCorrectResult()
    {
        // Same axis bug applied to ReduceMean — divides ReduceSum result by reduceSize.
        var grad = MakeAscending(new[] { 2, 4, 2 });
        var mean = _engine.ReduceMean(grad, new[] { 0 }, keepDims: false);
        Assert.Equal(new[] { 4, 2 }, mean._shape);
        // Mean = ReduceSum_result / 2 (since reduce dim is size 2)
        Assert.Equal(new[] { 5f, 6, 7, 8, 9, 10, 11, 12 }, mean.AsSpan().ToArray());
    }

    /// <summary>
    /// End-to-end integration: train a 2-layer FFN on rank-3 input
    /// (the original repro from FusedLinearBiasGradIntegrationTests).
    /// Loss must monotonically decrease — if any bias gradient is
    /// computed against the wrong reduction axis, SGD diverges.
    /// </summary>
    [Fact]
    public void TwoLayerFFL_Float_RankPromotedBias_LossDecreases()
    {
        const int B = 2, T = 4, FIn = 3, FH = 8, FOut = 2;
        var rng = new Random(99);

        var input = NewRandomFloat(new[] { B, T, FIn }, rng, 0.5f);
        var w1 = NewRandomFloat(new[] { FIn, FH }, rng, 0.1f);
        var b1 = NewRandomFloat(new[] { FH }, rng, 0.01f);
        var w2 = NewRandomFloat(new[] { FH, FOut }, rng, 0.1f);
        var b2 = NewRandomFloat(new[] { 1, FOut }, rng, 0.01f);
        var target = NewRandomFloat(new[] { B, T, FOut }, rng, 0.5f);

        float initialLoss = 0, finalLoss = 0;
        for (int step = 0; step < 5; step++)
        {
            using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
            var h = _engine.FusedLinear(input, w1, b1, FusedActivationType.ReLU);
            var y = _engine.FusedLinear(h, w2, b2, FusedActivationType.None);
            var diff = _engine.TensorSubtract(y, target);
            var sq = _engine.TensorMultiply(diff, diff);
            var loss = _engine.ReduceSum(sq, null);
            var grads = tape.ComputeGradients(loss, new[] { w1, b1, w2, b2 });

            if (step == 0) initialLoss = loss.AsSpan()[0];
            finalLoss = loss.AsSpan()[0];

            ApplyGradFloat(w1, grads[w1], 0.01f);
            ApplyGradFloat(b1, grads[b1], 0.01f);
            ApplyGradFloat(w2, grads[w2], 0.01f);
            ApplyGradFloat(b2, grads[b2], 0.01f);
        }

        Assert.True(finalLoss < initialLoss,
            $"Loss did not decrease (gradient direction wrong?): {initialLoss:G4} → {finalLoss:G4}");
    }

    private static Tensor<float> NewRandomFloat(int[] shape, Random rng, float scale)
    {
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)((rng.NextDouble() - 0.5) * 2 * scale);
        return t;
    }

    private static void ApplyGradFloat(Tensor<float> param, Tensor<float> grad, float lr)
    {
        var p = param.AsWritableSpan();
        var g = grad.AsSpan();
        for (int i = 0; i < p.Length; i++) p[i] -= lr * g[i];
    }
}
