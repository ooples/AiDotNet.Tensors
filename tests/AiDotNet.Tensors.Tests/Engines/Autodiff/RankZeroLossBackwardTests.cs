using System;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Guards backpropagation through a RANK-0 loss — the shape produced by any full reduction with
/// <c>keepDims: false</c>.
/// </summary>
/// <remarks>
/// <para>
/// ORIGIN: AiDotNet's DiceLoss reduced over all axes with <c>keepDims: false</c> (giving rank-0
/// <c>[]</c> tensors) and then divided them. The FORWARD divide was fine, but the BACKWARD was not:
/// the tape seeds a rank-1 <c>[1]</c> gradient, and <c>DivideBackward</c> passed that straight into
/// <c>TensorDivide</c> against a rank-0 operand, throwing
/// "Tensor shapes must match. Got [1] and []" from inside <c>ComputeGradients</c>.
/// </para>
/// <para>
/// That made the loss unusable for TRAINING, and the failure mode was badly misleading: the
/// exception surfaces mid-training-step, so callers saw "no parameters changed after training" and
/// blamed the model or the gradients rather than the loss. A sweep of all 245 backward functions
/// found the same unguarded pattern in Multiply, Log, Exp and ComplexMagnitudeSquared; all are now
/// routed through AlignGradRank.
/// </para>
/// <para>
/// Each test asserts gradients are produced, FINITE, and numerically CORRECT — not merely that no
/// exception is thrown, so a future "fix" that silently zeroes the gradient still fails here.
/// </para>
/// </remarks>
[Collection("EngineCurrentGlobalState")]
public class RankZeroLossBackwardTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<double> Param(params double[] values)
        => new Tensor<double>(values, new[] { values.Length });

    /// <summary>Full reduction with keepDims:false — the shape that triggered the bug.</summary>
    private Tensor<double> Rank0Sum(Tensor<double> t)
        => _engine.ReduceSum(t, new[] { 0 }, keepDims: false);

    [Fact]
    public void Divide_WithRankZeroOperands_Backpropagates()
    {
        var a = Param(2.0, 4.0);
        var b = Param(1.0, 3.0);

        using var tape = new GradientTape<double>();
        var num = Rank0Sum(a);                            // 6, shape []
        var den = Rank0Sum(b);                            // 4, shape []
        var loss = _engine.TensorDivide(num, den);        // 1.5

        Assert.Equal(0, loss.Shape.Length);

        var grads = tape.ComputeGradients(loss, new[] { a, b });

        // d(sum(a)/sum(b))/da_i = 1/sum(b) = 0.25
        var ga = grads[a];
        for (int i = 0; i < ga.Length; i++)
        {
            Assert.True(!double.IsNaN(ga[i]) && !double.IsInfinity(ga[i]), $"grad a[{i}] not finite: {ga[i]}");
            Assert.True(Math.Abs(ga[i] - 0.25) < 1e-9, $"grad a[{i}]={ga[i]}, expected 0.25");
        }

        // d/db_i = -sum(a)/sum(b)^2 = -6/16 = -0.375
        var gb = grads[b];
        for (int i = 0; i < gb.Length; i++)
        {
            Assert.True(!double.IsNaN(gb[i]) && !double.IsInfinity(gb[i]), $"grad b[{i}] not finite: {gb[i]}");
            Assert.True(Math.Abs(gb[i] + 0.375) < 1e-9, $"grad b[{i}]={gb[i]}, expected -0.375");
        }
    }

    [Fact]
    public void Multiply_WithRankZeroOperands_Backpropagates()
    {
        var a = Param(2.0, 3.0);
        var b = Param(4.0, 1.0);

        using var tape = new GradientTape<double>();
        var loss = _engine.TensorMultiply(Rank0Sum(a), Rank0Sum(b));   // 5 * 5 = 25

        Assert.Equal(0, loss.Shape.Length);

        var grads = tape.ComputeGradients(loss, new[] { a, b });

        // d(sum(a)*sum(b))/da_i = sum(b) = 5, and symmetrically for b.
        var ga = grads[a];
        for (int i = 0; i < ga.Length; i++)
            Assert.True(Math.Abs(ga[i] - 5.0) < 1e-9, $"grad a[{i}]={ga[i]}, expected 5");
        var gb = grads[b];
        for (int i = 0; i < gb.Length; i++)
            Assert.True(Math.Abs(gb[i] - 5.0) < 1e-9, $"grad b[{i}]={gb[i]}, expected 5");
    }

    [Fact]
    public void Log_WithRankZeroOperand_Backpropagates()
    {
        var a = Param(1.0, 3.0);

        using var tape = new GradientTape<double>();
        var loss = _engine.TensorLog(Rank0Sum(a));    // ln 4

        Assert.Equal(0, loss.Shape.Length);

        var grads = tape.ComputeGradients(loss, new[] { a });

        // d(log(sum(a)))/da_i = 1/sum(a) = 0.25
        var ga = grads[a];
        for (int i = 0; i < ga.Length; i++)
        {
            Assert.True(!double.IsNaN(ga[i]) && !double.IsInfinity(ga[i]));
            Assert.True(Math.Abs(ga[i] - 0.25) < 1e-9, $"grad a[{i}]={ga[i]}, expected 0.25");
        }
    }

    [Fact]
    public void Exp_WithRankZeroOperand_Backpropagates()
    {
        var a = Param(0.5, 0.25);

        using var tape = new GradientTape<double>();
        var loss = _engine.TensorExp(Rank0Sum(a));    // e^0.75

        Assert.Equal(0, loss.Shape.Length);

        var grads = tape.ComputeGradients(loss, new[] { a });

        // d(exp(sum(a)))/da_i = exp(sum(a))
        double expected = Math.Exp(0.75);
        var ga = grads[a];
        for (int i = 0; i < ga.Length; i++)
        {
            Assert.True(!double.IsNaN(ga[i]) && !double.IsInfinity(ga[i]));
            // 1e-6 matches the tolerance the neighbouring AbsSoftplusGradientRepro tests use: the
            // engine evaluates exp at ~float32 precision (measured 2.117000102996826 against the
            // exact 2.117000016612675), which is a precision characteristic, not a gradient error.
            Assert.True(Math.Abs(ga[i] - expected) < 1e-6, $"grad a[{i}]={ga[i]}, expected {expected}");
        }
    }

    /// <summary>
    /// The composite shape that broke AiDotNet's SAM/SAM2: one loss term reduces to rank-1 <c>[1]</c>
    /// (FocalLoss) and another to rank-0 <c>[]</c> (DiceLoss), and the two are summed.
    /// </summary>
    [Fact]
    public void MixedRankLossTerms_CanBeCombinedAndBackpropagated()
    {
        var a = Param(1.0, 2.0, 3.0);

        using var tape = new GradientTape<double>();
        var rank0 = Rank0Sum(a);                                        // shape []
        var rank1 = _engine.ReduceSum(a, new[] { 0 }, keepDims: true);  // shape [1]

        Assert.Equal(0, rank0.Shape.Length);
        Assert.Equal(1, rank1.Shape.Length);

        // Aligning then combining is exactly what a composite objective must do.
        var loss = _engine.TensorAdd(_engine.Reshape(rank0, new[] { 1 }), rank1);   // 2 * sum(a)

        var grads = tape.ComputeGradients(loss, new[] { a });

        // d(2*sum(a))/da_i = 2
        var ga = grads[a];
        for (int i = 0; i < ga.Length; i++)
        {
            Assert.True(!double.IsNaN(ga[i]) && !double.IsInfinity(ga[i]));
            Assert.True(Math.Abs(ga[i] - 2.0) < 1e-9, $"grad a[{i}]={ga[i]}, expected 2");
        }
    }

    /// <summary>
    /// A rank-0 loss reached through a CHAIN (divide then log), so the aligned gradient has to
    /// survive more than one backward hop.
    /// </summary>
    [Fact]
    public void ChainedRankZeroOps_Backpropagate()
    {
        var a = Param(2.0, 2.0);
        var b = Param(1.0, 1.0);

        using var tape = new GradientTape<double>();
        var ratio = _engine.TensorDivide(Rank0Sum(a), Rank0Sum(b));   // 4/2 = 2
        var loss = _engine.TensorLog(ratio);                          // ln 2

        Assert.Equal(0, loss.Shape.Length);

        var grads = tape.ComputeGradients(loss, new[] { a, b });

        // d(log(sum(a)/sum(b)))/da_i = 1/sum(a) = 0.25 ; d/db_i = -1/sum(b) = -0.5
        var ga = grads[a];
        for (int i = 0; i < ga.Length; i++)
            Assert.True(Math.Abs(ga[i] - 0.25) < 1e-9, $"grad a[{i}]={ga[i]}, expected 0.25");
        var gb = grads[b];
        for (int i = 0; i < gb.Length; i++)
            Assert.True(Math.Abs(gb[i] + 0.5) < 1e-9, $"grad b[{i}]={gb[i]}, expected -0.5");
    }
}
