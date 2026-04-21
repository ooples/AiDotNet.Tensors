using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Regression tests for issue #233 — <c>CpuEngine.BatchNormBackward&lt;T&gt;</c>
/// crashed on rank-3 inputs because forward routes 3D tensors through a
/// dedicated <c>BatchNorm3D</c> helper that treats <c>_shape[0]</c> as the
/// channel axis (mean/variance have length <c>_shape[0]</c>), but the
/// matching backward path only branched on <c>rank == 4</c>. Rank-3
/// inputs fell through to the 2D path which assumes
/// <c>_shape[1]</c> is the feature axis and indexes <c>meanData[f]</c> for
/// <c>f ∈ [0, _shape[1])</c> — out of bounds whenever <c>_shape[1] &gt; _shape[0]</c>.
/// </summary>
public class BatchNormBackward3DTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    /// <summary>
    /// The minimal repro from the issue: <c>[1, 8, 24]</c> double tensor,
    /// trivial gamma/beta. Pre-fix this threw IndexOutOfRangeException on
    /// the backward call; post-fix it returns shape-compatible gradients.
    /// </summary>
    [Fact]
    public void BatchNormBackward_Rank3Double_DoesNotCrash()
    {
        var input = new Tensor<double>(new[] { 1, 8, 24 });
        var gamma = new Tensor<double>(new[] { 1 });
        var beta = new Tensor<double>(new[] { 1 });

        for (int i = 0; i < input.Length; i++) input.AsWritableSpan()[i] = 0.5;
        gamma.AsWritableSpan()[0] = 1.0;
        beta.AsWritableSpan()[0] = 0.0;

        using var tape = new GradientTape<double>();
        var y = _engine.BatchNorm(input, gamma, beta, 1e-5,
                                  out var mean, out var variance);
        var loss = _engine.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { input, gamma, beta });

        Assert.Equal(input._shape, grads[input]._shape);
        Assert.Equal(gamma._shape, grads[gamma]._shape);
        Assert.Equal(beta._shape, grads[beta]._shape);
    }

    /// <summary>
    /// 3D backward gradient values must match a finite-difference reference.
    /// Shape <c>[C, H, W] = [3, 4, 5]</c> exercises non-trivial channels and
    /// spatial extent so the per-channel sum reduction is actually tested.
    /// </summary>
    [Fact]
    public void BatchNormBackward_Rank3Double_GradientMatchesFiniteDifference()
    {
        const int C = 3, H = 4, W = 5;
        var input = new Tensor<double>(new[] { C, H, W });
        var gamma = new Tensor<double>(new[] { C });
        var beta = new Tensor<double>(new[] { C });

        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++)
            input.AsWritableSpan()[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < C; i++)
        {
            gamma.AsWritableSpan()[i] = 0.5 + rng.NextDouble();
            beta.AsWritableSpan()[i] = rng.NextDouble() - 0.5;
        }

        // Analytical gradients via tape.
        using var tape = new GradientTape<double>();
        var y = _engine.BatchNorm(input, gamma, beta, 1e-5, out _, out _);
        var loss = _engine.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { input, gamma, beta });

        var dInputAnalytical = grads[input].AsSpan();
        var dGammaAnalytical = grads[gamma].AsSpan();
        var dBetaAnalytical = grads[beta].AsSpan();

        // Finite-difference reference for gamma + beta (cheap — only C=3 evaluations).
        const double h = 1e-5;
        for (int c = 0; c < C; c++)
        {
            double original = gamma.AsSpan()[c];
            gamma.AsWritableSpan()[c] = original + h;
            double lossPlus = LossOf(input, gamma, beta);
            gamma.AsWritableSpan()[c] = original - h;
            double lossMinus = LossOf(input, gamma, beta);
            gamma.AsWritableSpan()[c] = original;
            double finite = (lossPlus - lossMinus) / (2 * h);
            Assert.True(Math.Abs(finite - dGammaAnalytical[c]) < 1e-3,
                $"d(loss)/d(gamma[{c}]): finite={finite:G6}, analytical={dGammaAnalytical[c]:G6}");
        }
        for (int c = 0; c < C; c++)
        {
            double original = beta.AsSpan()[c];
            beta.AsWritableSpan()[c] = original + h;
            double lossPlus = LossOf(input, gamma, beta);
            beta.AsWritableSpan()[c] = original - h;
            double lossMinus = LossOf(input, gamma, beta);
            beta.AsWritableSpan()[c] = original;
            double finite = (lossPlus - lossMinus) / (2 * h);
            Assert.True(Math.Abs(finite - dBetaAnalytical[c]) < 1e-3,
                $"d(loss)/d(beta[{c}]): finite={finite:G6}, analytical={dBetaAnalytical[c]:G6}");
        }
        // Spot-check input gradient at a handful of positions to keep the
        // test fast (full 60-element sweep would do 120 forward passes).
        int[] probes = { 0, 7, 13, 28, 41, 59 };
        foreach (int idx in probes)
        {
            double original = input.AsSpan()[idx];
            input.AsWritableSpan()[idx] = original + h;
            double lossPlus = LossOf(input, gamma, beta);
            input.AsWritableSpan()[idx] = original - h;
            double lossMinus = LossOf(input, gamma, beta);
            input.AsWritableSpan()[idx] = original;
            double finite = (lossPlus - lossMinus) / (2 * h);
            Assert.True(Math.Abs(finite - dInputAnalytical[idx]) < 1e-3,
                $"d(loss)/d(input[{idx}]): finite={finite:G6}, analytical={dInputAnalytical[idx]:G6}");
        }
    }

    private double LossOf(Tensor<double> input, Tensor<double> gamma, Tensor<double> beta)
    {
        // No tape — pure forward. ReduceSum returns Tensor<double>; element[0] is the scalar.
        var y = _engine.BatchNorm(input, gamma, beta, 1e-5, out _, out _);
        var loss = _engine.ReduceSum(y, null);
        return loss.AsSpan()[0];
    }
}
