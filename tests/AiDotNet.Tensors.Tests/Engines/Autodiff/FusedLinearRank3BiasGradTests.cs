using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Regression tests for issue #234 — FusedLinear backward's bias gradient
/// shape on rank-3+ inputs and on Tensor&lt;double&gt;.
///
/// <para>The fast path (float, rank-2 inputs, BLAS available) handled bias
/// reduction correctly. The fallback only reduced axis 0, leaving any
/// non-batch leading dim (e.g. the time axis in [B, T, F]) unreduced.
/// Result: bias gradient came out of the tape with shape [T, F] when the
/// bias was [F] or [1, F], and the next optimizer step crashed with a
/// shape mismatch.</para>
///
/// <para>The fallback fires whenever any of the following is true:
/// T is double (BLAS path is float-only), input rank ≥ 3, weight rank ≥ 3,
/// BLAS unavailable, or gradOutput non-contiguous. Transformers and
/// sequence-modelling stacks with double precision hit it on every
/// FusedLinear in their feed-forward block.</para>
/// </summary>
public class FusedLinearRank3BiasGradTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    /// <summary>
    /// Minimal repro from #234: rank-3 input [1, 8, 1], weights [1, 24],
    /// bias [1, 24] → y [1, 8, 24]. Bias-grad must match bias shape.
    /// </summary>
    [Fact]
    public void FusedLinearBackward_Rank3Double_BiasGradMatchesBiasShape()
    {
        var input = new Tensor<double>(new[] { 1, 8, 1 });
        var weights = new Tensor<double>(new[] { 1, 24 });
        var bias = new Tensor<double>(new[] { 1, 24 });

        // Fill so gradients aren't trivially zero.
        var inSpan = input.AsWritableSpan();
        for (int i = 0; i < inSpan.Length; i++) inSpan[i] = 0.1 * (i + 1);
        var wSpan = weights.AsWritableSpan();
        for (int i = 0; i < wSpan.Length; i++) wSpan[i] = 0.01 * (i + 1);
        var bSpan = bias.AsWritableSpan();
        for (int i = 0; i < bSpan.Length; i++) bSpan[i] = 0.001 * (i + 1);

        using var tape = new GradientTape<double>();
        var y = _engine.FusedLinear(input, weights, bias, FusedActivationType.None);
        Assert.Equal(new[] { 1, 8, 24 }, y._shape);

        var loss = _engine.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { input, weights, bias });

        // The bug: bias-grad came back as [8, 24]. The fix: must match the
        // bias's own shape so the downstream optimizer-step TensorAdd
        // ("Tensor shapes must match. Got [1, 24] and [8, 24]") doesn't
        // throw.
        Assert.Equal(bias._shape, grads[bias]._shape);
    }

    /// <summary>
    /// dL/dbias must match PyTorch's reference: the gradient is the sum
    /// of gradOutput over every leading axis. With loss = sum(y) and y =
    /// x @ W + bias broadcast, dL/dbias[f] = batch * seq.
    /// </summary>
    [Fact]
    public void FusedLinearBackward_Rank3Double_BiasGradHasCorrectValues()
    {
        const int B = 2, T = 5, FIn = 3, FOut = 4;
        var input = new Tensor<double>(new[] { B, T, FIn });
        var weights = new Tensor<double>(new[] { FIn, FOut });
        var bias = new Tensor<double>(new[] { FOut });

        // Doesn't matter what the values are — dL/dbias under loss=sum(y)
        // is constant: each output element contributes 1 to its bias slot,
        // so dL/dbias[f] = B * T for every f.
        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++)
            input.AsWritableSpan()[i] = rng.NextDouble();
        for (int i = 0; i < weights.Length; i++)
            weights.AsWritableSpan()[i] = rng.NextDouble();

        using var tape = new GradientTape<double>();
        var y = _engine.FusedLinear(input, weights, bias, FusedActivationType.None);
        var loss = _engine.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { input, weights, bias });

        Assert.Equal(bias._shape, grads[bias]._shape);

        double expected = B * T;
        var biasGradSpan = grads[bias].AsSpan();
        for (int i = 0; i < biasGradSpan.Length; i++)
            Assert.True(Math.Abs(biasGradSpan[i] - expected) < 1e-9,
                $"bias-grad[{i}] = {biasGradSpan[i]}, expected {expected}");
    }

    /// <summary>
    /// Tensor&lt;double&gt; on plain rank-2 inputs also goes through the
    /// fallback (fast path is float-only). Verify it produces the right
    /// bias-grad shape and value here too.
    /// </summary>
    [Fact]
    public void FusedLinearBackward_Rank2Double_FallbackBiasGrad()
    {
        const int M = 3, K = 4, N = 5;
        var input = new Tensor<double>(new[] { M, K });
        var weights = new Tensor<double>(new[] { K, N });
        var bias = new Tensor<double>(new[] { N });

        var rng = new Random(7);
        for (int i = 0; i < input.Length; i++)
            input.AsWritableSpan()[i] = rng.NextDouble();
        for (int i = 0; i < weights.Length; i++)
            weights.AsWritableSpan()[i] = rng.NextDouble();

        using var tape = new GradientTape<double>();
        var y = _engine.FusedLinear(input, weights, bias, FusedActivationType.None);
        var loss = _engine.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { input, weights, bias });

        Assert.Equal(bias._shape, grads[bias]._shape);

        // dL/dbias[f] = M (each row contributes 1).
        var biasGradSpan = grads[bias].AsSpan();
        for (int i = 0; i < biasGradSpan.Length; i++)
            Assert.True(Math.Abs(biasGradSpan[i] - M) < 1e-9,
                $"bias-grad[{i}] = {biasGradSpan[i]}, expected {M}");
    }

    /// <summary>
    /// Bias declared as [1, F] (the LagLlama / MOIRAI pattern) must come
    /// back with the same shape — leading 1s preserved.
    /// </summary>
    [Fact]
    public void FusedLinearBackward_Rank3Double_BiasWithLeadingOnes_PreservesShape()
    {
        var input = new Tensor<double>(new[] { 1, 8, 1 });
        var weights = new Tensor<double>(new[] { 1, 24 });
        var bias = new Tensor<double>(new[] { 1, 24 });
        for (int i = 0; i < input.Length; i++) input.AsWritableSpan()[i] = 0.5;
        for (int i = 0; i < weights.Length; i++) weights.AsWritableSpan()[i] = 0.5;

        using var tape = new GradientTape<double>();
        var y = _engine.FusedLinear(input, weights, bias, FusedActivationType.None);
        var loss = _engine.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { input, weights, bias });

        Assert.Equal(new[] { 1, 24 }, grads[bias]._shape);
    }
}
