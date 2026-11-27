using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// <see cref="IEngine.TensorSoftShrink{T}"/>: PyTorch's softshrink, <c>x - lambda</c> above lambda, <c>x + lambda</c> below
/// -lambda and 0 between, with the gradient passing only where |x| &gt; lambda.
/// </summary>
public class SoftShrinkTests
{
    private readonly CpuEngine _engine = new();

    // torch.nn.functional.softshrink, written out: where(x > l, x - l, where(x < -l, x + l, 0)).
    private static double Reference(double x, double lambda)
        => x > lambda ? x - lambda : x < -lambda ? x + lambda : double.IsNaN(x) ? double.NaN : 0.0;

    // 37 elements: the SIMD body and a scalar tail; values on both sides of and exactly at the boundary.
    private static readonly double[] Inputs =
    {
        -3.0, -0.5000001, -0.5, -0.4999, -0.25, -0.0, 0.0, 0.25, 0.4999, 0.5, 0.5000001, 3.0,
        1e-30, -1e-30, 7.5, -7.5, 0.75, -0.75, 1.0, -1.0, 2.5, -2.5, 0.51, -0.51, 0.49, -0.49,
        100.0, -100.0, 0.125, -0.125, 0.875, -0.875, 4.0, -4.0, 0.6, -0.6, 1.5,
    };

    [Fact]
    public void Float_MatchesTheReferenceAcrossTheSimdBodyAndTail()
    {
        var x = new Tensor<float>(new[] { Inputs.Length });
        for (int i = 0; i < Inputs.Length; i++) x[i] = (float)Inputs[i];

        var y = _engine.TensorSoftShrink(x, 0.5f);

        for (int i = 0; i < Inputs.Length; i++)
        {
            float expected = (float)Reference((float)Inputs[i], 0.5);
            Assert.Equal(MathCompat.SingleToInt32Bits(expected), MathCompat.SingleToInt32Bits(y[i]));
        }
    }

    [Fact]
    public void Double_MatchesTheReferenceAcrossTheSimdBodyAndTail()
    {
        var x = new Tensor<double>(new[] { Inputs.Length });
        for (int i = 0; i < Inputs.Length; i++) x[i] = Inputs[i];

        var y = _engine.TensorSoftShrink(x, 0.5);

        for (int i = 0; i < Inputs.Length; i++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(Reference(Inputs[i], 0.5)), BitConverter.DoubleToInt64Bits(y[i]));
    }

    [Fact]
    public void NonFiniteInputs_FollowTorch()
    {
        var x = new Tensor<float>(new[] { 3 });
        x[0] = float.NaN; x[1] = float.PositiveInfinity; x[2] = float.NegativeInfinity;

        var y = _engine.TensorSoftShrink(x, 0.5f);

        Assert.True(float.IsNaN(y[0]));
        Assert.Equal(float.PositiveInfinity, y[1]);
        Assert.Equal(float.NegativeInfinity, y[2]);
    }

    [Fact]
    public void ShrunkValues_ArePositiveZero()
    {
        // torch returns +0 for every |x| <= lambda, -0 included.
        var x = new Tensor<float>(new[] { 3 });
        x[0] = -0.0f; x[1] = -0.3f; x[2] = 0.5f;

        var y = _engine.TensorSoftShrink(x, 0.5f);

        for (int i = 0; i < 3; i++) Assert.Equal(0, MathCompat.SingleToInt32Bits(y[i]));
    }

    [Fact]
    public void Half_UsesItsOwnArithmetic()
    {
#if NET5_0_OR_GREATER
        var x = new Tensor<Half>(new[] { 4 });
        x[0] = (Half)1.5f; x[1] = (Half)(-1.5f); x[2] = (Half)0.25f; x[3] = (Half)(-0.75f);

        var y = _engine.TensorSoftShrink(x, (Half)0.5f);

        Assert.Equal((Half)1.0f, y[0]);
        Assert.Equal((Half)(-1.0f), y[1]);
        Assert.Equal((Half)0.0f, y[2]);
        Assert.Equal((Half)(-0.25f), y[3]);
#endif
    }

    [Fact]
    public void ZeroLambda_IsTheIdentityAwayFromZero()
    {
        var x = new Tensor<double>(new[] { 3 });
        x[0] = -2.0; x[1] = 0.0; x[2] = 3.0;

        var y = _engine.TensorSoftShrink(x, 0.0);

        Assert.Equal(-2.0, y[0]);
        Assert.Equal(0.0, y[1]);
        Assert.Equal(3.0, y[2]);
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    public void InvalidLambda_Throws(double lambda)
    {
        var x = new Tensor<double>(new[] { 2 });
        Assert.Throws<ArgumentOutOfRangeException>(() => _engine.TensorSoftShrink(x, lambda));
    }

    [Fact]
    public void Gradient_PassesWhereTheMagnitudeExceedsLambda()
    {
        var x = new Tensor<double>(new[] { 6 });
        double[] values = { -2.0, -0.5, -0.2, 0.2, 0.5, 2.0 };
        for (int i = 0; i < values.Length; i++) x[i] = values[i];

        using var tape = new GradientTape<double>();
        var y = _engine.TensorSoftShrink(x, 0.5);
        var loss = _engine.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { x });

        // The boundary |x| == lambda is in the flat part, as torch's backward (x > l or x < -l) has it.
        double[] expected = { 1, 0, 0, 0, 0, 1 };
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], grads[x][i]);
    }

    [Fact]
    public void StridedInput_IsReadInLogicalOrder()
    {
        var m = new Tensor<float>(new[] { 2, 3 });
        for (int i = 0; i < 6; i++) m[i] = i - 2.5f;   // -2.5 .. 2.5
        var transposed = _engine.TensorTranspose(m);    // [3, 2], not contiguous

        var y = _engine.TensorSoftShrink(transposed, 1.0f);

        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 2; c++)
                Assert.Equal((float)Reference(transposed[r, c], 1.0), y[r, c]);
    }
}
