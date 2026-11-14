using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class GradCheckAdversarialTests
{
    private static Tensor<double> T(double value) => new(new[] { value }, new[] { 1 });

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void NumericalGradient_RejectsInvalidSteps(double epsilon)
    {
        var input = T(3);
        Assert.Throws<ArgumentOutOfRangeException>(() => GradCheck.NumericalGradient(x => x[0], input, epsilon));
        Assert.Equal(3.0, input[0]);
    }

    [Fact]
    public void Check_RejectsAFalsePassWhenFloatPerturbationsRoundToTheSameValue()
    {
        var input = new Tensor<float>(new[] { 1e20f }, new[] { 1 });
        Assert.Throws<InvalidOperationException>(() => GradCheck.Check(
            x => new Tensor<float>(x.ToArray(), new[] { 1 }), input));
        Assert.Equal(1e20f, input[0]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Check_RejectsAnEmptyElementBudget(int count)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => GradCheck.Check(x => x, T(1),
            new GradCheckOptions { MaxElementsPerInput = count }));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(-1.0)]
    public void Compare_RejectsInvalidTolerances(double tolerance)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => GradCheck.Compare(T(0), T(1), tolerance, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => GradCheck.Compare(T(0), T(1), 0, tolerance));
    }

    [Theory]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(double.NaN)]
    public void Compare_NonFiniteNumericalDerivativeCannotPass(double derivative)
    {
        var result = GradCheck.Compare(T(0), T(derivative), 1e-5, 1e-3);
        Assert.False(result.Passed);
        Assert.Single(result.WorstMismatches);
        Assert.Equal(double.PositiveInfinity, result.MaxAbsoluteError);
        Assert.Equal(double.PositiveInfinity, result.MaxRelativeError);
    }
}
