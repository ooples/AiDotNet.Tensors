using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public sealed class GammaBackwardDomainTests
{
    [Fact]
    public void NonfiniteInputsAndPoles_HaveExplicitDerivatives_ForFloatingTypes()
    {
        AssertNonfiniteInputs<double>();
        AssertNonfiniteInputs<float>();
        AssertNonfiniteInputs<Half>();
        AssertNonfiniteInputs<BFloat16>();
    }

    private static void AssertNonfiniteInputs<T>()
    {
        double[] input = { double.NegativeInfinity, double.PositiveInfinity, double.NaN, -2, 0, -0.0 };
        var (lgamma, digamma) = Backward<T>(input);
        double[] expectedLgamma = { double.NaN, double.PositiveInfinity, double.NaN, double.NaN,
            double.NegativeInfinity, double.PositiveInfinity };
        double[] expectedDigamma = { double.NaN, 0, double.NaN, double.PositiveInfinity,
            double.PositiveInfinity, double.PositiveInfinity };
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < input.Length; i++)
        {
            AssertSpecialValue(expectedLgamma[i], ops.ToDouble(lgamma[i]));
            AssertSpecialValue(expectedDigamma[i], ops.ToDouble(digamma[i]));
        }
    }

    [Theory]
    [InlineData(1.0, -0.5772156649015328606, 1.6449340668482264365)]
    [InlineData(-0.5, 0.0364899739785765206, 8.9348022005446793094)]
    public void FiniteDerivatives_MatchIndependentGammaIdentities(double x, double psi, double trigamma)
    {
        var (lgamma, digamma) = Backward<double>(new[] { x }, upstream: -1.75);
        Assert.InRange(Math.Abs(lgamma[0] - -1.75 * psi), 0, 1e-12);
        // The existing positive-domain trigamma asymptotic truncates after x^-7.
        Assert.InRange(Math.Abs(digamma[0] - -1.75 * trigamma), 0, 1e-8);
    }

    [Fact]
    public void LargeNegativeArguments_UseReflectionWithoutWalkingTheNegativeDomain()
    {
        var (lgamma, digamma) = Backward<double>(new[] { -1000000000000.5, -1e100 });
        // At a negative half-integer cot(pi*x) is zero; psi(1-x) approaches log(1-x)-1/(2*(1-x)).
        Assert.InRange(Math.Abs(lgamma[0] - (Math.Log(1e12) + 1e-12)), 0, 1e-12);
        // psi'(x) + psi'(1-x) = pi^2 / sin^2(pi*x), with psi'(1-x) = 1e-12 + O(1e-24).
        Assert.InRange(Math.Abs(digamma[0] - (Math.PI * Math.PI - 1e-12)), 0, 1e-13);
        // All representable doubles this far from zero are integers, hence poles.
        Assert.True(double.IsNaN(lgamma[1]));
        Assert.Equal(double.PositiveInfinity, digamma[1]);
    }

    [Theory]
    [InlineData(-0.9999999999999999)]
    [InlineData(-1.0000000000000002)]
    [InlineData(-1.9999999999999998)]
    [InlineData(-2.0000000000000004)]
    public void Adjacent_negative_poles_preserve_relative_distance(double x)
    {
        var (lgamma, digamma) = Backward<double>(new[] { x });
        double distance = x - Math.Round(x);
        // psi(-n + d) = -1/d + O(1), psi'(-n + d) = 1/d^2 + O(1).
        // At these adjacent representable inputs the regular terms contribute less than the scaled tolerance.
        Assert.InRange(Math.Abs(lgamma[0] * distance + 1), 0, 2e-15);
        Assert.InRange(Math.Abs(digamma[0] * distance * distance - 1), 0, 2e-15);
    }

    private static (Tensor<T> Lgamma, Tensor<T> Digamma) Backward<T>(double[] values, double upstream = 1)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var data = new T[values.Length];
        var gradient = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            data[i] = ops.FromDouble(values[i]);
            gradient[i] = ops.FromDouble(upstream);
        }
        var x = new Tensor<T>(data, new[] { data.Length });
        var go = new Tensor<T>(gradient, new[] { data.Length });
        var output = new Tensor<T>(new[] { data.Length });
        var engine = new CpuEngine();
        var lgamma = new Dictionary<Tensor<T>, Tensor<T>>();
        var digamma = new Dictionary<Tensor<T>, Tensor<T>>();
        BackwardFunctions<T>.LgammaBackward(go, new[] { x }, output, Array.Empty<object>(), engine, lgamma);
        BackwardFunctions<T>.DigammaBackward(go, new[] { x }, output, Array.Empty<object>(), engine, digamma);
        return (lgamma[x], digamma[x]);
    }

    private static void AssertSpecialValue(double expected, double actual)
    {
        if (double.IsNaN(expected)) Assert.True(double.IsNaN(actual), $"Expected NaN, got {actual:R}");
        else Assert.Equal(expected, actual);
    }
}
