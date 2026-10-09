using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// FusedOptimizer.TrySumOfSquaresHost / TryScaleHost: the parallel SIMD passes behind eager global-norm clipping and
/// the NaN/Inf gradient probe. Checked against a serial double loop on multi-chunk lengths, on an offset view, and on
/// non-finite input.
/// </summary>
public class GradNormHostTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(13)]
    [InlineData(200_003)]
    public void SumOfSquares_Float_MatchesSerialDouble(int n)
    {
        var rng = new Random(n);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 4 - 2);
        double expected = 0;
        foreach (var v in a) expected += (double)v * v;
        Assert.True(FusedOptimizer.TrySumOfSquaresHost(new Tensor<float>(a, new[] { n }), out double got));
        Assert.Equal(expected, got, expected * 1e-12);
    }

    [Fact]
    public void SumOfSquares_Double_OnOffsetView()
    {
        var a = new double[2 * 70_001];
        for (int i = 0; i < a.Length; i++) a[i] = i < 70_001 ? 1000.0 : 0.5;
        var view = new Tensor<double>(a, new[] { 2, 70_001 }).Slice(0, 1, 2);
        Assert.True(view.IsContiguous);
        Assert.True(FusedOptimizer.TrySumOfSquaresHost(view, out double got));
        Assert.Equal(70_001 * 0.25, got, 1e-6);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void SumOfSquares_IsNonFinite_WhenAnyElementIs(float bad)
    {
        var a = new float[150_000];
        a[149_999] = bad;
        Assert.True(FusedOptimizer.TrySumOfSquaresHost(new Tensor<float>(a, new[] { a.Length }), out double got));
        Assert.True(double.IsNaN(got) || double.IsInfinity(got));
    }

    [Fact]
    public void SumOfSquares_LargestFloat_StaysFinite()
    {
        var a = new float[16];
        for (int i = 0; i < a.Length; i++) a[i] = float.MaxValue;
        Assert.True(FusedOptimizer.TrySumOfSquaresHost(new Tensor<float>(a, new[] { 16 }), out double got));
        Assert.False(double.IsInfinity(got));
    }

    [Fact]
    public void Scale_Float_OnOffsetView_TouchesOnlyTheView()
    {
        var a = new float[2 * 100_000];
        for (int i = 0; i < a.Length; i++) a[i] = i;
        var view = new Tensor<float>(a, new[] { 2, 100_000 }).Slice(0, 1, 2);
        long v0 = view.Version;
        Assert.True(FusedOptimizer.TryScaleHost(view, 0.5));
        Assert.True(view.Version > v0, "scale must bump the version");
        for (int i = 0; i < 100_000; i++) Assert.Equal(i, a[i]);
        for (int i = 100_000; i < a.Length; i++) Assert.Equal(i * 0.5f, a[i]);
    }

    [Fact]
    public void UnsupportedElementType_ReturnsFalse()
    {
        var t = new Tensor<int>(new[] { 1, 2, 3 }, new[] { 3 });
        Assert.False(FusedOptimizer.TrySumOfSquaresHost(t, out _));
        Assert.False(FusedOptimizer.TryScaleHost(t, 2.0));
    }
}
