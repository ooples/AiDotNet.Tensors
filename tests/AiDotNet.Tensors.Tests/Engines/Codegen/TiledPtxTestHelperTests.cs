// Copyright (c) AiDotNet. All rights reserved.

using Xunit;
using Xunit.Sdk;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public sealed class TiledPtxTestHelperTests
{
    [Fact]
    public void AssertClose_AcceptsExactAndEmptyOutputs()
    {
        TiledPtxTestHelper.AssertClose(new[] { 1.0 }, new[] { 1f }, 0, "exact");
        TiledPtxTestHelper.AssertClose(
            System.Array.Empty<double>(), System.Array.Empty<float>(), 0, "empty");
    }

    [Fact]
    public void AssertClose_RejectsLengthMismatchBeforeIndexing()
    {
        Assert.Throws<EqualException>(() =>
            TiledPtxTestHelper.AssertClose(new[] { 1.0 }, [], 1e-3, "shape"));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AssertClose_RejectsNonFiniteValues(bool nonFiniteReference)
    {
        double[] expected = { nonFiniteReference ? double.NaN : 1.0 };
        float[] actual = { nonFiniteReference ? 1f : float.PositiveInfinity };

        Assert.ThrowsAny<XunitException>(() =>
            TiledPtxTestHelper.AssertClose(expected, actual, 1e-3, "finite"));
    }
}
