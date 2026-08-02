// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public sealed class CodegenOutputAgreementTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void ShapeMismatch_FailsClosedWithoutIndexingEitherArray(int candidateLength)
    {
        bool agrees = CodegenOutputAgreement.Agrees(
            new float[candidateLength], new float[2], 1e-3,
            out double deviation, out long worstIndex,
            out float actual, out float expected);

        Assert.False(agrees);
        Assert.Equal(double.PositiveInfinity, deviation);
        Assert.Equal(-1, worstIndex);
        Assert.Equal(0, actual);
        Assert.Equal(0, expected);
    }

    [Fact]
    public void EqualShapes_UseTheRequestedRelativeTolerance()
    {
        float[] reference = { 2f, -4f };

        Assert.True(CodegenOutputAgreement.Agrees(
            new[] { 2f, -4.002f }, reference, 1e-3,
            out double accepted, out _, out _, out _));
        Assert.False(CodegenOutputAgreement.Agrees(
            new[] { 2f, -4.02f }, reference, 1e-3,
            out double rejected, out long worstIndex,
            out float actual, out float expected));
        Assert.True(accepted <= 1e-3);
        Assert.True(rejected > 1e-3);
        Assert.Equal(1, worstIndex);
        Assert.Equal(-4.02f, actual);
        Assert.Equal(-4f, expected);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void NonFiniteValue_FailsClosedAndReportsItsLocation(bool candidateIsNonFinite)
    {
        float[] candidate = { 1f, candidateIsNonFinite ? float.NaN : 2f };
        float[] reference = { 1f, candidateIsNonFinite ? 2f : float.PositiveInfinity };

        bool agrees = CodegenOutputAgreement.Agrees(
            candidate, reference, 1e-3,
            out double deviation, out long worstIndex,
            out float actual, out float expected);

        Assert.False(agrees);
        Assert.Equal(double.PositiveInfinity, deviation);
        Assert.Equal(1, worstIndex);
        Assert.Equal(candidate[1], actual);
        Assert.Equal(reference[1], expected);
    }
}
