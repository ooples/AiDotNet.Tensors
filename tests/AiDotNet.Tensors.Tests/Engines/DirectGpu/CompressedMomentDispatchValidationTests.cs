using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public class CompressedMomentDispatchValidationTests
{
    [Theory]
    [InlineData(-1)]
    [InlineData(8388608)]
    [InlineData(int.MaxValue)]
    public void WorkSizeRejectsOverflow(int blocks)
        => Assert.Throws<ArgumentOutOfRangeException>(() => CompressedMomentBufferValidation.CheckedInt8WorkSize(blocks));

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(8388607)]
    public void WorkSizeAcceptsRepresentableValues(int blocks)
        => Assert.Equal((long)blocks * 256, CompressedMomentBufferValidation.CheckedInt8WorkSize(blocks));

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    [InlineData(-0.1f)]
    [InlineData(1.1f)]
    public void FactorsRejectInvalidValues(float invalid)
    {
        for (int i = 0; i < 4; i++)
        {
            var values = new[] { 0.1f, 0.01f, 0.1f, 0.01f };
            values[i] = invalid;
            Assert.Throws<ArgumentOutOfRangeException>(() => CompressedMomentHyperparameters.ValidateInt8Factors(values[0], values[1], values[2], values[3]));
        }
    }

    [Fact]
    public void FactorsValidateEndpoints()
    {
        CompressedMomentHyperparameters.ValidateInt8Factors(0f, 1f, 1f, 1f);
        Assert.Throws<ArgumentOutOfRangeException>(() => CompressedMomentHyperparameters.ValidateInt8Factors(0.1f, 0.1f, 0f, 1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => CompressedMomentHyperparameters.ValidateInt8Factors(0.1f, 0.1f, 1f, 0f));
    }
}
