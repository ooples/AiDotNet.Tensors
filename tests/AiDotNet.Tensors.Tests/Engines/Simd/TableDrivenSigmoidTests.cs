#if !NETFRAMEWORK

using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

public sealed class TableDrivenSigmoidTests
{
    [Fact]
    public unsafe void SigmoidArray_VectorPathPreservesInterpolationAndSaturationContract()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        float[] input =
        [
            float.NegativeInfinity,
            -16.0f,
            -15.999f,
            -15.9375f,
            -8.0f,
            8.0f,
            15.875f,
            15.9375f,
            15.999f,
            16.0f,
            float.PositiveInfinity,
            float.NaN,
            -0.125f,
            0.125f,
            -3.25f,
            3.25f,
        ];
        var output = new float[input.Length];

        fixed (float* inputPtr = input)
        fixed (float* outputPtr = output)
            TableDrivenSigmoid.SigmoidArray(inputPtr, outputPtr, input.Length);

        for (int i = 0; i < input.Length; i++)
        {
            float expected = TableDrivenSigmoid.Sigmoid(input[i]);
            if (float.IsNaN(expected))
            {
                Assert.True(float.IsNaN(output[i]), $"index {i}: expected NaN, got {output[i]:R}");
                continue;
            }

            Assert.True(
                MathF.Abs(expected - output[i]) <= 2.5e-7f,
                $"index {i}, x={input[i]:R}: scalar={expected:R}, vector={output[i]:R}");
        }
    }

    [Fact]
    public unsafe void SigmoidArray_UpperInterpolationCellsStayWithinAdvertisedAccuracy()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        const int length = 1024;
        var input = new float[length];
        var output = new float[length];
        for (int i = 0; i < length; i++)
            input[i] = 15.75f + 0.249f * i / (length - 1);

        fixed (float* inputPtr = input)
        fixed (float* outputPtr = output)
            TableDrivenSigmoid.SigmoidArray(inputPtr, outputPtr, input.Length);

        for (int i = 0; i < length; i++)
        {
            float expected = 1.0f / (1.0f + MathF.Exp(-input[i]));
            Assert.True(
                MathF.Abs(expected - output[i]) <= 2.2e-6f,
                $"index {i}, x={input[i]:R}: expected={expected:R}, actual={output[i]:R}");
        }
    }

    [Theory]
    [InlineData(-16f)]
    [InlineData(-14.375474f)]
    [InlineData(14.375474f)]
    [InlineData(16f)]
    public unsafe void SigmoidArray_CentralRangeDoesNotPrematurelySaturate(float value)
    {
        var input = new float[8];
        var output = new float[8];
        Array.Fill(input, value);

        fixed (float* inputPtr = input)
        fixed (float* outputPtr = output)
            TableDrivenSigmoid.SigmoidArray(inputPtr, outputPtr, input.Length);

        float expected = 1.0f / (1.0f + MathF.Exp(-value));
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(output[i] > 0f && output[i] < 1f,
                $"table sigmoid({value}) = {output[i]} — central inputs must not saturate");
            Assert.True(MathF.Abs(output[i] - expected) < 1e-6f,
                $"table sigmoid({value}) = {output[i]}, expected {expected}");
        }
    }
}

#endif
