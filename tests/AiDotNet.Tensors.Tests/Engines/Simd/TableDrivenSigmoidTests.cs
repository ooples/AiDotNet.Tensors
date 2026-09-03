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
            -8.0f,
            -7.999f,
            -7.9375f,
            7.875f,
            7.9375f,
            7.999f,
            8.0f,
            float.PositiveInfinity,
            float.NaN,
            -0.125f,
            0.125f,
            -3.25f,
            3.25f,
            -1.0f,
            1.0f,
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
            input[i] = 7.75f + 0.249f * i / (length - 1);

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
}

#endif
