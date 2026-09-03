#if !NETFRAMEWORK

using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

public sealed class CpuSigmoidKernelTests
{
    [Fact]
    public unsafe void ResolvedKernel_WritesTheCompleteNonVectorAlignedBuffer()
    {
        const int length = 65;
        var input = new float[length];
        var output = new float[length];
        Array.Fill(output, float.NaN);
        for (int i = 0; i < length; i++)
            input[i] = (i - 32) * 0.125f;

        CpuSigmoidKernel kernel = CpuSigmoidKernel.Resolve(length);
        fixed (float* inputPtr = input)
        fixed (float* outputPtr = output)
            kernel.Invoke(inputPtr, outputPtr);

        for (int i = 0; i < length; i++)
            Assert.False(float.IsNaN(output[i]), $"index {i} was not written");

        float expectedTail = 1.0f / (1.0f + MathF.Exp(-input[^1]));
        Assert.Equal(expectedTail, output[^1]);
    }
}

#endif
