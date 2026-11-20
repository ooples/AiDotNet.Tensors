using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("CompilationGlobalState")]
public sealed class RoundedWeightNormTests
{
    [Fact]
    public void NativeCpuReferenceNormPreservesDirectionIdentity()
    {
        // Frozen PyTorch2.14.1 CPU control, git5c488690, AVX2, eight threads.
        // The LCG emits exact FP32 input bits; its native row norm is0x4089c7cc.
        // The expected result comes from the independent native normalization,
        // rather than recomputing this implementation's reduction in the test.
        var values = new float[224];
        uint state = 1234;
        for (int i = 0; i < values.Length; i++)
        {
            state = unchecked(state * 1664525u + 1013904223u);
            values[i] = (state >> 8) * (1f / 16777216f) - .5f;
        }
        float nativeNorm = BitConverter.Int32BitsToSingle(0x4089c7cc);
        bool previous = CpuEngine.WeightNormRoundedNormEnabled;
        try
        {
            CpuEngine.WeightNormRoundedNormEnabled = true;
            using var direction = new Tensor<float>(values, new[] { 1, 224 });
            using var gain = new Tensor<float>(new[] { nativeNorm }, new[] { 1 });
            using var normalized = new CpuEngine().TensorWeightNorm(direction, gain, epsilon: 0);
            Assert.Equal(values.Select(BitConverter.SingleToInt32Bits), normalized.ToArray().Select(BitConverter.SingleToInt32Bits));
        }
        finally { CpuEngine.WeightNormRoundedNormEnabled = previous; }
    }
}
