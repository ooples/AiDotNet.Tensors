#if !NETFRAMEWORK

using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// <see cref="SimdKernels.SigmoidUnsafe"/> computes the same bits on every CPU vendor.
/// </summary>
/// <remarks>
/// An Intel-only table-driven sigmoid used to be selected when the CPU reported fast gathers. It
/// was accurate, but its values differed from the polynomial every other CPU ran, so a model trained
/// differently on Intel than on AMD: RepViT-SAM's ten-step conformance training diverged on Intel
/// runners and converged on AMD ones. Pinning every aligned eight-lane block to
/// <see cref="SimdKernels.FastSigmoid256"/> fails if any vendor-specific path returns.
/// </remarks>
public sealed class SigmoidVendorIndependenceTests
{
    [Theory]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(1000)]
    [InlineData(4096)]
    public unsafe void EveryVectorBlockMatchesThePortablePolynomial(int length)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        var rng = new Random(20260930 + length);
        var input = new float[length];
        for (int i = 0; i < length; i++) input[i] = (float)(rng.NextDouble() * 40.0 - 20.0);
        var output = new float[length];

        fixed (float* pIn = input)
        fixed (float* pOut = output)
        {
            SimdKernels.SigmoidUnsafe(pIn, pOut, length);

            for (int block = 0; block + 8 <= length; block += 8)
            {
                var expected = SimdKernels.FastSigmoid256(Avx.LoadVector256(pIn + block));
                for (int lane = 0; lane < 8; lane++)
                {
                    Assert.True(
                        BitConverter.SingleToInt32Bits(expected.GetElement(lane)) ==
                        BitConverter.SingleToInt32Bits(output[block + lane]),
                        $"length {length}, element {block + lane} (x={input[block + lane]:R}): " +
                        $"got {output[block + lane]:R}, portable polynomial gives {expected.GetElement(lane):R}");
                }
            }
        }
    }

    [Fact]
    public unsafe void SaturatesInfinitiesAndPropagatesNaN()
    {
        float[] input =
        [
            float.NegativeInfinity, -100f, -16f, -8f, -0.125f, 0f,
            0.125f, 8f, 16f, 100f, float.PositiveInfinity, float.NaN,
        ];
        var output = new float[input.Length];

        fixed (float* pIn = input)
        fixed (float* pOut = output)
        {
            SimdKernels.SigmoidUnsafe(pIn, pOut, input.Length);
        }

        Assert.True(output[0] >= 0f && output[0] <= 1e-37f, $"sigmoid(-inf) = {output[0]:R}");
        Assert.Equal(1f, output[10]);
        Assert.True(float.IsNaN(output[11]), $"NaN input produced {output[11]:R}");
        for (int i = 1; i < 10; i++)
        {
            double exact = 1.0 / (1.0 + Math.Exp(-input[i]));
            Assert.True(Math.Abs(output[i] - exact) <= 1e-6,
                $"sigmoid({input[i]:R}) = {output[i]:R}, exact {exact:R}");
        }
    }
}

#endif
