#if NET5_0_OR_GREATER
using System;
using System.Runtime.Intrinsics;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Verifies FastExp256ForSoftmax (4-term Horner) stays within the
/// expected ~1e-4 relative error band vs MathF.Exp across the typical
/// softmax input range. Lower precision than FastExp256 (1e-6) but fine
/// for softmax because the normalisation rescales per-element error.
///
/// <para>Guarded with NET5_0_OR_GREATER because the kernel (and its
/// <see cref="Vector256{T}"/> inputs) only compile on NET5+. On net471
/// the tests don't apply.</para>
/// </summary>
public class FastExp256ForSoftmaxTests
{
    [Theory]
    [InlineData(-5.0f)]
    [InlineData(-1.0f)]
    [InlineData(-0.1f)]
    [InlineData(0.0f)]
    [InlineData(0.1f)]
    [InlineData(1.0f)]
    [InlineData(5.0f)]
    public void MatchesMathFExpWithin1e3(float center)
    {
        // Test 8 values clustered around center
        var values = new float[8];
        for (int i = 0; i < 8; i++) values[i] = center + (i - 4) * 0.05f;
        var v = Vector256.Create(values[0], values[1], values[2], values[3],
                                  values[4], values[5], values[6], values[7]);

        var actual = SimdKernels.FastExp256ForSoftmax(v);
        for (int i = 0; i < 8; i++)
        {
            float expected = MathF.Exp(values[i]);
            float got = actual.GetElement(i);
            float relErr = MathF.Abs(got - expected) / expected;
            Assert.True(relErr < 1e-3,
                $"FastExp256ForSoftmax({values[i]}) = {got}, MathF.Exp = {expected}, rel err = {relErr:F6}");
        }
    }

    [Fact]
    public void SoftmaxOutputMatchesReferenceWithin1Percent()
    {
        // End-to-end: run SimdKernels.Softmax and verify per-row sum=1
        // and per-element rel error vs scalar MathF.Exp reference is small.
        const int rows = 32, cols = 256;
        var rng = new Random(0xE40);
        var input = new float[rows * cols];
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(rng.NextDouble() * 6.0 - 3.0);

        var output = new float[rows * cols];
        SimdKernels.Softmax(input, output, rows, cols);

        // Reference: scalar MathF.Exp softmax
        var reference = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            int off = r * cols;
            float max = float.NegativeInfinity;
            for (int c = 0; c < cols; c++) if (input[off + c] > max) max = input[off + c];
            float sum = 0;
            for (int c = 0; c < cols; c++)
            {
                reference[off + c] = MathF.Exp(input[off + c] - max);
                sum += reference[off + c];
            }
            for (int c = 0; c < cols; c++) reference[off + c] /= sum;
        }

        // Per-row sum must be ≈ 1 (within float error)
        for (int r = 0; r < rows; r++)
        {
            float sum = 0;
            for (int c = 0; c < cols; c++) sum += output[r * cols + c];
            Assert.True(MathF.Abs(sum - 1f) < 1e-3, $"row {r} sum = {sum}");
        }

        // Per-element max relative error vs reference
        float maxRelErr = 0;
        for (int i = 0; i < output.Length; i++)
        {
            if (reference[i] > 1e-6f)
            {
                float rel = MathF.Abs(output[i] - reference[i]) / reference[i];
                if (rel > maxRelErr) maxRelErr = rel;
            }
        }
        Assert.True(maxRelErr < 0.01, $"max relative error vs scalar reference = {maxRelErr:F6}");
    }

    // The AVX-512 and AVX2 softmax rows must give the same bits: their exps are the same polynomial lane for lane,
    // so only the reduction of the per-lane sums can differ. Both must reduce as the AVX-512 path does, where lane
    // group g (elements 8g..8g+7 of each 32) pairs with group g+2: (g0 + g2) + (g1 + g3). This reference runs on
    // either path, so an AVX-512 runner checks the AVX-512 row and an AVX2 machine the AVX2 row against one order.
    [SkippableTheory]
    [InlineData(64)]
    [InlineData(2048)]
    [InlineData(2047)]
    public unsafe void SoftmaxRow_ReducesInTheAvx512Order_OnEveryPath(int length)
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported,
            "the vector softmax row needs AVX2 and FMA");
        var rng = RandomHelper.CreateSeededRandom(0x5AB + length);
        for (int row = 0; row < 16; row++)
        {
            var input = new float[length];
            for (int i = 0; i < length; i++) input[i] = (float)(rng.NextDouble() * 16.0 - 8.0);
            var output = new float[length];
            fixed (float* pi = input, po = output) SimdKernels.SoftmaxRowUnsafe(pi, po, length);

            float max = float.NegativeInfinity;
            foreach (float v in input) if (v > max) max = v;
            var vmax = Vector256.Create(max);
            var g = new Vector256<float>[4];
            var exps = new float[length];
            int simdLen = length & ~31;
            for (int i = 0; i < simdLen; i += 32)
                for (int k = 0; k < 4; k++)
                {
                    var e = SimdKernels.FastExp256ForSoftmax(Vector256.Create(input, i + 8 * k) - vmax);
                    e.CopyTo(exps, i + 8 * k);
                    g[k] += e;
                }
            float sum = SimdKernels.HorizontalSum((g[0] + g[2]) + (g[1] + g[3]));
            for (int i = simdLen; i < length; i++) { exps[i] = MathF.Exp(input[i] - max); sum += exps[i]; }
            float inv = 1f / sum;
            for (int i = 0; i < length; i++)
            {
                float expected = exps[i] * inv;
                if (BitConverter.SingleToInt32Bits(output[i]) != BitConverter.SingleToInt32Bits(expected))
                    Assert.Fail($"row {row}, element {i}: expected {expected:R}, got {output[i]:R}");
            }
        }
    }
}
#endif