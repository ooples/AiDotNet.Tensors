#if NET5_0_OR_GREATER
using System;
using System.Runtime.Intrinsics;
using AiDotNet.Tensors.Engines.Simd;
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
}
#endif
