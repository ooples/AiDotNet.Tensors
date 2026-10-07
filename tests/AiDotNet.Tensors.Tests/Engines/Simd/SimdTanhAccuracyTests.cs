using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// The float tanh kernel (vectorized with AVX2+FMA, libm otherwise) against double-precision tanh: a bounded ULP
/// error across the whole float range, exact special values, and results that do not depend on an element's
/// position in the call (so any split of a buffer -- parallel chunks, fused tiles -- produces the same bits).
/// </summary>
public class SimdTanhAccuracyTests
{
    /// <summary>The vector kernel's measured maximum over all 2^32 inputs is 1 ULP (see SimdKernels.Tanh256). Without
    /// AVX2+FMA the kernel is the platform libm, held to a looser bound. The op-parity contract allows 64.</summary>
    private static long MaxUlp =>
#if NET5_0_OR_GREATER
        System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported ? 1 : 4;
#else
        4;
#endif

    private static long UlpDistance(float actual, double exact)
    {
        float e = (float)exact;
        if (float.IsNaN(e)) return float.IsNaN(actual) ? 0 : long.MaxValue;
        if (actual == e) return 0;
        int ia = TestHelpers.MathCompat.SingleToInt32Bits(actual);
        int ie = TestHelpers.MathCompat.SingleToInt32Bits(e);
        if ((ia < 0) != (ie < 0)) return long.MaxValue;
        return Math.Abs((long)ia - ie);
    }

    [Fact]
    public unsafe void TanhUnsafe_IsWithinMaxUlp_OverASweepOfEveryBinade()
    {
        const int count = 1 << 16;
        var input = new float[count];
        var output = new float[count];
        long worst = 0;
        float worstX = 0;
        // Bit patterns stepped by an odd stride cover every exponent of both signs with varied mantissas.
        for (long start = 0; start <= uint.MaxValue; start += (long)count * 997)
        {
            for (int i = 0; i < count; i++)
            {
                long bits = start + (long)i * 997;
                input[i] = BitsToSingle((uint)(bits & 0xFFFFFFFF));
            }
            fixed (float* pi = input, po = output) SimdKernels.TanhUnsafe(pi, po, count);
            for (int i = 0; i < count; i++)
            {
                if (float.IsNaN(input[i])) { Assert.True(float.IsNaN(output[i]), $"tanh(NaN) = {output[i]:R}"); continue; }
                long ulp = UlpDistance(output[i], Math.Tanh(input[i]));
                if (ulp > worst) { worst = ulp; worstX = input[i]; }
            }
        }
        Assert.True(worst <= MaxUlp, $"max error {worst} ULP at x = {worstX:R} (tanh = {Math.Tanh(worstX):R})");
    }

    [Fact]
    public unsafe void TanhUnsafe_SpecialValues()
    {
        var input = new[] { 0f, -0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, 1e-30f, -1e-30f,
            float.Epsilon, -float.Epsilon, 9f, -9f, 20f, -20f, 1e30f, -1e30f, 0.625f, -0.625f, 0.62499994f };
        var output = new float[input.Length];
        fixed (float* pi = input, po = output) SimdKernels.TanhUnsafe(pi, po, input.Length);
        Assert.Equal(0, TestHelpers.MathCompat.SingleToInt32Bits(output[0]));
        Assert.Equal(TestHelpers.MathCompat.SingleToInt32Bits(-0f), TestHelpers.MathCompat.SingleToInt32Bits(output[1]));
        Assert.Equal(1f, output[2]);
        Assert.Equal(-1f, output[3]);
        Assert.True(float.IsNaN(output[4]));
        Assert.Equal(1e-30f, output[5]);
        Assert.Equal(-1e-30f, output[6]);
        Assert.Equal(float.Epsilon, output[7]);
        Assert.Equal(-float.Epsilon, output[8]);
        for (int i = 9; i < input.Length; i++)
            Assert.True(UlpDistance(output[i], Math.Tanh(input[i])) <= MaxUlp, $"tanh({input[i]:R}) = {output[i]:R}, expected {Math.Tanh(input[i]):R}");
    }

    [Fact]
    public unsafe void TanhUnsafe_ResultDoesNotDependOnPosition()
    {
        var rng = new Random(7);
        var input = new float[1003];
        for (int i = 0; i < input.Length; i++) input[i] = (float)((rng.NextDouble() * 2 - 1) * 12);
        var whole = new float[input.Length];
        var single = new float[input.Length];
        fixed (float* pi = input, pw = whole, ps = single)
        {
            SimdKernels.TanhUnsafe(pi, pw, input.Length);
            for (int i = 0; i < input.Length; i++) SimdKernels.TanhUnsafe(pi + i, ps + i, 1);
        }
        for (int i = 0; i < input.Length; i++)
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(whole[i]) == TestHelpers.MathCompat.SingleToInt32Bits(single[i]),
                $"element {i}: whole-buffer {whole[i]:R} vs single-element {single[i]:R}");
    }

    private static unsafe float BitsToSingle(uint bits) => *(float*)&bits;
}
