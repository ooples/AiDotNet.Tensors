#if NET5_0_OR_GREATER
using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// <see cref="AccurateTanh"/> against the correctly rounded double-precision tanh. An exhaustive run over
/// all 2^32 float bit patterns measured a maximum error of 1 ULP with exact NaN / ±0 / ±∞ handling; this
/// test sweeps every 251st bit pattern (about 17M inputs, every exponent and both signs) plus the edges.
/// </summary>
public class AccurateTanhTests
{
    [SkippableFact]
    public unsafe void MatchesDoubleTanh_WithinTwoUlp_AcrossTheFloatRange()
    {
        Skip.IfNot(AccurateTanh.IsSupported, "AVX2/FMA not available.");
        const uint Stride = 251;
        const int Batch = 1 << 16;
        long total = (1L << 32) / Stride;
        int batches = (int)((total + Batch - 1) / Batch);
        long worst = 0;
        uint worstBits = 0;
        object gate = new();

        Parallel.For(0, batches, batch =>
        {
            var input = new float[Batch];
            var output = new float[Batch];
            int count = 0;
            for (long k = (long)batch * Batch; k < Math.Min(total, (long)(batch + 1) * Batch); k++)
                input[count++] = BitConverter.UInt32BitsToSingle((uint)(k * Stride));
            fixed (float* pi = input, po = output) AccurateTanh.Tanh(pi, po, count);

            long localWorst = 0;
            uint localBits = 0;
            for (int i = 0; i < count; i++)
            {
                long ulp = UlpError(input[i], output[i]);
                if (ulp > localWorst) { localWorst = ulp; localBits = BitConverter.SingleToUInt32Bits(input[i]); }
            }
            lock (gate) if (localWorst > worst) { worst = localWorst; worstBits = localBits; }
        });

        Assert.True(worst <= 2,
            $"max error {worst} ULP at x = {BitConverter.UInt32BitsToSingle(worstBits):R}");
    }

    [SkippableTheory]
    [InlineData(0f)]
    [InlineData(-0f)]
    [InlineData(1e-45f)]          // smallest denormal
    [InlineData(-1e-30f)]
    [InlineData(0.6249999f)]      // just below the polynomial / exponential switch
    [InlineData(0.625f)]
    [InlineData(9f)]              // tanh(9) rounds to 0.99999994, not 1
    [InlineData(10f)]
    [InlineData(-88.7f)]
    [InlineData(float.MaxValue)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    [InlineData(float.NaN)]
    public unsafe void EdgeValues_MatchDoubleTanhExactlyOrWithinOneUlp(float x)
    {
        Skip.IfNot(AccurateTanh.IsSupported, "AVX2/FMA not available.");
        // Fill a full vector so the SIMD path (not the scalar tail) handles the value.
        var input = new float[16];
        var output = new float[16];
        Array.Fill(input, x);
        fixed (float* pi = input, po = output) AccurateTanh.Tanh(pi, po, input.Length);

        Assert.True(UlpError(x, output[0]) <= 1,
            $"tanh({x:R}) = {output[0]:R}, expected {(float)Math.Tanh(x):R}");
    }

    private static long UlpError(float x, float got)
    {
        float want = (float)Math.Tanh(x);
        if (float.IsNaN(want)) return float.IsNaN(got) ? 0 : long.MaxValue;
        if (float.IsNaN(got)) return long.MaxValue;
        // Distinguish -0 from +0: tanh is odd, so the sign of zero must be preserved.
        int wantBits = BitConverter.SingleToInt32Bits(want), gotBits = BitConverter.SingleToInt32Bits(got);
        if (want == 0 && got == 0) return wantBits == gotBits ? 0 : long.MaxValue;
        if (Math.Sign(want) != Math.Sign(got)) return long.MaxValue;
        return Math.Abs((long)gotBits - wantBits);
    }
}
#endif
