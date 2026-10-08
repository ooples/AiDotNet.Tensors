#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The vectorized float tanh (used when MKL VML is absent) must stay inside the op-parity ULP budget
/// for Tanh (16) everywhere, without the near-zero cancellation of the old 2*sigmoid(2x)-1 form.
/// </summary>
public class TanhAccuracyTests
{
    private readonly ITestOutputHelper _out;
    public TanhAccuracyTests(ITestOutputHelper o) => _out = o;

    private static long UlpDistance(float a, float b)
    {
        int ia = BitConverter.SingleToInt32Bits(a), ib = BitConverter.SingleToInt32Bits(b);
        if (ia < 0) ia = int.MinValue - ia;
        if (ib < 0) ib = int.MinValue - ib;
        return Math.Abs((long)ia - ib);
    }

    [Fact]
    public unsafe void TanhUnsafe_IsWithinAFewUlpOfDoubleTanh_AcrossTheRange()
    {
        // Every 64th float bit pattern in [0, 20], both signs: about 16.7M values, all exponents.
        uint maxBits = BitConverter.SingleToUInt32Bits(20f);
        int count = (int)(maxBits / 64) + 1;
        var input = new float[2 * count];
        for (int i = 0; i < count; i++)
        {
            float v = BitConverter.UInt32BitsToSingle((uint)i * 64);
            input[2 * i] = v;
            input[2 * i + 1] = -v;
        }
        var output = new float[input.Length];
        fixed (float* pi = input)
        fixed (float* po = output)
        {
            SimdKernels.TanhUnsafe(pi, po, input.Length);
        }

        long worst = 0; float worstAt = 0;
        for (int i = 0; i < input.Length; i++)
        {
            float expected = (float)Math.Tanh(input[i]);
            long d = UlpDistance(expected, output[i]);
            if (d > worst) { worst = d; worstAt = input[i]; }
        }
        _out.WriteLine($"max ULP {worst} at x = {worstAt:R} over {input.Length} values");
        Assert.True(worst <= 4, $"max ULP {worst} at x = {worstAt:R}");
    }

    [Fact]
    public unsafe void TanhUnsafe_HandlesSpecialValues()
    {
        var input = new[] { 0f, -0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, 9f, -9f, 1e-30f, -1e-30f };
        var output = new float[input.Length];
        fixed (float* pi = input)
        fixed (float* po = output)
        {
            SimdKernels.TanhUnsafe(pi, po, input.Length);
        }
        Assert.Equal(BitConverter.SingleToInt32Bits(0f), BitConverter.SingleToInt32Bits(output[0]));
        Assert.Equal(BitConverter.SingleToInt32Bits(-0f), BitConverter.SingleToInt32Bits(output[1]));
        Assert.Equal(1f, output[2]);
        Assert.Equal(-1f, output[3]);
        Assert.True(float.IsNaN(output[4]));
        Assert.Equal((float)Math.Tanh(9), output[5]);
        Assert.Equal((float)Math.Tanh(-9), output[6]);
        Assert.Equal(1e-30f, output[7]);
        Assert.Equal(-1e-30f, output[8]);
    }

    [Fact]
    public void EngineTanh_MatchesDoubleTanh()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 4, 300 });
        for (int i = 0; i < x.Length; i++) x[i] = (i - 600) * 0.0137f;
        var y = engine.Tanh(x);
        for (int i = 0; i < x.Length; i++)
            Assert.True(UlpDistance((float)Math.Tanh(x[i]), y[i]) <= 4, $"x = {x[i]}: got {y[i]}");
    }
}

#endif
