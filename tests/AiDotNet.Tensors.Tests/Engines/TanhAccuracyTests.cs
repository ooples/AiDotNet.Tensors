#if NET5_0_OR_GREATER
using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The vectorized float tanh and mish (used when MKL VML is absent) must stay inside their op-parity ULP
/// budgets (16 and 64) everywhere, give the same bits for a value wherever it sits in a buffer, and get
/// the special values right.
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

    private static double MishRef(double x) => x * Math.Tanh(x > 20 ? x : Math.Log(1 + Math.Exp(x)));

    /// <summary>
    /// Evaluates <paramref name="kernel"/> on every 256th float bit pattern in [0, <paramref name="limit"/>]
    /// and its negation, eight at a time, and returns the worst ULP error against <paramref name="reference"/>
    /// (values within <paramref name="absFloor"/> of a reference below 1e-3 are exempt). The kernel is
    /// called directly, so a host with MKL VML still tests the vector code, and nothing large is allocated.
    /// </summary>
    private static (long Ulp, float At, int Count) Sweep(Func<Vector256<float>, Vector256<float>> kernel,
        Func<double, double> reference, float limit, double absFloor)
    {
        long worst = 0; float at = 0; int count = 0;
        uint maxBits = BitConverter.SingleToUInt32Bits(limit);
        var lanes = new float[8];
        int filled = 0;
        for (uint bits = 0; bits <= maxBits; bits += 256)
        {
            float v = BitConverter.UInt32BitsToSingle(bits);
            lanes[filled++] = v;
            lanes[filled++] = -v;
            if (filled < 8) continue;
            filled = 0;
            var y = kernel(Vector256.Create(lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]));
            for (int k = 0; k < 8; k++)
            {
                double expected = reference(lanes[k]);
                float got = y.GetElement(k);
                count++;
                if (Math.Abs(expected) < 1e-3 && Math.Abs(expected - got) <= absFloor) continue;
                long d = UlpDistance((float)expected, got);
                if (d > worst) { worst = d; at = lanes[k]; }
            }
        }
        return (worst, at, count);
    }

    [SkippableFact]
    public void AccurateTanh256_IsWithinOneUlpOfDoubleTanh()
    {
        Skip.IfNot(Avx2.IsSupported && Fma.IsSupported, "The vector kernel needs AVX2 and FMA.");
        var (ulp, at, count) = Sweep(SimdKernels.AccurateTanh256, Math.Tanh, 20f, 0);
        _out.WriteLine($"tanh: max ULP {ulp} at x = {at:R} over {count} values");
        Assert.True(ulp <= 1, $"max ULP {ulp} at x = {at:R}");
    }

    [SkippableFact]
    public void AccurateMish256_IsWithinItsParityBudget()
    {
        Skip.IfNot(Avx2.IsSupported && Fma.IsSupported, "The vector kernel needs AVX2 and FMA.");
        var (ulp, at, count) = Sweep(SimdKernels.AccurateMish256, MishRef, 30f, 1e-6);
        _out.WriteLine($"mish: max ULP {ulp} (|mish| >= 1e-3, else 1e-6 abs) at x = {at:R} over {count} values");
        Assert.True(ulp <= 8, $"max ULP {ulp} at x = {at:R}");
    }

    private static readonly float[] Specials =
        { 0f, -0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, 9f, -9f, 25f, -25f, -100f, 1e-30f, -1e-30f };

    /// <summary>
    /// Runs <paramref name="kernel"/> on buffers of every length from 1 to 15 with each special value at every
    /// position, so the value lands in vector lanes and in the tail, and requires the same bits each time.
    /// </summary>
    private static unsafe float[] SameBitsAtEveryPosition(delegate*<float*, float*, int, void> kernel, string name)
    {
        var first = new float[Specials.Length];
        for (int s = 0; s < Specials.Length; s++)
        {
            int? bits = null;
            for (int len = 1; len <= 15; len++)
                for (int pos = 0; pos < len; pos++)
                {
                    var input = new float[len];
                    for (int k = 0; k < len; k++) input[k] = 0.5f;
                    input[pos] = Specials[s];
                    var output = new float[len];
                    fixed (float* pi = input)
                    fixed (float* po = output)
                    {
                        kernel(pi, po, len);
                    }
                    int got = BitConverter.SingleToInt32Bits(output[pos]);
                    if (bits is null) { bits = got; first[s] = output[pos]; continue; }
                    Assert.True(bits == got,
                        $"{name}({Specials[s]:R}) gave {BitConverter.Int32BitsToSingle(bits.Value):R} and {output[pos]:R} at length {len}, position {pos}");
                }
        }
        return first;
    }

    [Fact]
    public unsafe void TanhUnsafe_SpecialValues_AreRight_AtEveryPosition()
    {
        var y = SameBitsAtEveryPosition(&SimdKernels.TanhUnsafe, "tanh");
        Assert.Equal(BitConverter.SingleToInt32Bits(0f), BitConverter.SingleToInt32Bits(y[0]));
        Assert.Equal(BitConverter.SingleToInt32Bits(-0f), BitConverter.SingleToInt32Bits(y[1]));
        Assert.Equal(1f, y[2]);
        Assert.Equal(-1f, y[3]);
        Assert.True(float.IsNaN(y[4]));
        Assert.Equal((float)Math.Tanh(9), y[5]);
        Assert.Equal((float)Math.Tanh(-9), y[6]);
        Assert.Equal(1f, y[7]);
        Assert.Equal(-1f, y[8]);
        Assert.Equal(-1f, y[9]);
        Assert.Equal(1e-30f, y[10]);
        Assert.Equal(-1e-30f, y[11]);
    }

    [Fact]
    public unsafe void MishUnsafe_SpecialValues_AreRight_AtEveryPosition()
    {
        var y = SameBitsAtEveryPosition(&SimdKernels.MishUnsafe, "mish");
        Assert.Equal(BitConverter.SingleToInt32Bits(0f), BitConverter.SingleToInt32Bits(y[0]));
        Assert.Equal(BitConverter.SingleToInt32Bits(-0f), BitConverter.SingleToInt32Bits(y[1]));
        Assert.Equal(float.PositiveInfinity, y[2]);
        Assert.Equal(0f, y[3]); // mish(-infinity) = -0
        Assert.True(float.IsNaN(y[4]));
        Assert.Equal(25f, y[7]);
        Assert.True(Math.Abs(y[8]) <= 1e-6f, $"mish(-25) = {y[8]}");
        Assert.Equal(0f, y[9]);
    }

    [Fact]
    public void EngineTanh_MatchesDoubleTanh()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 4, 300 });
        for (int i = 0; i < x.Length; i++) x[i] = (i - 600) * 0.0137f;
        var y = engine.Tanh(x);
        for (int i = 0; i < x.Length; i++)
            Assert.True(UlpDistance((float)Math.Tanh(x[i]), y[i]) <= 1, $"x = {x[i]}: got {y[i]}");
    }
}
#endif