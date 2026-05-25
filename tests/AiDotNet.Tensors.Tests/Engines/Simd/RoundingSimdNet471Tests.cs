using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Parity tests for the SIMD Floor/Ceiling kernels (audit-2026-05 phase 5 §D).
/// On net471 these exercise the new BCL Vector&lt;float&gt; truncate-and-correct
/// path (the BCL has no Vector&lt;T&gt;.Floor before .NET 7). Edge cases cover
/// negatives, exact integers, values past 2^23 (already integer-valued in
/// float), ±Inf, and a non-multiple-of-lane-width tail.
/// </summary>
public unsafe class RoundingSimdNet471Tests
{
    // Raw IEEE-754 bit comparison — `==` treats +0.0 and -0.0 as equal, so it
    // can't catch a signed-zero regression (floor(-0.0)=-0.0, ceil(-0.5)=-0.0).
    // net471 has no SingleToInt32Bits, so round-trip via GetBytes/ToInt32.
    private static int BitsF(float v) => BitConverter.ToInt32(BitConverter.GetBytes(v), 0);
    private static long BitsD(double v) => BitConverter.DoubleToInt64Bits(v);

    private static float[] EdgeCases() => new[]
    {
        0f, -0f, 0.3f, -0.3f, 2.7f, -2.7f, 3.0f, -3.0f, 5f, -5f,
        0.9999999f, -0.9999999f, 1.0000001f, -1.0000001f,
        123.456f, -123.456f, 8388608f /*2^23*/, -8388608f,
        16777216f /*2^24*/, -16777216f, 1e7f, -1e7f,
        float.PositiveInfinity, float.NegativeInfinity,
        // pad to exercise lanes + a tail (37 total)
        1.5f, -1.5f, 2.5f, -2.5f, 100.1f, -100.1f, 0.001f, -0.001f,
        42f, -42f, 7.7f, -7.7f, 9.2f,
    };

    [Fact]
    public void Floor_Float_MatchesMathF()
    {
        var input = EdgeCases();
        var got = new float[input.Length];
        fixed (float* ip = input, op = got)
            SimdKernels.FloorUnsafe(ip, op, input.Length);

        for (int i = 0; i < input.Length; i++)
        {
            float expected = MathF.Floor(input[i]);
            if (float.IsNaN(expected)) Assert.True(float.IsNaN(got[i]));
            else Assert.True(BitsF(got[i]) == BitsF(expected),
                $"Floor({input[i]:G9}) = {got[i]:G9} (0x{BitsF(got[i]):X8}), expected {expected:G9} (0x{BitsF(expected):X8})");
        }
    }

    [Fact]
    public void Ceiling_Float_MatchesMathF()
    {
        var input = EdgeCases();
        var got = new float[input.Length];
        fixed (float* ip = input, op = got)
            SimdKernels.CeilingUnsafe(ip, op, input.Length);

        for (int i = 0; i < input.Length; i++)
        {
            float expected = MathF.Ceiling(input[i]);
            if (float.IsNaN(expected)) Assert.True(float.IsNaN(got[i]));
            else Assert.True(BitsF(got[i]) == BitsF(expected),
                $"Ceiling({input[i]:G9}) = {got[i]:G9} (0x{BitsF(got[i]):X8}), expected {expected:G9} (0x{BitsF(expected):X8})");
        }
    }

    private static double[] EdgeCasesD() => new[]
    {
        0d, -0d, 0.3, -0.3, 2.7, -2.7, 3.0, -3.0, 5d, -5d,
        0.9999999999, -0.9999999999, 1.0000000001, -1.0000000001,
        123.456, -123.456, 4503599627370496.0 /*2^52*/, -4503599627370496.0,
        9007199254740992.0 /*2^53*/, 1e15, -1e15,
        double.PositiveInfinity, double.NegativeInfinity,
        1.5, -1.5, 2.5, -2.5, 100.1, -100.1, 0.001, -0.001, 42d, -42d, 7.7, -7.7, 9.2,
    };

    [Fact]
    public void Floor_Double_MatchesMath()
    {
        var input = EdgeCasesD();
        var got = new double[input.Length];
        fixed (double* ip = input, op = got)
            SimdKernels.FloorUnsafe(ip, op, input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            double expected = Math.Floor(input[i]);
            if (double.IsNaN(expected)) Assert.True(double.IsNaN(got[i]));
            else Assert.True(BitsD(got[i]) == BitsD(expected),
                $"Floor({input[i]:G17}) = {got[i]:G17} (0x{BitsD(got[i]):X16}), expected {expected:G17} (0x{BitsD(expected):X16})");
        }
    }

    [Fact]
    public void Ceiling_Double_MatchesMath()
    {
        var input = EdgeCasesD();
        var got = new double[input.Length];
        fixed (double* ip = input, op = got)
            SimdKernels.CeilingUnsafe(ip, op, input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            double expected = Math.Ceiling(input[i]);
            if (double.IsNaN(expected)) Assert.True(double.IsNaN(got[i]));
            else Assert.True(BitsD(got[i]) == BitsD(expected),
                $"Ceiling({input[i]:G17}) = {got[i]:G17} (0x{BitsD(got[i]):X16}), expected {expected:G17} (0x{BitsD(expected):X16})");
        }
    }
}
