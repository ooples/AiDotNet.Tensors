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
            else Assert.True(got[i] == expected, $"Floor({input[i]:G9}) = {got[i]:G9}, expected {expected:G9}");
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
            else Assert.True(got[i] == expected, $"Ceiling({input[i]:G9}) = {got[i]:G9}, expected {expected:G9}");
        }
    }
}
