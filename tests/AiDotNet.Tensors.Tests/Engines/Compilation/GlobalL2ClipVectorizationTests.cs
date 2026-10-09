using System;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Parity tests for the SIMD-vectorized global-L2 gradient-clip helpers
/// (<see cref="CompiledTrainingPlan{T}.SumSquaresVectorized(double[],int)"/> and
/// <c>ScaleInPlaceVectorized</c>). The clip is a per-step, full-gradient pass that
/// was ~19% of a large-model fused training step as a scalar loop with per-element
/// virtual numOps dispatch (PerfView). These tests pin the vectorized helpers to the
/// exact scalar reference across lengths that exercise the SIMD tail (len % width != 0).
/// </summary>
[Collection("BlasManaged-Stats-Serial")]   // one test varies the process-wide MaxDegreeOfParallelism
public class GlobalL2ClipVectorizationTests
{
    // Lengths chosen to hit: below one vector width, exact multiples, and multiples+tail,
    // for both float (width 8 on AVX2) and double (width 4 on AVX2).
    public static TheoryData<int> Lengths => new() { 0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 1000, 4097 };

    // Around and well past one ClipChunkElements chunk (65536), with SIMD tails.
    public static TheoryData<int> ChunkedLengths => new() { 100, 65_535, 65_536, 65_537, 131_079, 401_413 };

    [Theory]
    [MemberData(nameof(ChunkedLengths))]
    public void SumSquaresChunked_IsTheOrderedSumOfChunkPartials_AtEveryThreadCount(int len)
    {
        var a = MakeFloat(len, seed: 4242 + len);
        int chunk = CompiledTrainingPlan<float>.ClipChunkElements;
        double expected = 0.0;
        for (int s = 0; s < len; s += chunk)
            expected += CompiledTrainingPlan<float>.SumSquaresRange(a, s, Math.Min(len, s + chunk));
        if (len <= chunk)   // one chunk: exactly the unchunked sum
            Assert.Equal(BitConverter.DoubleToInt64Bits(CompiledTrainingPlan<float>.SumSquaresVectorized(a, len)),
                BitConverter.DoubleToInt64Bits(expected));

        int before = AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            foreach (int threads in new[] { 1, 3, 16 })
            {
                AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = threads;
                double actual = CompiledTrainingPlan<float>.SumSquaresChunked(a, len);
                Assert.True(BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual),
                    $"len={len} threads={threads}: {actual:R} != {expected:R}");
            }
        }
        finally
        {
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = before;
        }

        double scalar = 0.0;
        for (int i = 0; i < len; i++) { double v = a[i]; scalar += v * v; }
        Assert.True(Math.Abs(scalar - expected) <= Math.Abs(scalar) * 1e-12, $"len={len}: {expected:R} vs scalar {scalar:R}");
    }

    [Fact]
    public void SumSquaresChunked_AddsPartialsInChunkOrder()
    {
        // Chunks of very different magnitude, so the association of the partials shows in the low bits.
        int chunk = CompiledTrainingPlan<float>.ClipChunkElements;
        int len = 5 * chunk + 3;
        var rng = new Random(17);
        var a = new float[len];
        for (int i = 0; i < len; i++)
            a[i] = (float)((rng.NextDouble() + 0.1) * Math.Pow(37.0, i / chunk));
        var partials = new double[(len + chunk - 1) / chunk];
        for (int c = 0; c < partials.Length; c++)
            partials[c] = CompiledTrainingPlan<float>.SumSquaresRange(a, c * chunk, Math.Min(len, (c + 1) * chunk));
        double inOrder = 0.0, reversed = 0.0;
        for (int c = 0; c < partials.Length; c++) inOrder += partials[c];
        for (int c = partials.Length - 1; c >= 0; c--) reversed += partials[c];
        Assert.NotEqual(BitConverter.DoubleToInt64Bits(inOrder), BitConverter.DoubleToInt64Bits(reversed));   // the fixture discriminates

        Assert.Equal(BitConverter.DoubleToInt64Bits(inOrder),
            BitConverter.DoubleToInt64Bits(CompiledTrainingPlan<float>.SumSquaresChunked(a, len)));
    }

    [Theory]
    [MemberData(nameof(ChunkedLengths))]
    public void ScaleInPlaceChunked_IsBitIdenticalToTheSerialScale(int len)
    {
        var serial = MakeFloat(len, seed: 99 + len);
        var chunked = (float[])serial.Clone();
        CompiledTrainingPlan<float>.ScaleInPlaceVectorized(serial, len, 0.3711f);
        CompiledTrainingPlan<float>.ScaleInPlaceChunked(chunked, len, 0.3711f);
        for (int i = 0; i < serial.Length; i++)   // includes the 5-element pad, which neither may touch
            Assert.True(SingleBits(serial[i]) == SingleBits(chunked[i]),
                $"len={len} [{i}]: {chunked[i]:R} != {serial[i]:R}");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(65_535)]
    [InlineData(65_536)]
    [InlineData(200_003)]
    [InlineData(401_412)]
    public unsafe void AllFiniteParallel_FindsANonFiniteValueAnywhere(int position)
    {
        const int len = 401_413;
        var a = MakeFloat(len, seed: 5);
        fixed (float* p = a) Assert.True(FusedOptimizer.AllFiniteParallel(p, len));
        foreach (float bad in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            float saved = a[position];
            a[position] = bad;
            fixed (float* p = a) Assert.False(FusedOptimizer.AllFiniteParallel(p, len), $"{bad} at {position} not found");
            a[position] = saved;
        }
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void SumSquares_Double_MatchesScalarReference(int len)
    {
        var a = MakeDouble(len, seed: 12345 + len);
        double expected = 0.0;
        for (int i = 0; i < len; i++) expected += a[i] * a[i];

        double actual = CompiledTrainingPlan<double>.SumSquaresVectorized(a, len);

        // Double accumulation is bit-identical order aside; allow a tiny relative epsilon
        // for the SIMD reassociation.
        Assert.Equal(expected, actual, 9);
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void SumSquares_Float_MatchesScalarReference(int len)
    {
        var a = MakeFloat(len, seed: 777 + len);
        // Reference mirrors the prior scalar path: widen each float to double, then square.
        double expected = 0.0;
        for (int i = 0; i < len; i++) { double v = a[i]; expected += v * v; }

        double actual = CompiledTrainingPlan<float>.SumSquaresVectorized(a, len);

        double tol = Math.Max(1e-6, Math.Abs(expected) * 1e-6);
        Assert.True(Math.Abs(expected - actual) <= tol,
            $"len={len}: expected={expected}, actual={actual}");
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void ScaleInPlace_Double_MatchesScalarReference(int len)
    {
        const double scale = 0.375;
        var a = MakeDouble(len, seed: 999 + len);
        var reference = (double[])a.Clone();
        for (int i = 0; i < len; i++) reference[i] *= scale;
        double origPad = a.Length > len ? a[len] : 0.0;   // first pool-padding element

        CompiledTrainingPlan<double>.ScaleInPlaceVectorized(a, len, scale);

        for (int i = 0; i < len; i++) Assert.Equal(reference[i], a[i], 12);
        // Pool padding beyond len must be untouched by the SIMD path.
        if (a.Length > len) Assert.Equal(origPad, a[len], 12);
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void ScaleInPlace_Float_MatchesScalarReference(int len)
    {
        const float scale = 0.375f;
        var a = MakeFloat(len, seed: 4242 + len);
        var reference = (float[])a.Clone();
        for (int i = 0; i < len; i++) reference[i] *= scale;

        CompiledTrainingPlan<float>.ScaleInPlaceVectorized(a, len, scale);

        for (int i = 0; i < len; i++) Assert.Equal(reference[i], a[i], 6);
    }

    // Allocate a buffer LARGER than len (simulating pool padding) so the SIMD path's
    // logical-length bound is exercised — the tail beyond len must never be read/written.
    private static double[] MakeDouble(int len, int seed)
    {
        var rng = new Random(seed);
        var a = new double[len + 5];
        for (int i = 0; i < a.Length; i++) a[i] = (rng.NextDouble() * 2 - 1) * 10;
        return a;
    }

    // BitConverter.SingleToInt32Bits is not on net471.
    private static int SingleBits(float f) => BitConverter.ToInt32(BitConverter.GetBytes(f), 0);

    private static float[] MakeFloat(int len, int seed)
    {
        var rng = new Random(seed);
        var a = new float[len + 5];
        for (int i = 0; i < a.Length; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * 10);
        return a;
    }
}
