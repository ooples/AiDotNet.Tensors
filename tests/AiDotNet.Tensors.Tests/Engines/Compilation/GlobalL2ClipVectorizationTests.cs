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
public class GlobalL2ClipVectorizationTests
{
    // Lengths chosen to hit: below one vector width, exact multiples, and multiples+tail,
    // for both float (width 8 on AVX2) and double (width 4 on AVX2).
    public static TheoryData<int> Lengths => new() { 0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 1000, 4097 };

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

    private static float[] MakeFloat(int len, int seed)
    {
        var rng = new Random(seed);
        var a = new float[len + 5];
        for (int i = 0; i < a.Length; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * 10);
        return a;
    }
}
