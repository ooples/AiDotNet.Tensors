using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Regression tests for the five HRR binding kernels from issue #248.
/// Each kernel is checked against a naive scalar reference on inputs
/// chosen to exercise both the AVX-aligned fast path (length divisible
/// by 4) and the scalar-tail path (length with a 1–3 element remainder).
/// Tolerances are set tight enough to catch accumulation-order bugs but
/// loose enough to allow the FMA path (which has strictly less rounding
/// error than the scalar reference) to pass.
/// </summary>
public class SimdHrrKernelsTests
{
    private const double Tol = 1e-10;

    // ─── ComplexMultiplyDouble ────────────────────────────────────────

    [Theory]
    [InlineData(8, false)]   // AVX-aligned, non-conjugate
    [InlineData(8, true)]    // AVX-aligned, conjugate
    [InlineData(17, false)]  // 4×4 + 1 tail, non-conjugate
    [InlineData(17, true)]   // 4×4 + 1 tail, conjugate
    [InlineData(3, false)]   // all-scalar (below AVX threshold)
    [InlineData(3, true)]
    public void ComplexMultiplyDouble_MatchesNaiveReference(int n, bool conjugateB)
    {
        var rng = new Random(0xC9);
        var aR = new double[n]; var aI = new double[n];
        var bR = new double[n]; var bI = new double[n];
        for (int i = 0; i < n; i++)
        {
            aR[i] = rng.NextDouble() * 2 - 1; aI[i] = rng.NextDouble() * 2 - 1;
            bR[i] = rng.NextDouble() * 2 - 1; bI[i] = rng.NextDouble() * 2 - 1;
        }
        var cR = new double[n]; var cI = new double[n];
        SimdComplexKernels.ComplexMultiplyDouble(aR, aI, bR, bI, cR, cI, conjugateB);

        for (int i = 0; i < n; i++)
        {
            double expR, expI;
            if (conjugateB)
            {
                expR = aR[i] * bR[i] + aI[i] * bI[i];
                expI = aI[i] * bR[i] - aR[i] * bI[i];
            }
            else
            {
                expR = aR[i] * bR[i] - aI[i] * bI[i];
                expI = aR[i] * bI[i] + aI[i] * bR[i];
            }
            Assert.True(Math.Abs(cR[i] - expR) < Tol, $"cR[{i}] = {cR[i]}, expected {expR}");
            Assert.True(Math.Abs(cI[i] - expI) < Tol, $"cI[{i}] = {cI[i]}, expected {expI}");
        }
    }

    [Fact]
    public void ComplexMultiplyDouble_UnbindRecoversOriginal()
    {
        // HRR unbind round-trip: (a ⊙ b) ⊙ conj(b) should recover a
        // (up to a normalization factor) when b is unit-magnitude.
        const int n = 16;
        var rng = new Random(0xAB1);
        var aR = new double[n]; var aI = new double[n];
        var bR = new double[n]; var bI = new double[n];
        for (int i = 0; i < n; i++)
        {
            aR[i] = rng.NextDouble() * 2 - 1; aI[i] = rng.NextDouble() * 2 - 1;
            // Unit-magnitude b: use random phases so |b| = 1.
            double phase = rng.NextDouble() * 2 * Math.PI;
            bR[i] = Math.Cos(phase); bI[i] = Math.Sin(phase);
        }
        var bound = (new double[n], new double[n]);
        var recovered = (new double[n], new double[n]);
        SimdComplexKernels.ComplexMultiplyDouble(aR, aI, bR, bI, bound.Item1, bound.Item2);
        SimdComplexKernels.ComplexMultiplyDouble(
            bound.Item1, bound.Item2, bR, bI, recovered.Item1, recovered.Item2, conjugateB: true);

        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(recovered.Item1[i] - aR[i]) < 1e-12);
            Assert.True(Math.Abs(recovered.Item2[i] - aI[i]) < 1e-12);
        }
    }

    [Fact]
    public void ComplexMultiplyDouble_LengthMismatch_Throws()
    {
        var a = new double[4];
        var b = new double[5]; // wrong length
        var c = new double[4];
        Assert.Throws<ArgumentException>(() =>
            SimdComplexKernels.ComplexMultiplyDouble(a, a, b, b, c, c));
    }

    // ─── GatherDouble ─────────────────────────────────────────────────

    [Fact]
    public void GatherDouble_Permutation_MatchesNaive()
    {
        var input = new double[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var indices = new[] { 3, 1, 7, 0, 5, 2, 6, 4 };
        var output = new double[indices.Length];
        SimdHrrKernels.GatherDouble(input, indices, output);

        for (int i = 0; i < indices.Length; i++)
            Assert.Equal(input[indices[i]], output[i]);
    }

    [Fact]
    public void GatherDouble_RepeatedIndices_Allowed()
    {
        var input = new double[] { 10, 20, 30 };
        var indices = new[] { 0, 0, 2, 1, 2 };
        var output = new double[indices.Length];
        SimdHrrKernels.GatherDouble(input, indices, output);
        Assert.Equal(new double[] { 10, 10, 30, 20, 30 }, output);
    }

    [Fact]
    public void GatherDouble_OutOfRangeIndex_Throws()
    {
        var input = new double[] { 1, 2, 3 };
        var bad = new[] { 0, 3 }; // 3 is out of range
        var output = new double[2];
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimdHrrKernels.GatherDouble(input, bad, output));
    }

    [Fact]
    public void GatherDouble_NegativeIndex_Throws()
    {
        var input = new double[] { 1, 2, 3 };
        var bad = new[] { -1 };
        var output = new double[1];
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimdHrrKernels.GatherDouble(input, bad, output));
    }

    [Fact]
    public void GatherDoubleParallel_MatchesSequential()
    {
        // Large enough to trigger the parallel path.
        int n = 10_000;
        var rng = new Random(0xD7);
        var input = new double[n];
        var indices = new int[n];
        for (int i = 0; i < n; i++)
        {
            input[i] = rng.NextDouble();
            indices[i] = rng.Next(0, n);
        }
        var parOut = new double[n];
        var seqOut = new double[n];
        SimdHrrKernels.GatherDoubleParallel(input, indices, parOut);
        SimdHrrKernels.GatherDouble(input, indices, seqOut);
        Assert.Equal(seqOut, parOut);
    }

    // ─── UnitPhaseCodebookDouble ──────────────────────────────────────

    [Fact]
    public void UnitPhaseCodebookDouble_IsUnitMagnitude()
    {
        const int V = 16, D = 64;
        var outR = new double[V * D];
        var outI = new double[V * D];
        SimdHrrKernels.UnitPhaseCodebookDouble(outR, outI, seed: 42, V, D);

        for (int i = 0; i < V * D; i++)
        {
            double mag2 = outR[i] * outR[i] + outI[i] * outI[i];
            // Unit magnitude up to fp rounding (~1e-15 at this scale).
            Assert.True(Math.Abs(mag2 - 1.0) < 1e-12,
                $"Entry {i} has |z|² = {mag2}, expected ≈ 1");
        }
    }

    [Fact]
    public void UnitPhaseCodebookDouble_IsDeterministicPerSeed()
    {
        const int V = 8, D = 32;
        var a1 = new double[V * D]; var b1 = new double[V * D];
        var a2 = new double[V * D]; var b2 = new double[V * D];
        SimdHrrKernels.UnitPhaseCodebookDouble(a1, b1, seed: 123, V, D);
        SimdHrrKernels.UnitPhaseCodebookDouble(a2, b2, seed: 123, V, D);
        Assert.Equal(a1, a2);
        Assert.Equal(b1, b2);
    }

    [Fact]
    public void UnitPhaseCodebookDouble_DifferentSeedsDiffer()
    {
        const int V = 8, D = 32;
        var a1 = new double[V * D]; var b1 = new double[V * D];
        var a2 = new double[V * D]; var b2 = new double[V * D];
        SimdHrrKernels.UnitPhaseCodebookDouble(a1, b1, seed: 1, V, D);
        SimdHrrKernels.UnitPhaseCodebookDouble(a2, b2, seed: 2, V, D);
        Assert.NotEqual(a1, a2);
    }

    [Fact]
    public void UnitPhaseCodebookDouble_KPsk_SnapsToLattice()
    {
        const int V = 8, D = 32;
        const int k = 4; // 4-PSK: phases ∈ {0, π/2, π, 3π/2}
        var outR = new double[V * D];
        var outI = new double[V * D];
        SimdHrrKernels.UnitPhaseCodebookDouble(outR, outI, seed: 7, V, D, kPsk: true, k: k);

        double step = 2 * Math.PI / k;
        for (int i = 0; i < V * D; i++)
        {
            double phase = Math.Atan2(outI[i], outR[i]);
            if (phase < 0) phase += 2 * Math.PI;
            // Check: phase should be within 1 ULP of some integer multiple
            // of step. Measure the residual after integer-division (the
            // "distance to nearest lattice point") mod step, tolerating
            // both directions of the wrap (2π → 0).
            double residual = phase % step;
            double distance = Math.Min(residual, step - residual);
            Assert.True(distance < 1e-10,
                $"Entry {i} phase {phase} is not on the 4-PSK lattice (distance {distance})");
        }
    }

    // ─── PhaseCoherenceDecodeDouble ───────────────────────────────────

    [Fact]
    public void PhaseCoherenceDecodeDouble_MatchesNaive()
    {
        const int V = 6, D = 12;
        var rng = new Random(0x9E3);
        var codesR = new double[V * D]; var codesI = new double[V * D];
        var queryR = new double[D]; var queryI = new double[D];
        for (int i = 0; i < V * D; i++) { codesR[i] = rng.NextDouble() * 2 - 1; codesI[i] = rng.NextDouble() * 2 - 1; }
        for (int d = 0; d < D; d++) { queryR[d] = rng.NextDouble() * 2 - 1; queryI[d] = rng.NextDouble() * 2 - 1; }

        var scores = new double[V];
        SimdHrrKernels.PhaseCoherenceDecodeDouble(codesR, codesI, queryR, queryI, scores, V, D);

        // Naive reference: scores[v] = Σ_d Re( query[d] · conj(code[v][d]) )
        //                           = Σ_d ( qR·cR + qI·cI )
        for (int v = 0; v < V; v++)
        {
            double expected = 0;
            for (int d = 0; d < D; d++)
                expected += queryR[d] * codesR[v * D + d] + queryI[d] * codesI[v * D + d];
            Assert.True(Math.Abs(scores[v] - expected) < Tol,
                $"scores[{v}] = {scores[v]}, expected {expected}");
        }
    }

    [Fact]
    public void PhaseCoherenceDecodeDouble_SelfCodeGivesMaxScore()
    {
        // A code matched against itself should have a large score;
        // matched against a random other code the score should be
        // smaller in expectation. This is the core recall property.
        const int V = 8, D = 256;
        var codesR = new double[V * D];
        var codesI = new double[V * D];
        SimdHrrKernels.UnitPhaseCodebookDouble(codesR, codesI, seed: 0xABCD, V, D);

        // Query = code[3]
        int target = 3;
        var queryR = new double[D]; var queryI = new double[D];
        Array.Copy(codesR, target * D, queryR, 0, D);
        Array.Copy(codesI, target * D, queryI, 0, D);

        var scores = new double[V];
        SimdHrrKernels.PhaseCoherenceDecodeDouble(codesR, codesI, queryR, queryI, scores, V, D);

        // Target score = D (self-inner-product on unit codes), others ≪ D.
        Assert.True(Math.Abs(scores[target] - D) < 1e-10,
            $"Self-match score = {scores[target]}, expected {D}");
        for (int v = 0; v < V; v++)
        {
            if (v == target) continue;
            Assert.True(scores[v] < scores[target] * 0.5,
                $"Non-match score[{v}] = {scores[v]} too close to self-match {scores[target]}");
        }
    }

    [Fact]
    public void PhaseCoherenceDecodeDoubleParallel_MatchesSequential()
    {
        const int V = 32, D = 512;
        var codesR = new double[V * D];
        var codesI = new double[V * D];
        var queryR = new double[D];
        var queryI = new double[D];
        SimdHrrKernels.UnitPhaseCodebookDouble(codesR, codesI, seed: 11, V, D);
        SimdHrrKernels.UnitPhaseCodebookDouble(queryR.AsSpan(), queryI.AsSpan(), seed: 22, 1, D);

        var parScores = new double[V];
        var seqScores = new double[V];
        SimdHrrKernels.PhaseCoherenceDecodeDoubleParallel(codesR, codesI, queryR, queryI, parScores, V, D);
        SimdHrrKernels.PhaseCoherenceDecodeDouble(codesR, codesI, queryR, queryI, seqScores, V, D);
        for (int v = 0; v < V; v++)
            Assert.True(Math.Abs(parScores[v] - seqScores[v]) < Tol);
    }

    // ─── HRRBindAccumulateDouble ──────────────────────────────────────

    [Fact]
    public void HRRBindAccumulateDouble_MatchesNaive()
    {
        const int D = 32;
        const int nKeys = 4, nVals = 5;
        var rng = new Random(0xBEEF);

        var keyR = new double[nKeys * D]; var keyI = new double[nKeys * D];
        var valR = new double[nVals * D]; var valI = new double[nVals * D];
        for (int i = 0; i < keyR.Length; i++) { keyR[i] = rng.NextDouble() * 2 - 1; keyI[i] = rng.NextDouble() * 2 - 1; }
        for (int i = 0; i < valR.Length; i++) { valR[i] = rng.NextDouble() * 2 - 1; valI[i] = rng.NextDouble() * 2 - 1; }

        var keyIds = new[] { 0, 2, 3, 1, 0 };
        var valIds = new[] { 4, 1, 0, 3, 2 };
        var memR = new double[D]; var memI = new double[D];

        SimdHrrKernels.HRRBindAccumulateDouble(
            keyR, keyI, valR, valI, keyIds, valIds, memR, memI, D);

        var expR = new double[D]; var expI = new double[D];
        for (int n = 0; n < keyIds.Length; n++)
        {
            int kOff = keyIds[n] * D;
            int vOff = valIds[n] * D;
            for (int d = 0; d < D; d++)
            {
                double aR = keyR[kOff + d], aI = keyI[kOff + d];
                double bR = valR[vOff + d], bI = valI[vOff + d];
                expR[d] += aR * bR - aI * bI;
                expI[d] += aR * bI + aI * bR;
            }
        }
        for (int d = 0; d < D; d++)
        {
            Assert.True(Math.Abs(memR[d] - expR[d]) < Tol, $"memR[{d}] = {memR[d]}, expected {expR[d]}");
            Assert.True(Math.Abs(memI[d] - expI[d]) < Tol, $"memI[{d}] = {memI[d]}, expected {expI[d]}");
        }
    }

    [Fact]
    public void HRRBindAccumulateDouble_EmptyPairs_LeavesMemoryUnchanged()
    {
        const int D = 8;
        var keyR = new double[D]; var keyI = new double[D];
        var valR = new double[D]; var valI = new double[D];
        var memR = new double[D]; var memI = new double[D];
        for (int i = 0; i < D; i++) { memR[i] = i; memI[i] = -i; }

        SimdHrrKernels.HRRBindAccumulateDouble(
            keyR, keyI, valR, valI,
            Array.Empty<int>(), Array.Empty<int>(),
            memR, memI, D);

        for (int i = 0; i < D; i++)
        {
            Assert.Equal(i, memR[i]);
            Assert.Equal(-i, memI[i]);
        }
    }

    [Fact]
    public void HRRBindAccumulateDouble_OutOfRangeKeyId_Throws()
    {
        const int D = 4;
        var keyR = new double[2 * D]; var keyI = new double[2 * D];
        var valR = new double[2 * D]; var valI = new double[2 * D];
        var memR = new double[D]; var memI = new double[D];
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimdHrrKernels.HRRBindAccumulateDouble(
                keyR, keyI, valR, valI,
                new[] { 99 }, new[] { 0 },
                memR, memI, D));
    }

    [Fact]
    public void HRRBindAccumulateDouble_DimensionMismatch_Throws()
    {
        const int D = 4;
        var keyR = new double[2 * D]; var keyI = new double[2 * D];
        var valR = new double[2 * D]; var valI = new double[2 * D];
        var memR = new double[D + 1]; var memI = new double[D + 1]; // wrong
        Assert.Throws<ArgumentException>(() =>
            SimdHrrKernels.HRRBindAccumulateDouble(
                keyR, keyI, valR, valI,
                new[] { 0 }, new[] { 0 },
                memR, memI, D));
    }
}
