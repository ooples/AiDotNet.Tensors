using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The multiresolution hash encoding against hand values from tiny-cuda-nn's GridEncoding
/// (include/tiny-cuda-nn/common_device.h: coherent_prime_hash, grid_index, grid_scale, grid_resolution, pos_fract) and
/// Müller et al. 2022. Every expected value here differs from what the previous implementation produced: it hashed
/// with Teschner's primes 73856093/19349663/83492791, hashed every level, and placed the grid at x * resolution.
/// </summary>
public class MultiresolutionHashEncodingTinyCudaNnTests
{
    private const double Tolerance = 1e-6;

    // A table whose entry i holds i in its single feature, so a point exactly on a vertex encodes to that vertex's index.
    private static Tensor<float> IdentityTable(int tableSize)
    {
        var data = new float[tableSize];
        for (int i = 0; i < tableSize; i++) data[i] = i;
        return new Tensor<float>(data, new[] { tableSize, 1 });
    }

    // With tiny-cuda-nn's grid, pos = x * scale + 0.5, so x = (k - 0.5) / scale lands exactly on vertex k.
    private static float OnVertex(int k, float scale) => (k - 0.5f) / scale;

    [Theory]
    // index = (x ^ y*2654435761 ^ z*805459861) mod 2^32 mod 4096, hand-computed:
    [InlineData(1, 2, 3, 1500)]      // 1 ^ 1013904226 ^ 2416379583 = 2892625372; Teschner gave 2630
    [InlineData(7, 11, 13, 2061)]    // 3162261517 mod 4096; Teschner gave 1045
    [InlineData(2, 1, 1, 3622)]      // 2 ^ 2654435761 ^ 805459861 = 2922720806; Teschner gave 1938
    [InlineData(64, 64, 64, 2368)]   // 2370537792 mod 4096; Teschner gave 3392
    public void HashedLevel_UsesCoherentPrimeHashWithXor(int x, int y, int z, int expectedIndex)
    {
        // N_min = 65: scale = 65 - 1 = 64, 65 vertices per axis, 65^3 = 274625 > 4096 entries, so the level is hashed.
        const float scale = 64f;
        var positions = new Tensor<float>(
            new[] { OnVertex(x, scale), OnVertex(y, scale), OnVertex(z, scale) }, new[] { 1, 3 });

        var encoded = new CpuEngine().MultiresolutionHashEncoding(positions, new[] { IdentityTable(4096) }, 65, 2.0, 1);

        Assert.Equal(expectedIndex, encoded.ToArray()[0], Tolerance);
    }

    [Theory]
    [InlineData(0u, 0u, 0u, 0u)]
    [InlineData(1u, 0u, 0u, 1u)]          // x factor is 1
    [InlineData(0u, 1u, 0u, 2481u)]       // 2654435761 mod 4096
    [InlineData(0u, 0u, 1u, 1941u)]       // 805459861 mod 4096
    [InlineData(1u, 2u, 3u, 1500u)]
    [InlineData(2u, 1u, 1u, 3622u)]
    public void HashGridIndex_MatchesCoherentPrimeHash(uint x, uint y, uint z, uint expected)
    {
        // 2048 vertices per axis exceeds MAX_BASES[3] = 1625, so the level is always hashed.
        Assert.Equal(expected, InstantNGPOperations.HashGridIndex(x, y, z, 2048, 4096));
    }

    [Fact]
    public void CoarseLevel_ThatFitsTheTable_IsIndexedDensely()
    {
        // N_min = 5: scale 4, 5 vertices per axis, 5^3 = 125 <= 128 entries: dense, x + 5y + 25z = 1 + 10 + 75 = 86.
        const float scale = 4f;
        var positions = new Tensor<float>(
            new[] { OnVertex(1, scale), OnVertex(2, scale), OnVertex(3, scale) }, new[] { 1, 3 });
        var engine = new CpuEngine();

        var dense = engine.MultiresolutionHashEncoding(positions, new[] { IdentityTable(128) }, 5, 2.0, 1);
        Assert.Equal(86.0, dense.ToArray()[0], Tolerance);

        // One entry fewer than 125 and grid_index hashes: (1 ^ 2*2654435761 ^ 3*805459861) mod 2^32 mod 124 = 120.
        var hashed = engine.MultiresolutionHashEncoding(positions, new[] { IdentityTable(124) }, 5, 2.0, 1);
        Assert.Equal(120.0, hashed.ToArray()[0], Tolerance);

        Assert.True(InstantNGPOperations.UsesDenseHashGridIndex(5, 125));
        Assert.False(InstantNGPOperations.UsesDenseHashGridIndex(5, 124));
        Assert.Equal(86u, InstantNGPOperations.HashGridIndex(1, 2, 3, 5, 128));
        // Past MAX_BASES[3] = 0x659 the cube overflows uint32 and the level is hashed whatever the table size.
        Assert.False(InstantNGPOperations.UsesDenseHashGridIndex(0x65A, int.MaxValue));
    }

    [Fact]
    public void LevelGeometry_MatchesGridScaleAndGridResolution()
    {
        // grid_scale = 2^(l * log2 b) * N_min - 1, grid_resolution = ceil(scale) + 1.
        var level0 = InstantNGPOperations.ComputeHashGridLevel(0, 16, 1.3819, HashGridAlignment.TinyCudaNn);
        Assert.Equal(15f, level0.Scale);
        Assert.Equal(0.5f, level0.PositionOffset);
        Assert.Equal(16, level0.GridResolution);

        var level1 = InstantNGPOperations.ComputeHashGridLevel(1, 16, 1.3819, HashGridAlignment.TinyCudaNn);
        Assert.Equal(16 * 1.3819 - 1, level1.Scale, 1e-4);   // 21.1104
        Assert.Equal(23, level1.GridResolution);              // ceil(21.1104) + 1

        var level3 = InstantNGPOperations.ComputeHashGridLevel(3, 5, 2.0, HashGridAlignment.TinyCudaNn);
        Assert.Equal(39f, level3.Scale);                      // 8 * 5 - 1
        Assert.Equal(40, level3.GridResolution);

        // The paper's eq. 2: N_l = floor(N_min * b^l), no offset, N_l + 1 vertices.
        var paper = InstantNGPOperations.ComputeHashGridLevel(1, 16, 1.3819, HashGridAlignment.Paper);
        Assert.Equal(22f, paper.Scale);
        Assert.Equal(0f, paper.PositionOffset);
        Assert.Equal(23, paper.GridResolution);
    }

    [Theory]
    // tiny-cuda-nn keeps per_level_scale as a float (encodings/grid.h m_per_level_scale) and passes log2f of it, so
    // grid_scale is exp2f(l * log2f((float)b)) * N_min - 1. Taking log2 of the double b gives 0.33333334 instead of
    // 0.33333337 for b = 2^(1/3), hence scale 31 and R = 32 at level 3, and 29.554932 at level 2 of b = 128^(1/15).
    [InlineData(3, 16, 1.2599210498948732, 31.0000038f, 33)]
    [InlineData(2, 16, 1.3819128799677762, 29.5549297f, 31)]
    public void LevelGeometry_TakesLog2OfTheFloatPerLevelScale(
        int level, int baseResolution, double perLevelScale, float expectedScale, int expectedResolution)
    {
        var grid = InstantNGPOperations.ComputeHashGridLevel(level, baseResolution, perLevelScale, HashGridAlignment.TinyCudaNn);

        Assert.True(grid.Scale == expectedScale, $"scale {grid.Scale:R}, tiny-cuda-nn {expectedScale:R}");
        Assert.Equal(expectedResolution, grid.GridResolution);
    }

    [Fact]
    public void PaperAlignment_GivesTheIntegerLevelThatRealArithmeticGives()
    {
        // b = exp(ln(4096 / 32) / 7) evaluates to 1.9999999999999998, so N_min * b^l falls just below 64, 128, ...
        // 4096 in double; eq. 2's floor of the real product is the integer itself (N_{L-1} = N_max).
        double b = 1.9999999999999998;
        Assert.True(32 * b < 64);
        var level1 = InstantNGPOperations.ComputeHashGridLevel(1, 32, b, HashGridAlignment.Paper);
        Assert.Equal(64f, level1.Scale);
        Assert.Equal(65, level1.GridResolution);
        Assert.Equal(4096f, InstantNGPOperations.ComputeHashGridLevel(7, 32, b, HashGridAlignment.Paper).Scale);

        // N_min = 16, N_max = 4096, L = 16: the finest level is N_max.
        double b16 = Math.Exp(Math.Log(4096.0 / 16.0) / 15.0);
        Assert.Equal(4096f, InstantNGPOperations.ComputeHashGridLevel(15, 16, b16, HashGridAlignment.Paper).Scale);

        // A level that is not an integer is still floored: 16 * 1.3819 = 22.1104.
        Assert.Equal(22f, InstantNGPOperations.ComputeHashGridLevel(1, 16, 1.3819, HashGridAlignment.Paper).Scale);
    }

    [Theory]
    // fmaf(scale, x, 0.5f), rounded once. (float)((double)scale * x + 0.5) double-rounds these to 0.501644254,
    // 0.501079798 and 0.50147903: the double sum is inexact and lands on a float tie the exact value is not on.
    [InlineData(16.0008984f, 0.000102761951f, 0.501644313f)]
    [InlineData(16.0011921f, 6.74804687e-05f, 0.501079738f)]
    [InlineData(16.0013943f, 9.24294363e-05f, 0.50147897f)]
    public void GridPosition_IsOneFusedMultiplyAdd(float scale, float x, float expectedFraction)
    {
        Assert.True(InstantNGPOperations.FusedMultiplyAdd(scale, x, 0.5f) == expectedFraction);

        InstantNGPOperations.HashGridAxis(x, new InstantNGPOperations.HashGridLevel(scale, 0.5f, 18), out uint cell, out float fraction);

        Assert.Equal(0u, cell);
        Assert.True(fraction == expectedFraction, $"fraction {fraction:R}, fmaf {expectedFraction:R}");
    }

    [Fact]
    public void FarFace_IsEncodedAsTinyCudaNnDoes_WithoutPullingXBelowOne()
    {
        // tiny-cuda-nn's kernel_grid does not clamp: x = 1 on scale 4 is pos = 4.5, cell 4, fraction exactly 0.5, so every
        // corner weighs 1/8. Clamping x to 0.999999 gave fraction 0.4999959 and weights 0.1250031 / 0.1249969.
        // Corners (4|5)^3 of the dense 5^3 grid, x + 5y + 25z mod 128, in tiny-cuda-nn's corner order.
        var positions = new Tensor<float>(new[] { 1f, 1f, 1f }, new[] { 1, 3 });
        var table = OneHotCornerTable(128, new[] { 124, 125, 1, 2, 21, 22, 26, 27 });

        var weights = new CpuEngine().MultiresolutionHashEncoding(positions, new[] { table }, 5, 2.0, 8).ToArray();

        for (int c = 0; c < 8; c++)
            Assert.True(Math.Abs(0.125 - weights[c]) < 1e-9, $"corner {c}: expected 0.125, got {weights[c]:R}");
    }

    // One feature per corner: feature c is 1 only at corner c's entry, so output c is corner c's trilinear weight.
    private static Tensor<float> OneHotCornerTable(int tableSize, int[] cornerEntries)
    {
        var data = new float[tableSize * 8];
        for (int c = 0; c < 8; c++) data[cornerEntries[c] * 8 + c] = 1f;
        return new Tensor<float>(data, new[] { tableSize, 8 });
    }

    private static int[] DenseCorners(int x0, int y0, int z0, int r)
    {
        var entries = new int[8];
        for (int c = 0; c < 8; c++)
            entries[c] = (x0 + (c & 1)) + (y0 + ((c >> 1) & 1)) * r + (z0 + ((c >> 2) & 1)) * r * r;
        return entries;
    }

    [Fact]
    public void GridPosition_AddsHalfCellOffset_AndWeightsAreTrilinear()
    {
        // N_min = 5: scale 4. pos = 4x + 0.5: (0.3, 0.55, 0.6) -> (1.7, 2.7, 2.9), cell (1, 2, 2), fraction
        // (0.7, 0.7, 0.9). x * resolution would have given (1.2, 2.2, 2.4) instead. 125 <= 128: dense.
        var positions = new Tensor<float>(new[] { 0.3f, 0.55f, 0.6f }, new[] { 1, 3 });
        var table = OneHotCornerTable(128, DenseCorners(1, 2, 2, 5));

        var weights = new CpuEngine().MultiresolutionHashEncoding(positions, new[] { table }, 5, 2.0, 8).ToArray();

        // Corner c takes fx on x when bit 0 is set, else 1 - fx; likewise y (bit 1) and z (bit 2).
        double[] expected = { 0.3 * 0.3 * 0.1, 0.7 * 0.3 * 0.1, 0.3 * 0.7 * 0.1, 0.7 * 0.7 * 0.1,
                              0.3 * 0.3 * 0.9, 0.7 * 0.3 * 0.9, 0.3 * 0.7 * 0.9, 0.7 * 0.7 * 0.9 };
        for (int c = 0; c < 8; c++)
            Assert.True(Math.Abs(expected[c] - weights[c]) < 1e-6, $"corner {c}: expected {expected[c]}, got {weights[c]}");
    }

    [Fact]
    public void PaperAlignment_ScalesByFloorOfNminTimesBToTheL_WithoutOffset()
    {
        // N_0 = 4: pos = 4x: (0.3, 0.55, 0.6) -> (1.2, 2.2, 2.4), cell (1, 2, 2), fraction (0.2, 0.2, 0.4).
        // 5 vertices per axis, 125 <= 128: dense.
        var positions = new Tensor<float>(new[] { 0.3f, 0.55f, 0.6f }, new[] { 1, 3 });
        var table = OneHotCornerTable(128, DenseCorners(1, 2, 2, 5));

        var weights = new CpuEngine().MultiresolutionHashEncoding(
            positions, new[] { table }, 4, 2.0, 8, HashGridAlignment.Paper).ToArray();

        double[] expected = { 0.8 * 0.8 * 0.6, 0.2 * 0.8 * 0.6, 0.8 * 0.2 * 0.6, 0.2 * 0.2 * 0.6,
                              0.8 * 0.8 * 0.4, 0.2 * 0.8 * 0.4, 0.8 * 0.2 * 0.4, 0.2 * 0.2 * 0.4 };
        for (int c = 0; c < 8; c++)
            Assert.True(Math.Abs(expected[c] - weights[c]) < 1e-6, $"corner {c}: expected {expected[c]}, got {weights[c]}");
    }

    [Fact]
    public void Backward_PutsTheTrilinearWeightOnEachDenseCornerEntry()
    {
        // The point of GridPosition_AddsHalfCellOffset_AndWeightsAreTrilinear with a unit upstream gradient: each
        // corner's entry (dense index x + 5y + 25z) receives exactly that corner's weight, every other entry zero.
        var positions = new Tensor<double>(new[] { 0.3, 0.55, 0.6 }, new[] { 1, 3 });
        var table = new Tensor<double>(new double[128], new[] { 128, 1 });
        var upstream = new Tensor<double>(new[] { 1.0 }, new[] { 1, 1 });

        var gradient = new CpuEngine().MultiresolutionHashEncodingBackward(
            positions, new[] { table }, 5, 2.0, 1, upstream)[0].ToArray();

        var expected = new double[128];
        int[] corners = DenseCorners(1, 2, 2, 5);
        double[] w = { 0.3 * 0.3 * 0.1, 0.7 * 0.3 * 0.1, 0.3 * 0.7 * 0.1, 0.7 * 0.7 * 0.1,
                       0.3 * 0.3 * 0.9, 0.7 * 0.3 * 0.9, 0.3 * 0.7 * 0.9, 0.7 * 0.7 * 0.9 };
        for (int c = 0; c < 8; c++) expected[corners[c]] += w[c];
        for (int i = 0; i < 128; i++)
            Assert.True(Math.Abs(expected[i] - gradient[i]) < 1e-6, $"entry {i}: expected {expected[i]}, got {gradient[i]}");
    }

    [Fact]
    public void Backward_MatchesCentralFiniteDifferences()
    {
        // Two levels: N_min = 3 (scale 2, 3^3 = 27 <= 32 entries, dense) and scale 5 (6^3 = 216 > 32, hashed, with
        // collisions among 12 points). The loss sum(g * encode(tables)) is linear in the tables, so a central
        // difference is exact up to rounding.
        const int points = 12, features = 2, tableSize = 32, levels = 2;
        var rng = RandomHelper.CreateSeededRandom(2490);
        var positionData = new double[points * 3];
        for (int i = 0; i < positionData.Length; i++) positionData[i] = rng.NextDouble();
        var positions = new Tensor<double>(positionData, new[] { points, 3 });
        var tables = new Tensor<double>[levels];
        for (int l = 0; l < levels; l++)
        {
            var data = new double[tableSize * features];
            for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble() * 2 - 1;
            tables[l] = new Tensor<double>(data, new[] { tableSize, features });
        }
        var upstreamData = new double[points * levels * features];
        for (int i = 0; i < upstreamData.Length; i++) upstreamData[i] = rng.NextDouble() * 2 - 1;
        var upstream = new Tensor<double>(upstreamData, new[] { points, levels * features });
        var engine = new CpuEngine();

        double Loss()
        {
            var encoded = engine.MultiresolutionHashEncoding(positions, tables, 3, 2.0, features).ToArray();
            double sum = 0;
            for (int i = 0; i < encoded.Length; i++) sum += encoded[i] * upstreamData[i];
            return sum;
        }

        var analytic = engine.MultiresolutionHashEncodingBackward(positions, tables, 3, 2.0, features, upstream);
        const double h = 1e-3;
        bool anyNonZero = false;
        for (int l = 0; l < levels; l++)
        {
            var gradient = analytic[l].ToArray();
            for (int i = 0; i < tableSize * features; i++)
            {
                double original = tables[l].GetFlat(i);
                tables[l].SetFlat(i, original + h);
                double plus = Loss();
                tables[l].SetFlat(i, original - h);
                double minus = Loss();
                tables[l].SetFlat(i, original);
                double numeric = (plus - minus) / (2 * h);
                Assert.True(Math.Abs(numeric - gradient[i]) < 1e-8,
                    $"level {l}, element {i}: finite difference {numeric:R}, backward {gradient[i]:R}");
                anyNonZero |= gradient[i] != 0.0;
            }
        }
        Assert.True(anyNonZero, "The gradient is all zero, so the check proves nothing.");
    }
}
