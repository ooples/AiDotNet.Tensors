using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-issue C (#371) task C.1: verifies the tiny-shape bypass in
/// <see cref="BlasManagedLib.Gemm{T}"/> produces bit-exact output vs the regular
/// route and is limited to outputs that cannot form a complete SIMD tile.
/// </summary>
[Collection("BlasManaged-Perf-Serial")]
public class TinyShapeBypassTest
{
    /// <summary>
    /// A shape below the work threshold must produce the same result through the bypass and non-bypass routes.
    /// Use PackingMode.ForcePackBoth on one side to exclude the bypass.
    /// </summary>
    [Fact]
    public void Bypass_BitExact_Vs_ForcedFullPath_FP32()
    {
        // N deliberately is not a valid FP32 JIT width. Otherwise the default call tests the
        // earlier JIT shortcut rather than the tiny-shape bypass this test is meant to isolate.
        const int M = 8, N = 6, K = 4;  // 192 work, the documented bypass win region
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var cBypass = new float[M * N];
        var cForcedPath = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Default options route through the bypass.
        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cBypass, N, M, N, K);

        // ForcePackBoth excludes the bypass even for tiny shapes.
        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cForcedPath, N, M, N, K,
            new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth });

        for (int i = 0; i < cBypass.Length; i++)
            Assert.True(cBypass[i] == cForcedPath[i],
                $"Tiny-shape bypass mismatch at [{i / N}, {i % N}]: " +
                $"bypass={cBypass[i]:G9} forced={cForcedPath[i]:G9}");
    }

    [Fact]
    public void Bypass_BitExact_Vs_ForcedFullPath_FP64()
    {
        // N deliberately is not a valid FP64 JIT width, for the same route-isolation reason.
        const int M = 8, N = 6, K = 4;
        var rng = new Random(42);
        var a = new double[M * K];
        var b = new double[K * N];
        var cBypass = new double[M * N];
        var cForcedPath = new double[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        BlasManagedLib.Gemm<double>(a, K, false, b, N, false, cBypass, N, M, N, K);
        BlasManagedLib.Gemm<double>(a, K, false, b, N, false, cForcedPath, N, M, N, K,
            new BlasOptions<double> { PackingMode = PackingMode.ForcePackBoth });

        for (int i = 0; i < cBypass.Length; i++)
            Assert.Equal(cBypass[i], cForcedPath[i]);
    }

    [Fact]
    public void Bypass_IsLimitedToShapesWithoutACompleteSimdTile()
    {
        BlasOptions<float> automatic = default;
        var forced = new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth };

        Assert.True(BlasManagedLib.IsTinyShapeBypassEligible(
            m: 8, n: 6, k: 4, in automatic, deterministic: true));
        Assert.False(BlasManagedLib.IsTinyShapeBypassEligible(
            m: 16, n: 16, k: 16, in automatic, deterministic: true));
        Assert.False(BlasManagedLib.IsTinyShapeBypassEligible(
            m: 8, n: 6, k: 4, in forced, deterministic: true));
        Assert.False(BlasManagedLib.IsTinyShapeBypassEligible(
            m: 128, n: 128, k: 128, in automatic, deterministic: true));
    }

    [Fact]
    public void Bypass_Skipped_When_Above_Threshold()
    {
        // Bigger shape — bypass must NOT fire (M*N*K > threshold). Verify by
        // ensuring the full path's output is still correct (this is the path
        // most tests already exercise; mostly a smoke check).
        const int M = 128, N = 128, K = 128;  // 2M work, well above threshold
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K);

        // Sanity: result should be non-zero (would be all-zero if c.Clear hit but
        // the kernel never ran).
        bool anyNonZero = false;
        for (int i = 0; i < c.Length; i++) if (c[i] != 0) { anyNonZero = true; break; }
        Assert.True(anyNonZero, "Result must be non-zero — kernel didn't run?");
    }
}
