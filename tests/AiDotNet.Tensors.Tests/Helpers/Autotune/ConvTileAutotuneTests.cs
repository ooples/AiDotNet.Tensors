using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

/// <summary>
/// GPU-free tests for <see cref="ConvTileAutotune"/> — the tile-candidate
/// generator and cache-key conventions for the tiled 1x1 conv GEMM. The final
/// test composes the whole autotune foundation (fingerprint -> kernel id + shape
/// -> candidates -> Resolve -> parse-back -> cache reuse) exactly as conv
/// dispatch will, with an injected benchmark, so it runs without a GPU. Shares
/// the <c>AutotuneCacheTests</c> collection because it redirects the cache root.
/// </summary>
[Collection("AutotuneCacheTests")]
public sealed class ConvTileAutotuneTests : IDisposable
{
    private const string EnvVar = "AIDOTNET_AUTOTUNE_CACHE_PATH";
    private readonly string _tempRoot;
    private readonly string? _originalEnv;

    public ConvTileAutotuneTests()
    {
        _originalEnv = Environment.GetEnvironmentVariable(EnvVar);
        _tempRoot = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "aidotnet-conv-tile-test-" + Guid.NewGuid().ToString("N"));
        Environment.SetEnvironmentVariable(EnvVar, _tempRoot);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(EnvVar, _originalEnv);
        try { if (System.IO.Directory.Exists(_tempRoot)) System.IO.Directory.Delete(_tempRoot, true); }
        catch { /* best effort */ }
    }

    private static List<int> Variants(IReadOnlyList<AutotuneCandidate> candidates)
    {
        var v = new List<int>(candidates.Count);
        foreach (AutotuneCandidate c in candidates)
        {
            Assert.True(ConvTileAutotune.TryParseTile(c.Variant, out int t));
            v.Add(t);
        }
        return v;
    }

    [Fact]
    public void ResNet_c64_AdmitsAllDefaultTiles_WithSixteenFirst()
    {
        // N32/K64/C64/HW3136 (56x56): 64 and 3136 are divisible by 4, 8, 16, 32.
        IReadOnlyList<AutotuneCandidate> c = ConvTileAutotune.Candidates(64, 64, 3136);
        Assert.Equal(new[] { 16, 32, 8, 4 }, Variants(c)); // preference order; tile-16 is the default fallback
    }

    [Fact]
    public void ResNet_c128_ExcludesThirtyTwo_BecauseHwIs784()
    {
        // K128/C128/HW784 (28x28): 784 % 32 == 16, so tile-32 cannot launch.
        IReadOnlyList<AutotuneCandidate> c = ConvTileAutotune.Candidates(128, 128, 784);
        Assert.Equal(new[] { 16, 8, 4 }, Variants(c));
    }

    [Fact]
    public void ResNet_c256_DefaultTiles_UsesFour_WhenLargerTilesCannotLaunch()
    {
        // K256/C256/HW196 (14x14): only the default tile-4 divides every axis.
        IReadOnlyList<AutotuneCandidate> candidates =
            ConvTileAutotune.Candidates(256, 256, 196);
        Assert.Equal(new[] { 4 }, Variants(candidates));
        Assert.True(ConvTileAutotune.HasLaunchableTile(256, 256, 196));
    }

    [Fact]
    public void Candidates_RespectThreadPerBlockLimit()
    {
        // With a 256-thread cap, tile-32 (1024 threads) is excluded even though it divides.
        IReadOnlyList<AutotuneCandidate> c = ConvTileAutotune.Candidates(
            64, 64, 3136, maxThreadsPerBlock: 256);
        Assert.Equal(new[] { 16, 8, 4 }, Variants(c)); // 16->256 threads ok; 32->1024 excluded
    }

    [Fact]
    public void Candidates_DedupsAndIgnoresNonPositiveEdges()
    {
        IReadOnlyList<AutotuneCandidate> c = ConvTileAutotune.Candidates(
            64, 64, 3136, tileEdges: new[] { 16, 16, -4, 0, 16 });
        Assert.Equal(new[] { 16 }, Variants(c));
    }

    [Fact]
    public void CandidateFor_CarriesTileParameter()
    {
        AutotuneCandidate c = ConvTileAutotune.CandidateFor(16);
        Assert.Equal("tile-16", c.Variant);
        Assert.Equal("16", c.Parameters[ConvTileAutotune.TileParameter]);
    }

    [Theory]
    [InlineData("tile-16", true, 16)]
    [InlineData("tile-8", true, 8)]
    [InlineData("tile-", false, 0)]
    [InlineData("tile-abc", false, 0)]
    [InlineData("warps-4", false, 0)]
    [InlineData("", false, 0)]
    public void TryParseTile_Behaves(string variant, bool ok, int expected)
    {
        Assert.Equal(ok, ConvTileAutotune.TryParseTile(variant, out int t));
        if (ok) Assert.Equal(expected, t);
    }

    [Fact]
    public void KernelId_IsConvCategory_AndKeyedPerDevice()
    {
        var fp = GpuDeviceFingerprint.FromCuda("NVIDIA GeForce RTX 3080", "uuidZ", 8, 6, 12030);
        KernelId id = ConvTileAutotune.KernelId(fp);
        Assert.Equal("conv2d", id.Category);
        Assert.Equal(ConvTileAutotune.TiledOneByOneName + "@" + fp.LocalKey, id.Name);
    }

    [Fact]
    public void EndToEnd_ComposesFoundation_PicksBestTile_ParsesBack_AndReuses()
    {
        // Exactly the call sequence conv dispatch will make — minus the GPU.
        var fp = GpuDeviceFingerprint.FromCuda("NVIDIA GeForce RTX 3080", "uuidE2E", 8, 6, 12030);
        KernelId id = ConvTileAutotune.KernelId(fp);
        ShapeProfile shape = ConvTileAutotune.Shape(32, 64, 64, 3136);
        IReadOnlyList<AutotuneCandidate> candidates = ConvTileAutotune.Candidates(64, 64, 3136);

        // Sweep: pretend tile-32 wins on this device.
        AutotuneResolution first = GpuFirstRunAutotuner.Resolve(
            id, shape, candidates,
            c => c.Variant == "tile-32" ? 1200.0 : 600.0,
            autotuneEnabled: true);

        Assert.True(first.Measured);
        Assert.True(ConvTileAutotune.TryGetTile(first, out int tile));
        Assert.Equal(32, tile);

        // Next run reuses the cached winner without re-benchmarking.
        AutotuneResolution second = GpuFirstRunAutotuner.Resolve(
            id, shape, candidates,
            _ => throw new InvalidOperationException("must not benchmark on a cache hit"),
            autotuneEnabled: true);

        Assert.True(second.FromCache);
        Assert.True(ConvTileAutotune.TryGetTile(second, out int reusedTile));
        Assert.Equal(32, reusedTile);
    }
}
