// Copyright (c) AiDotNet. All rights reserved.
using System;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

// HipBackendTestFixture is .NET 6+ only (as are the other HIP hardware tests); the key tests below run everywhere.
#if NET6_0_OR_GREATER
/// <summary>
/// The CLBlast indirect GEMM on HIP (#1266) against a double-precision CPU reference, through
/// <see cref="HipBackend.Gemm"/> itself, with a run counter proving which shapes took the indirect route.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class HipClBlastIndirectGemmTests : IClassFixture<HipBackendTestFixture>
{
    private readonly HipBackendTestFixture _fixture;
    private HipBackend Backend => _fixture.Backend;

    public HipClBlastIndirectGemmTests(HipBackendTestFixture fixture) { _fixture = fixture; }

    private bool EnsureReady()
    {
        if (Backend.IsAvailable && !Backend.IsHipBlasAvailable && Backend.EnsureClBlastIndirectGemm()) return true;
        if (string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException(
                "GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but the HIP CLBlast indirect GEMM was unavailable.");
        return false;
    }

    [SkippableTheory]
    [InlineData(448, 448, 448, 1f, 0f)]      // exactly MinIndirectSize: no padding on this device's tiles
    [InlineData(447, 448, 61, 1f, 0f)]       // N alone reaches the threshold; M and K pad
    [InlineData(500, 700, 300, 0.5f, 0.25f)] // every dimension pads; beta reads C through the padded copy
    [InlineData(1024, 1024, 1024, 1f, 1f)]
    [InlineData(4096, 64, 257, 2f, -1f)]     // skinny N, ragged K
    public void Gemm_LargeShapes_TakeTheIndirectRoute_AndMatchReference(int m, int n, int k, float alpha, float beta)
    {
        Skip.IfNot(EnsureReady(), "HIP device with the CLBlast indirect GEMM and no hipBLAS is required.");
        var (a, b, c0) = Inputs(m, n, k, seed: m * 31 + n * 7 + k);
        using var ga = Backend.AllocateBuffer(a);
        using var gb = Backend.AllocateBuffer(b);
        using var gc = Backend.AllocateBuffer(c0);

        long before = Backend.ClBlastIndirectGemmRuns;
        Backend.Gemm(ga, gb, gc, m, n, k, alpha, beta);
        Assert.Equal(before + 1, Backend.ClBlastIndirectGemmRuns);
        AssertMatches(Backend.DownloadBuffer(gc), a, b, c0, m, n, k, alpha, beta);
    }

    [SkippableTheory]
    [InlineData(447, 447, 512)]
    [InlineData(1, 4096, 512)]               // vector products stay on the HIP kernels
    [InlineData(4096, 1, 512)]
    [InlineData(8, 512, 512)]               // past the threshold, but padding 8 to a 64 tile is 8x the work
    [InlineData(448, 8, 4096)]
    [InlineData(16, 1024, 1024)]
    [InlineData(3, 2000, 5)]
    public void Gemm_ShapesTheIndirectPathWouldLose_KeepTheHipKernels(int m, int n, int k)
    {
        Skip.IfNot(EnsureReady(), "HIP device with the CLBlast indirect GEMM and no hipBLAS is required.");
        var (a, b, c0) = Inputs(m, n, k, seed: 11);
        using var ga = Backend.AllocateBuffer(a);
        using var gb = Backend.AllocateBuffer(b);
        using var gc = Backend.AllocateBuffer(c0);

        long before = Backend.ClBlastIndirectGemmRuns;
        Backend.Gemm(ga, gb, gc, m, n, k);
        Assert.Equal(before, Backend.ClBlastIndirectGemmRuns);
        AssertMatches(Backend.DownloadBuffer(gc), a, b, c0, m, n, k, 1f, 0f);
    }

    [SkippableFact]
    public void Gemm_ReusedScratch_ShrinkingThenGrowingShapes_StayCorrect()
    {
        Skip.IfNot(EnsureReady(), "HIP device with the CLBlast indirect GEMM and no hipBLAS is required.");
        foreach (var (m, n, k) in new[] { (1024, 1024, 512), (480, 520, 100), (1100, 900, 700) })
        {
            var (a, b, c0) = Inputs(m, n, k, seed: m + n + k);
            using var ga = Backend.AllocateBuffer(a);
            using var gb = Backend.AllocateBuffer(b);
            using var gc = Backend.AllocateBuffer(c0);
            Backend.Gemm(ga, gb, gc, m, n, k, 1f, 0.5f);
            AssertMatches(Backend.DownloadBuffer(gc), a, b, c0, m, n, k, 1f, 0.5f);
        }
    }

    [SkippableFact]
    public void HipBlasAndClBlast_AreNotBuiltUntilAGemmNeedsThem()
    {
        Skip.IfNot(EnsureReady(), "HIP device with the CLBlast indirect GEMM and no hipBLAS is required.");
        using var backend = new HipBackend();
        Assert.True(backend.IsAvailable);
        Assert.False(backend.HipBlasInitializationAttempted);

        using var a = backend.AllocateBuffer(new float[4]);
        using var b = backend.AllocateBuffer(new float[4]);
        using var c = backend.AllocateBuffer(new float[4]);
        Assert.False(backend.HipBlasInitializationAttempted);

        backend.Gemm(a, b, c, 2, 2, 2);
        Assert.True(backend.HipBlasInitializationAttempted);
        // Below MinIndirectSize the threshold is decided from the database alone: no CLBlast module is compiled.
        Assert.False(backend.ClBlastModulesAttempted);
    }
    [SkippableFact]
    public void DeployedConfiguration_RunsItsShape_AndLeavesOtherShapesOnTheDatabaseConfiguration()
    {
        Skip.IfNot(EnsureReady(), "HIP device with the CLBlast indirect GEMM and no hipBLAS is required.");
        using var backend = new HipBackend();
        Assert.True(backend.EnsureClBlastIndirectGemm());
        GemmConfig database = Assert.NotNull(backend.ClBlastDefaultConfig);
        // The database configuration with its stride flags swapped: a different kernel, same tiles.
        GemmConfig swapped = database with { StrideM = !database.StrideM, StrideN = !database.StrideN };
        Assert.True(backend.DeployClBlastGemm(640, 576, 320, swapped));

        Assert.Equal(swapped, backend.ClBlastConfigFor(640, 576, 320));
        Assert.Equal(database, backend.ClBlastConfigFor(576, 640, 320));
        foreach (var (m, n, k) in new[] { (640, 576, 320), (576, 640, 320) })
        {
            var (a, b, c0) = Inputs(m, n, k, seed: m + 3 * n);
            using var ga = backend.AllocateBuffer(a);
            using var gb = backend.AllocateBuffer(b);
            using var gc = backend.AllocateBuffer(c0);
            long before = backend.ClBlastIndirectGemmRuns;
            backend.Gemm(ga, gb, gc, m, n, k, 1f, 0.5f);
            Assert.Equal(before + 1, backend.ClBlastIndirectGemmRuns);
            AssertMatches(backend.DownloadBuffer(gc), a, b, c0, m, n, k, 1f, 0.5f);
        }
    }
    private static (float[] A, float[] B, float[] C) Inputs(int m, int n, int k, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        float[] Fill(int len)
        {
            var x = new float[len];
            for (int i = 0; i < len; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            return x;
        }
        return (Fill(m * k), Fill(k * n), Fill(m * n));
    }

    private static void AssertMatches(float[] got, float[] a, float[] b, float[] c0, int m, int n, int k, float alpha, float beta)
    {
        // FP32 accumulation of k terms of magnitude <= 1: error grows like sqrt(k) ulps of the row sum.
        double tol = 1e-5 * Math.Sqrt(k) * Math.Max(1.0, Math.Abs(alpha)) + 1e-6;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int p = 0; p < k; p++) s += (double)a[i * k + p] * b[p * n + j];
                double expected = alpha * s + beta * c0[i * n + j];
                double err = Math.Abs(expected - got[i * n + j]);
                if (err > tol * Math.Max(1.0, Math.Abs(expected) + Math.Sqrt(k)))
                    Assert.Fail($"C[{i},{j}] = {got[i * n + j]}, expected {expected} (|err| {err:E2}) for {m}x{n}x{k}");
            }
        }
    }
}
#endif

/// <summary>The CLBlast database key a HIP backend builds must be the one OpenCL builds for the same card.</summary>
public sealed class HipClBlastDeviceKeyTests
{
    [Fact]
    public void ForAmdGpu_HipNames_ResolveTheSameTuningAsOpenCl()
    {
        // OpenCL on an RX 5500 XT: architecture = device name "gfx1012:xnack-", device = board name with the vendor.
        var openCl = new ClBlastDeviceInfo("GPU", "AMD", "gfx1012:xnack-", "AMD Radeon RX 5500 XT");
        // HIP on the same card: the same gcnArchName, the marketing name without "AMD ".
        var hip = ClBlastDeviceInfo.ForAmdGpu("gfx1012:xnack-", "Radeon RX 5500 XT");

        Assert.Equal(openCl.Architecture, hip.Architecture);
        Assert.Equal(openCl.DeviceName, hip.DeviceName);
        Assert.Equal(448, ClBlastGemmRoutineDatabase.GetXgemmMinIndirectSize(hip));
        Assert.True(ClBlastXgemmDatabase.TryGetConfig(openCl, out var expected));
        Assert.True(ClBlastXgemmDatabase.TryGetConfig(hip, out var actual));
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ForAmdGpu_NameAlreadyCarryingTheVendor_IsNotPrefixedTwice()
    {
        Assert.Equal("AMD Radeon RX 5500 XT", ClBlastDeviceInfo.ForAmdGpu("gfx1012:xnack-", "AMD Radeon RX 5500 XT").DeviceName);
    }
}

/// <summary>HIP builds only CLBlast GEMMK 0 configurations within CLBlast's own xgemm constraints.</summary>
public sealed class HipClBlastGemmConstraintTests
{
    private static readonly GemmConfig RadeonRx5500Xt = new()
    {
        KernelTemplate = GemmKernelTemplate.ClBlastBaselineK0, KernelName = "test",
        TileM = 64, TileN = 64, TileK = 16, ThreadTileM = 8, ThreadTileN = 8, MdimaSize = 16, NdimbSize = 8,
        KUnroll = 2, KReg = 1, VectorWidthM = 4, VectorWidthN = 4, StrideN = true, CacheA = true, CacheB = true
    };

    [Fact]
    public void DatabaseConfiguration_IsBuildable()
    {
        Assert.True(HipClBlastGemmKernels.Supports(RadeonRx5500Xt));
        Assert.True(ClBlastXgemmDatabase.TryGetConfig(ClBlastDeviceInfo.ForAmdGpu("gfx1012:xnack-", "Radeon RX 5500 XT"), out var database));
        Assert.True(HipClBlastGemmKernels.Supports(database));
    }

    [Fact]
    public void ConfigurationsOutsideClBlastsConstraints_AreRejected()
    {
        // HIP has no 8-wide float vector.
        Assert.False(HipClBlastGemmKernels.Supports(RadeonRx5500Xt with { VectorWidthM = 8 }));
        // GEMMK 1 is a different kernel.
        Assert.False(HipClBlastGemmKernels.Supports(RadeonRx5500Xt with { KernelTemplate = GemmKernelTemplate.ClBlastBaselineK1 }));
        // KWG % KWI.
        Assert.False(HipClBlastGemmKernels.Supports(RadeonRx5500Xt with { TileK = 16, KUnroll = 32 }));
        // MWG % (MDIMC * VWM): 64 % (32 * 4).
        Assert.False(HipClBlastGemmKernels.Supports(RadeonRx5500Xt with { ThreadTileM = 32 }));
        // MWG % (MDIMA * VWM): 64 % (32 * 4), the cooperative load DynamicGemmKernel.ValidateConfig does not check.
        Assert.False(HipClBlastGemmKernels.Supports(RadeonRx5500Xt with { MdimaSize = 32 }));
        Assert.Null(DynamicGemmKernel.ValidateConfig(RadeonRx5500Xt with { MdimaSize = 32 }));
        // KWG % ((MDIMC * NDIMC) / NDIMB): 16 % (64 / 2).
        Assert.False(HipClBlastGemmKernels.Supports(RadeonRx5500Xt with { NdimbSize = 2, VectorWidthN = 1 }));
    }
}