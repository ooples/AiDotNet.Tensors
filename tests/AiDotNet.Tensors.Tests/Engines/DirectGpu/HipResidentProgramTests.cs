#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1448: the resident program on HIP, one persistent cooperative launch with a grid barrier between stages, like
/// CUDA's. Here also: one barrier returns to zero launch after launch, and the stage kernel alone matches the
/// reference on a device without cooperative launch (an RDNA1 card under the Windows HIP SDK reports none), so the
/// HIP source is exercised wherever HIP runs.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class HipResidentProgramTests : ResidentProgramTestBase<HipBackend>, IClassFixture<HipBackendTestFixture>
{
    private readonly HipBackendTestFixture _fixture;
    private IGpuBuffer? _barrier;

    public HipResidentProgramTests(HipBackendTestFixture fixture) => _fixture = fixture;

    private protected override HipBackend RequireBackend()
    {
        var backend = _fixture.Backend;
        Skip.IfNot(backend.IsAvailable && backend.SupportsResidentProgram, "HIP is not available.");
        _barrier = backend.AllocateResidentBarrier();
        return backend;
    }

    private protected override int RunResident(HipBackend backend, ResidentStage[] stages)
        => backend.RunResidentProgram(stages, _barrier ?? throw new InvalidOperationException("No barrier allocated."));

    private protected override void RunSeparately(HipBackend backend, ResidentStage[] stages) => backend.RunStagesSeparately(stages);

    // The fixture owns the backend.
    private protected override void ReleaseBackend(HipBackend backend)
    {
        _barrier?.Dispose();
        _barrier = null;
    }

    // Only a chain that is not row-local takes the cooperative grid-barrier program.
    [SkippableFact]
    public void OneBarrier_ReturnsToZeroAcrossManyLaunches()
    {
        var backend = RequireBackend();
        try
        {
            Skip.IfNot(backend.SupportsCooperativeResidentProgram, "HIP cooperative launch is not available.");
            using var chain = new Chain(backend, 128, 512, 11);
            for (int i = 0; i < 300; i++) { chain.Poison(); Assert.Equal(1, RunResident(backend, chain.StagesNotRowLocal)); }
            chain.Download();
            var counts = backend.DownloadBuffer(_barrier ?? throw new InvalidOperationException("No barrier allocated."));
            Assert.Equal(0f, counts[0]);   // the arrival count (read as float bits: zero is zero)
        }
        finally
        {
            ReleaseBackend(backend);
        }
    }

    [SkippableTheory]
    [InlineData(64, 256)]
    [InlineData(300, 77)]
    [InlineData(70000, 2)]
    [InlineData(4097, 4096)]
    public void StagesRunSeparately_MatchAReference(int rows, int cols)
    {
        var backend = _fixture.Backend;
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        using var chain = new Chain(backend, rows, cols, rows * 7 + cols);
        chain.Poison();
        RunSeparately(backend, chain.Stages);
        AssertMatchesReference(chain, chain.Download());
    }
}
#endif
