using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #890 phase 3: the resident program (<see cref="CudaBackend.RunResidentProgram"/>) runs a declared chain as one
/// persistent cooperative launch with a grid barrier between stages (<see cref="ResidentProgramTestBase{TBackend}"/>
/// holds the shared tests). Here also: one barrier stays correct launch after launch, since it resets itself.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CudaResidentProgramTests : ResidentProgramTestBase<CudaBackend>
{
    private IGpuBuffer? _barrier;

    private protected override CudaBackend RequireBackend()
    {
        var backend = new CudaBackend();
        if (!backend.IsAvailable || !backend.SupportsResidentProgram)
        {
            backend.Dispose();
            Skip.If(true, "CUDA with cooperative launch is not available.");
        }
        _barrier = backend.AllocateResidentBarrier();
        return backend;
    }

    private protected override int RunResident(CudaBackend backend, ResidentStage[] stages)
        => backend.RunResidentProgram(stages, _barrier ?? throw new InvalidOperationException("No barrier allocated."));

    private protected override void RunSeparately(CudaBackend backend, ResidentStage[] stages) => backend.RunStagesSeparately(stages);

    // CUDA keeps its persistent grid-barrier program for every chain (and returns its grid, not a launch count).
    private protected override bool HasRowProgram => false;

    private protected override void ReleaseBackend(CudaBackend backend)
    {
        _barrier?.Dispose();
        _barrier = null;
        backend.Dispose();
    }

    // The barrier's count must return to zero at every barrier and its generation keep counting, or a later launch
    // would release early (a race) or hang. Hundreds of back-to-back launches on one barrier, then the count.
    [SkippableFact]
    public void OneBarrier_ReturnsToZeroAcrossManyLaunches()
    {
        var backend = RequireBackend();
        try
        {
            using var chain = new Chain(backend, 128, 512, 11);
            for (int i = 0; i < 300; i++) { chain.Poison(); RunResident(backend, chain.Stages); }
            chain.Download();
            var counts = backend.DownloadBuffer(_barrier ?? throw new InvalidOperationException("No barrier allocated."));
            Assert.Equal(0f, counts[0]);   // the arrival count (read as float bits: zero is zero)
        }
        finally
        {
            ReleaseBackend(backend);
        }
    }
}
