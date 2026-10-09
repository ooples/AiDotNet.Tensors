using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers.Autotune;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// The per-shape route tuner for the direct conv kernels: a real (exhaustive) tuning run picks and activates a
/// validated configuration that the engine's routing then obeys, and a persisted winner re-activates in a fresh state.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class DirectConvEvolutionAutotunerTests : IDisposable
{
    private static readonly DirectConvShape Shape =
        new(DirectConvPass.Forward, 4, 32, 32, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1);

    public void Dispose() => DirectConvTuning.Clear();

    [SkippableFact]
    public async Task TuneAsync_MeasuresEveryConfiguration_AndRoutingObeysTheActivatedWinner()
    {
        Skip.IfNot(DirectConvAvx2.IsSupported, "needs AVX2 and FMA");
        var store = new MemoryStore();

        EvolutionKernelTuningResult<DirectConvConfiguration> result = await DirectConvEvolutionAutotuner.TuneAsync(
            Shape, new KernelSearchSpaceVersion(1), new KernelBenchmarkProtocolVersion(1),
            deploymentRegistry: new KernelTuningDeploymentRegistry<DirectConvConfiguration>(), store: store);

        DirectConvConfiguration winner = result.ActiveDeployment.Configuration;
        Assert.Contains(winner, DirectConvEvolutionAutotuner.GetSearchSpace(Shape));
        Assert.True(DirectConvTuning.TryGet(Shape, out DirectConvConfiguration active));
        Assert.Equal(winner, active);
        Assert.Equal(winner.Route == DirectConvRoute.Direct, DirectConvAvx2.TryChoose(Shape, out int tasks));
        Assert.Equal(winner.TargetTasks, tasks);

        // An activated configuration overrides the measured default either way.
        DirectConvTuning.Activate(Shape, new DirectConvConfiguration(DirectConvRoute.Im2Col, 0));
        Assert.False(DirectConvAvx2.TryChoose(Shape, out _));
        DirectConvTuning.Activate(Shape, new DirectConvConfiguration(DirectConvRoute.Direct, 512));
        Assert.True(DirectConvAvx2.TryChoose(Shape, out tasks));
        Assert.Equal(512, tasks);
    }

    [SkippableFact]
    public async Task PersistedWinner_ReactivatesAfterTheTableIsCleared()
    {
        Skip.IfNot(DirectConvAvx2.IsSupported, "needs AVX2 and FMA");
        var store = new MemoryStore();
        var ssv = new KernelSearchSpaceVersion(1);
        var bpv = new KernelBenchmarkProtocolVersion(1);
        var result = await DirectConvEvolutionAutotuner.TuneAsync(Shape, ssv, bpv,
            deploymentRegistry: new KernelTuningDeploymentRegistry<DirectConvConfiguration>(), store: store);
        Assert.True(store.StoreCount > 0);

        DirectConvTuning.Clear();
        Assert.True(DirectConvEvolutionAutotuner.TryActivatePersisted(Shape, ssv, bpv, store));
        Assert.True(DirectConvTuning.TryGet(Shape, out DirectConvConfiguration active));
        Assert.Equal(result.ActiveDeployment.Configuration, active);

        // Another protocol version is another identity: nothing to activate.
        DirectConvTuning.Clear();
        Assert.False(DirectConvEvolutionAutotuner.TryActivatePersisted(Shape, ssv, new KernelBenchmarkProtocolVersion(2), store));
        Assert.False(DirectConvTuning.TryGet(Shape, out _));
    }

    [Fact]
    public void Validation_RejectsConfigurationsOutsideTheSearchSpace()
    {
        Assert.Null(DirectConvEvolutionAutotuner.ValidateConfiguration(Shape, new DirectConvConfiguration(DirectConvRoute.Im2Col, 0)));
        Assert.NotNull(DirectConvEvolutionAutotuner.ValidateConfiguration(Shape, new DirectConvConfiguration(DirectConvRoute.Im2Col, 64)));
        Assert.NotNull(DirectConvEvolutionAutotuner.ValidateConfiguration(Shape, new DirectConvConfiguration((DirectConvRoute)7, 0)));
        if (DirectConvAvx2.IsSupported)
        {
            Assert.Null(DirectConvEvolutionAutotuner.ValidateConfiguration(Shape, new DirectConvConfiguration(DirectConvRoute.Direct, 256)));
            Assert.NotNull(DirectConvEvolutionAutotuner.ValidateConfiguration(Shape, new DirectConvConfiguration(DirectConvRoute.Direct, 100)));
            var kernelShape = Shape with { Pass = DirectConvPass.BackwardKernel };
            Assert.Null(DirectConvEvolutionAutotuner.ValidateConfiguration(kernelShape, new DirectConvConfiguration(DirectConvRoute.Direct, 0)));
            Assert.NotNull(DirectConvEvolutionAutotuner.ValidateConfiguration(kernelShape, new DirectConvConfiguration(DirectConvRoute.Direct, 128)));
        }
        // A shape the kernels cannot run (3 input channels) only offers the existing route.
        var unaligned = Shape with { InChannels = 3 };
        Assert.Equal(new[] { new DirectConvConfiguration(DirectConvRoute.Im2Col, 0) }, DirectConvEvolutionAutotuner.GetSearchSpace(unaligned));
    }

    private sealed class MemoryStore : IKernelTuningStore<DirectConvConfiguration>
    {
        private readonly Dictionary<string, KernelTuningDeploymentSnapshot<DirectConvConfiguration>> _entries = new();

        public int StoreCount { get; private set; }

        public bool TryLoad(KernelTuningIdentity identity, IEvolutionGenomeCodec<DirectConvConfiguration> codec,
            out KernelTuningDeploymentSnapshot<DirectConvConfiguration>? snapshot)
            => _entries.TryGetValue(identity.StableKey, out snapshot);

        public bool TryStore(KernelTuningDeploymentSnapshot<DirectConvConfiguration> snapshot, IEvolutionGenomeCodec<DirectConvConfiguration> codec)
        {
            _entries[snapshot.Identity.StableKey] = snapshot;
            StoreCount++;
            return true;
        }
    }
}
