using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public sealed class GpuOptimizerStateInitializationTests
{
    [Fact]
    public void StateIsFilledOnDevice()
    {
        var state = new MockBackendState();
        using var buffer = CompiledTrainingPlan<float>.AllocateGpuOptimizerState(MockDirectGpuBackend.Create(state), 13);
        Assert.Equal(1, state.FillCalls);
        Assert.Single(state.AllocatedBuffers);
        Assert.All(((MockGpuBuffer)buffer).Data, value => Assert.Equal(0f, value));
    }

    [Fact]
    public void UnsupportedFill_ReleasesTemporaryAndUsesZeroUpload()
    {
        var state = new MockBackendState();
        state.ThrowOnCall["Fill"] = new NotSupportedException("No device fill");
        using var buffer = CompiledTrainingPlan<float>.AllocateGpuOptimizerState(MockDirectGpuBackend.Create(state), 13);
        Assert.Equal(2, state.AllocatedBuffers.Count);
        Assert.Equal(1, state.AllocatedBuffers[0].DisposeCount);
        Assert.All(((MockGpuBuffer)buffer).Data, value => Assert.Equal(0f, value));
    }

    [Fact]
    public void FailedFill_ReleasesTemporaryAndPropagates()
    {
        var state = new MockBackendState();
        var failure = new InvalidOperationException("Device failure");
        state.ThrowOnCall["Fill"] = failure;
        Assert.Same(failure, Assert.Throws<InvalidOperationException>(() =>
            CompiledTrainingPlan<float>.AllocateGpuOptimizerState(MockDirectGpuBackend.Create(state), 13)));
        Assert.Equal(1, Assert.Single(state.AllocatedBuffers).DisposeCount);
    }
}
