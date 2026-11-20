using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CompletedScratchHandoffTests
{
    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void HandoffPreservesUnreadLossGradientsViewsAndAdamTrajectory(bool retainView, bool inputGradients)
    {
        using var gpu = CompiledGpuSharedWorkspaceTests.CreateGpu();
        var previous = AiDotNetEngine.Current;
        var previousShare = CompiledTrainingPlan<float>.ShareTransientWorkspaceForTests;
        var previousInputs = CompiledTrainingPlan<float>.CompiledInputGradientsForTests;
        var previousHandoff = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_BACKEND_SCRATCH_HANDOFF");
        var previousBackward = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_SHARED_BACKWARD_SCRATCH");
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_BACKEND_SCRATCH_HANDOFF", "0");
            var expected = CompiledGpuSharedWorkspaceTests.Run(new CpuEngine(), false, retainView, inputGradients);
            var baseline = CompiledGpuSharedWorkspaceTests.Run(gpu, false, retainView, inputGradients);
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_SHARED_BACKWARD_SCRATCH", "1");
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_BACKEND_SCRATCH_HANDOFF", "1");
            var handoff = CompiledGpuSharedWorkspaceTests.Run(gpu, true, retainView, inputGradients);
            Assert.True(handoff.Handoff > 0, "This fixture must return completed scratch to the backend cache.");
            Assert.Equal(0, gpu.CompiledWorkspaceBroker.RetainedBytes);
            Assert.Equal(expected.Samples.Count, handoff.Samples.Count);
            for (int i = 0; i < handoff.Samples.Count; i++)
            {
                CompiledGpuSharedWorkspaceTests.AssertClose(expected.Samples[i], baseline.Samples[i], "CPU baseline");
                Assert.Equal(baseline.Samples[i], handoff.Samples[i]);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_BACKEND_SCRATCH_HANDOFF", previousHandoff);
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_SHARED_BACKWARD_SCRATCH", previousBackward);
            AiDotNetEngine.Current = previous;
            CompiledTrainingPlan<float>.ShareTransientWorkspaceForTests = previousShare;
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = previousInputs;
        }
    }
}
