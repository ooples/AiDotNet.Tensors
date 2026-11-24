using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Activation checkpointing inside a compiled training plan on whatever DirectGpu backend this machine has (the POCL lane
/// runs it on OpenCL; <see cref="CompiledGpuCheckpointSegmentTests"/> adds CUDA's captured-graph case): the segments are
/// released after their forward and recomputed in the backward, and training matches the plain plan exactly. A host
/// plan refuses checkpoint segments, as releasing a host plan's fixed forward arrays saves nothing.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class CompiledCheckpointSegmentDeviceTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public CompiledCheckpointSegmentDeviceTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    // A lane that requires GPU tests (the POCL shard) fails when the device did not come up, instead of skipping.
    private void RequireDevice()
    {
        if (_fixture.IsAvailable) return;
        Assert.False(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1",
            "This lane requires GPU tests, and no DirectGpu device came up: " + _fixture.InitializationException?.Message);
        Skip.If(true, "No GPU device.");
    }

    [SkippableTheory]
    [InlineData(-0.01f, false)]
    [InlineData(-0.01f, true)]
    public void CheckpointSegments_RecomputeAndTrainLikeThePlainPlan(float nudge, bool deferNudge)
    {
        RequireDevice();
        var engine = _fixture.Engine!;
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            const int steps = 5;
            // A run first, discarded: a backend that tunes its kernels on first use (OpenCL) has settled them before the
            // two runs compared below, so both run the same kernels.
            CompiledGpuCheckpointSegmentTests.Train(engine, segmentBlocks: 2, steps: 2, nudge: nudge, deferNudge: deferNudge);
            var plain = CompiledGpuCheckpointSegmentTests.Train(engine, segmentBlocks: 0, steps, nudge: nudge, deferNudge: deferNudge);
            var checkpointed = CompiledGpuCheckpointSegmentTests.Train(engine, segmentBlocks: 2, steps, nudge: nudge, deferNudge: deferNudge);
            Assert.Equal(0, plain.Segments);
            Assert.Equal(3, checkpointed.Segments);
            Assert.True(checkpointed.Recomputes > 0, "the segmented plan never recomputed a segment");
            CompiledGpuCheckpointSegmentTests.AssertClose(plain.Losses, checkpointed.Losses, 1e-6, "loss");
            CompiledGpuCheckpointSegmentTests.AssertClose(plain.Parameters, checkpointed.Parameters, 1e-6, "parameter");
            CompiledGpuCheckpointSegmentTests.AssertClose(plain.State, checkpointed.State, 1e-6, "state");
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    [SkippableFact]
    public void CheckpointSegment_ThatReadsItsStateAfterUpdatingIt_IsRefused()
    {
        RequireDevice();
        var engine = _fixture.Engine!;
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var ex = Assert.Throws<NotSupportedException>(() =>
                CompiledGpuCheckpointSegmentTests.Train(engine, segmentBlocks: 2, steps: 1, stateReadAfterNudge: true));
            Assert.Contains("updates a state in place and then reads it", ex.Message);
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    [Fact]
    public void CheckpointSegments_OnAHostPlan_AreRefused()
    {
        var engine = new CpuEngine();
        var ex = Assert.Throws<NotSupportedException>(() =>
            CompiledGpuCheckpointSegmentTests.Train(engine, segmentBlocks: 2, steps: 1));
        Assert.Contains("DirectGpu engine", ex.Message);
    }
}
