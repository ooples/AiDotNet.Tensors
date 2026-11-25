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
        var engine = TestRequire.NotNull(_fixture.Engine, "the fixture's engine");
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

    /// <summary>
    /// A dropout inside a segment is recomputed in the backward, and must drop what its forward dropped (PyTorch's
    /// checkpoint restores the RNG state for the same reason): the segment's output, kept from the forward, was computed
    /// with the forward's mask. With w = 1, loss = sum(dropout(w) * c) and its gradient is the kept mask times c, so one
    /// SGD step moves the weights by learningRate * loss in total exactly when both used one mask.
    /// </summary>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void CheckpointSegment_WithDropout_BackpropagatesTheMaskItsForwardDrew(bool segmented)
    {
        RequireDevice();
        var engine = _fixture.Engine ?? throw new InvalidOperationException("The fixture has no engine.");
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            const int n = 4096;
            const float learningRate = 1e-3f;
            var weights = new Tensor<float>(new[] { 1, n });
            var ones = new Tensor<float>(new[] { 1, n });
            var column = new Tensor<float>(new[] { 1, n });
            for (int i = 0; i < n; i++)
            {
                weights[i] = 1f;
                ones[i] = 1f;
                column[i] = 1f + (i % 97) / 97f;   // distinct enough that two masks give two sums
            }

            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { weights }))
            {
                GraphMode.CheckpointSegmentScope segment = segmented ? GraphMode.BeginCheckpointSegment() : default;
                var scaled = engine.TensorMultiply(weights, ones);
                var dropped = engine.Dropout(scaled, 0.5, true, out _);
                var output = engine.TensorMultiply(dropped, column);
                segment.Dispose();
                var loss = engine.ReduceSum(output, null);
                plan = (CompiledTrainingPlan<float>)scope.CompileTraining(new[] { weights }, loss);
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: learningRate);
                Assert.Equal(segmented ? 1 : 0, plan.CheckpointSegmentCount);
                for (int step = 0; step < 3; step++)
                {
                    var before = weights.ToArray();
                    float loss = plan.Step().ToArray()[0];
                    var after = weights.ToArray();
                    // loss = sum(w * mask * c) / keep and the gradient is mask * c / keep, so with the same mask
                    // sum(w * gradient) is the loss.
                    double fromUpdate = 0;
                    for (int i = 0; i < n; i++) fromUpdate += (double)before[i] * (before[i] - after[i]) / learningRate;
                    Assert.True(loss > 0, "the dropout dropped everything");
                    Assert.True(Math.Abs(fromUpdate - loss) <= 2e-3 * loss,
                        $"step {step}: the loss {loss:R} used one mask and the update another (sum(w * gradient) = {fromUpdate:R})");
                }
                if (segmented) Assert.True(plan.CheckpointRecomputeRuns > 0, "the segment was never recomputed");
            }
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
        var engine = TestRequire.NotNull(_fixture.Engine, "the fixture's engine");
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

    /// <summary>
    /// A segmented plan reports what its schedule releases and recomputes, and refuses the frozen-weight forward
    /// rebuild: the schedule names forward actions by index, and specialized actions would write elsewhere.
    /// </summary>
    [SkippableFact]
    public void CheckpointedPlan_ReportsItsSchedule_AndRefusesTheFrozenWeightRebuild()
    {
        RequireDevice();
        var engine = TestRequire.NotNull(_fixture.Engine, "the fixture's engine");
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            const int n = 256;
            var weights = new Tensor<float>(new[] { 1, n });
            var ones = new Tensor<float>(new[] { 1, n });
            var column = new Tensor<float>(new[] { 1, n });
            for (int i = 0; i < n; i++) { weights[i] = 0.5f + i / (float)n; ones[i] = 1f; column[i] = 1f + (i % 7) / 7f; }

            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { weights }))
            {
                var segment = GraphMode.BeginCheckpointSegment();
                var scaled = engine.TensorMultiply(weights, ones);
                var hidden = engine.TensorMultiply(engine.Tanh(scaled), column);
                segment.Dispose();
                var loss = engine.ReduceSum(hidden, null);
                plan = (CompiledTrainingPlan<float>)scope.CompileTraining(new[] { weights }, loss);
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-3f);
                Assert.Equal(1, plan.CheckpointSegmentCount);
                Assert.True(plan.CheckpointReleasedActionCount > 0, "the segment released no forward action");
                Assert.True(plan.CheckpointRecomputedActionCount > 0, "the segment recomputes no action");
                Assert.Single(plan.CheckpointRecomputeSteps);
                Assert.NotEmpty(plan.CheckpointRecomputeSteps[0]);

                var ex = Assert.Throws<NotSupportedException>(() => plan.EnableFrozenWeightOptimizations());
                Assert.Contains("does not compose with activation checkpoint segments", ex.Message);
                plan.Step();   // the refused rebuild left the plan usable
                Assert.True(plan.CheckpointRecomputeRuns > 0, "the segment was never recomputed");
            }
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
