using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Activation checkpointing inside a compiled GPU training plan (GraphMode.BeginCheckpointSegment), the compiled
/// counterpart of <c>torch.utils.checkpoint</c> under <c>torch.compile</c>. A stack of residual blocks, each with a
/// router-like state its forward reads and then nudges in place and an auxiliary term read only after the last block,
/// is trained twice from the same weights: once plain, once with every two blocks marked as a segment. The segmented
/// plan must release the segments' activations after their forward and recompute them in the backward (inside the
/// captured CUDA graph), and train to the same parameters, losses and state as the plain plan: the recompute reads the
/// state the forward read, not the nudged one, and the nudge happens once per step.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CompiledGpuCheckpointSegmentTests
{
    private const int Rows = 256, Width = 128, Hidden = 512, Experts = 16, Blocks = 6;

    private static CudaBackend RequireCuda()
    {
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            bool required = Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1"
                || Environment.GetEnvironmentVariable("AIDOTNET_EXPECT_GPU_BACKEND") == "CUDA";
            Assert.False(required, "The validation lane requires a working CUDA backend.");
            Skip.If(true, "No CUDA device available.");
        }
        return backend;
    }

    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    internal sealed class Run
    {
        internal List<float> Losses = new();
        internal float[] Parameters = Array.Empty<float>();
        internal float[] State = Array.Empty<float>();
        internal long ScratchBytes;
        internal int Segments;
        internal long Recomputes;
        internal long Replays;
        internal string? CaptureFailure;
    }

    /// <summary>
    /// Traces the block stack; <paramref name="segmentBlocks"/> &gt; 0 marks every that many blocks as one checkpoint
    /// segment. Each block: h += W2 silu(W1 h) * (1 + g), g = sigmoid(h Wr) + state (state read, then nudged in place
    /// from g's column sums); the auxiliary term sum(g * g) of every block is added to the loss after the last block.
    /// </summary>
    internal static Run Train(IEngine engine, int segmentBlocks, int steps, bool stateReadAfterNudge = false, float nudge = -0.01f,
        bool deferNudge = false)
    {
        var x = Filled(new[] { Rows, Width }, 1, 2f);
        var parameters = new List<Tensor<float>>();
        var states = new List<Tensor<float>>();
        for (int b = 0; b < Blocks; b++)
        {
            parameters.Add(Filled(new[] { Width, Hidden }, 10 + 3 * b, 0.2f));
            parameters.Add(Filled(new[] { Hidden, Width }, 11 + 3 * b, 0.2f));
            parameters.Add(Filled(new[] { Width, Experts }, 12 + 3 * b, 0.2f));
            states.Add(Filled(new[] { 1, Experts }, 40 + b, 0.1f));
        }
        var spread = Filled(new[] { Experts, Width }, 90, 0.5f);   // a constant mixing the gate into the width

        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters.ToArray()))
        {
            var h = x;
            var aux = new List<Tensor<float>>();
            var deferred = new List<(Tensor<float> State, Tensor<float> Gate)>();
            GraphMode.CheckpointSegmentScope segment = default;
            for (int b = 0; b < Blocks; b++)
            {
                if (segmentBlocks > 0 && b % segmentBlocks == 0) segment = GraphMode.BeginCheckpointSegment();
                var w1 = parameters[3 * b]; var w2 = parameters[3 * b + 1]; var wr = parameters[3 * b + 2];
                var state = states[b];
                var g = engine.TensorAdd(engine.Sigmoid(engine.TensorMatMul(h, wr)), state);      // reads the state
                if (deferNudge) deferred.Add((state, g));
                else
                    using (new NoGradScope<float>())
                    {
                        var colSum = engine.ReduceSum(g, new[] { 0 }, true);
                        engine.TensorAddInPlace(state, engine.TensorMultiplyScalar(engine.TensorSign(colSum), nudge));
                    }
                if (stateReadAfterNudge) g = engine.TensorAdd(g, engine.TensorMultiply(state, state));
                var gate = engine.TensorAddScalar(engine.TensorMatMul(g, spread), 1f);
                var inner = engine.TensorMultiply(engine.TensorMatMul(engine.TensorSiLU(engine.TensorMatMul(h, w1)), w2), gate);
                h = engine.TensorAdd(h, engine.TensorMultiplyScalar(inner, 0.1f));
                aux.Add(engine.ReduceSum(engine.TensorMultiply(g, g), null));
                // A segment of segmentBlocks blocks closes after its last block.
                if (segmentBlocks > 0 && (b + 1) % segmentBlocks == 0) segment.Dispose();
            }
            // A deferred update runs after the last block (as a router's bias nudge after the forward does): it reads each
            // block's gate, which is internal to that block's segment, long after the segment's other values are dead.
            foreach (var (state, gate) in deferred)
                using (new NoGradScope<float>())
                {
                    var colSum = engine.ReduceSum(gate, new[] { 0 }, true);
                    engine.TensorAddInPlace(state, engine.TensorMultiplyScalar(engine.TensorSign(colSum), nudge));
                }
            var loss = engine.ReduceSum(engine.TensorMultiply(h, h), null);
            foreach (var term in aux) loss = engine.TensorAdd(loss, engine.TensorMultiplyScalar(term, 1e-3f));
            plan = (CompiledTrainingPlan<float>)scope.CompileTraining(parameters.ToArray(), loss);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-4f);
            long replaysBefore = CompiledTrainingPlan<float>.GraphReplayCount;
            var run = new Run();
            for (int s = 0; s < steps; s++) run.Losses.Add(plan.Step().ToArray()[0]);
            var all = new List<float>();
            foreach (var p in parameters) all.AddRange(p.ToArray());
            run.Parameters = all.ToArray();
            var st = new List<float>();
            foreach (var s in states) st.AddRange(s.ToArray());
            run.State = st.ToArray();
            run.ScratchBytes = plan.ScratchBufferBytes;
            run.Segments = plan.CheckpointSegmentCount;
            run.Recomputes = plan.CheckpointRecomputeRuns;
            run.Replays = CompiledTrainingPlan<float>.GraphReplayCount - replaysBefore;
            run.CaptureFailure = plan.GraphCaptureFailureReason;
            return run;
        }
    }

    internal static void AssertClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual, double tolerance, string name)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int i = 0; i < expected.Count; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * (1 + Math.Abs(expected[i])),
                $"{name}[{i}]: plain {expected[i]:R}, checkpointed {actual[i]:R}");
    }

    [SkippableTheory]
    [InlineData(-0.01f, false)]   // the state moves every step
    [InlineData(0f, false)]       // the state never moves
    [InlineData(-0.01f, true)]    // the state moves after the whole forward
    public void CheckpointSegments_TrainLikeThePlainPlan_WithLessScratch_InsideTheCapturedGraph(float nudge, bool deferNudge)
    {
        using var backendProof = RequireCuda();
        using var engine = new DirectGpuTensorEngine();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            const int steps = 8;   // past the graph warmup: the last steps replay the captured graph
            var plain = Train(engine, segmentBlocks: 0, steps, nudge: nudge, deferNudge: deferNudge);
            var checkpointed = Train(engine, segmentBlocks: 2, steps, nudge: nudge, deferNudge: deferNudge);

            Assert.Equal(0, plain.Segments);
            Assert.Equal(Blocks / 2, checkpointed.Segments);
            Assert.True(checkpointed.Recomputes > 0, "the segmented plan never recomputed a segment");
            Assert.True(checkpointed.Replays > 0, $"the segmented plan never replayed a captured graph: {checkpointed.CaptureFailure}");
            Assert.True(plain.Replays > 0, plain.CaptureFailure);

            // The same kernels on the same values: the recompute reproduces the forward exactly.
            AssertClose(plain.Losses, checkpointed.Losses, 1e-6, "loss");
            AssertClose(plain.Parameters, checkpointed.Parameters, 1e-6, "parameter");
            AssertClose(plain.State, checkpointed.State, 1e-6, "state");

            Assert.True(checkpointed.ScratchBytes < plain.ScratchBytes,
                $"checkpointed scratch {checkpointed.ScratchBytes} B is not below the plain plan's {plain.ScratchBytes} B");
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    [SkippableFact]
    public void CheckpointSegment_ThatReadsItsStateAfterUpdatingIt_IsRefused()
    {
        using var backendProof = RequireCuda();
        using var engine = new DirectGpuTensorEngine();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var ex = Assert.Throws<NotSupportedException>(() => Train(engine, segmentBlocks: 2, steps: 1, stateReadAfterNudge: true));
            Assert.Contains("updates a state in place and then reads it", ex.Message);
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }
}
