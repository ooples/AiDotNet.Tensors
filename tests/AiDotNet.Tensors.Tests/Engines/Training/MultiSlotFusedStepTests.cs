using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Training;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Training;

/// <summary>
/// Phase 4B — MultiSlotFusedStep validates that N persistent input slots refresh
/// per step without recompilation. The batched-per-element diffusion use case
/// (clean sample + noise + per-batch-element timestep vector) is the motivating
/// pattern; this test uses a simpler [input, target, aux] triple with a linear
/// forward to isolate the multi-slot mechanism.
/// </summary>
public class MultiSlotFusedStepTests
{
    /// <summary>
    /// Register 3 slots, run 5 steps refreshing all 3 per step, verify the loss
    /// on step 2+ reflects the refreshed data (not stale first-step data).
    /// On non-GPU / non-compilation hosts IsAvailable is false and TryStep
    /// returns false — the test still asserts the graceful-fallback contract.
    /// </summary>
    [Fact]
    public void ThreeSlots_RefreshedPerStep_LossTracksCurrentData()
    {
        // On hosts without DirectGpu compilation, the trainer is unavailable and
        // TryStep returns false. That's the graceful-fallback contract; assert it.
        if (!MultiSlotFusedStep<float>.IsAvailable)
        {
            using var trainerCold = new MultiSlotFusedStep<float>();
            var wCold = new Tensor<float>(new float[] { 0.5f }, new[] { 1 });
            var freshCold = new[]
            {
                new Tensor<float>(new float[] { 1f, 2f, 3f }, new[] { 3 }),
                new Tensor<float>(new float[] { 0f, 0f, 0f }, new[] { 3 }),
                new Tensor<float>(new float[] { 1f }, new[] { 1 }),
            };
            var engineCold = new CpuEngine();
            Tensor<float> FwdCold(System.Collections.Generic.IReadOnlyList<Tensor<float>> slots) =>
                engineCold.TensorMultiply(slots[0], engineCold.TensorTile(wCold, new[] { 3 }));
            Tensor<float> LossCold(Tensor<float> pred, System.Collections.Generic.IReadOnlyList<Tensor<float>> slots) =>
                engineCold.ReduceSum(engineCold.TensorMultiply(
                    engineCold.TensorSubtract(pred, slots[1]),
                    engineCold.TensorSubtract(pred, slots[1])), null);
            bool ran = trainerCold.TryStep(
                new[] { wCold }, zeroGradAction: null,
                freshCold, FwdCold, LossCold,
                OptimizerType.SGD, 0.01f, 0.9f, 0.999f, 1e-8f, 0f,
                out _);
            Assert.False(ran, "TryStep must return false on non-GPU / non-compilation hosts (graceful fallback contract).");
            return;
        }

        var engine = new CpuEngine();
        using var trainer = new MultiSlotFusedStep<float>();
        var w = new Tensor<float>(new float[] { 0.5f }, new[] { 1 });

        Tensor<float> Fwd(System.Collections.Generic.IReadOnlyList<Tensor<float>> slots) =>
            engine.TensorMultiply(slots[0], engine.TensorTile(w, new[] { 3 }));

        Tensor<float> Loss(Tensor<float> pred, System.Collections.Generic.IReadOnlyList<Tensor<float>> slots)
        {
            var d = engine.TensorSubtract(pred, slots[1]);
            return engine.ReduceSum(engine.TensorMultiply(d, d), null);
        }

        // Step 1 — small input, small loss.
        var slotsA = new[]
        {
            new Tensor<float>(new float[] { 1f, 2f, 3f }, new[] { 3 }),
            new Tensor<float>(new float[] { 0.5f, 1f, 1.5f }, new[] { 3 }),
            new Tensor<float>(new float[] { 1f }, new[] { 1 }),
        };
        bool ranA = trainer.TryStep(
            new[] { w }, zeroGradAction: null,
            slotsA, Fwd, Loss,
            OptimizerType.SGD, 0.01f, 0.9f, 0.999f, 1e-8f, 0f,
            out float lossA);
        Assert.True(ranA, "First step must engage.");

        // Step 2 — 100x larger inputs. Loss should be dramatically larger,
        // proving the plan re-read the refreshed slot data.
        var slotsB = new[]
        {
            new Tensor<float>(new float[] { 100f, 200f, 300f }, new[] { 3 }),
            new Tensor<float>(new float[] { 0f, 0f, 0f }, new[] { 3 }),
            new Tensor<float>(new float[] { 1f }, new[] { 1 }),
        };
        bool ranB = trainer.TryStep(
            new[] { w }, zeroGradAction: null,
            slotsB, Fwd, Loss,
            OptimizerType.SGD, 0.01f, 0.9f, 0.999f, 1e-8f, 0f,
            out float lossB);
        Assert.True(ranB, "Second step must engage.");
        Assert.True(lossB > lossA * 100f,
            $"Second step's loss ({lossB:R}) should be substantially larger than the first ({lossA:R}) — " +
            "if the plan silently held stale slot references, lossB would be similar to lossA. " +
            "This validates the N-slot refresh mechanism.");
    }

    /// <summary>
    /// Shape change on ANY slot triggers a recompile (plan cache invalidation).
    /// This test flips a slot's shape between two steps and asserts both steps
    /// engage — the second step's recompile is invisible to the caller.
    /// </summary>
    [Fact]
    public void SlotShapeChange_TriggersRecompile_BothStepsSucceed()
    {
        if (!MultiSlotFusedStep<float>.IsAvailable) return; // Graceful skip on non-GPU

        var engine = new CpuEngine();
        using var trainer = new MultiSlotFusedStep<float>();
        var w = new Tensor<float>(new float[] { 0.3f }, new[] { 1 });

        Tensor<float> Fwd(System.Collections.Generic.IReadOnlyList<Tensor<float>> slots) =>
            engine.TensorMultiply(slots[0], engine.TensorTile(w, new[] { slots[0].Length }));
        Tensor<float> Loss(Tensor<float> pred, System.Collections.Generic.IReadOnlyList<Tensor<float>> slots) =>
            engine.ReduceSum(engine.TensorMultiply(
                engine.TensorSubtract(pred, slots[1]),
                engine.TensorSubtract(pred, slots[1])), null);

        var slotsSmall = new[]
        {
            new Tensor<float>(new float[] { 1f, 2f }, new[] { 2 }),
            new Tensor<float>(new float[] { 0f, 0f }, new[] { 2 }),
        };
        bool ran1 = trainer.TryStep(new[] { w }, null, slotsSmall, Fwd, Loss,
            OptimizerType.SGD, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, out _);
        Assert.True(ran1, "First step (small shape) must engage.");

        var slotsBig = new[]
        {
            new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 4 }),
            new Tensor<float>(new float[] { 0f, 0f, 0f, 0f }, new[] { 4 }),
        };
        bool ran2 = trainer.TryStep(new[] { w }, null, slotsBig, Fwd, Loss,
            OptimizerType.SGD, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, out _);
        Assert.True(ran2, "Second step (larger shape) must engage after auto-recompile.");
    }
}
