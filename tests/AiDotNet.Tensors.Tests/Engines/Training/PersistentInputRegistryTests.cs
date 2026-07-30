using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Training;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Training;

/// <summary>
/// Phase 4B — multi-slot persistent-input registry validates that a compiled plan
/// captures each slot's tensor reference once, and subsequent RefreshSlot calls
/// make the plan read new data on replay without recompilation.
/// </summary>
public class PersistentInputRegistryTests
{
    /// <summary>
    /// Register a slot, run a forward that captures it, refresh the slot with new
    /// data, and confirm re-running the forward reads the fresh data. This is the
    /// core contract: reference stability + data-refresh.
    /// </summary>
    [Fact]
    public void RefreshSlot_ReferenceStableDataChanges()
    {
        var engine = new CpuEngine();
        using var registry = new PersistentInputRegistry<float>();

        var (slot0, idx0) = registry.Register(new[] { 3 });
        // Populate with initial values.
        var v1 = new Tensor<float>(new float[] { 1f, 2f, 3f }, new[] { 3 });
        registry.RefreshSlot(idx0, v1);

        // Sum via engine — captures the slot0 reference.
        var sumV1 = engine.ReduceSum(slot0, null);
        Assert.Equal(6f, sumV1.ToArray()[0]);

        // Refresh with new data; the reference is the SAME tensor, so a second call
        // reads the new values.
        var v2 = new Tensor<float>(new float[] { 10f, 20f, 30f }, new[] { 3 });
        registry.RefreshSlot(idx0, v2);
        Assert.Same(slot0, registry.GetSlot(idx0));  // reference stable

        var sumV2 = engine.ReduceSum(slot0, null);
        Assert.Equal(60f, sumV2.ToArray()[0]);
    }

    /// <summary>
    /// Diffusion-training pattern: three slots — noisy sample, target noise,
    /// per-batch-element timesteps — all refreshed per step. Confirms that
    /// registering multiple slots with different shapes works.
    /// </summary>
    [Fact]
    public void MultipleSlots_DifferentShapes_AllRefreshable()
    {
        using var registry = new PersistentInputRegistry<float>();

        var (noisy, iNoisy) = registry.Register(new[] { 4, 8 });   // [B, features]
        var (target, iTarget) = registry.Register(new[] { 4, 8 });  // [B, features] — noise
        var (timesteps, iTs) = registry.Register(new[] { 4 });      // [B] — per-element timesteps

        Assert.Equal(3, registry.SlotCount);
        Assert.Equal(32, noisy.Length);
        Assert.Equal(32, target.Length);
        Assert.Equal(4, timesteps.Length);

        // Refresh each slot independently.
        var freshNoisy = new Tensor<float>(new float[32], new[] { 4, 8 });
        for (int i = 0; i < 32; i++) freshNoisy.AsWritableSpan()[i] = i * 0.1f;
        registry.RefreshSlot(iNoisy, freshNoisy);
        Assert.Equal(0.1f, noisy.AsSpan()[1]);

        var freshTs = new Tensor<float>(new float[] { 100f, 200f, 300f, 400f }, new[] { 4 });
        registry.RefreshSlot(iTs, freshTs);
        Assert.Equal(300f, timesteps.AsSpan()[2]);

        // The target slot is still the initial zero tensor — refreshing others
        // doesn't touch it.
        Assert.Equal(0f, target.AsSpan()[0]);
    }

    /// <summary>
    /// A compiled persistent-tape plan captures the slot's reference; subsequent
    /// RefreshSlot calls change the loss the plan produces without any
    /// invalidate-and-recompile. This is the full end-to-end validation of the
    /// batched-per-element timestep use case.
    /// </summary>
    [Fact]
    public void CompiledPlanReadsFreshDataAfterRefresh()
    {
        var engine = new CpuEngine();
        using var registry = new PersistentInputRegistry<float>();
        var (input, iInput) = registry.Register(new[] { 3 });
        var weight = new Tensor<float>(new float[] { 0.5f }, new[] { 1 });

        // Initial data
        registry.RefreshSlot(iInput, new Tensor<float>(new float[] { 1f, 2f, 3f }, new[] { 3 }));

        // Compile the plan referring to the persistent-registered slot.
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var y = engine.TensorMultiply(input, engine.TensorTile(weight, new[] { 3 }));
            engine.ReduceSum(y, null);
            plan = scope.CompileTraining(new[] { weight });
        }

        using (plan)
        {
            // Step 1: loss = sum([1,2,3] * 0.5) = 3.
            var loss1 = plan.Step()[0];
            Assert.True(System.MathF.Abs(loss1 - 3.0f) < 1e-4f, $"Step 1 loss expected 3.0, got {loss1:R}");

            // Refresh the input slot WITHOUT recompiling; step 2 must read the new data.
            registry.RefreshSlot(iInput, new Tensor<float>(new float[] { 10f, 20f, 30f }, new[] { 3 }));

            var loss2 = plan.Step()[0];
            // With Adam having taken one step on weight (starting at 0.5, grad = [1,2,3]·1 → sum = 6, step drives weight down)
            // the point of this test isn't the exact value — it's that loss2 REFLECTS the fresh input data
            // (would be ~30x larger absent the weight update). If the plan silently held stale references, loss2
            // would still be near 3. Assert loss2 is much larger than loss1 to confirm refresh took effect.
            Assert.True(loss2 > loss1 * 3.0f,
                $"After 10x-scaled input, loss2 ({loss2:R}) should be substantially larger than loss1 ({loss1:R}). " +
                "Compiled plan may be holding stale input reference (regresses Phase 4B).");
        }
    }
}
