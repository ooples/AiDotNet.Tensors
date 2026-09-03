using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public sealed class BackwardStorageReleasePlanTests
{
    [Fact]
    public void AliasStorage_IsReleasedAfterFinalSavedStateUse_NotAliasStep()
    {
        using var saved = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        using var alias = saved.Reshape(new[] { 4 });
        using var laterOutput = new Tensor<float>(new[] { 5f }, new[] { 1 });

        var steps = new[]
        {
            new BackwardStep<float>
            {
                Output = alias,
                Inputs = Array.Empty<Tensor<float>>(),
                Backward = null!
            },
            new BackwardStep<float>
            {
                Output = laterOutput,
                Inputs = Array.Empty<Tensor<float>>(),
                Backward = null!,
                SavedState = new object[] { saved }
            }
        };

        var plan = BackwardStorageReleasePlan<float>.Create(steps, steps.Length);

        Assert.False(plan.IsStorageScheduledAfterStep(alias, 0));
        Assert.True(plan.IsStorageScheduledAfterStep(alias, 1));
    }

    [Fact]
    public void NestedAndContextSavedTensors_ExtendStorageLifetimeAndArePinned()
    {
        using var saved = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        using var alias = saved.Reshape(new[] { 4 });
        using var laterOutput = new Tensor<float>(new[] { 5f }, new[] { 1 });
        var context = new AutogradContext();
        context.SaveForBackward(saved);
        var savedState = new object[] { new object[] { context } };

        var entry = new TapeEntry<float> { SavedState = savedState };
        DifferentiableOps.PinSavedStateTensors(ref entry);
        Assert.True(entry.SavedStatePinsHeld);
        Assert.True(saved._pinnedByTape);

        var steps = new[]
        {
            new BackwardStep<float>
            {
                Output = alias,
                Inputs = Array.Empty<Tensor<float>>(),
                Backward = null!
            },
            new BackwardStep<float>
            {
                Output = laterOutput,
                Inputs = Array.Empty<Tensor<float>>(),
                Backward = null!,
                SavedState = savedState
            }
        };
        var plan = BackwardStorageReleasePlan<float>.Create(steps, steps.Length);

        Assert.False(plan.IsStorageScheduledAfterStep(saved, 0));
        Assert.True(plan.IsStorageScheduledAfterStep(saved, 1));

        DifferentiableOps.UnpinSavedStateTensors(ref entry);
        Assert.False(entry.SavedStatePinsHeld);
        Assert.False(saved._pinnedByTape);
    }
}
