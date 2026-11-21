using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public sealed class CompiledWeightNormForwardSharingGuardTests
{
    [Theory]
    [InlineData("retained")]
    [InlineData("loss")]
    [InlineData("parameter")]
    [InlineData("saved")]
    [InlineData("saved_independent_array_alias")]
    [InlineData("raw_saved_array_alias")]
    [InlineData("external_view")]
    [InlineData("same_array_separate_storage")]
    [InlineData("partial_output")]
    [InlineData("strided_output")]
    [InlineData("strided_source")]
    [InlineData("partial_source")]
    [InlineData("different_epsilon")]
    public void Sharing_preserves_observation_alias_and_layout_contracts(string exclusion)
    {
        IEngine engine = new CpuEngine();
        var direction = new Tensor<float>(new[] { .5f, -.75f, .125f, .375f, -.625f, -.5f, .25f, .625f, -.125f, .375f }, new[] { 2, 5 });
        var gain = new Tensor<float>(new[] { .75f, -1.25f }, new[] { 2 });
        var first = new Tensor<float>(new[] { 2, 5 });
        var second = new Tensor<float>(new[] { 2, 5 });
        var loss = new Tensor<float>(new[] { 1 });
        var retained = new HashSet<Tensor<float>>();
        Tensor<float>? externalView = null;
        Tensor<float>? savedAlias = null;
        float[]? savedArray = null;
        if (exclusion == "retained") retained.Add(first);
        if (exclusion == "loss") loss = first;
        if (exclusion == "parameter") first = direction;
        if (exclusion == "external_view") externalView = first.Reshape(10);
        if (exclusion == "same_array_separate_storage")
        {
            var array = new float[10];
            first = new Tensor<float>(array, new[] { 2, 5 });
            second = new Tensor<float>(array, new[] { 2, 5 });
            Assert.NotSame(first._storage, second._storage);
            Assert.Same(first.GetBackingArrayForCacheLookupUnsafe(), second.GetBackingArrayForCacheLookupUnsafe());
        }
        if (exclusion == "partial_output") first = Tensor<float>.FromMemory(new float[12].AsMemory(1, 10), new[] { 2, 5 });
        if (exclusion == "strided_output") first = new Tensor<float>(new[] { 5, 2 }).Transpose(new[] { 1, 0 });
        if (exclusion == "strided_source") direction = new Tensor<float>(direction.ToArray(), new[] { 5, 2 }).Transpose(new[] { 1, 0 });
        if (exclusion == "partial_source")
        {
            var padded = new float[12];
            direction.ToArray().CopyTo(padded, 1);
            direction = Tensor<float>.FromMemory(padded.AsMemory(1, 10), new[] { 2, 5 });
        }
        var steps = new List<CompiledStep<float>>
        {
            WeightNormSharingFixture.Step(first, direction, gain),
            WeightNormSharingFixture.Step(second, direction, gain, exclusion == "different_epsilon" ? .125 : .03125),
        };
        if (exclusion == "saved")
            steps.Add(new CompiledStep<float>("saved-output-reader", (_, destination) => destination[0] = 1f,
                new Tensor<float>(new[] { 1 }), Array.Empty<Tensor<float>>(), savedState: new object[] { first }));
        if (exclusion == "saved_independent_array_alias")
        {
            savedAlias = new Tensor<float>(second.GetBackingArrayForCacheLookupUnsafe()!, new[] { 2, 5 });
            Assert.NotSame(second._storage, savedAlias._storage);
            Assert.Same(second.GetBackingArrayForCacheLookupUnsafe(), savedAlias.GetBackingArrayForCacheLookupUnsafe());
            var alias = savedAlias;
            steps.Add(new CompiledStep<float>("saved-output-reader", (_, destination) => destination[0] = alias[0],
                new Tensor<float>(new[] { 1 }), Array.Empty<Tensor<float>>(), savedState: new object[] { savedAlias }));
        }
        if (exclusion == "raw_saved_array_alias")
        {
            savedArray = second.GetBackingArrayForCacheLookupUnsafe()!;
            var alias = savedArray;
            steps.Add(new CompiledStep<float>("saved-array-reader", (_, destination) => destination[0] = alias[0],
                new Tensor<float>(new[] { 1 }), Array.Empty<Tensor<float>>(), savedState: new object[] { savedArray }));
        }
        object firstStorage = first._storage;
        object secondStorage = second._storage;
        bool fixedSourceView = exclusion is "strided_source" or "partial_source";
        float[] directionBefore = direction.ToArray();
        float[] gainBefore = gain.ToArray();
        Assert.Equal(fixedSourceView ? 1 : 0, WeightNormSharingFixture.Apply(steps, direction, gain, loss, retained, engine));
        Assert.Same(firstStorage, first._storage);
        if (fixedSourceView) Assert.Same(first._storage, second._storage);
        else Assert.Same(secondStorage, second._storage);
        foreach (var step in steps)
        {
            float[]? expected = step.OpName == "TensorWeightNorm"
                ? engine.TensorWeightNorm(step.Inputs[0], step.Inputs[1], (double)step.SavedState![0]).ToArray() : null;
            step.Execute(engine, step.OutputBuffer);
            if (expected is not null) Assert.Equal(expected, step.OutputBuffer.ToArray());
        }
        if (fixedSourceView)
        {
            Assert.Equal(directionBefore, direction.ToArray());
            Assert.Equal(gainBefore, gain.ToArray());
        }
        if (savedAlias is not null) Assert.Equal(second[0], steps[steps.Count - 1].OutputBuffer[0]);
        if (savedArray is not null) Assert.Equal(second[0], steps[steps.Count - 1].OutputBuffer[0]);
        GC.KeepAlive(externalView);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Source_write_or_unknown_host_state_between_producers_prevents_sharing(bool hostState)
    {
        var engine = new CpuEngine();
        var direction = new Tensor<float>(new[] { .5f, -.75f, .125f, .375f, -.625f }, new[] { 1, 5 });
        var gain = new Tensor<float>(new[] { .75f }, new[] { 1 });
        var first = new Tensor<float>(direction.Shape.ToArray());
        var second = new Tensor<float>(direction.Shape.ToArray());
        var write = new CompiledStep<float>("state-write", (_, destination) =>
        {
            direction[0] += .75f;
            if (hostState) destination[0] = 1f;
        }, hostState ? new Tensor<float>(new[] { 1 }) : direction,
            hostState ? Array.Empty<Tensor<float>>() : new[] { direction }) { ReplaysHostState = hostState };
        var steps = new List<CompiledStep<float>> { WeightNormSharingFixture.Step(first, direction, gain), write,
            WeightNormSharingFixture.Step(second, direction, gain) };
        Assert.Equal(0, WeightNormSharingFixture.Apply(steps, direction, gain, new Tensor<float>(new[] { 1 }), new(), engine));
        float[] before = engine.TensorWeightNorm(direction, gain, .03125).ToArray();
        foreach (var step in steps) step.Execute(engine, step.OutputBuffer);
        Assert.Equal(before, first.ToArray());
        Assert.Equal(engine.TensorWeightNorm(direction, gain, .03125).ToArray(), second.ToArray());
        Assert.NotEqual(first[0], second[0]);
    }

    [Fact]
    public void A_write_after_shared_pair_starts_a_new_buffer_and_keeps_prior_values()
    {
        var engine = new CpuEngine();
        var direction = new Tensor<float>(new[] { .5f, -.75f, .125f, .375f, -.625f }, new[] { 1, 5 });
        var gain = new Tensor<float>(new[] { .75f }, new[] { 1 });
        var outputs = new[] { new Tensor<float>(new[] { 1, 5 }), new Tensor<float>(new[] { 1, 5 }),
            new Tensor<float>(new[] { 1, 5 }), new Tensor<float>(new[] { 1, 5 }) };
        var steps = new List<CompiledStep<float>>
        {
            WeightNormSharingFixture.Step(outputs[0], direction, gain),
            WeightNormSharingFixture.Step(outputs[1], direction, gain),
            new("source-write", (_, destination) => destination[0] += .75f, direction, new[] { direction }),
            WeightNormSharingFixture.Step(outputs[2], direction, gain),
            WeightNormSharingFixture.Step(outputs[3], direction, gain),
        };
        Assert.Equal(2, WeightNormSharingFixture.Apply(steps, direction, gain, new Tensor<float>(new[] { 1 }), new(), engine));
        Assert.Same(outputs[0]._storage, outputs[1]._storage);
        Assert.Same(outputs[2]._storage, outputs[3]._storage);
        Assert.NotSame(outputs[0]._storage, outputs[2]._storage);
        float[] expectedBefore = engine.TensorWeightNorm(direction, gain, .03125).ToArray();
        foreach (var step in steps) step.Execute(engine, step.OutputBuffer);
        float[] expectedAfter = engine.TensorWeightNorm(direction, gain, .03125).ToArray();
        Assert.NotEqual(expectedBefore[0], expectedAfter[0]);
        Assert.Equal(expectedBefore, outputs[0].ToArray());
        Assert.Equal(expectedBefore, outputs[1].ToArray());
        Assert.Equal(expectedAfter, outputs[2].ToArray());
        Assert.Equal(expectedAfter, outputs[3].ToArray());
    }
}

internal static class WeightNormSharingFixture
{
    internal static CompiledStep<float> Step(Tensor<float> output, Tensor<float> direction, Tensor<float> gain, double epsilon = .03125)
        => new("TensorWeightNorm", (engine, destination) => engine.TensorWeightNormInto(destination, direction, gain, epsilon),
            output, new[] { direction, gain }, BackwardFunctions<float>.WeightNormBackward, new object[] { epsilon });

    internal static int Apply(IReadOnlyList<CompiledStep<float>> steps, Tensor<float> direction, Tensor<float> gain,
        Tensor<float> loss, HashSet<Tensor<float>> retained, IEngine engine)
    {
        const string flag = "AIDOTNET_COMPILED_WEIGHTNORM_FORWARD_SHARE";
        string? previous = Environment.GetEnvironmentVariable(flag);
        Environment.SetEnvironmentVariable(flag, null);
        try { return CompiledTrainingPlan<float>.ShareWeightNormForwardStorage(steps, new[] { direction, gain }, loss, retained, engine); }
        finally { Environment.SetEnvironmentVariable(flag, previous); }
    }
}
