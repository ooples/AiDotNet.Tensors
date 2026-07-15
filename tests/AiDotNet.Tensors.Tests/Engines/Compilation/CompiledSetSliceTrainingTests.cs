using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Regression coverage for neighborhood-assembly graphs used by mesh models.
/// </summary>
[Collection("CompiledTrainingPlanSerial")]
public class CompiledSetSliceTrainingTests
{
    [Fact]
    public void TensorConcatenate_ContiguousLastAxis_CopiesRowBlocksInOrder()
    {
        var engine = new CpuEngine();
        var left = Tensor(new[] { 2, 2 }, 1f, 2f, 3f, 4f);
        var middle = Tensor(new[] { 2, 1 }, 5f, 6f);
        var right = Tensor(new[] { 2, 2 }, 7f, 8f, 9f, 10f);

        var result = engine.TensorConcatenate(new[] { left, middle, right }, axis: 1);

        Assert.Equal(new[] { 2, 5 }, result.Shape.ToArray());
        Assert.Equal(
            new[] { 1f, 2f, 5f, 7f, 8f, 3f, 4f, 6f, 9f, 10f },
            result.AsSpan().ToArray());
    }

    [Fact]
    public void TensorSplit_EagerBackward_RestoresChunkGradientAtOriginalOffset()
    {
        var engine = new CpuEngine();
        var input = Tensor(
            new[] { 2, 4 },
            1f, 2f, 3f, 4f,
            5f, 6f, 7f, 8f);

        using var tape = new GradientTape<float>();
        var chunks = engine.TensorSplit(input, numSplits: 2, axis: 1);
        var loss = engine.ReduceSum(chunks[1], null);
        var gradient = tape.ComputeGradients(loss, new[] { input })[input];

        Assert.Equal(
            new[] { 0f, 0f, 1f, 1f, 0f, 0f, 1f, 1f },
            gradient.AsSpan().ToArray());
    }

    [Fact]
    public void TensorSetSlice_EagerBackward_ChainedOverwritesRouteExactGradients()
    {
        var engine = new CpuEngine();
        var destination = Tensor(
            new[] { 2, 4 },
            1f, 2f, 3f, 4f,
            5f, 6f, 7f, 8f);
        var sourceA = Tensor(new[] { 2, 1 }, 9f, 10f);
        var sourceB = Tensor(new[] { 2, 2 }, 11f, 12f, 13f, 14f);

        using var tape = new GradientTape<float>();
        var withA = engine.TensorSetSlice(destination, sourceA, new[] { 0, 0 });
        var withB = engine.TensorSetSlice(withA, sourceB, new[] { 0, 2 });
        var loss = engine.ReduceSum(withB, null);
        var gradients = tape.ComputeGradients(loss, new[] { destination, sourceA, sourceB });

        Assert.Equal(
            new[] { 0f, 1f, 0f, 0f, 0f, 1f, 0f, 0f },
            gradients[destination].AsSpan().ToArray());
        Assert.Equal(new[] { 1f, 1f }, gradients[sourceA].AsSpan().ToArray());
        Assert.Equal(new[] { 1f, 1f, 1f, 1f }, gradients[sourceB].AsSpan().ToArray());
    }

    [Fact]
    public void TensorSetSlice_CompiledBackward_ConnectsOverwrittenSourceToLoss()
    {
        var engine = new CpuEngine();
        var input = Tensor(new[] { 2, 2 }, 0.2f, -0.4f, 0.7f, 0.3f);
        var upstreamWeight = Tensor(new[] { 2, 2 }, 0.5f, -0.2f, 0.1f, 0.8f);
        var readoutWeight = Tensor(new[] { 4, 1 }, 0.4f, -0.6f, 0.9f, 0.2f);
        var destination = new Tensor<float>(new[] { 2, 4 });

        using ICompiledTrainingPlan<float> plan = Compile(
            engine,
            new[] { upstreamWeight, readoutWeight },
            () =>
            {
                var source = engine.TensorMatMul(input, upstreamWeight);
                var assembled = engine.TensorSetSlice(destination, source, new[] { 0, 1 });
                var output = engine.TensorMatMul(assembled, readoutWeight);
                return engine.ReduceSum(engine.TensorMultiply(output, output), null);
            });

        plan.Step();

        var upstreamGradient = plan.Gradients[0].AsSpan().ToArray();
        Assert.All(upstreamGradient, value => Assert.True(IsFinite(value)));
        Assert.Contains(upstreamGradient, value => MathF.Abs(value) > 1e-6f);
    }

    [Fact]
    public void AdamStep_DisconnectedParameter_RemainsFiniteAndUnchanged()
    {
        var engine = new CpuEngine();
        var input = Tensor(new[] { 2, 2 }, 0.2f, -0.4f, 0.7f, 0.3f);
        var connectedWeight = Tensor(new[] { 2, 1 }, 0.5f, -0.2f);
        var disconnectedWeight = Tensor(new[] { 2, 2 }, 0.1f, 0.2f, 0.3f, 0.4f);
        var before = disconnectedWeight.AsSpan().ToArray();

        using ICompiledTrainingPlan<float> plan = Compile(
            engine,
            new[] { connectedWeight, disconnectedWeight },
            () =>
            {
                var output = engine.TensorMatMul(input, connectedWeight);
                return engine.ReduceSum(engine.TensorMultiply(output, output), null);
            });

        plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.01f);
        plan.Step();

        var after = disconnectedWeight.AsSpan().ToArray();
        Assert.All(after, value => Assert.True(IsFinite(value)));
        Assert.Equal(before, after);
    }

    private static ICompiledTrainingPlan<float> Compile(
        CpuEngine engine,
        Tensor<float>[] parameters,
        Func<Tensor<float>> forward)
    {
        using var scope = GraphMode.Enable();
        _ = forward();
        return scope.CompileTraining(parameters);
    }

    private static Tensor<float> Tensor(int[] shape, params float[] values)
        => new(values, shape);

    private static bool IsFinite(float value)
        => !float.IsNaN(value) && !float.IsInfinity(value);
}
