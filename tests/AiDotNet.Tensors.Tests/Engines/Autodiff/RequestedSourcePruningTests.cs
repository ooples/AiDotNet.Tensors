using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Verifies that requested-source reachability can omit frozen parameter gradients without
/// changing gradients for trainable parameters earlier in the graph.
/// </summary>
public sealed class RequestedSourcePruningTests
{
    [Fact]
    public void Conv2D_FrozenDownstreamKernel_PreservesUpstreamTrainableGradient()
    {
        var complete = Run(includeFrozenKernel: true);
        var pruned = Run(includeFrozenKernel: false);

        Assert.Equal(complete.TrainableGradient, pruned.TrainableGradient);
        Assert.True(complete.FrozenGradientProduced);
        Assert.False(pruned.FrozenGradientProduced);
    }

    [Fact]
    public void EmptyRequestedSources_ProducesNoGradients()
    {
        var engine = new CpuEngine();
        var input = Sequence([1, 2, 4, 4], 0.01f, 0.002f);
        var kernel = Sequence([2, 2, 3, 3], -0.02f, 0.001f);

        using var tape = new GradientTape<float>();
        var output = engine.Conv2D(input, kernel, stride: 1, padding: 1, dilation: 1);
        var loss = engine.ReduceSum(output);
        var gradients = tape.ComputeGradients(loss, Array.Empty<Tensor<float>>());

        Assert.Empty(gradients);
        Assert.Null(input.Grad);
        Assert.Null(kernel.Grad);
    }

    private static (float[] TrainableGradient, bool FrozenGradientProduced) Run(
        bool includeFrozenKernel)
    {
        var engine = new CpuEngine();
        var input = Sequence([1, 2, 5, 5], -0.05f, 0.003f);
        var trainableKernel = Sequence([3, 2, 3, 3], 0.02f, -0.0007f);
        var frozenKernel = Sequence([2, 3, 3, 3], -0.01f, 0.0005f);

        using var tape = new GradientTape<float>();
        var hidden = engine.Conv2D(
            input, trainableKernel, stride: 1, padding: 1, dilation: 1);
        var output = engine.Conv2D(
            hidden, frozenKernel, stride: 1, padding: 1, dilation: 1);
        var loss = engine.ReduceSum(engine.TensorMultiply(output, output));
        Tensor<float>[] sources = includeFrozenKernel
            ? [trainableKernel, frozenKernel]
            : [trainableKernel];

        var gradients = tape.ComputeGradients(loss, sources);

        return (
            gradients[trainableKernel].AsSpan().ToArray(),
            gradients.ContainsKey(frozenKernel));
    }

    private static Tensor<float> Sequence(int[] shape, float start, float step)
    {
        int length = shape.Aggregate(1, (product, dimension) => product * dimension);
        var values = new float[length];
        for (int i = 0; i < values.Length; i++) values[i] = start + i * step;
        return new Tensor<float>(values, shape);
    }
}
