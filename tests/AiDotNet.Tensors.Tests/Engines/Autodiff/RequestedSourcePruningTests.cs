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
[Collection("EngineCurrentGlobalState")]
public sealed class RequestedSourcePruningTests : IDisposable
{
    private readonly IEngine _priorEngine;
    private readonly CpuEngine _engine;

    public RequestedSourcePruningTests()
    {
        _priorEngine = AiDotNetEngine.Current;
        _engine = new CpuEngine();
        AiDotNetEngine.Current = _engine;
    }

    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

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
        var input = Sequence([1, 2, 4, 4], 0.01f, 0.002f);
        var kernel = Sequence([2, 2, 3, 3], -0.02f, 0.001f);

        using var tape = new GradientTape<float>();
        var output = _engine.Conv2D(input, kernel, stride: 1, padding: 1, dilation: 1);
        var loss = _engine.ReduceSum(output);
        var gradients = tape.ComputeGradients(loss, Array.Empty<Tensor<float>>());

        Assert.Empty(gradients);
        Assert.Null(input.Grad);
        Assert.Null(kernel.Grad);
    }

    [Fact]
    public void SelectiveFusedLinear_WritesPaddedRentalBackingStorage()
    {
        const int size = 33; // 1089 elements: ArrayPool normally returns a padded 2048-slot array.
        var input = new Tensor<float>(new float[size * size], [size, size]);
        var weightData = new float[size * size];
        for (int i = 0; i < size; i++) weightData[i * size + i] = 1f;
        var weight = new Tensor<float>(weightData, [size, size]);
        var bias = new Tensor<float>(new float[size], [size]);

        using var tape = new GradientTape<float>();
        var output = _engine.FusedLinear(input, weight, bias, FusedActivationType.None);
        var loss = _engine.ReduceSum(output);
        var gradients = tape.ComputeGradients(loss, [input]);

        Assert.True(gradients.ContainsKey(input));
        foreach (float value in gradients[input].AsSpan()) Assert.Equal(1f, value, 5);
    }

    [Fact]
    public void LargeSourceFilter_PreservesRetainedAndHookedIntermediate()
    {
        var requested = new Tensor<float>([2f], [1]);
        var unrequested = new Tensor<float>([3f], [1]);
        using var tape = new GradientTape<float>(
            new GradientTapeOptions { EnableHooks = true });

        Tensor<float> requestedBranch = requested;
        Tensor<float> retainedBranch = unrequested;
        for (int i = 0; i < 105; i++)
        {
            requestedBranch = _engine.TensorMultiplyScalar(requestedBranch, 1f);
            retainedBranch = _engine.TensorMultiplyScalar(retainedBranch, 1f);
        }

        bool hookCalled = false;
        tape.RetainGrad(retainedBranch);
        tape.RegisterHook(retainedBranch, gradient =>
        {
            hookCalled = true;
            return gradient;
        });

        var loss = _engine.ReduceSum(_engine.TensorAdd(requestedBranch, retainedBranch));
        var gradients = tape.ComputeGradients(loss, [requested]);

        Assert.True(hookCalled);
        Assert.NotNull(retainedBranch.Grad);
        Assert.Equal(1f, retainedBranch.Grad![0], 5);
        Assert.Equal(1f, gradients[requested][0], 5);
    }

    private (float[] TrainableGradient, bool FrozenGradientProduced) Run(
        bool includeFrozenKernel)
    {
        var input = Sequence([1, 2, 5, 5], -0.05f, 0.003f);
        var trainableKernel = Sequence([3, 2, 3, 3], 0.02f, -0.0007f);
        var frozenKernel = Sequence([2, 3, 3, 3], -0.01f, 0.0005f);

        using var tape = new GradientTape<float>();
        var hidden = _engine.Conv2D(
            input, trainableKernel, stride: 1, padding: 1, dilation: 1);
        var output = _engine.Conv2D(
            hidden, frozenKernel, stride: 1, padding: 1, dilation: 1);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(output, output));
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
