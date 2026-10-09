using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A compiled training step keeps the previous step's gradient buffers in its map and clears each on its first write
/// of the step. SignBackward used to return as soon as its input had an entry, so a buffer whose only contribution was
/// a Sign kept the previous step's values instead of zero.
/// </summary>
public class SignBackwardStaleGradientTests
{
    [Fact]
    public void SignBackward_ZeroesAStaleBufferFromAnEarlierStep()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new float[] { -2f, 0.5f, 3f, -0.25f }, new[] { 4 });
        var output = new Tensor<float>(new float[] { -1f, 1f, 1f, -1f }, new[] { 4 });
        var gradOutput = new Tensor<float>(new float[] { 1f, 1f, 1f, 1f }, new[] { 4 });
        var stale = new Tensor<float>(new float[] { 7f, 7f, 7f, 7f }, new[] { 4 });
        var grads = new Dictionary<Tensor<float>, Tensor<float>> { [input] = stale };

        int previous = DifferentiableOps.GradWriteGeneration;
        try
        {
            DifferentiableOps.GradWriteGeneration = DifferentiableOps.NextGradWriteGeneration();
            BackwardFunctions<float>.SignBackward(gradOutput, new[] { input }, output, new object[0], engine, grads);
        }
        finally
        {
            DifferentiableOps.GradWriteGeneration = previous;
        }

        var grad = grads[input].ToArray();
        Assert.Equal(new float[] { 0f, 0f, 0f, 0f }, grad);
    }

    [Fact]
    public void SignBackward_LeavesAGradientWrittenThisStepUnchanged()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new float[] { -2f, 0.5f }, new[] { 2 });
        var output = new Tensor<float>(new float[] { -1f, 1f }, new[] { 2 });
        var gradOutput = new Tensor<float>(new float[] { 1f, 1f }, new[] { 2 });
        var grads = new Dictionary<Tensor<float>, Tensor<float>>();

        int previous = DifferentiableOps.GradWriteGeneration;
        try
        {
            DifferentiableOps.GradWriteGeneration = DifferentiableOps.NextGradWriteGeneration();
            DifferentiableOps.AccumulateGrad(grads, input, new Tensor<float>(new float[] { 4f, -5f }, new[] { 2 }), engine);
            BackwardFunctions<float>.SignBackward(gradOutput, new[] { input }, output, new object[0], engine, grads);
        }
        finally
        {
            DifferentiableOps.GradWriteGeneration = previous;
        }

        Assert.Equal(new float[] { 4f, -5f }, grads[input].ToArray());
    }
}
