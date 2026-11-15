using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public class BackwardOwnershipReviewTests
{
    [SkippableFact]
    public void Unfiltered_nonleaf_gradient_survives_tape_disposal()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "GPU required.");
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var x = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
            Tensor<float> intermediate;
            Dictionary<Tensor<float>, Tensor<float>> grads;
            using (var tape = new GradientTape<float>())
            {
                intermediate = gpu.TensorMultiply(x, x);
                var loss = gpu.ReduceSum(gpu.TensorMultiply(intermediate, intermediate));
                grads = tape.ComputeGradients(loss);
            }
            Assert.Equal(new[] { 2f, 8f }, grads[intermediate].ToArray());
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [SkippableFact]
    public void Opaque_backward_published_captured_leaf_gradient_survives_scope()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "GPU required.");
        var leaf = new Tensor<float>(new[] { 2f, 3f }, new[] { 2 });
        var loss = new Tensor<float>(new[] { 1f }, new[] { 1 });
        var chain = new CompiledDelegateChain<float>(new[]
        {
            new BackwardStep<float>
            {
                Output = loss,
                Inputs = Array.Empty<Tensor<float>>(),
                Backward = (_, _, _, _, engine, grads) => grads[leaf] = engine.TensorMultiply(leaf, leaf)
            }
        });
        var result = chain.Execute(loss, null, gpu);
        Assert.Equal(new[] { 4f, 9f }, result[leaf].ToArray());
    }
}
