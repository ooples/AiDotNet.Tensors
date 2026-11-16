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
    public void Nested_saved_storage_and_live_gradient_aliases_are_not_released()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "GPU required.");
        var leaf = new Tensor<float>(new[] { 2f, 3f }, new[] { 2 });
        var saved = gpu.TensorMultiply(leaf, leaf);
        var first = new Tensor<float>(new[] { 2 });
        var second = new Tensor<float>(new[] { 2 });
        var loss = new Tensor<float>(new[] { 1 });
        var context = new AutogradContext();
        context.SaveForBackward(saved);
        var steps = new[] { new BackwardStep<float> { Output = first, Inputs = Array.Empty<Tensor<float>>(),
            SavedState = new object[] { new object[] { context } }, Backward = null! } };
        var releaser = new DeviceGradientReleaser<float>(steps, 1, new HashSet<object>(), loss, new[] { leaf }, null);
        var grads = new Dictionary<Tensor<float>, Tensor<float>> { [first] = saved };
        releaser.Observe(grads, Array.Empty<Tensor<float>>(), true);
        releaser.AfterStep(grads, first, gpu);
        Assert.True(grads.ContainsKey(first));
        Assert.Equal(new[] { 4f, 9f }, saved.ToArray());

        var shared = gpu.TensorAdd(leaf, leaf);
        grads[first] = shared;
        grads[second] = shared.Reshape(2);
        releaser.Observe(grads, new[] { first, second }, false);
        releaser.AfterStep(grads, first, gpu);
        Assert.True(grads.ContainsKey(first));
        grads[second] = gpu.TensorMultiply(leaf, leaf);
        releaser.Observe(grads, new[] { second }, false);
        releaser.AfterStep(grads, first, gpu);
        Assert.False(grads.ContainsKey(first));
        Assert.Equal(new[] { 4f, 9f }, grads[second].ToArray());
    }

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
