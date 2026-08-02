using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Integration tests proving Tensor.Reshape() propagates GradFn for gradient tape.
/// Verifies fix for issue #123.
/// </summary>
public class ReshapeGradientTests
{
    [Fact]
    public void Reshape_SetsGradFn_WhenTapeActive()
    {
        using var tape = new GradientTape<float>();
        var original = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var reshaped = original.Reshape(3, 2);

        Assert.NotNull(reshaped.GradFn);
    }

    [Fact]
    public void Reshape_NoGradFn_WhenNoTape()
    {
        var original = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var reshaped = original.Reshape(4);

        Assert.Null(reshaped.GradFn);
    }

    [Fact]
    public void Reshape_GradientsFlowBackward()
    {
        var engine = new CpuEngine();

        // weight is the parameter we want gradients for
        var weight = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var input = new Tensor<float>(new float[] { 1, 0, 0, 1 }, new[] { 2, 2 });

        using var tape = new GradientTape<float>();

        // Forward: matmul -> reshape -> sum
        var matmulResult = engine.TensorMatMul(input, weight); // [2,2]
        var reshaped = matmulResult.Reshape(4);                // [4] — this was breaking gradients
        var loss = engine.ReduceSum(reshaped, new[] { 0 }, false); // scalar

        var grads = tape.ComputeGradients(loss);

        // Gradient must flow through reshape back to weight
        Assert.True(grads.ContainsKey(weight),
            "Gradient for weight must exist — Reshape must not break the gradient chain");
        Assert.True(grads[weight].Length == 4, "Gradient shape must match weight shape");

        // Verify gradient is non-zero (sum of all outputs = gradient of 1 everywhere)
        var gradData = grads[weight].GetDataArray();
        bool anyNonZero = false;
        for (int i = 0; i < gradData.Length; i++)
            if (gradData[i] != 0f) anyNonZero = true;
        Assert.True(anyNonZero, "Gradient values must be non-zero");
    }

    [Fact]
    public void Reshape_MultipleReshapes_GradientsFlowThrough()
    {
        var engine = new CpuEngine();
        var weight = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });

        using var tape = new GradientTape<float>();

        // Multiple reshapes in sequence
        var r1 = weight.Reshape(3, 2);    // [3,2]
        var r2 = r1.Reshape(6);           // [6]
        var r3 = r2.Reshape(2, 3);        // [2,3]
        var loss = engine.ReduceSum(r3, new[] { 0, 1 }, false);

        var grads = tape.ComputeGradients(loss);

        Assert.True(grads.ContainsKey(weight),
            "Gradient must flow through multiple reshapes");
    }

    [Fact]
    public void Reshape_3DTo2D_GradientsFlowForTraining()
    {
        // Simulates the common pattern: batch reshape in Forward method
        var engine = new CpuEngine();
        var input = new Tensor<float>(
            new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
            new[] { 2, 2, 3 }); // [batch=2, seq=2, features=3]
        var weight = new Tensor<float>(
            new float[] { 1, 0, 0, 1, 0, 0 },
            new[] { 3, 2 }); // [features=3, output=2]

        using var tape = new GradientTape<float>();

        // Reshape [2,2,3] -> [4,3] for matmul (common in transformer layers)
        var flat = input.Reshape(4, 3);
        var output = engine.TensorMatMul(flat, weight); // [4, 2]
        var loss = engine.ReduceSum(output, new[] { 0, 1 }, false);

        var grads = tape.ComputeGradients(loss);

        Assert.True(grads.ContainsKey(weight),
            "Weight gradient must exist after reshape+matmul");
        Assert.True(grads.ContainsKey(input),
            "Input gradient must flow through reshape");
    }

    [Fact]
    public void MetadataViewChain_ExpandSqueezeAndPermute_PropagatesGradient()
    {
        var engine = new CpuEngine();
        var parameter = new Tensor<float>(new[] { 2f, 3f }, new[] { 1, 2 });

        using var tape = new GradientTape<float>();
        var expanded = parameter.ExpandDims(0);       // [1,1,2]
        var squeezed = expanded.Squeeze(0);          // [1,2]
        var permuted = squeezed.Transpose(new[] { 1, 0 }); // [2,1]
        var loss = engine.ReduceSum(permuted, null);

        Assert.NotNull(expanded.GradFn);
        Assert.NotNull(squeezed.GradFn);
        Assert.NotNull(permuted.GradFn);

        var gradients = tape.ComputeGradients(loss);

        Assert.True(gradients.ContainsKey(parameter));
        Assert.Equal(new[] { 1f, 1f }, gradients[parameter].ToArray());
    }

    [Fact]
    public void SubTensorViewChain_ScattersGradientToFixedLeadingIndices()
    {
        var engine = new CpuEngine();
        var parameter = new Tensor<float>(
            new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f },
            new[] { 2, 2, 2 });

        using var tape = new GradientTape<float>();
        var selected = parameter.SubTensor(1, 0);
        var loss = engine.ReduceSum(selected, null);
        var gradients = tape.ComputeGradients(loss);

        Assert.Equal(
            new[] { 0f, 0f, 0f, 0f, 1f, 1f, 0f, 0f },
            gradients[parameter].ToArray());
    }

    [Fact]
    public void NarrowView_ScattersGradientToSelectedRange()
    {
        var engine = new CpuEngine();
        var parameter = new Tensor<float>(
            new[] { 1f, 2f, 3f, 4f, 5f, 6f },
            new[] { 2, 3 });

        using var tape = new GradientTape<float>();
        var selected = parameter.Slice(axis: 1, start: 1, end: 3);
        var loss = engine.ReduceSum(selected, null);
        var gradients = tape.ComputeGradients(loss);

        Assert.Equal(
            new[] { 0f, 1f, 1f, 0f, 1f, 1f },
            gradients[parameter].ToArray());
    }

    [Fact]
    public void NonContiguousReshape_RecordsMaterializationAndViewAsDistinctEdges()
    {
        var engine = new CpuEngine();
        var parameter = new Tensor<float>(
            new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 });

        using var tape = new GradientTape<float>();
        var transposed = parameter.Transpose();
        var reshaped = transposed.Reshape(6);
        var loss = engine.ReduceSum(reshaped, null);

        var reshapeNode = Assert.IsType<GradNode<float>>(reshaped.GradFn);
        var materialized = Assert.IsType<Tensor<float>>(reshapeNode.Input0);
        Assert.NotSame(transposed, materialized);
        Assert.Same(transposed, Assert.IsType<GradNode<float>>(materialized.GradFn).Input0);

        var gradients = tape.ComputeGradients(loss);
        Assert.Equal(new[] { 1f, 1f, 1f, 1f, 1f, 1f }, gradients[parameter].ToArray());
    }

    [Fact]
    public void NonContiguousReshape_CompiledReplayPreservesGradientChain()
    {
        var engine = new CpuEngine();
        var parameter = new Tensor<float>(
            new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 });
        var parameters = new[] { parameter };

        using var scope = GraphMode.EnableTraining(parameters);
        var reshaped = parameter.Transpose().Reshape(6);
        var loss = engine.ReduceSum(engine.TensorMultiply(reshaped, reshaped), null);
        using var plan = scope.CompileTraining(parameters, loss);
        plan.ConfigureOptimizer(OptimizerType.SGD, 0.01f, 0.9f, 0.999f, 1e-8f, 0f);

        plan.Step();

        var compiled = Assert.IsType<CompiledTrainingPlan<float>>(plan);
        Assert.Equal(new[] { 2f, 4f, 6f, 8f, 10f, 12f }, compiled.Gradients[0].ToArray());
    }

    [Fact]
    public void SubTensor_InvalidLaterIndex_DoesNotConstructPartialViews()
    {
        var tensor = new Tensor<float>(new float[8], new[] { 2, 2, 2 });
        int refCountBefore = tensor._storage.RefCount;

        Assert.Throws<ArgumentOutOfRangeException>(() => tensor.SubTensor(1, 2));

        Assert.Equal(refCountBefore, tensor._storage.RefCount);
    }
}
