using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Regression tests for issue #228 — CompiledInferencePlan silently returned the
/// last optimized step's OutputBuffer instead of the caller's returned tensor
/// whenever the forward ended in a pure metadata-view op (Reshape on contiguous
/// data, Squeeze, Unsqueeze, ...) or had host-side control flow that
/// conditionally appended such a view. Fixed by making the API take
/// Func&lt;Tensor&lt;T&gt;&gt; and recording metadata views as lazy nodes so the
/// explicit output identity flows through the plan.
/// </summary>
public class CompiledPlanFinalOutputReproTest
{
    private readonly ITestOutputHelper _output;
    public CompiledPlanFinalOutputReproTest(ITestOutputHelper output) { _output = output; }

    /// Forward ends in a Reshape view of an intermediate tensor.
    /// Reshape on contiguous data does NOT record a lazy node unless GraphMode
    /// is explicitly handled, so the compiled plan would otherwise pick the
    /// last pre-reshape op's output buffer and replay returns its shape
    /// instead of the reshape's shape.
    [Fact]
    public void CompiledPlan_ForwardReturnsReshape_ReturnsWrongShape()
    {
        using var cache = new CompiledModelCache<float>();
        var engine = new CpuEngine();

        var input = new Tensor<float>(new[] { 1, 6 });
        var weights = new Tensor<float>(new[] { 6, 10 });
        for (int i = 0; i < input.Length; i++) input[i] = 1.0f;
        for (int i = 0; i < weights.Length; i++) weights[i] = 0.1f;

        var expectedEager = engine.TensorMatMul(input, weights).Reshape(new[] { 10 });
        Assert.Equal(new[] { 10 }, expectedEager.Shape.ToArray());

        var plan = cache.GetOrCompileInference(input._shape, () =>
        {
            var matmul = engine.TensorMatMul(input, weights);   // lazy node
            var result = matmul.Reshape(new[] { 10 });          // metadata view
            return result;
        });

        var compiledOutput = plan.Execute();
        Assert.Equal(new[] { 10 }, compiledOutput.Shape.ToArray());
    }

    /// Host-side branch — the exact pattern in AiDotNet ResNet / VGG / CNN
    /// forwards (promote rank-3 input to rank-4, then optionally strip the
    /// synthetic batch dim with Reshape).
    [Fact]
    public void CompiledPlan_ForwardWithHostSideBranch_ReturnsWrongShape()
    {
        using var cache = new CompiledModelCache<float>();
        var engine = new CpuEngine();

        var input = new Tensor<float>(new[] { 1, 6 });
        var weights = new Tensor<float>(new[] { 6, 10 });
        for (int i = 0; i < input.Length; i++) input[i] = 1.0f;
        for (int i = 0; i < weights.Length; i++) weights[i] = 0.1f;

        var plan = cache.GetOrCompileInference(input._shape, () =>
        {
            var output = engine.TensorMatMul(input, weights);
            if (output.Rank == 2 && output.Shape[0] == 1)
                output = output.Reshape(new[] { output.Shape[1] });
            return output;
        });

        var compiledOutput = plan.Execute();
        Assert.Equal(new[] { 10 }, compiledOutput.Shape.ToArray());
    }

    /// Canary: Transpose ends the forward. TensorTranspose goes through
    /// LazyTensorScope.Record* so a proper lazy node IS created. Must keep
    /// working after the fix.
    [Fact]
    public void CompiledPlan_ForwardReturnsTranspose_ReturnsCorrectShape()
    {
        using var cache = new CompiledModelCache<float>();
        var engine = new CpuEngine();

        var input = new Tensor<float>(new[] { 4, 6 });
        var weights = new Tensor<float>(new[] { 6, 10 });
        for (int i = 0; i < input.Length; i++) input[i] = 1.0f;
        for (int i = 0; i < weights.Length; i++) weights[i] = 0.1f;

        var plan = cache.GetOrCompileInference(input._shape, () =>
        {
            var mm = engine.TensorMatMul(input, weights);
            var t = engine.TensorTranspose(mm);
            return t;
        });

        var compiledOutput = plan.Execute();
        Assert.Equal(new[] { 10, 4 }, compiledOutput.Shape.ToArray());
    }
}
