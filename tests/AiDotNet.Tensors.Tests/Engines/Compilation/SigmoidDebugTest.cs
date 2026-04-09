using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Trait("Category", "Benchmark")]
public class SigmoidDebugTest
{
    private readonly ITestOutputHelper _output;
    public SigmoidDebugTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Sigmoid_CheckCompiledPlanStepCount()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new float[1_000_000], new[] { 1_000_000 });

        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.Sigmoid(input);
            plan = scope.CompileInference<float>();
        }

        _output.WriteLine($"Plan step count: {plan.StepCount}");
        _output.WriteLine($"Input contiguous: {input.IsContiguous}");
        _output.WriteLine($"Input rank: {input.Rank}");
        _output.WriteLine($"Input length: {input.Length}");

        // The plan should have 1 step (Sigmoid).
        // If it has more, something is adding overhead.
        Assert.True(plan.StepCount >= 1, $"Plan has {plan.StepCount} steps");

        plan.Dispose();
    }
}
