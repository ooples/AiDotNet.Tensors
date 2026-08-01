using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public sealed class CompiledStopGradientTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;

    public CompiledStopGradientTests()
    {
        AiDotNetEngine.Current = new CpuEngine();
    }

    public void Dispose()
    {
        AiDotNetEngine.Current = _priorEngine;
    }

    [Fact]
    public void InferenceReplay_RefreshesDetachedForwardValue()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 2.0f }, new[] { 1 });
        var bias = new Tensor<float>(new[] { 10.0f }, new[] { 1 });

        ICompiledPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var negated = engine.TensorNegate(input);
            var detached = engine.StopGradient(negated);
            var output = engine.TensorAdd(detached, bias);
            plan = scope.CompileInference<float>(output, input);
        }

        using (plan)
        {
            Assert.Equal(8.0f, plan.Execute()[0]);

            input[0] = 4.0f;
            Assert.Equal(6.0f, plan.Execute()[0]);
        }
    }

    [Fact]
    public void InferenceReplay_RefreshesDetachedNonContiguousView()
    {
        var engine = new CpuEngine();
        var source = new Tensor<float>(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 2, 2 });
        var transposed = source.Transpose(new[] { 1, 0 });
        Assert.False(transposed.IsContiguous);

        ICompiledPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var detached = engine.StopGradient(transposed);
            plan = scope.CompileInference<float>(detached, source);
        }

        using (plan)
        {
            Assert.Equal(new[] { 1.0f, 3.0f, 2.0f, 4.0f }, plan.Execute().ToArray());

            source[1] = 20.0f;
            Assert.Equal(new[] { 1.0f, 3.0f, 20.0f, 4.0f }, plan.Execute().ToArray());
        }
    }

    [Fact]
    public void TrainingReplay_RefreshesStraightThroughCorrectionWithoutBackpropagatingIt()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 3.0f }, new[] { 1 });
        var parameter = new Tensor<float>(new[] { 2.0f }, new[] { 1 });
        var two = new Tensor<float>(new[] { 2.0f }, new[] { 1 });
        var parameters = new[] { parameter };

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var primitive = engine.TensorMultiply(parameter, input);
            var stableForward = engine.TensorMultiply(primitive, two);
            var correction = engine.StopGradient(engine.TensorSubtract(stableForward, primitive));
            var loss = engine.ReduceSum(engine.TensorAdd(primitive, correction), null);
            plan = scope.CompileTraining(parameters, loss);
        }

        using (plan)
        {
            Assert.Equal(12.0f, plan.Step()[0]);
            Assert.Equal(3.0f, plan.Gradients[0][0]);

            input[0] = 4.0f;
            Assert.Equal(16.0f, plan.Step()[0]);
            Assert.Equal(4.0f, plan.Gradients[0][0]);
        }
    }
}
