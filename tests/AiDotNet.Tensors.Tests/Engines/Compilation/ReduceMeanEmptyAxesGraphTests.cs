using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Empty axes mean "reduce every axis" in eager ReduceMean. Graph mode recorded them as NO reduced axes - an
/// unreduced output shape - while the recorded node replays the eager full reduction into it, so a traced
/// ReduceMean(x, []) disagreed with the eager one.
/// </summary>
public class ReduceMeanEmptyAxesGraphTests
{
    [Fact]
    public void A_traced_reduce_mean_over_empty_axes_is_the_eager_full_reduction()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 });
        var eager = engine.ReduceMean(x, System.Array.Empty<int>(), keepDims: false);

        CompiledInferencePlan<float> plan;
        Tensor<float> traced;
        using (var scope = GraphMode.Enable())
        {
            traced = engine.ReduceMean(x, System.Array.Empty<int>(), keepDims: false);
            plan = scope.CompileInference<float>();
        }
        using (plan)
        {
            Assert.Equal(eager.Shape.ToArray(), traced.Shape.ToArray());
            var result = plan.Execute();
            Assert.Equal(eager.Length, result.Length);
            Assert.Equal(3.5f, result.ToArray()[0], 5);
        }
    }
}
