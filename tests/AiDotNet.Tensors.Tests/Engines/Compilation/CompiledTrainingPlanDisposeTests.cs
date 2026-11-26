using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Releasing a compiled training plan runs every stage even when one fails. Before, Dispose ran its stages in sequence
/// with no guard, so a failure while taking back lent step memory skipped the graph teardown and every release after
/// it, and the plan, already marked disposed, could never release them.
/// </summary>
public sealed class CompiledTrainingPlanDisposeTests
{
    private static Tensor<float> Filled(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    [Fact]
    public void A_failing_stage_does_not_skip_the_rest_of_the_teardown_and_its_failure_is_rethrown()
    {
        var w = Filled(new[] { 4, 3 }, 1);
        var x = Filled(new[] { 2, 4 }, 2);
        var parameters = new[] { w };
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var e = AiDotNetEngine.Current;
            var y = e.TensorMatMul(x, w);
            var loss = e.ReduceSum(e.TensorMultiply(y, y), null);
            plan = (CompiledTrainingPlan<float>)scope.CompileTraining(parameters, loss);
        }
        plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.01f);
        plan.Step();

        // Once: later stages take back lent memory too, and must not fail again.
        bool fired = false;
        plan.TestFaultInIdleReclaim = () =>
        {
            if (fired) return;
            fired = true;
            throw new InvalidOperationException("injected reclaim fault");
        };
        var ex = Assert.Throws<InvalidOperationException>(() => plan.Dispose());
        Assert.Equal("injected reclaim fault", ex.Message);
        Assert.True(plan.TeardownCompletedForTests, "the stages after the failing one did not run");
        plan.Dispose();   // already disposed: a no-op, not a second throw
    }
}
