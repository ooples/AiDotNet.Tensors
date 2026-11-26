using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Releasing a compiled training plan runs every stage even when one fails. Before, Dispose ran its stages in sequence
/// with no guard, so a failure while taking back lent step memory skipped the graph teardown and every release after
/// it, and the plan, already marked disposed, could never release them. And it suppressed the finalizer before any
/// stage ran, so a graph whose teardown threw had no owner left to destroy it.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class CompiledTrainingPlanDisposeTests
{
    private static Tensor<float> Filled(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    private static CompiledTrainingPlan<float> TrainedCpuPlan()
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
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
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.01f);
            plan.Step();
            return plan;
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    private static void FailOnce(CompiledTrainingPlan<float> plan)
    {
        // Once: later stages take back lent memory too, and must not fail again.
        bool fired = false;
        plan.TestFaultInIdleReclaim = () =>
        {
            if (fired) return;
            fired = true;
            throw new InvalidOperationException("injected reclaim fault");
        };
    }

    [Fact]
    public void A_failing_stage_does_not_skip_the_rest_of_the_teardown_and_its_failure_is_rethrown()
    {
        var plan = TrainedCpuPlan();
        var held = plan.ReleasableStateForTests;
        Assert.True(held.OptimizerState, "the trained plan holds no optimizer state, so the test would prove nothing");

        FailOnce(plan);
        CompiledTrainingPlan<float>.RecordTeardownStagesForTests = true;
        InvalidOperationException ex;
        try { ex = Assert.Throws<InvalidOperationException>(() => plan.Dispose()); }
        finally { CompiledTrainingPlan<float>.RecordTeardownStagesForTests = false; }
        Assert.Equal("injected reclaim fault", ex.Message);
        // Every stage ran, in order, and the state they release is gone.
        Assert.Equal(CompiledTrainingPlan<float>.TeardownStages, plan.TeardownStagesForTests);
        Assert.Equal((false, false, 0, false), plan.ReleasableStateForTests);
        plan.Dispose();   // already disposed: a no-op, not a second throw
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long DisposeAndDrop(bool failAStage)
    {
        var plan = TrainedCpuPlan();
        if (failAStage)
        {
            FailOnce(plan);
            Assert.Throws<InvalidOperationException>(() => plan.Dispose());
        }
        else plan.Dispose();
        return plan.IdForTests;
    }

    private static void Collect()
    {
        for (int i = 0; i < 3; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    }

    [Fact]
    public void A_plan_whose_teardown_failed_keeps_its_finalizer_and_a_clean_one_does_not()
    {
        CompiledTrainingPlan<float>.TrackFinalizationForTests = true;
        try
        {
            long clean = DisposeAndDrop(failAStage: false);
            long failed = DisposeAndDrop(failAStage: true);
            Collect();
            Assert.False(CompiledTrainingPlan<float>.FinalizedPlansForTests.ContainsKey(clean),
                "a plan released cleanly still ran its finalizer");
            Assert.True(CompiledTrainingPlan<float>.FinalizedPlansForTests.ContainsKey(failed),
                "the finalizer was suppressed although a teardown stage failed, so nothing is left to destroy what it held");
        }
        finally { CompiledTrainingPlan<float>.TrackFinalizationForTests = false; }
    }
}
