using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Issue #348 — integration tests for the fused-compiled training path
/// with per-step LR schedules and per-parameter-group rates.
///
/// All tests use a minimal ReduceSum(weight) graph: loss = Σ weight[i],
/// so ∂L/∂weight[i] = 1 for every element. With SGD this gives a closed-
/// form expected trajectory: weight[i]_after_N_steps = weight[i]_initial -
/// Σ_{t=1..N} lr_t, which lets us assert the schedule actually drives
/// per-step rate changes (not just a captured initial value).
/// </summary>
public class FusedAdaptiveLrPlanTests
{
    private static ICompiledTrainingPlan<float> CompileReduceSumPlan(
        CpuEngine engine, Tensor<float> weight)
    {
        using var scope = GraphMode.Enable();
        engine.ReduceSum(weight, null);
        return scope.CompileTraining(new[] { weight });
    }

    [Fact]
    public void ConfigureOptimizer_ConstantLr_TakesExpectedSteps()
    {
        var engine = new CpuEngine();
        var weight = new Tensor<float>(
            new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, new[] { 3, 2 });
        var initial = weight.AsSpan().ToArray();

        using var plan = CompileReduceSumPlan(engine, weight);
        plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.01f);

        for (int i = 0; i < 10; i++) plan.Step();

        // After 10 steps at lr=0.01 with grad=1 each step: weight[i] -= 0.10.
        for (int i = 0; i < initial.Length; i++)
            Assert.Equal(initial[i] - 0.10f, weight.AsSpan()[i], 5);
    }

    [Fact]
    public void ConfigureOptimizer_CosineSchedule_AppliesPerStepLr()
    {
        var engine = new CpuEngine();
        var weight = new Tensor<float>(new float[] { 10.0f, 10.0f }, new[] { 2 });
        float initial = weight.AsSpan()[0];

        var schedule = LrSchedule.Cosine(lrMax: 0.1, totalSteps: 100, lrMin: 0.0);
        using var plan = CompileReduceSumPlan(engine, weight);
        plan.ConfigureOptimizer(OptimizerType.SGD, schedule);

        const int steps = 100;
        double expectedDelta = 0.0;
        for (int t = 1; t <= steps; t++)
        {
            plan.Step();
            expectedDelta += schedule.GetLr(t);
        }

        float finalVal = weight.AsSpan()[0];
        float expectedFinal = initial - (float)expectedDelta;
        Assert.Equal(expectedFinal, finalVal, 4);

        // Sanity: a constant-0.1 schedule would deliver finalVal = initial - 10.0,
        // but cosine averages roughly half of that. Make sure we did NOT just
        // burn the constant lr the whole way.
        Assert.True(System.Math.Abs(finalVal - (initial - 10.0f)) > 1.0f,
            $"Cosine schedule produced same trajectory as constant lr — schedule didn't take effect.");
    }

    [Fact]
    public void ConfigureOptimizer_LinearWarmupCosine_FollowsScheduleExactly()
    {
        var engine = new CpuEngine();
        var weight = new Tensor<float>(new float[] { 0.0f }, new[] { 1 });

        var schedule = LrSchedule.LinearWarmupCosine(
            lrMax: 0.5, warmupSteps: 5, totalSteps: 50, lrMin: 0.0);
        using var plan = CompileReduceSumPlan(engine, weight);
        plan.ConfigureOptimizer(OptimizerType.SGD, schedule);

        double cumDelta = 0.0;
        for (int t = 1; t <= 50; t++)
        {
            plan.Step();
            cumDelta += schedule.GetLr(t);
            // After each step, weight should equal -cumDelta to fp precision.
            Assert.Equal(-(float)cumDelta, weight.AsSpan()[0], 4);
        }
    }

    [Fact]
    public void ConfigureOptimizer_ConstantViaLrSchedule_MatchesFloatOverload()
    {
        // Float overload routes through LrSchedule.Constant — same numerics.
        var engine = new CpuEngine();
        var w1 = new Tensor<float>(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var w2 = new Tensor<float>(new float[] { 1.0f, 2.0f }, new[] { 2 });

        using var plan1 = CompileReduceSumPlan(engine, w1);
        using var plan2 = CompileReduceSumPlan(engine, w2);
        plan1.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.05f);
        plan2.ConfigureOptimizer(OptimizerType.SGD, LrSchedule.Constant(0.05));

        for (int i = 0; i < 20; i++)
        {
            plan1.Step();
            plan2.Step();
        }

        for (int i = 0; i < w1.Length; i++)
            Assert.Equal(w1.AsSpan()[i], w2.AsSpan()[i], 6);
    }

    [Fact]
    public void ConfigureOptimizerGrouped_AppliesDistinctLrsPerGroup()
    {
        // ReduceSum over a concat of two weight tensors gives grad=1 for both.
        // We compile with two separate parameters — backbone and head — and
        // assign different lr schedules.
        var engine = new CpuEngine();
        var backbone = new Tensor<float>(new float[] { 100.0f, 100.0f }, new[] { 2 });
        var head = new Tensor<float>(new float[] { 100.0f, 100.0f }, new[] { 2 });

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            // Sum each separately, then add the scalars — both contribute grad=1
            // to their respective parameter.
            var l1 = engine.ReduceSum(backbone, null);
            var l2 = engine.ReduceSum(head, null);
            engine.TensorAdd(l1, l2);
            plan = scope.CompileTraining(new[] { backbone, head });
        }

        using (plan)
        {
            // backbone group: lr=0.001; head group: lr=0.1 (100× larger).
            plan.ConfigureOptimizerGrouped(
                OptimizerType.SGD,
                new LrSchedule[] { LrSchedule.Constant(0.001), LrSchedule.Constant(0.1) },
                new int[] { 0, 1 });

            for (int i = 0; i < 100; i++) plan.Step();

            // backbone: 100 steps × 0.001 = 0.1 delta. head: 100 × 0.1 = 10.0.
            Assert.Equal(99.9f, backbone.AsSpan()[0], 3);
            Assert.Equal(90.0f, head.AsSpan()[0], 3);
        }
    }

    [Fact]
    public void ConfigureOptimizerGrouped_RejectsMismatchedGroupCount()
    {
        var engine = new CpuEngine();
        var w = new Tensor<float>(new float[] { 1.0f }, new[] { 1 });
        using var plan = CompileReduceSumPlan(engine, w);

        // Two parameters in the compiled plan: 1. Group array size is 2 — should throw.
        Assert.Throws<ArgumentException>(() =>
            plan.ConfigureOptimizerGrouped(
                OptimizerType.SGD,
                new LrSchedule[] { LrSchedule.Constant(0.1) },
                new int[] { 0, 0 }));
    }

    [Fact]
    public void ConfigureOptimizer_RejectsIneligibleOptimizer()
    {
        var engine = new CpuEngine();
        var w = new Tensor<float>(new float[] { 1.0f }, new[] { 1 });
        using var plan = CompileReduceSumPlan(engine, w);

        // LBFGS isn't fused-eligible — closure-based. Caller should get a
        // clear error, not a silent fall-through to the eager path.
        Assert.Throws<NotSupportedException>(() =>
            plan.ConfigureOptimizer(OptimizerType.LBFGS, LrSchedule.Constant(0.01)));
    }

    [Theory]
    [InlineData(OptimizerType.SparseAdam)]
    [InlineData(OptimizerType.ScheduleFreeSGD)]
    public void ConfigureOptimizer_RejectsOptimizersPlanCantDispatch(OptimizerType opt)
    {
        // Tensors #500/#499: the fused dispatch now covers the full elementwise
        // kernel-backed set PLUS the global-state optimizers HypergradientSGD and
        // DAdaptationSGD (via a two-phase global-reduction pass). What remains
        // rejected needs an execution model the plan can't provide: SparseAdam
        // (sparse-gradient indices) and ScheduleFreeSGD (a y-buffer written before
        // the forward pass). These must fail FAST at configure time, not surprise
        // the caller on the first Step().
        var engine = new CpuEngine();
        var w = new Tensor<float>(new float[] { 1.0f }, new[] { 1 });
        using var plan = CompileReduceSumPlan(engine, w);
        Assert.Throws<NotSupportedException>(() =>
            plan.ConfigureOptimizer(opt, LrSchedule.Constant(0.01)));
    }

    [Fact]
    public void ConfigureOptimizerGrouped_RejectsNullScheduleSlotEvenIfUnreferenced()
    {
        // PR #349 review #2: the grouped hot path evaluates GetLr() on
        // every schedule slot per step (not just the ones referenced).
        // A null slot must fail at configure, not crash inside Step().
        var engine = new CpuEngine();
        var w = new Tensor<float>(new float[] { 1.0f }, new[] { 1 });
        using var plan = CompileReduceSumPlan(engine, w);
        var schedules = new LrSchedule[] { LrSchedule.Constant(0.1), null! };
        Assert.Throws<ArgumentException>(() =>
            plan.ConfigureOptimizerGrouped(OptimizerType.SGD, schedules, new int[] { 0 }));
    }

    [Fact]
    public void ConfigureOptimizer_AdamWithSchedule_DrivesProductiveTraining()
    {
        // Train two Adam optimizers on the same problem with two
        // different schedules. The "good" schedule (cosine) must beat
        // the "control" schedule (lr=0 throughout) by a wide margin —
        // this proves the schedule is actually evaluated each step,
        // not captured-once-and-ignored. We don't compare against a
        // PyTorch reference because Adam's sign-normalized trajectory
        // is hyperparameter-sensitive on a small-batch quadratic; the
        // contrast against the zero-lr control is the proof.
        var engine = new CpuEngine();
        var target = new Tensor<float>(new float[] { 1.0f, -1.0f, 2.0f, -2.0f }, new[] { 4 });

        float LossOf(Tensor<float> wState)
        {
            float s = 0f;
            for (int i = 0; i < wState.Length; i++)
            {
                float d = wState.AsSpan()[i] - target.AsSpan()[i];
                s += d * d;
            }
            return s;
        }

        ICompiledTrainingPlan<float> BuildPlan(Tensor<float> wParam)
        {
            using var scope = GraphMode.Enable();
            var diff = engine.TensorSubtract(wParam, target);
            var sq = engine.TensorMultiply(diff, diff);
            engine.ReduceSum(sq, null);
            return scope.CompileTraining(new[] { wParam });
        }

        var wScheduled = new Tensor<float>(new float[] { 5.0f, 5.0f, 5.0f, 5.0f }, new[] { 4 });
        var wControl   = new Tensor<float>(new float[] { 5.0f, 5.0f, 5.0f, 5.0f }, new[] { 4 });
        float initialLoss = LossOf(wScheduled);

        using (var planScheduled = BuildPlan(wScheduled))
        using (var planControl   = BuildPlan(wControl))
        {
            planScheduled.ConfigureOptimizer(OptimizerType.Adam,
                LrSchedule.LinearWarmupCosine(lrMax: 0.02, warmupSteps: 50, totalSteps: 500));
            planControl.ConfigureOptimizer(OptimizerType.Adam, LrSchedule.Constant(0.0));

            for (int i = 0; i < 500; i++)
            {
                planScheduled.Step();
                planControl.Step();
            }

            float scheduledLoss = LossOf(wScheduled);
            float controlLoss = LossOf(wControl);

            // Scheduled run must materially reduce loss; control (lr=0)
            // is mathematically constrained to NOT change parameters.
            Assert.Equal(initialLoss, controlLoss, 4);
            Assert.True(scheduledLoss < initialLoss * 0.2f,
                $"Adam+schedule didn't drive training: initial={initialLoss}, scheduled={scheduledLoss}, control={controlLoss}");
        }
    }
}
