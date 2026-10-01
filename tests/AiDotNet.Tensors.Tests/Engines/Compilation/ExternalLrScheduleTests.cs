using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// <see cref="ExternalLrSchedule"/>: a learning rate the caller sets between steps without reconfiguring the plan
/// (which would restart its moments). The loss is the sum of the weight, so every gradient is 1 and an SGD step
/// lowers each weight by exactly the rate it used.
/// </summary>
public class ExternalLrScheduleTests
{
    private static ICompiledTrainingPlan<double> Compile(Tensor<double> weight)
    {
        var engine = new CpuEngine();
        using var scope = GraphMode.Enable();
        engine.ReduceSum(weight, null);
        return scope.CompileTraining(new[] { weight });
    }

    private static double StepDrop(ICompiledTrainingPlan<double> plan, Tensor<double> weight)
    {
        double before = weight[0];
        plan.Step();
        return before - weight[0];
    }

    [Fact]
    public void A_new_rate_applies_from_the_next_step()
    {
        var weight = Tensor<double>.CreateRandom([3, 2]);
        using var plan = Compile(weight);
        var schedule = LrSchedule.External(0.25);
        plan.ConfigureOptimizer(OptimizerType.SGD, schedule);

        Assert.Equal(0.25, StepDrop(plan, weight), 12);
        schedule.LearningRate = 0.0625;
        Assert.Equal(0.0625, StepDrop(plan, weight), 12);
        Assert.Equal(0.0625, StepDrop(plan, weight), 12);
    }

    [Fact]
    public void An_imported_plan_hands_back_an_external_schedule_at_the_saved_rate()
    {
        var sourceWeight = Tensor<double>.CreateRandom([3, 2]);
        using var source = Compile(sourceWeight);
        var sourceSchedule = LrSchedule.External(0.5);
        source.ConfigureOptimizer(OptimizerType.Adam, sourceSchedule);
        source.Step();
        sourceSchedule.LearningRate = 0.125;
        byte[] state = source.ExportOptimizerState() ?? throw new InvalidOperationException("Adam plan exported no state.");

        var targetWeight = Tensor<double>.CreateRandom([3, 2]);
        using var target = Compile(targetWeight);
        target.ConfigureOptimizer(OptimizerType.Adam, 1e-3f);
        target.ImportOptimizerState(state);

        var introspection = Assert.IsAssignableFrom<ICompiledTrainingPlanIntrospection<double>>(target);
        var restored = Assert.IsType<ExternalLrSchedule>(Assert.Single(introspection.LearningRateSchedules));
        Assert.NotSame(sourceSchedule, restored);
        Assert.Equal(0.125, restored.LearningRate);

        // The restored instance drives the restored plan, and it still exports as an external rate.
        restored.LearningRate = 0.0;
        double before = targetWeight[0];
        target.Step();
        Assert.Equal(before, targetWeight[0]);
        Assert.Equal(state.Length, target.ExportOptimizerState()!.Length);
    }

    [Fact]
    public void Continuing_another_plans_optimizer_keeps_the_callers_schedule()
    {
        var weight = Tensor<double>.CreateRandom([3, 2]);
        var schedule = LrSchedule.External(0.25);
        using var first = Compile(weight);
        first.ConfigureOptimizer(OptimizerType.SGD, schedule);
        first.Step();

        using var second = Compile(weight);
        second.ContinueOptimizerFrom(first);

        var live = Assert.Single(Assert.IsAssignableFrom<ICompiledTrainingPlanIntrospection<double>>(second).LearningRateSchedules);
        Assert.Same(schedule, live);
        schedule.LearningRate = 0.03125;
        Assert.Equal(0.03125, StepDrop(second, weight), 12);
    }

    [Fact]
    public void Continuing_by_copy_keeps_the_callers_schedule()
    {
        // A grouped configuration is not shared between plans, so ContinueOptimizerFrom copies it through a checkpoint.
        var weight = Tensor<double>.CreateRandom([3, 2]);
        var schedule = LrSchedule.External(0.25);
        using var first = Compile(weight);
        first.ConfigureOptimizerGrouped(OptimizerType.SGD, new LrSchedule[] { schedule }, new[] { 0 });
        first.Step();

        using var second = Compile(weight);
        second.ContinueOptimizerFrom(first);

        var live = Assert.Single(Assert.IsAssignableFrom<ICompiledTrainingPlanIntrospection<double>>(second).LearningRateSchedules);
        Assert.Same(schedule, live);
        schedule.LearningRate = 0.03125;
        Assert.Equal(0.03125, StepDrop(second, weight), 12);
    }

    [Fact]
    public void Schedules_are_empty_before_an_optimizer_is_configured()
    {
        using var plan = Compile(Tensor<double>.CreateRandom([3, 2]));
        Assert.Empty(Assert.IsAssignableFrom<ICompiledTrainingPlanIntrospection<double>>(plan).LearningRateSchedules);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(-1e-3)]
    public void An_invalid_rate_is_rejected(double rate)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => LrSchedule.External(rate));
        var schedule = LrSchedule.External(0.1);
        Assert.Throws<ArgumentOutOfRangeException>(() => schedule.LearningRate = rate);
        Assert.Equal(0.1, schedule.LearningRate);
    }
}
