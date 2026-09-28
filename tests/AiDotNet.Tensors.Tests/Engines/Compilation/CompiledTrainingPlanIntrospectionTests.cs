using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// <see cref="ICompiledTrainingPlanIntrospection{T}"/>: the optimizer step a plan's schedule was evaluated at, and the
/// parameter tensors its update writes. A caller uses them to tell a legitimately zero update from a plan that no
/// longer writes the model's tensors.
/// </summary>
public class CompiledTrainingPlanIntrospectionTests
{
    private static ICompiledTrainingPlan<float> Compile(Tensor<float> weight)
    {
        var engine = new CpuEngine();
        using var scope = GraphMode.Enable();
        engine.ReduceSum(weight, null);
        return scope.CompileTraining(new[] { weight });
    }

    [Fact]
    public void OptimizerStep_CountsAppliedUpdates_FromZero()
    {
        var weight = Tensor<float>.CreateRandom([3, 2]);
        using var plan = Compile(weight);
        plan.ConfigureOptimizer(OptimizerType.Adam, 1e-3f);
        var introspection = Assert.IsAssignableFrom<ICompiledTrainingPlanIntrospection<float>>(plan);

        Assert.Equal(0, introspection.OptimizerStep);
        plan.Step();
        Assert.Equal(1, introspection.OptimizerStep);
        plan.Step();
        plan.Step();
        Assert.Equal(3, introspection.OptimizerStep);
    }

    [Fact]
    public void OptimizerStep_TakesTheImportedStep()
    {
        var sourceWeight = Tensor<float>.CreateRandom([3, 2]);
        using var source = Compile(sourceWeight);
        source.ConfigureOptimizer(OptimizerType.Adam, 1e-3f);
        for (int i = 0; i < 4; i++) source.Step();
        byte[] state = source.ExportOptimizerState() ?? throw new InvalidOperationException("Adam plan exported no state.");

        var targetWeight = Tensor<float>.CreateRandom([3, 2]);
        using var target = Compile(targetWeight);
        target.ConfigureOptimizer(OptimizerType.Adam, 1e-3f);
        target.ImportOptimizerState(state);

        Assert.Equal(4, Assert.IsAssignableFrom<ICompiledTrainingPlanIntrospection<float>>(target).OptimizerStep);
    }

    [Fact]
    public void OptimizedParameters_AreTheCompiledTensors_AndCannotBeRewritten()
    {
        var weight = Tensor<float>.CreateRandom([3, 2]);
        using var plan = Compile(weight);
        var parameters = Assert.IsAssignableFrom<ICompiledTrainingPlanIntrospection<float>>(plan).OptimizedParameters;

        var only = Assert.Single(parameters);
        Assert.Same(weight, only);

        // A read-only view: casting back to a mutable list must not reach the plan's own slots.
        Assert.False(parameters is Tensor<float>[], "the plan's parameter array must not be handed out");
        Assert.Throws<NotSupportedException>(() => ((IList<Tensor<float>>)parameters)[0] = Tensor<float>.CreateRandom([3, 2]));
    }
}