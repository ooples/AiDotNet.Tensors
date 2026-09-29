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
    public static TheoryData<string> MomentStorageModes => new() { "float32", "bfloat16", "int8", "amsgrad" };

    /// <summary>
    /// Exporting and importing optimizer state works in every moment-storage mode. Each mode keeps its moments in a
    /// different per-parameter array (float, bfloat16 ushort, int8 byte plus double scales) and leaves the others
    /// allocated with null entries, which made ExportOptimizerState throw NullReferenceException before the slot check.
    /// </summary>
    [Theory]
    [MemberData(nameof(MomentStorageModes))]
    public void OptimizerState_RoundTrips_InEveryMomentStorageMode(string mode)
    {
        ICompiledTrainingPlan<float> Build()
        {
            // 64 x 64 = 4096 elements: the default int8 minimum, so the int8 mode really quantizes.
            var plan = Compile(Tensor<float>.CreateRandom([64, 64]));
            if (mode == "bfloat16") plan.RequestBf16MomentStorage(true);
            if (mode == "int8") plan.RequestInt8MomentStorage(true, blockSize: 256);
            plan.ConfigureOptimizer(mode == "amsgrad" ? OptimizerType.AMSGrad : OptimizerType.Adam, 1e-3f);
            return plan;
        }

        using var source = Build();
        for (int i = 0; i < 3; i++) source.Step();
        byte[] state = source.ExportOptimizerState() ?? throw new InvalidOperationException($"{mode} plan exported no state.");

        using var target = Build();
        target.ImportOptimizerState(state);
        Assert.Equal(3, Assert.IsAssignableFrom<ICompiledTrainingPlanIntrospection<float>>(target).OptimizerStep);

        // The imported state is the source's: exporting it again reproduces the payload byte for byte.
        Assert.Equal(state, target.ExportOptimizerState());
    }

    [Fact]
    public void OptimizerState_RoundTrips_ForADoublePlan()
    {
        ICompiledTrainingPlan<double> Build()
        {
            var engine = new CpuEngine();
            var weight = Tensor<double>.CreateRandom([3, 2]);
            ICompiledTrainingPlan<double> plan;
            using (var scope = GraphMode.Enable())
            {
                engine.ReduceSum(weight, null);
                plan = scope.CompileTraining(new[] { weight });
            }
            plan.ConfigureOptimizer(OptimizerType.Adam, 1e-3f);
            return plan;
        }

        using var source = Build();
        for (int i = 0; i < 2; i++) source.Step();
        byte[] state = source.ExportOptimizerState() ?? throw new InvalidOperationException("double plan exported no state.");

        using var target = Build();
        target.ImportOptimizerState(state);
        Assert.Equal(2, Assert.IsAssignableFrom<ICompiledTrainingPlanIntrospection<double>>(target).OptimizerStep);
        Assert.Equal(state, target.ExportOptimizerState());
    }
}