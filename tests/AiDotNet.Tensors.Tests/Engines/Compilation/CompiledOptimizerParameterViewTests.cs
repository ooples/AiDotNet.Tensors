using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Integration coverage for compiled fused optimizers over parameter views.
/// View addressing must survive graph compilation, gradient clipping, grouped
/// dispatch, COW isolation, and both supported floating-point precisions.
/// </summary>
public sealed class CompiledOptimizerParameterViewTests
{
    private static ICompiledTrainingPlan<T> CompileReduceSum<T>(CpuEngine engine, Tensor<T> parameter)
    {
        using var scope = GraphMode.Enable();
        engine.ReduceSum(parameter, null);
        return scope.CompileTraining(new[] { parameter });
    }

    [Fact]
    public void FloatSgd_ContiguousMemorySlice_UpdatesOnlyLogicalWindow()
    {
        var engine = new CpuEngine();
        var backing = new[] { 91f, 1f, 2f, 92f };
        var parameter = Tensor<float>.FromMemory(new Memory<float>(backing, 1, 2), new[] { 2 });

        using var plan = CompileReduceSum(engine, parameter);
        plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1f);
        plan.Step();

        Assert.Equal(91f, backing[0]);
        Assert.Equal(0.9f, backing[1], 6);
        Assert.Equal(1.9f, backing[2], 6);
        Assert.Equal(92f, backing[3]);
    }

    [Fact]
    public void FloatGroupedSgd_NonContiguousView_PreservesCowPeerAndGroupRates()
    {
        var engine = new CpuEngine();
        var source = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var peer = (Tensor<float>)source.CloneShared();
        var transposedParameter = source.Transpose(new[] { 1, 0 });
        var slicedBacking = new[] { 71f, 10f, 20f, 72f };
        var slicedParameter = Tensor<float>.FromMemory(
            new Memory<float>(slicedBacking, 1, 2), new[] { 2 });

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var viewLoss = engine.ReduceSum(transposedParameter, null);
            var sliceLoss = engine.ReduceSum(slicedParameter, null);
            engine.TensorAdd(viewLoss, sliceLoss);
            plan = scope.CompileTraining(new[] { transposedParameter, slicedParameter });
        }

        using (plan)
        {
            plan.ConfigureOptimizerGrouped(
                OptimizerType.SGD,
                new[] { LrSchedule.Constant(0.2), LrSchedule.Constant(0.1) },
                new[] { 0, 1 });
            for (int step = 0; step < 3; step++)
                plan.Step();
        }

        var expectedSource = new[] { 0.4f, 1.4f, 2.4f, 3.4f };
        var actualSource = source.ToArray();
        for (int i = 0; i < expectedSource.Length; i++)
            Assert.Equal(expectedSource[i], actualSource[i], 5);
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, peer.ToArray());
        Assert.Equal(71f, slicedBacking[0]);
        Assert.Equal(9.7f, slicedBacking[1], 5);
        Assert.Equal(19.7f, slicedBacking[2], 5);
        Assert.Equal(72f, slicedBacking[3]);
    }

    [Fact]
    public void DoubleAdam_OffsetParameter_MatchesDenseMultiTensorSemantics()
    {
        var engine = new CpuEngine();
        var dense = new Tensor<double>(new[] { 3.0, -2.0 }, new[] { 2 });
        var backing = new[] { 101.0, 3.0, -2.0, 102.0 };
        var offset = Tensor<double>.FromMemory(new Memory<double>(backing, 1, 2), new[] { 2 });

        using var densePlan = CompileReduceSum(engine, dense);
        using var offsetPlan = CompileReduceSum(engine, offset);
        densePlan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.01f);
        offsetPlan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.01f);

        densePlan.Step();
        offsetPlan.Step();

        Assert.Equal(dense.ToArray()[0], backing[1], 12);
        Assert.Equal(dense.ToArray()[1], backing[2], 12);
        Assert.Equal(101.0, backing[0]);
        Assert.Equal(102.0, backing[3]);
    }

    [Fact]
    public void DoubleGroupedSgd_NonContiguousView_UpdatesUnderlyingLogicalElements()
    {
        var engine = new CpuEngine();
        var source = new Tensor<double>(new[] { 2.0, 4.0, 6.0, 8.0 }, new[] { 2, 2 });
        var view = source.Transpose(new[] { 1, 0 });
        var dense = new Tensor<double>(new[] { 10.0, 20.0 }, new[] { 2 });

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var viewLoss = engine.ReduceSum(view, null);
            var denseLoss = engine.ReduceSum(dense, null);
            engine.TensorAdd(viewLoss, denseLoss);
            plan = scope.CompileTraining(new[] { view, dense });
        }

        using (plan)
        {
            plan.ConfigureOptimizerGrouped(
                OptimizerType.SGD,
                new[] { LrSchedule.Constant(0.05), LrSchedule.Constant(0.2) },
                new[] { 0, 1 });
            plan.Step();
        }

        Assert.Equal(new[] { 1.95, 3.95, 5.95, 7.95 }, source.ToArray());
        Assert.Equal(new[] { 9.8, 19.8 }, dense.ToArray());
    }

    [Fact]
    public void GradientClip_DoesNotClassifyParameterViewAsUnmaterialized()
    {
        var engine = new CpuEngine();
        var source = new Tensor<float>(new[] { 1f, 1f, 1f, 1f }, new[] { 2, 2 });
        var view = source.Transpose(new[] { 1, 0 });

        using var plan = CompileReduceSum(engine, view);
        plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1f);
        plan.SetMaxGradNorm(1.0);
        plan.Step();

        // ||[1,1,1,1]||₂=2, so clipping to 1 scales every gradient to ~0.5.
        foreach (float value in source.ToArray())
            Assert.Equal(0.95f, value, 5);
    }

    [Fact]
    public void MatMulForwardBackward_NonContiguousParameter_UsesStrideAwareFallback()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 1f, 0f, 0f, 1f }, new[] { 2, 2 });
        var source = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var parameterView = source.Transpose(new[] { 1, 0 });

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.TensorMatMul(input, parameterView);
            engine.ReduceSum(output, null);
            plan = scope.CompileTraining(new[] { parameterView });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1f);
            plan.Step();
        }

        Assert.Equal(new[] { 0.9f, 1.9f, 2.9f, 3.9f }, source.ToArray());
    }
}
