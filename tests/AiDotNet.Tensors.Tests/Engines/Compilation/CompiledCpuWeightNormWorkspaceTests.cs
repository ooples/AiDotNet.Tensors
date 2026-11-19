using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public sealed class CompiledCpuWeightNormWorkspaceTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    public void Workspace_preserves_five_updates_and_retained_kernel_values(bool retainFirst, bool rebuildForward)
    {
        var baseline = Run(false, retainFirst, rebuildForward);
        var checkpoint = Run(true, retainFirst, rebuildForward);
        Assert.Equal(!retainFirst, checkpoint.WorkspaceActive);
        Assert.Equal(retainFirst ? 2 : 1, checkpoint.KernelArrays);
        Assert.Equal(2, baseline.KernelArrays);
        Assert.Equal(baseline.Values.Count, checkpoint.Values.Count);
        for (int row = 0; row < baseline.Values.Count; row++)
        {
            var expected = baseline.Values[row]; var actual = checkpoint.Values[row];
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
        }
    }

    private sealed record Result(bool WorkspaceActive, int KernelArrays, List<float[]> Values);

    private sealed class CustomCpuEngine : CpuEngine { }

    [Fact]
    public void Custom_cpu_engine_keeps_independent_kernel_storage()
    {
        var baseline = Run(false, false, false, true);
        var guarded = Run(true, false, false, true);
        Assert.False(guarded.WorkspaceActive);
        Assert.Equal(2, guarded.KernelArrays);
        Assert.Equal(baseline.Values.Count, guarded.Values.Count);
        for (int i = 0; i < baseline.Values.Count; i++) Assert.Equal(baseline.Values[i], guarded.Values[i]);
    }

    private static Result Run(bool enabled, bool retainFirst, bool rebuildForward, bool customEngine = false)
    {
        var previousEngine = AiDotNetEngine.Current;
        bool previousDeferred = LazyTensorScope.DeferCpuTrainingOutputs;
        bool previousWorkspace = CompiledTrainingPlan<float>.CpuWeightNormWorkspaceEnabled;
        try
        {
            AiDotNetEngine.Current = customEngine ? new CustomCpuEngine() : new CpuEngine();
            var engine = AiDotNetEngine.Current;
            LazyTensorScope.DeferCpuTrainingOutputs = true;
            CompiledTrainingPlan<float>.CpuWeightNormWorkspaceEnabled = enabled;
            using var directionA = new Tensor<float>(Enumerable.Range(0, 48).Select(i => (i % 11 - 5) * .125f).ToArray(), new[] { 4, 4, 1, 3 });
            using var directionB = new Tensor<float>(Enumerable.Range(0, 48).Select(i => (i % 13 - 6) * .0625f).ToArray(), new[] { 4, 4, 1, 3 });
            using var gainA = new Tensor<float>(new[] { .75f, -.5f, 1.25f, .875f }, new[] { 4 });
            using var gainB = new Tensor<float>(new[] { -.625f, 1.125f, .5f, -.75f }, new[] { 4 });
            using var input = new Tensor<float>(Enumerable.Range(0, 64).Select(i => (i % 9 - 4) * .125f).ToArray(), new[] { 1, 4, 1, 16 });
            using var seed = new Tensor<float>(Enumerable.Range(0, 64).Select(i => (i % 7 - 3) * .125f).ToArray(), new[] { 1, 4, 1, 16 });
            var parameters = new[] { directionA, gainA, directionB, gainB };
            Tensor<float> kernelA;
            CompiledTrainingPlan<float> plan;
            using (var graph = GraphMode.EnableTraining(parameters))
            {
                kernelA = engine.TensorWeightNorm(directionA, gainA, .03125);
                var first = engine.Conv2D(input, kernelA, new[] { 1, 1 }, new[] { 0, 1 }, new[] { 1, 1 });
                var kernelB = engine.TensorWeightNorm(directionB, gainB, .03125);
                var second = engine.Conv2D(input, kernelB, new[] { 1, 1 }, new[] { 0, 1 }, new[] { 1, 1 });
                if (retainFirst) graph.RetainOutput(kernelA);
                var loss = engine.ReduceSum(engine.TensorMultiply(engine.TensorAdd(first, second), seed));
                plan = graph.CompileTraining(parameters, loss);
            }
            using (plan)
            {
                bool active = typeof(CompiledTrainingPlan<float>).GetField("_weightNormWorkspace", BindingFlags.NonPublic | BindingFlags.Instance)!
                    .GetValue(plan) is not null;
                var steps = (CompiledStep<float>[])typeof(CompiledTrainingPlan<float>)
                    .GetField("_forwardSteps", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(plan)!;
                int arrays = steps.Where(step => step.OpType == OpType.TensorWeightNorm)
                    .Select(step => step.OutputBuffer.GetBackingArrayForCacheLookupUnsafe()).Distinct().Count();
                if (rebuildForward) plan.EnableFrozenWeightOptimizations();
                plan.ConfigureOptimizer(OptimizerType.Adam, .003, .9, .999, 1e-8);
                var values = new List<float[]>();
                for (int iteration = 0; iteration < 5; iteration++)
                {
                    values.Add(plan.Step().ToArray());
                    foreach (var parameter in parameters)
                    {
                        values.Add(parameter.ToArray());
                        Assert.NotNull(parameter.Grad);
                        values.Add(parameter.Grad!.ToArray());
                    }
                    if (retainFirst) values.Add(kernelA.ToArray());
                }
                return new Result(active, arrays, values);
            }
        }
        finally
        {
            CompiledTrainingPlan<float>.CpuWeightNormWorkspaceEnabled = previousWorkspace;
            LazyTensorScope.DeferCpuTrainingOutputs = previousDeferred;
            AiDotNetEngine.Current = previousEngine;
        }
    }
}
