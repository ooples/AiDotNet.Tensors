using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CompiledGpuWeightNormWorkspaceTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public CompiledGpuWeightNormWorkspaceTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableTheory]
    [InlineData(false, false, false)]
    [InlineData(false, true, false)]
    [InlineData(true, false, false)]
    [InlineData(false, false, true)]
    public void Device_banks_preserve_five_updates_and_retained_outputs(bool retain, bool rebuild, bool transpose)
    {
        Skip.IfNot(_fixture.IsAvailable, "No native GPU backend.");
        var baseline = Run(false, retain, rebuild, transpose);
        var repeated = Run(false, retain, rebuild, transpose);
        var candidate = Run(true, retain, rebuild, transpose);
        for (int row = 0; row < baseline.Values.Count; row++)
            Assert.True(baseline.Values[row].SequenceEqual(repeated.Values[row]), $"Unchanged control record {row} differs.");
        Assert.Equal(!retain, candidate.Active);
        if (candidate.Active)
        {
            Assert.True(candidate.WorkspaceBytes < candidate.OriginalBytes);
            Assert.Equal(1, candidate.AllocationRoots);
        }
        Assert.Equal(baseline.Values.Count, candidate.Values.Count);
        for (int row = 0; row < baseline.Values.Count; row++)
            Assert.True(baseline.Values[row].SequenceEqual(candidate.Values[row]), $"Workspace record {row} differs.");
    }

    private sealed record Result(bool Active, long OriginalBytes, long WorkspaceBytes,
        int AllocationRoots, List<float[]> Values);

    private Result Run(bool enabled, bool retain, bool rebuild, bool transpose)
    {
        var oldEngine = AiDotNetEngine.Current;
        bool oldEnabled = CompiledTrainingPlan<float>.GpuWeightNormWorkspaceEnabled;
        var engine = _fixture.Engine!;
        AiDotNetEngine.Current = engine;
        CompiledTrainingPlan<float>.GpuWeightNormWorkspaceEnabled = enabled;
        try
        {
            using var directionA = new Tensor<float>(Enumerable.Range(0, 48).Select(i => (i % 11 - 5) * .125f).ToArray(), new[] { 4, 4, 1, 3 });
            using var directionB = new Tensor<float>(Enumerable.Range(0, 80).Select(i => (i % 13 - 6) * .0625f).ToArray(), new[] { 4, 4, 1, 5 });
            using var gainA = new Tensor<float>(new[] { .75f, -.5f, 1.25f, .875f }, new[] { 4 });
            using var gainB = new Tensor<float>(new[] { -.625f, 1.125f, .5f, -.75f }, new[] { 4 });
            using var biasA = new Tensor<float>(new[] { .1f, -.2f, .3f, -.4f }, new[] { 4 });
            using var biasB = new Tensor<float>(new[] { -.3f, .5f, .2f, -.1f }, new[] { 4 });
            using var input = new Tensor<float>(Enumerable.Range(0, 64).Select(i => (i % 9 - 4) * .125f).ToArray(), new[] { 1, 4, 1, 16 });
            using var seed = new Tensor<float>(Enumerable.Range(0, 64).Select(i => (i % 7 - 3) * .125f).ToArray(), new[] { 1, 4, 1, 16 });
            var parameters = new[] { directionA, gainA, directionB, gainB, biasA, biasB };
            foreach (var tensor in parameters.Concat(new[] { input, seed })) engine.EnsureResidentInput(tensor);
            CompiledTrainingPlan<float> plan;
            Tensor<float> kernelA;
            using (var graph = GraphMode.EnableTraining(parameters))
            {
                kernelA = engine.TensorWeightNorm(directionA, gainA, .03125);
                var first = transpose
                    ? engine.ConvTranspose2D(input, kernelA, new[] { 1, 1 }, new[] { 0, 1 }, new[] { 0, 0 })
                    : engine.Conv2D(input, kernelA, new[] { 1, 1 }, new[] { 0, 1 }, new[] { 1, 1 });
                var kernelB = engine.TensorWeightNorm(directionB, gainB, .03125);
                var second = transpose
                    ? engine.ConvTranspose2D(input, kernelB, new[] { 1, 1 }, new[] { 0, 2 }, new[] { 0, 0 })
                    : engine.Conv2D(input, kernelB, new[] { 1, 1 }, new[] { 0, 2 }, new[] { 1, 1 });
                first = engine.TensorChannelBiasAdd(first, biasA);
                second = engine.TensorChannelBiasAdd(second, biasB);
                if (retain) graph.RetainOutput(kernelA);
                var loss = engine.ReduceSum(engine.TensorMultiply(engine.TensorAdd(first, second), seed));
                plan = graph.CompileTraining(parameters, loss);
            }
            using (plan)
            {
                plan.DisableGraphStep();
                if (rebuild) plan.EnableFrozenWeightOptimizations();
                var workspace = typeof(CompiledTrainingPlan<float>).GetField("_weightNormWorkspace", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(plan);
                long Bytes(string name) => workspace is null ? 0 : (long)workspace.GetType().GetField(name, BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(workspace)!;
                var steps = (CompiledStep<float>[])typeof(CompiledTrainingPlan<float>).GetField("_forwardSteps", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(plan)!;
                if (!transpose) Assert.Contains(steps, step => step.OpName == "FusedConv2DBias");
                plan.ConfigureOptimizer(OptimizerType.Adam, .003, .9, .999, 1e-8);
                var values = new List<float[]>();
                int allocationRoots = 0;
                for (int iteration = 0; iteration < 5; iteration++)
                {
                    values.Add(plan.Step().ToArray());
                    foreach (var parameter in parameters)
                    {
                        values.Add(parameter.ToArray());
                        values.Add(parameter.Grad!.ToArray());
                    }
                    if (retain) values.Add(kernelA.ToArray());
                    if (workspace is not null)
                    {
                        var kernels = steps.Where(step => step.OpType == OpType.TensorWeightNorm).Select(step => step.OutputBuffer).ToArray();
                        Assert.All(kernels, tensor => Assert.Null(tensor.GetBackingArrayForCacheLookupUnsafe()));
                        allocationRoots = kernels.Select(tensor => GpuBufferViewRoot.Of(tensor._gpuBuffer!)).Distinct().Count();
                    }
                }
                return new Result(workspace is not null, Bytes("OriginalBytes"), Bytes("WorkspaceBytes"), allocationRoots, values);
            }
        }
        finally
        {
            CompiledTrainingPlan<float>.GpuWeightNormWorkspaceEnabled = oldEnabled;
            AiDotNetEngine.Current = oldEngine;
        }
    }
}
