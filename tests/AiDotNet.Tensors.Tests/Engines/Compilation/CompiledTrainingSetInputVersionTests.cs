// Copyright (c) CrowdTrainer LLC. All rights reserved.
// SetInputs / SetInput write the captured input through an unmarked span. The whole-step CUDA graph's step I/O
// re-stages a device-resident input only when its version changes, so a setter that copied without advancing the
// version would leave the next replay training on the previous input. These pin the version bump on both setters.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public class CompiledTrainingSetInputVersionTests
{
    private static Tensor<float> Filled(float start)
    {
        var data = new float[16];
        for (int i = 0; i < data.Length; i++) data[i] = start + i;
        return new Tensor<float>(data, new[] { 4, 4 });
    }

    private static (CompiledTrainingPlan<float> Plan, Tensor<float> Input) Compile()
    {
        var engine = new CpuEngine();
        var input = Filled(1f);
        var w = Filled(0.5f);
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h = engine.TensorMatMul(input, w);
            engine.ReduceSum(h, null);
            plan = scope.CompileTraining(new[] { w });
        }
        return (plan, input);
    }

    [Fact]
    public void SetInputs_AdvancesTheCapturedInputsVersion()
    {
        var (plan, input) = Compile();
        try
        {
            int before = input.GpuCacheVersion;   // reading it starts tracking, as the step I/O does
            plan.SetInputs(new[] { Filled(100f) });
            Assert.NotEqual(before, input.GpuCacheVersion);
            Assert.Equal(100f, input.AsSpan()[0]);
        }
        finally { plan.Dispose(); }
    }

    [Fact]
    public void SetInput_AdvancesTheCapturedInputsVersion()
    {
        var (plan, input) = Compile();
        try
        {
            int before = input.GpuCacheVersion;
            plan.SetInput(Filled(200f));
            Assert.NotEqual(before, input.GpuCacheVersion);
            Assert.Equal(200f, input.AsSpan()[0]);
        }
        finally { plan.Dispose(); }
    }
}
