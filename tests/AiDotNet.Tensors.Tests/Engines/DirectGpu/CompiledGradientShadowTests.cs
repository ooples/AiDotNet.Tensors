using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CompiledGradientShadowTests
{
    private static Tensor<float> Values(int rows, int columns, int seed)
    {
        var random = new Random(seed);
        var values = new float[rows * columns];
        for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() - .5) * .1f;
        return new Tensor<float>(values, new[] { rows, columns });
    }

    private static CompiledTrainingPlan<float> Compile(IEngine engine, Tensor<float> input,
        Tensor<float> weight, bool disableCapture = true)
    {
        var parameters = new[] { weight };
        using var graph = GraphMode.EnableTraining(parameters);
        var value = engine.TensorMatMul(input, weight);
        for (int i = 0; i < 5; i++) value = engine.Tanh(engine.TensorMultiplyScalar(value, .9f));
        var loss = engine.ReduceMean(engine.TensorMultiply(value, value), new[] { 0, 1 }, false);
        var plan = Assert.IsType<CompiledTrainingPlan<float>>(graph.CompileTraining(parameters, loss));
        if (disableCapture) plan.DisableGraphStep();
        plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: .001);
        return plan;
    }

    private static (List<float[]> Values, long Bytes, long Steps, long Reuses, long Delayed, long PhaseBytes) Run(DirectGpuTensorEngine gpu,
        bool enabled, bool gradientView, bool inputGradients)
    {
        Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRADIENT_SHADOWS", enabled ? "1" : "0");
        CompiledTrainingPlan<float>.CompiledInputGradientsForTests = inputGradients;
        AiDotNetEngine.Current = gpu;
        using var inputA = Values(8, 8, 1);
        using var inputB = Values(4, 8, 2);
        using var weight = Values(8, 8, 3);
        using var a = Compile(gpu, inputA, weight);
        using var b = Compile(gpu, inputB, weight);
        var values = new List<float[]>();
        for (int step = 0; step < 4; step++)
        {
            inputA[0] = .1f + .01f * step;
            inputB[0] = -.15f - .01f * step;
            var lossA = a.Step();
            var gradientA = a.Gradients[0];
            var inputGradientA = inputGradients ? inputA.Grad : null;
            using var view = gradientView ? new CpuEngine().Reshape(gradientA, new[] { 64 }) : null;
            if (enabled) Assert.Null(gradientA._gpuBuffer);
            var lossB = b.Step(); // May overwrite native storage the A step lent; no A value was read yet.
            values.Add(lossA.ToArray());
            values.Add(lossB.ToArray());
            values.Add(gradientA.ToArray());
            values.Add(weight.ToArray());
            if (view is not null) values.Add(view.ToArray());
            if (inputGradientA is not null) values.Add(inputGradientA.ToArray());
        }
        long bytes = a.GradientShadowBytes + b.GradientShadowBytes;
        long steps = a.GradientShadowCompletedSteps + b.GradientShadowCompletedSteps;
        long reuses = a.GradientShadowReuses + b.GradientShadowReuses;
        var last = a.Gradients[0];
        var expected = last.ToArray();
        a.Dispose();
        Assert.Equal(expected, last.ToArray());
        return (values, bytes, steps, reuses,
            a.GradientPhaseDelayedBindings + b.GradientPhaseDelayedBindings,
            a.GradientPhaseReuseBytes + b.GradientPhaseReuseBytes);
    }

    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void ExactShadowsPreserveUnreadGradientsViewsAndAdamTrajectory(bool gradientView, bool inputGradients)
    {
        using var gpu = CreateGpu();
        var oldEngine = AiDotNetEngine.Current;
        var oldInputs = CompiledTrainingPlan<float>.CompiledInputGradientsForTests;
        var oldFlag = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRADIENT_SHADOWS");
        try
        {
            var baseline = Run(gpu, false, gradientView, inputGradients);
            var shadow = Run(gpu, true, gradientView, inputGradients);
            Assert.Equal(0, baseline.Bytes);
            Assert.Equal(8, shadow.Steps);
            Assert.True(shadow.Bytes > 0);
            Assert.True(shadow.Reuses > 0, "Native backing must actually move between completed plans.");
            if (Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRADIENT_PHASE_REUSE") == "1")
            {
                Assert.True(shadow.Delayed > 0, "The first-use path must actually execute.");
                Assert.True(shadow.PhaseBytes > 0, "Dead forward or backward backing must actually serve public gradients.");
            }
            Assert.Equal(baseline.Values.Count, shadow.Values.Count);
            for (int i = 0; i < shadow.Values.Count; i++) Assert.Equal(baseline.Values[i], shadow.Values[i]);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRADIENT_SHADOWS", oldFlag);
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = oldInputs;
            AiDotNetEngine.Current = oldEngine;
        }
    }

    [SkippableFact]
    public void CapturablePlanRetainsItsPrivateDeviceGradients()
    {
        using var gpu = CreateGpu();
        var oldEngine = AiDotNetEngine.Current;
        var oldFlag = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRADIENT_SHADOWS");
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRADIENT_SHADOWS", "1");
            AiDotNetEngine.Current = gpu;
            using var input = Values(8, 8, 4);
            using var weight = Values(8, 8, 5);
            using var plan = Compile(gpu, input, weight, disableCapture: false);
            for (int i = 0; i < 4; i++) Assert.True(float.IsFinite(plan.Step().ToArray()[0]));
            Assert.Equal(0, plan.GradientShadowBytes);
            Assert.Equal(0, plan.GradientShadowCompletedSteps);
            Assert.Equal(0, plan.GradientPhaseDelayedBindings);
            Assert.Equal(0, plan.GradientPhaseReuseBytes);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRADIENT_SHADOWS", oldFlag);
            AiDotNetEngine.Current = oldEngine;
        }
    }

    private static DirectGpuTensorEngine CreateGpu()
    {
        var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
            "Actual CUDA required.");
        return gpu;
    }
}
