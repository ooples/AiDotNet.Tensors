using System;
using AiDotNet.Tensors.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public sealed class UnusedCompiledGradientTests
{
    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void DetachedForwardBranchesDoNotRetainUnusedGradientsAndPreserveResults(bool cuda, bool inputGradients)
    {
        var previousEngine = AiDotNetEngine.Current;
        bool? previousInputGradients = CompiledTrainingPlan<float>.CompiledInputGradientsForTests;
        var previousViews = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_VIEWS");
        DirectGpuTensorEngine? gpu = null;
        try
        {
            IEngine engine = new CpuEngine();
            if (cuda)
            {
                try { gpu = new DirectGpuTensorEngine(); }
                catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
                {
                    Skip.If(true, "CUDA unavailable: " + ex.GetType().Name);
                    return;
                }
                Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                    "CUDA backend required.");
                engine = gpu;
            }
            AiDotNetEngine.Current = engine;
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = inputGradients;
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_VIEWS", "1");
            var x = new Tensor<float>(new[] { 0.5f, 1f, -0.5f, 2f }, new[] { 2, 2 });
            var p = new Tensor<float>(new[] { 1f, 0.5f, -2f, 0.25f }, new[] { 2, 2 });
            var frozen = new Tensor<float>(new[] { 2f, -1f, 0.5f, 1.5f }, new[] { 2, 2 });
            var frozenBefore = frozen.ToArray();
            var parameters = new[] { frozen, p };
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                Tensor<float> target;
                using (new NoGradScope<float>())
                {
                    var first = engine.TensorMultiply(x, frozen);
                    var second = engine.TensorMultiply(first, first);
                    target = engine.TensorMultiplyScalar(second, 0.5f);
                }
                var predicted = engine.TensorMultiply(x, p);
                var sum = engine.TensorAdd(predicted, target);
                var loss = engine.ReduceSum(engine.TensorMultiply(sum, sum), null);
                plan = scope.CompileTraining(parameters, loss);
            }
            var compiled = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            using (plan)
            {
                Assert.True(compiled.PrunedGradientBufferCount > 0,
                    "The detached chain's internal gradients have no backward reads or writes.");
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0f);
                for (int step = 0; step < 3; step++)
                {
                    var inputs = new[] { 0.5f + step * 0.125f, 1f, -0.5f, 2f - step * 0.25f };
                    inputs.CopyTo(x.AsWritableSpan());
                    var weights = p.ToArray();
                    float expectedLoss = 0;
                    var expectedParameterGradient = new float[inputs.Length];
                    var expectedInputGradient = new float[inputs.Length];
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        float detached = inputs[i] * frozenBefore[i];
                        float sum = inputs[i] * weights[i] + 0.5f * detached * detached;
                        expectedLoss += sum * sum;
                        expectedParameterGradient[i] = 2f * sum * inputs[i];
                        expectedInputGradient[i] = 2f * sum * weights[i];
                    }
                    AssertClose(new[] { expectedLoss }, plan.Step().ToArray());
                    AssertClose(expectedParameterGradient, plan.Gradients[1].ToArray());
                    Assert.Equal(frozenBefore, frozen.ToArray());
                    Assert.All(plan.Gradients[0].ToArray(), value => Assert.Equal(0f, value));
                    if (inputGradients)
                    {
                        Assert.NotNull(x.Grad);
                        if (x.Grad is { } gradient) AssertClose(expectedInputGradient, gradient.ToArray());
                    }
                }
            }
            // Parameter and explicitly requested input results remain readable after residency is released.
            Assert.All(plan.Gradients[0].ToArray(), value => Assert.Equal(0f, value));
            if (inputGradients)
            {
                Assert.NotNull(x.Grad);
                if (x.Grad is { } gradient) Assert.All(gradient.ToArray(), value => Assert.True(!float.IsNaN(value) && !float.IsInfinity(value)));
            }
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = previousInputGradients;
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_VIEWS", previousViews);
            gpu?.Dispose();
        }
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-4f * (1f + Math.Abs(expected[i])),
                $"[{i}]: expected {expected[i]}, actual {actual[i]}");
    }
}
