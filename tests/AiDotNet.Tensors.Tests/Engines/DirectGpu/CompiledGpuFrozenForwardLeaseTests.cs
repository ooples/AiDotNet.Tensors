using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CompiledGpuFrozenForwardLeaseTests
{
    private static Tensor<float> Values(int seed)
    {
        var random = new Random(seed);
        var values = new float[64 * 64];
        for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() - 0.5) * 0.25f;
        return new Tensor<float>(values, new[] { 64, 64 });
    }

    private static (float[] Weights, float[] Losses, long ScratchBytes, int Candidates) Train(
        IEngine engine, bool noGrad, bool captured)
    {
        var frozen = Values(1);
        var input = Values(2);
        var weight = Values(3);
        var parameters = new[] { weight };
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            Tensor<float> FrozenChain()
            {
                var value = frozen;
                for (int i = 0; i < 6; i++) value = engine.Tanh(engine.TensorMultiplyScalar(value, 0.9f));
                return value;
            }
            Tensor<float> frozenResult;
            if (noGrad)
            {
                using (new NoGradScope<float>()) frozenResult = FrozenChain();
            }
            else frozenResult = FrozenChain(); // Unselected leaves produce no backward, even outside NoGrad.
            // The active multiply's backward still needs the frozen value through this view: its buffer must stay live.
            var throughView = engine.Reshape(engine.Reshape(frozenResult, new[] { 1, 64, 64 }), new[] { 64, 64 });
            var predicted = engine.TensorMatMul(input, weight);
            var gated = engine.TensorMultiply(predicted, throughView);
            var loss = engine.ReduceMean(engine.TensorMultiply(gated, gated), null);
            plan = scope.CompileTraining(parameters, loss);
        }
        using (plan)
        {
            var compiled = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            if (!captured) compiled.DisableGraphStep();
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.01f);
            var losses = new List<float>();
            for (int step = 0; step < (captured ? 8 : 4); step++)
            {
                input[0] = 0.125f + step * 0.01f;
                losses.Add(plan.Step().ToArray()[0]);
            }
            return (weight.ToArray(), losses.ToArray(), compiled.ScratchBufferBytes, compiled.ForwardSpanLeaseCandidates);
        }
    }

    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void FrozenChainsReleaseIntermediateStorageButKeepValuesNeededByActiveBackward(bool noGrad, bool captured)
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "CUDA unavailable: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "CUDA backend required.");
            var previousEngine = AiDotNetEngine.Current;
            bool previousLeases = CompiledTrainingPlan<float>.LeaseForwardOutputs;
            bool? previousInputGradients = CompiledTrainingPlan<float>.CompiledInputGradientsForTests;
            try
            {
                CompiledTrainingPlan<float>.CompiledInputGradientsForTests = false;
                AiDotNetEngine.Current = new CpuEngine();
                var expected = Train(AiDotNetEngine.Current, noGrad, captured);
                AiDotNetEngine.Current = gpu;
                CompiledTrainingPlan<float>.LeaseForwardOutputs = false;
                var kept = Train(gpu, noGrad, captured);
                CompiledTrainingPlan<float>.LeaseForwardOutputs = true;
                long replaysBefore = CompiledTrainingPlan<float>.GraphReplayCount;
                var leased = Train(gpu, noGrad, captured);
                if (captured)
                    Assert.True(CompiledTrainingPlan<float>.GraphReplayCount > replaysBefore,
                        "The candidate must replay a captured graph.");
                Assert.True(leased.Candidates >= 10, $"Only {leased.Candidates} forward buffers qualified.");
                Assert.True(leased.ScratchBytes < kept.ScratchBytes,
                    $"Leased {leased.ScratchBytes} scratch bytes, kept {kept.ScratchBytes}.");
                AssertClose(kept.Weights, leased.Weights);
                AssertClose(kept.Losses, leased.Losses);
                AssertClose(expected.Losses, leased.Losses);
                AssertClose(expected.Weights, leased.Weights);
            }
            finally
            {
                AiDotNetEngine.Current = previousEngine;
                CompiledTrainingPlan<float>.LeaseForwardOutputs = previousLeases;
                CompiledTrainingPlan<float>.CompiledInputGradientsForTests = previousInputGradients;
            }
        }
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-6f + 2e-3f * Math.Abs(expected[i]),
                $"[{i}]: expected {expected[i]}, actual {actual[i]}");
    }
}
