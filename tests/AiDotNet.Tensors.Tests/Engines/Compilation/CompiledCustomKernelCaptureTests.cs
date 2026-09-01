using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Custom differentiable kernels must remain live graph nodes during compiled training.
/// An eager-only implementation can appear correct on the trace batch while freezing its output
/// as a leaf, which severs both upstream gradients and subsequent replay updates.
/// </summary>
public class CompiledCustomKernelCaptureTests
{
    private delegate Tensor<float> Forward(CpuEngine engine, Tensor<float>[] parameters);

    private static Tensor<float> Values(int[] shape, float start = 0.08f, float step = 0.013f)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = start + step * (i % 11);
        return tensor;
    }

    private static void AssertCompiledParity(
        string operationName,
        Tensor<float>[] parameters,
        Forward forward,
        float lossTolerance = 2e-4f,
        float gradientTolerance = 2e-3f,
        CpuEngine? executionEngine = null,
        Action? mutateReplayInput = null)
    {
        var engine = executionEngine ?? new CpuEngine();
        var (eagerLoss, eagerGradients) = Eager(engine, parameters, forward);

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var compiledOutput = forward(engine, parameters);
            var compiledLoss = engine.ReduceSum(compiledOutput, null);
            plan = scope.CompileTraining(parameters, compiledLoss);
        }

        using (plan)
        {
            float replayedLoss = plan.Step()[0];
            AssertFiniteAndClose(operationName + " loss", eagerLoss, replayedLoss, lossTolerance);

            var compiled = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            AssertGradients(operationName, parameters, eagerGradients, compiled.Gradients, gradientTolerance);

            // This specifically distinguishes a graph edge from a plausible-looking value frozen at trace time.
            float originalLoss = replayedLoss;
            parameters[0][0] += 0.19f;
            var (updatedEagerLoss, updatedEagerGradients) = Eager(engine, parameters, forward);
            float updatedReplayLoss = plan.Step()[0];
            AssertFiniteAndClose(operationName + " updated loss", updatedEagerLoss, updatedReplayLoss, lossTolerance);
            Assert.True(Math.Abs(updatedReplayLoss - originalLoss) > 1e-7f,
                $"{operationName} replay ignored a live upstream parameter change.");
            AssertGradients(operationName + " updated", parameters, updatedEagerGradients,
                compiled.Gradients, gradientTolerance);

            if (mutateReplayInput is not null)
            {
                float lossBeforeInputMutation = updatedReplayLoss;
                mutateReplayInput();
                var (inputUpdatedEagerLoss, inputUpdatedEagerGradients) = Eager(engine, parameters, forward);
                float inputUpdatedReplayLoss = plan.Step()[0];
                AssertFiniteAndClose(operationName + " live input loss",
                    inputUpdatedEagerLoss, inputUpdatedReplayLoss, lossTolerance);
                Assert.True(Math.Abs(inputUpdatedReplayLoss - lossBeforeInputMutation) > 1e-7f,
                    $"{operationName} replay ignored a live non-parameter input change.");
                AssertGradients(operationName + " live input", parameters, inputUpdatedEagerGradients,
                    compiled.Gradients, gradientTolerance);
            }
        }
    }

    private static (float Loss, Dictionary<Tensor<float>, Tensor<float>> Gradients) Eager(
        CpuEngine engine,
        Tensor<float>[] parameters,
        Forward forward)
    {
        using var tape = new GradientTape<float>();
        var loss = engine.ReduceSum(forward(engine, parameters), null);
        return (loss[0], tape.ComputeGradients(loss, parameters));
    }

    private static void AssertGradients(
        string operationName,
        Tensor<float>[] parameters,
        IReadOnlyDictionary<Tensor<float>, Tensor<float>> expected,
        IReadOnlyList<Tensor<float>> actual,
        float tolerance)
    {
        Assert.Equal(parameters.Length, actual.Count);
        for (int parameterIndex = 0; parameterIndex < parameters.Length; parameterIndex++)
        {
            Assert.True(expected.TryGetValue(parameters[parameterIndex], out var expectedTensor),
                $"{operationName} eager backward omitted parameter {parameterIndex}.");
            var expectedValues = expectedTensor!.ToArray();
            var actualValues = actual[parameterIndex].ToArray();
            Assert.Equal(expectedValues.Length, actualValues.Length);
            for (int element = 0; element < expectedValues.Length; element++)
                AssertFiniteAndClose(
                    $"{operationName} gradient {parameterIndex}[{element}]",
                    expectedValues[element], actualValues[element], tolerance);
        }
    }

    private static void AssertFiniteAndClose(string label, float expected, float actual, float tolerance)
    {
        Assert.True(MathCompat.IsFinite(expected), $"{label}: eager value {expected} is not finite.");
        Assert.True(MathCompat.IsFinite(actual), $"{label}: compiled value {actual} is not finite.");
        float allowed = tolerance * Math.Max(1f, Math.Abs(expected));
        Assert.True(Math.Abs(actual - expected) <= allowed,
            $"{label}: compiled {actual:G9} != eager {expected:G9}; allowed {allowed:G9}.");
    }

    [Fact]
    public void AbcScan_IsCapturedWithGradientsAndLiveReplay()
    {
        var p = new[]
        {
            Values(new[] { 1, 3, 4 }), Values(new[] { 1, 3, 4 }, 0.11f),
            Values(new[] { 1, 3, 4 }, 0.17f), Values(new[] { 1, 3, 2 }, 0.55f, 0.02f),
            Values(new[] { 2, 2, 2 }, 0.09f)
        };
        AssertCompiledParity("ABC", p, (e, x) => e.AbcScanForward(x[0], x[1], x[2], x[3], x[4], 2));
    }

    [Fact]
    public void GatedDeltaNetScan_IsCapturedWithGradientsAndLiveReplay()
    {
        var p = new[]
        {
            Values(new[] { 1, 3, 4 }), Values(new[] { 1, 3, 4 }, 0.12f),
            Values(new[] { 1, 3, 4 }, 0.18f), Values(new[] { 1, 3, 2 }, 0.65f, 0.01f),
            Values(new[] { 1, 3, 2 }, 0.35f, 0.02f)
        };
        AssertCompiledParity("GatedDeltaNet", p,
            (e, x) => e.GatedDeltaNetScanForward(x[0], x[1], x[2], x[3], x[4], 2));
    }

    [Fact]
    public void GlaScan_IsCapturedWithGradientsAndLiveReplay()
    {
        var p = new[]
        {
            Values(new[] { 1, 3, 4 }), Values(new[] { 1, 3, 4 }, 0.12f),
            Values(new[] { 1, 3, 4 }, 0.18f), Values(new[] { 1, 3, 2 }, 0.65f, 0.01f)
        };
        AssertCompiledParity("GLA", p, (e, x) => e.GlaScanForward(x[0], x[1], x[2], x[3], 2));
    }

    [Fact]
    public void DirectGpuScan_IsCapturedWithGradientsAndLiveReplay()
    {
        using var engine = new DirectGpuTensorEngine();
        var p = new[]
        {
            Values(new[] { 1, 3, 4 }), Values(new[] { 1, 3, 4 }, 0.12f),
            Values(new[] { 1, 3, 4 }, 0.18f), Values(new[] { 1, 3, 2 }, 0.65f, 0.01f)
        };
        AssertCompiledParity(
            "DirectGpu GLA", p, (e, x) => e.GlaScanForward(x[0], x[1], x[2], x[3], 2),
            lossTolerance: 4e-4f, gradientTolerance: 3e-3f, executionEngine: engine);
    }

    [Fact]
    public void Interpolate_IsCapturedWithGradientsAndLiveReplay()
    {
        var p = new[] { Values(new[] { 1, 1, 2, 2 }, 0.1f, 0.17f) };
        AssertCompiledParity(
            "bilinear interpolate", p,
            (e, x) => e.Interpolate(x[0], new[] { 4, 4 }, InterpolateMode.Bilinear));
    }

    [Fact]
    public void DirectGpuInterpolate_IsCapturedWithGradientsAndLiveReplay()
    {
        using var engine = new DirectGpuTensorEngine();
        var p = new[] { Values(new[] { 1, 1, 2, 2 }, 0.1f, 0.17f) };
        AssertCompiledParity(
            "DirectGpu bilinear interpolate", p,
            (e, x) => e.Interpolate(x[0], new[] { 4, 4 }, InterpolateMode.Bilinear),
            lossTolerance: 4e-4f,
            gradientTolerance: 3e-3f,
            executionEngine: engine);
    }

    [Fact]
    public void MambaScan_IsCapturedWithGradientsAndLiveReplay()
    {
        var p = new[]
        {
            Values(new[] { 1, 3, 4 }), Values(new[] { 1, 3, 4 }, 0.04f, 0.006f),
            Values(new[] { 4, 2 }, -0.2f, 0.02f), Values(new[] { 1, 3, 2 }, 0.1f),
            Values(new[] { 1, 3, 2 }, 0.13f), Values(new[] { 4 }, 0.2f)
        };
        AssertCompiledParity("Mamba", p,
            (e, x) => e.MambaSelectiveScanForward(x[0], x[1], x[2], x[3], x[4], x[5]));
    }

    [Fact]
    public void Mamba2Scan_IsCapturedWithGradientsAndLiveReplay()
    {
        var p = new[]
        {
            Values(new[] { 1, 3, 4 }), Values(new[] { 1, 3, 2 }, 0.04f, 0.006f),
            Values(new[] { 2 }, -0.2f, 0.03f), Values(new[] { 1, 3, 2 }, 0.1f),
            Values(new[] { 1, 3, 2 }, 0.13f), Values(new[] { 2 }, 0.2f)
        };
        AssertCompiledParity("Mamba2", p,
            (e, x) => e.Mamba2SsdScanForward(x[0], x[1], x[2], x[3], x[4], x[5], 2));
    }

    [Fact]
    public void MesaScan_IsCapturedWithGradientsAndLiveReplay()
    {
        var p = new[]
        {
            Values(new[] { 1, 3, 4 }, 0.04f), Values(new[] { 1, 3, 4 }, 0.06f),
            Values(new[] { 1, 3, 4 }, 0.08f), Values(new[] { 2, 2, 2 }, 0.03f)
        };
        AssertCompiledParity("Mesa", p,
            (e, x) => e.MesaScanForward(x[0], x[1], x[2], x[3], 0.75f, 2),
            gradientTolerance: 4e-3f);
    }

    [Fact]
    public void RgLruScan_IsCapturedWithGradientsAndLiveReplay()
    {
        var p = new[]
        {
            Values(new[] { 1, 3, 4 }), Values(new[] { 1, 3, 4 }, 0.55f, 0.01f),
            Values(new[] { 1, 3, 4 }, 0.45f, 0.01f), Values(new[] { 4 }, 0.1f)
        };
        AssertCompiledParity("RG-LRU", p, (e, x) => e.RgLruScanForward(x[0], x[1], x[2], x[3]));
    }

    [Fact]
    public void Rwkv4Scan_IsCapturedWithGradientsAndLiveReplay()
    {
        var p = new[]
        {
            Values(new[] { 1, 3, 4 }, -0.1f), Values(new[] { 1, 3, 4 }, 0.06f),
            Values(new[] { 1, 3, 4 }, 0.09f), Values(new[] { 4 }, -0.3f, 0.03f),
            Values(new[] { 4 }, 0.12f)
        };
        AssertCompiledParity("RWKV4", p,
            (e, x) => e.Rwkv4WkvForward(x[0], x[1], x[2], x[3], x[4]));
    }

    [Fact]
    public void XLstmScan_IsCapturedWithGradientsAndLiveReplay()
    {
        var p = new[]
        {
            Values(new[] { 1, 3, 4 }, 0.04f), Values(new[] { 1, 3, 4 }, 0.06f),
            Values(new[] { 1, 3, 4 }, 0.08f), Values(new[] { 1, 3, 2 }, 0.7f, 0.02f),
            Values(new[] { 1, 3, 2 }, 0.6f, 0.01f), Values(new[] { 1, 3, 2 }, 0.55f, 0.01f)
        };
        AssertCompiledParity("xLSTM", p,
            (e, x) => e.XLstmScanForward(x[0], x[1], x[2], x[3], x[4], x[5], 2));
    }

    [Fact]
    public void ComplexDiagonalSsmScan_IsCapturedWithGradientsAndLiveReplay()
    {
        var p = new[]
        {
            Values(new[] { 1, 3, 1, 2 }), Values(new[] { 1, 2 }, 0.2f),
            Values(new[] { 1, 2 }, 0.03f), Values(new[] { 1, 2, 2 }, 0.08f),
            Values(new[] { 1, 2, 2 }, 0.02f), Values(new[] { 1, 2, 2 }, 0.09f),
            Values(new[] { 1, 2, 2 }, 0.01f), Values(new[] { 1, 2 }, 0.15f)
        };
        AssertCompiledParity("complex diagonal SSM", p,
            (e, x) => e.ComplexDiagonalSsmScanForward(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]));
    }

    [Fact]
    public void RoutedDiagonalSsmScan_IsCapturedWithGradientsAndLiveReplay()
    {
        var p = new[]
        {
            Values(new[] { 1, 3, 2 }), Values(new[] { 1, 3, 2 }, 0.4f, 0.03f),
            Values(new[] { 2, 2 }, 0.2f), Values(new[] { 2, 2, 2 }, 0.08f),
            Values(new[] { 2, 2, 2 }, 0.09f), Values(new[] { 2, 2 }, 0.15f)
        };
        AssertCompiledParity("routed diagonal SSM", p,
            (e, x) => e.RoutedDiagonalSsmScanForward(x[0], x[1], x[2], x[3], x[4], x[5]));
    }

    [Fact]
    public void FusedLinearCrossEntropy_IndexLabelsStayLiveAcrossReplays()
    {
        var engine = new CpuEngine();
        var parameters = new[]
        {
            Values(new[] { 2, 3 }, 0.1f), Values(new[] { 3, 4 }, -0.08f, 0.04f),
            Values(new[] { 4 }, -0.03f, 0.02f)
        };
        var targetIds = new Tensor<int>(new[] { 0, 2 }, new[] { 2 });
        Forward forward = (e, x) => e.FusedLinearCrossEntropyWithLogits(x[0], x[1], x[2], targetIds);
        var (expectedLoss, expectedGradients) = Eager(engine, parameters, forward);

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var compiledLoss = forward(engine, parameters);
            plan = scope.CompileTraining(parameters, compiledLoss);
        }

        using (plan)
        {
            float actualLoss = plan.Step()[0];
            var compiled = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            AssertFiniteAndClose("indexed fused CE loss", expectedLoss, actualLoss, 2e-5f);
            AssertGradients("indexed fused CE", parameters, expectedGradients, compiled.Gradients, 2e-5f);

            targetIds[0] = 1;
            var (updatedExpectedLoss, updatedExpectedGradients) = Eager(engine, parameters, forward);
            float updatedActualLoss = plan.Step()[0];
            AssertFiniteAndClose("indexed fused CE updated loss", updatedExpectedLoss, updatedActualLoss, 2e-5f);
            Assert.True(Math.Abs(updatedActualLoss - actualLoss) > 1e-7f,
                "Compiled fused CE reused the labels from tracing instead of the current batch.");
            AssertGradients("indexed fused CE updated", parameters, updatedExpectedGradients,
                compiled.Gradients, 2e-5f);
        }
    }

    [Fact]
    public void FusedLinearCrossEntropy_DenseTargetsAreCaptured()
    {
        var parameters = new[]
        {
            Values(new[] { 2, 3 }, 0.1f), Values(new[] { 3, 4 }, -0.08f, 0.04f),
            Values(new[] { 4 }, -0.03f, 0.02f)
        };
        var target = new Tensor<float>(
            new[] { 1f, 0f, 0f, 0f, 0f, 0f, 1f, 0f }, new[] { 2, 4 });
        AssertCompiledParity("dense fused CE", parameters,
            (e, x) => e.FusedLinearCrossEntropyWithLogits(x[0], x[1], x[2], target),
            lossTolerance: 2e-5f, gradientTolerance: 2e-5f,
            mutateReplayInput: () =>
            {
                for (int i = 0; i < target.Length; i++) target[i] = 0f;
                target[1] = 1f;
                target[7] = 1f;
            });
    }

    [Fact]
    public async Task ConcurrentStepCalls_AreSerializedAcrossForwardAndBackward()
    {
        var engine = new CpuEngine();
        var parameter = Values(new[] { 1 });
        var parameters = new[] { parameter };
        int activeForwards = 0;
        int maximumConcurrentForwards = 0;

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var output = scope.RecordUnary(
                LazyNodeType.Custom,
                "ConcurrentReplayProbe",
                parameter,
                new[] { 1 },
                (_, destination) =>
                {
                    int active = Interlocked.Increment(ref activeForwards);
                    int observed;
                    do
                    {
                        observed = maximumConcurrentForwards;
                        if (active <= observed) break;
                    }
                    while (Interlocked.CompareExchange(
                        ref maximumConcurrentForwards, active, observed) != observed);

                    Thread.Sleep(40);
                    destination[0] = parameter[0];
                    Interlocked.Decrement(ref activeForwards);
                },
                BackwardFunctions<float>.ReshapeBackward,
                new object[] { new[] { 1 } });
            var loss = engine.ReduceSum(output, null);
            plan = scope.CompileTraining(parameters, loss);
        }

        using (plan)
        using (var start = new ManualResetEventSlim(false))
        {
            Task Replay() => Task.Run(() =>
            {
                start.Wait();
                _ = plan.Step()[0];
            });

            var first = Replay();
            var second = Replay();
            start.Set();
            await Task.WhenAll(first, second);
        }

        Assert.Equal(1, maximumConcurrentForwards);
    }
}
