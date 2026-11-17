using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CompiledGpuSharedWorkspaceTests
{
    private static Tensor<float> Values(int rows, int columns, int seed)
    {
        var random = new Random(seed);
        var data = new float[rows * columns];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(random.NextDouble() - 0.5) * 0.1f;
        return new Tensor<float>(data, new[] { rows, columns });
    }

    private static CompiledTrainingPlan<float> Compile(IEngine engine, Tensor<float> input, Tensor<float> weight,
        bool retainView, bool disableCapture = true)
    {
        var parameters = new[] { weight };
        using var scope = GraphMode.EnableTraining(parameters);
        var value = engine.TensorMatMul(input, weight);
        for (int i = 0; i < 5; i++) value = engine.Tanh(engine.TensorMultiplyScalar(value, 0.9f));
        if (retainView)
        {
            var view = engine.TensorNarrow(value, 0, 1, 2);
            GradientRecording.RetainOutput("public", engine.Reshape(view, new[] { 2, 8 }));
        }
        var loss = engine.ReduceMean(engine.TensorMultiply(value, value), new[] { 0, 1 }, false);
        var plan = Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(parameters, loss));
        if (disableCapture) plan.DisableGraphStep();
        plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.001);
        return plan;
    }

    private static (List<float[]> Samples, long Shared, long Reuses) Run(IEngine engine, bool share, bool retainView,
        bool inputGradients)
    {
        AiDotNetEngine.Current = engine;
        CompiledTrainingPlan<float>.ShareTransientWorkspaceForTests = share;
        CompiledTrainingPlan<float>.CompiledInputGradientsForTests = inputGradients;
        var aInput = Values(8, 8, 1);
        var bInput = Values(4, 8, 2);
        var weight = Values(8, 8, 3); // Both plans read and update the same public parameter storage.
        using var a = Compile(engine, aInput, weight, retainView);
        using var b = Compile(engine, bInput, weight, false);
        var samples = new List<float[]>();
        long shared = 0;
        for (int step = 0; step < 4; step++)
        {
            aInput[0] = 0.1f + 0.01f * step;
            bInput[0] = -0.15f - 0.01f * step;
            Tensor<float> aLoss;
            Tensor<float>? publicView = null;
            using (var retained = RetainedOutputCapture<float>.Begin())
            {
                aLoss = a.Step();
                retained.Complete();
                if (retainView) publicView = retained.Get("public");
            }
            shared += a.SharedWorkspaceBytesLastStep;
            var aGradient = a.Gradients[0];
            var aInputGradient = inputGradients ? aInput.Grad : null;
            var bLoss = b.Step();
            shared += b.SharedWorkspaceBytesLastStep;
            // Delay every public read until B has overwritten any incorrectly lent A storage.
            samples.Add(aLoss.ToArray());
            samples.Add(bLoss.ToArray());
            samples.Add(aGradient.ToArray());
            samples.Add(weight.ToArray());
            if (retainView) samples.Add(publicView!.ToArray());
            if (inputGradients) samples.Add(Assert.IsType<Tensor<float>>(aInputGradient).ToArray());
        }
        return (samples, shared, engine is DirectGpuTensorEngine gpu ? gpu.CompiledWorkspaceBroker.ReuseCount : 0);
    }

    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void AlternatingDifferentShapePlansPreserveUnreadLossGradientsViewsAndAdamTrajectory(bool retainView,
        bool inputGradients)
    {
        using var gpu = CreateGpu();
        var previous = AiDotNetEngine.Current;
        var previousShare = CompiledTrainingPlan<float>.ShareTransientWorkspaceForTests;
        var previousInputs = CompiledTrainingPlan<float>.CompiledInputGradientsForTests;
        try
        {
            var expected = Run(new CpuEngine(), false, retainView, inputGradients);
            var isolated = Run(gpu, false, retainView, inputGradients);
            var shared = Run(gpu, true, retainView, inputGradients);
            Assert.True(shared.Shared > 0, "This fixture must actually lend native forward storage.");
            Assert.True(shared.Reuses > isolated.Reuses, "The opposite plan must actually reuse returned backing.");
            Assert.Equal(expected.Samples.Count, shared.Samples.Count);
            for (int sample = 0; sample < shared.Samples.Count; sample++)
            {
                AssertClose(expected.Samples[sample], shared.Samples[sample]);
                AssertClose(isolated.Samples[sample], shared.Samples[sample]);
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            CompiledTrainingPlan<float>.ShareTransientWorkspaceForTests = previousShare;
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = previousInputs;
        }
    }

    [SkippableTheory]
    [InlineData("capturable")]
    [InlineData("capture_failure")]
    [InlineData("observer")]
    [InlineData("sanitize")]
    public void PotentialCaptureAndObservationKeepPrivateBacking(string mode)
    {
        using var gpu = CreateGpu();
        var previous = AiDotNetEngine.Current;
        var previousObserver = CompiledTrainingPlan<float>.ForwardStepObserver;
        bool previousSanitize = CompiledTrainingPlan<float>.SanitizeForwardBuffers;
        try
        {
            AiDotNetEngine.Current = gpu;
            using var plan = Compile(gpu, Values(8, 8, 1), Values(8, 8, 2), false,
                mode is not ("capturable" or "capture_failure"));
            if (mode == "capture_failure") plan.FailNextCaptureForTesting = true;
            if (mode == "observer") CompiledTrainingPlan<float>.ForwardStepObserver = (_, _, _) => { };
            if (mode == "sanitize") CompiledTrainingPlan<float>.SanitizeForwardBuffers = true;
            for (int i = 0; i < (mode == "capture_failure" ? 8 : 4); i++)
            {
                Assert.True(float.IsFinite(plan.Step().ToArray()[0]));
                Assert.Equal(0, plan.SharedWorkspaceBytesLastStep);
            }
            if (mode == "capture_failure")
            {
                Assert.Contains("capture failure forced for testing", Assert.IsType<string>(plan.GraphCaptureFailureReason));
                // A later explicit disable cannot authorize backing that has already participated in graph preparation.
                plan.DisableGraphStep();
                Assert.True(float.IsFinite(plan.Step().ToArray()[0]));
                Assert.Equal(0, plan.SharedWorkspaceBytesLastStep);
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            CompiledTrainingPlan<float>.ForwardStepObserver = previousObserver;
            CompiledTrainingPlan<float>.SanitizeForwardBuffers = previousSanitize;
        }
    }

    private static DirectGpuTensorEngine CreateGpu()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "CUDA unavailable: " + ex.GetType().Name);
            throw;
        }
        Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
            "CUDA backend required.");
        return gpu;
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-6f + 3e-3f * Math.Abs(expected[i]),
                $"[{i}]: expected {expected[i]}, actual {actual[i]}");
    }
}
