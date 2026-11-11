using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// RReLU on a GPU engine keeps its gradient. Unseeded training RReLU compiles as it does on the CPU: the GPU override hands a traced call to the
/// CpuEngine op, which records the node, instead of refusing the trace. Each step draws new slopes, and the backward
/// uses the slopes its forward applied (for x &lt; 0 the gradient of sum(y) is y / x).
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledGpuRreluTests
{
    private static DirectGpuTensorEngine Gpu()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex)
        {
            throw new SkipException("No GPU engine: " + ex.GetType().Name);
        }

        if (!gpu.IsGpuAvailable)
        {
            gpu.Dispose();
            Skip.If(true, "No GPU device.");
        }
        return gpu;
    }

    /// <summary>
    /// The eager tape on a GPU engine: the device kernel records its own backward, which rebuilds the drawn slopes
    /// from the forward's seed. In evaluation every negative slope is the midpoint (0.25 here). The op must run on the
    /// device: a CPU fallback would pass the slope checks without exercising the device kernel or its backward.
    /// </summary>
    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void Rrelu_OnTheGpuEngine_HasTheGradientOfTheSlopesItApplied(bool training)
    {
        using var gpu = Gpu();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var x = new Tensor<float>(new[] { -1f, -2f, -0.5f, -3f, 1f, -1.5f, -0.25f, 2f }, new[] { 8 });
            float[] output, grad;
            GpuLaunchProbe.Reset();
            using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
            {
                var y = gpu.TensorRrelu(x, 0.1, 0.4, training, seed: training ? null : 3);
                // Counted before ReduceSum and the backward launch work of their own.
                Assert.True(GpuLaunchProbe.Count > 0, "RReLU launched no GPU work.");
                var grads = tape.ComputeGradients(gpu.ReduceSum(y, null), new[] { x });
                output = y.ToArray();
                Assert.True(grads.ContainsKey(x), "The GPU engine's RReLU recorded no gradient for its input.");
                grad = grads[x].ToArray();
            }
            // Fallbacks entries read "{count}x {op}: {reason}".
            static bool IsRreluFallback(string entry) => entry.IndexOf("x TensorRrelu:", StringComparison.Ordinal) >= 0;
            Assert.True(IsRreluFallback("1x TensorRrelu: guard or route declined"), "The fallback matcher misses the probe's format.");
            Assert.DoesNotContain(GpuLaunchProbe.Fallbacks, IsRreluFallback);

            for (int i = 0; i < x.Length; i++)
            {
                float slope = x[i] > 0 ? 1f : output[i] / x[i];
                if (x[i] < 0 && !training) Assert.Equal(0.25f, slope, 5);
                Assert.True(Math.Abs(grad[i] - slope) <= 1e-5f,
                    $"element {i}: the forward applied slope {slope:R}, the backward {grad[i]:R}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    [SkippableFact]
    public void UnseededTrainingRrelu_CompilesOnTheGpuEngine_WithTheForwardsSlopesInTheBackward()
    {
        using var gpu = Gpu();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var x = new Tensor<float>(new[] { -1f, -2f, -0.5f, -3f, 1f, -1.5f, -0.25f, 2f }, new[] { 8 });
            Tensor<float> y;
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { x }))
            {
                y = gpu.TensorRrelu(x, 0.1, 0.4, training: true, seed: null);
                gpu.ReduceSum(y, null);
                plan = scope.CompileTraining(new[] { x });
            }

            using (plan)
            {
                var firstSlopes = new float[x.Length];
                bool anySlopeChanged = false;
                for (int step = 0; step < 2; step++)
                {
                    plan.Step();
                    var grad = plan.Gradients[0].ToArray();
                    var output = y.ToArray();
                    for (int i = 0; i < x.Length; i++)
                    {
                        float slope = x[i] > 0 ? 1f : output[i] / x[i];
                        if (x[i] < 0)
                            Assert.True(slope >= 0.1f - 1e-6f && slope <= 0.4f + 1e-6f, $"step {step}: slope {slope} at {i}");
                        Assert.True(Math.Abs(grad[i] - slope) <= 1e-5f,
                            $"step {step}, element {i}: the forward applied slope {slope:R}, the backward {grad[i]:R}");
                        if (step == 0) firstSlopes[i] = slope;
                        else if (x[i] < 0 && firstSlopes[i] != slope) anySlopeChanged = true;
                    }
                }
                // Each step draws a new seed. One slope can repeat by chance; all six negative ones cannot.
                Assert.True(anySlopeChanged, "Every slope of the second step repeated the first step's: the seed was not redrawn.");
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    /// <summary>
    /// A captured CUDA graph replays kernels with their capture-time arguments and does not re-run host code, so an
    /// unseeded RReLU's per-step seed would freeze. Its plan must never be graph-captured; the same plan with a seed has
    /// no per-step host state and keeps its eligibility (the control, so the check is not vacuous).
    /// </summary>
    [SkippableFact]
    public void UnseededTrainingRrelu_KeepsThePlanOutOfCudaGraphCapture()
    {
        using var gpu = Gpu();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            bool Eligible(int? seed)
            {
                var x = new Tensor<float>(new[] { -1f, -2f, -0.5f, -3f, 1f, -1.5f, -0.25f, 2f }, new[] { 8 });
                ICompiledTrainingPlan<float> plan;
                using (var scope = GraphMode.EnableTraining(new[] { x }))
                {
                    gpu.ReduceSum(gpu.TensorRrelu(x, 0.1, 0.4, training: true, seed: seed), null);
                    plan = scope.CompileTraining(new[] { x });
                }
                using (plan)
                    return Assert.IsType<CompiledTrainingPlan<float>>(plan).IsGraphStepEligible;
            }

            Assert.True(Eligible(7), "Control: the seeded RReLU plan is not graph-step eligible, so the check below proves nothing.");
            Assert.False(Eligible(null), "The unseeded RReLU plan is graph-step eligible: a captured graph would freeze its seed.");
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }
}
