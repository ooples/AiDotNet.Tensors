using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A declared output holds the last step's value on the device until it is read. Anything that retires the plan's
/// device state between the step and the read (reconfiguring the optimizer, leaving graph replay) must land that value
/// on the host first, as it does the loss and the gradients, instead of dropping its pending download.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CompiledRetainedOutputRetirementTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    private const int Steps = 8;

    // capture: the plan must be replaying a captured CUDA step graph when it is retired (the case the landing exists
    // for); otherwise it never captures, and retirement meets a resident or eager step instead.
    private static DirectGpuTensorEngine RequireDevice(bool capture)
    {
        DirectGpuTensorEngine? device = null;
        try { device = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(device is null || !device.IsGpuAvailable, "No direct GPU backend available.");
        if (capture)
        {
            Skip.If(device!.GetBackend() is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "Whole-step graph capture requires CUDA.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");
        }
        return device!;
    }

    // retained = input + parameter, loss = sum(retained^2); runs Steps steps with input = step.
    private static CompiledTrainingPlan<float> CompileAndRun(DirectGpuTensorEngine device, int length, bool capture,
        out Tensor<float> retained)
    {
        AiDotNetEngine.Current = device;
        var input = new Tensor<float>([length]);
        var parameter = new Tensor<float>([length]);
        parameter.AsWritableSpan().Fill(0.25f);
        parameter.Gpu();
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { parameter }))
        {
            retained = device.TensorAdd(input, parameter);
            GradientRecording.RetainOutput(retained);
            device.ReduceSum(device.TensorMultiply(retained, retained), null);
            plan = scope.CompileTraining(new[] { parameter });
        }
        if (!capture) plan.DisableGraphStep();
        long replaysBefore = CompiledTrainingPlan<float>.GraphReplayCount;
        for (int step = 1; step <= Steps; step++)
        {
            input.AsWritableSpan().Fill(step);
            input.IncrementVersion();
            plan.Step();
            if (step == 2) foreach (float value in retained.ToArray()) Assert.Equal(2.25f, value);
        }
        if (capture)
        {
            // Without a replaying graph, retirement never reaches the landing this test is about.
            Assert.True(plan.StepGraphNodeCounts is not null, plan.GraphCaptureFailureReason ?? "No step graph was captured.");
            Assert.True(CompiledTrainingPlan<float>.GraphReplayCount > replaysBefore,
                plan.GraphCaptureFailureReason ?? "No CUDA graph replay occurred.");
        }
        return plan;
    }

    [SkippableTheory]
    [InlineData(false, 16, true)]
    [InlineData(true, 16, true)]
    [InlineData(false, 65536, true)]
    [InlineData(true, 65536, true)]
    [InlineData(false, 16, false)]
    [InlineData(true, 16, false)]
    [InlineData(false, 65536, false)]
    [InlineData(true, 65536, false)]
    public void Declared_output_unread_since_the_step_survives_retiring_the_device_step(bool reconfigure, int length, bool capture)
    {
        using var device = RequireDevice(capture);
        using var plan = CompileAndRun(device, length, capture, out var retained);
        if (reconfigure) plan.ConfigureOptimizer(OptimizerType.SGD, 0.0);
        else plan.DisableGraphStep();
        foreach (float value in retained.ToArray()) Assert.Equal(Steps + 0.25f, value);
    }

    // A declared output whose landing fails (a CUDA copy error throws InvalidOperationException) must not come back as
    // an earlier step's host copy: retirement keeps its download pending, so the caller's read reports the failure.
    // Dispose retires the device step too, and must not throw.
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Declared_output_whose_landing_fails_reports_the_failure_on_read(bool dispose)
    {
        using var device = RequireDevice(capture: true);
        var plan = CompileAndRun(device, 16, capture: true, out var retained);
        const string failure = "simulated failed device-to-host copy";
        var key = (object?)retained.GetBackingArrayForCacheLookupUnsafe() ?? retained.DataVector;
        try
        {
            // Stands in for the step's own pending download, as a failed transfer leaves it.
            HostSync.Register(key, _ => throw new InvalidOperationException(failure));
            if (dispose) plan.Dispose();
            else plan.ConfigureOptimizer(OptimizerType.SGD, 0.0);
            var error = Assert.Throws<InvalidOperationException>(() => retained.ToArray());
            Assert.Contains(failure, error.Message);
            Assert.True(HostSync.IsPending(key), "the failed download was dropped");
        }
        finally
        {
            HostSync.Remove(key);
            plan.Dispose();
        }
    }
}
