using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
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

    [SkippableTheory]
    [InlineData(false, 16)]
    [InlineData(true, 16)]
    [InlineData(false, 65536)]
    [InlineData(true, 65536)]
    public void Declared_output_unread_since_the_step_survives_retiring_the_device_step(bool reconfigure, int length)
    {
        DirectGpuTensorEngine? device = null;
        try { device = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(device is null || !device.IsGpuAvailable, "No direct GPU backend available.");
        using (device)
        {
            AiDotNetEngine.Current = device!;
            var input = new Tensor<float>([length]);
            var parameter = new Tensor<float>([length]);
            parameter.AsWritableSpan().Fill(0.25f);
            parameter.Gpu();
            Tensor<float> retained;
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { parameter }))
            {
                retained = device!.TensorAdd(input, parameter);
                GradientRecording.RetainOutput(retained);
                device.ReduceSum(device.TensorMultiply(retained, retained), null);
                plan = scope.CompileTraining(new[] { parameter });
            }
            using (plan)
            {
                // Past the warmup steps, so a backend with graph capture is replaying its step graph by the last one.
                const int steps = 8;
                for (int step = 1; step <= steps; step++)
                {
                    input.AsWritableSpan().Fill(step);
                    input.IncrementVersion();
                    plan.Step();
                    if (step == 2) foreach (float value in retained.ToArray()) Assert.Equal(2.25f, value);
                }
                if (reconfigure) plan.ConfigureOptimizer(OptimizerType.SGD, 0.0);
                else plan.DisableGraphStep();
                foreach (float value in retained.ToArray()) Assert.Equal(steps + 0.25f, value);
            }
        }
    }
}
