using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("DirectGpuSerial")]
public sealed class CompiledRetainedOutputTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void Declared_view_survives_later_forward_and_backward_work(bool gpu, bool capture)
    {
        DirectGpuTensorEngine? device = null;
        if (gpu)
        {
            try { device = new DirectGpuTensorEngine(); }
            catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
            Skip.If(device is null || !device.IsGpuAvailable, "No direct GPU backend available.");
        }
        using (device)
        {
            IEngine engine = device is null ? new CpuEngine() : device;
            AiDotNetEngine.Current = engine;
            // Cross both CPU forward-pooling and GPU span-lease size thresholds.
            const int length = 65536;
            var input = new Tensor<float>([256, 256]);
            var parameter = new Tensor<float>([256, 256]);
            parameter.AsWritableSpan().Fill(0.25f);
            if (gpu) parameter.Gpu();
            Tensor<float> retained;
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { parameter }))
            {
                var added = engine.TensorAdd(input, parameter);
                retained = engine.Reshape(added, new[] { length });
                GradientRecording.RetainOutput(retained);
                GradientRecording.RetainOutput(retained); // Idempotent registration.
                var doubled = engine.TensorAdd(retained, retained);
                engine.ReduceSum(engine.TensorAdd(doubled, doubled), null);
                plan = scope.CompileTraining(new[] { parameter });
            }
            using (plan)
            {
                if (!capture) ((CompiledTrainingPlan<float>)plan).DisableGraphStep();
                object? storage = null;
                for (int step = 0; step < 10; step++)
                {
                    input.AsWritableSpan().Fill(step + 1);
                    input.IncrementVersion();
                    float expected = step + 1.25f;
                    Assert.Equal(4f * length * expected, plan.Step().ToArray()[0]);
                    var values = retained.ToArray();
                    Assert.Equal(length, values.Length);
                    foreach (float value in values) Assert.Equal(expected, value);
                    foreach (float gradient in plan.Gradients[0].ToArray()) Assert.Equal(4f, gradient);
                    // Retention reserves stable bounded storage, rather than allocating one result per step.
                    if (storage is null) storage = retained._storage;
                    else Assert.Same(storage, retained._storage);
                }
            }
        }
    }

    [Fact]
    public void Declared_matmul_result_remains_observable_across_linear_activation_fusion()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var input = new Tensor<float>([4, 4]);
        var weight = new Tensor<float>([4, 4]);
        var bias = new Tensor<float>([4, 4]);
        weight.AsWritableSpan().Fill(0.25f);
        bias.AsWritableSpan().Fill(0.5f);
        Tensor<float> retained;
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { weight }))
        {
            retained = engine.TensorMatMul(input, weight);
            GradientRecording.RetainOutput(retained);
            engine.ReduceSum(engine.ReLU(engine.TensorAdd(retained, bias)), null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
            for (int step = 1; step <= 3; step++)
            {
                input.AsWritableSpan().Fill(step);
                input.IncrementVersion();
                Assert.Equal(16f * (step + 0.5f), plan.Step().ToArray()[0]);
                foreach (float value in retained.ToArray()) Assert.Equal((float)step, value);
                foreach (float gradient in plan.Gradients[0].ToArray()) Assert.Equal(4f * step, gradient);
            }
    }

    [Fact]
    public void Declared_output_outside_the_loss_ancestry_is_recomputed_each_step()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var input = new Tensor<float>([8]);
        var parameter = new Tensor<float>([8]);
        parameter.AsWritableSpan().Fill(0.25f);
        Tensor<float> retained;
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { parameter }))
        {
            var loss = engine.ReduceSum(parameter, null);
            // Recorded after the explicit loss, with no path back to it.
            retained = engine.TensorAdd(input, parameter);
            GradientRecording.RetainOutput(retained);
            plan = scope.CompileTraining(new[] { parameter }, loss);
        }
        using (plan)
            for (int step = 1; step <= 4; step++)
            {
                input.AsWritableSpan().Fill(step);
                input.IncrementVersion();
                Assert.Equal(2f, plan.Step().ToArray()[0]);
                foreach (float value in retained.ToArray()) Assert.Equal(step + 0.25f, value);
                foreach (float gradient in plan.Gradients[0].ToArray()) Assert.Equal(1f, gradient);
            }
    }


    [Fact]
    public void Registration_requires_an_active_uncompiled_training_forward()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<float>([2]);
        Assert.Throws<ArgumentNullException>(() => GradientRecording.RetainOutput<float>(null!));
        Assert.Throws<InvalidOperationException>(() => GradientRecording.RetainOutput(parameter));
        using (var inference = GraphMode.EnableInference())
        {
            var output = engine.TensorAdd(parameter, parameter);
            Assert.Throws<InvalidOperationException>(() => GradientRecording.RetainOutput(output));
        }
        var scope = GraphMode.EnableTraining(new[] { parameter });
        try
        {
            Assert.Throws<ArgumentException>(() => GradientRecording.RetainOutput(parameter));
            var output = engine.TensorAdd(parameter, parameter);
            GradientRecording.RetainOutput(output);
            engine.ReduceSum(output, null);
            using var plan = scope.CompileTraining(new[] { parameter });
            Assert.Throws<InvalidOperationException>(() => scope.RetainOutput(output));
        }
        finally { scope.Dispose(); }
        Assert.Throws<ObjectDisposedException>(() => scope.RetainOutput(parameter));
    }
}
