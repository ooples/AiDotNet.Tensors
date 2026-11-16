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
            if (capture)
            {
                Skip.If(device!.GetBackend() is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                    "Whole-step graph capture requires CUDA.");
                Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                    "Graph capture is disabled for this process.");
            }

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
                long replayCountBefore = CompiledTrainingPlan<float>.GraphReplayCount;

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
                if (capture)
                {
                    Assert.NotNull(((CompiledTrainingPlan<float>)plan).StepGraphNodeCounts);
                    Assert.True(CompiledTrainingPlan<float>.GraphReplayCount > replayCountBefore,
                        ((CompiledTrainingPlan<float>)plan).GraphCaptureFailureReason ?? "No CUDA graph replay occurred.");
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
    public void Retained_upstream_activation_does_not_accumulate_stale_convolution_gradients()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var input = new Tensor<float>([1, 1, 4, 4]);
        var kernel = new Tensor<float>(new[] { 2f }, new[] { 1, 1, 1, 1 });
        var bias = new Tensor<float>(new[] { 1f }, new[] { 1 });
        var parameters = new[] { input, kernel, bias };
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var upstream = engine.TensorMultiplyScalar(input, 3f);
            GradientRecording.RetainOutput(upstream);
            var conv = engine.Conv2D(upstream, kernel, 1, 0, 1);
            engine.ReduceSum(engine.ReLU(engine.TensorChannelBiasAdd(conv, bias)), null);
            plan = scope.CompileTraining(parameters);
        }
        using (plan)
            for (int step = 1; step <= 3; step++)
            {
                input.AsWritableSpan().Fill(step);
                input.IncrementVersion();
                Assert.Equal(16f * (6f * step + 1f), plan.Step().ToArray()[0]);
                foreach (var value in plan.Gradients[0].ToArray()) Assert.Equal(6f, value);
                Assert.Equal(48f * step, plan.Gradients[1].ToArray()[0]);
                Assert.Equal(16f, plan.Gradients[2].ToArray()[0]);
            }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Observation_only_division_does_not_poison_loss_gradients(bool pool)
    {
        using var options = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Push(
            new AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions { EnableBackwardGradientPooling = pool });
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<float>([8]);
        parameter.AsWritableSpan().Fill(1f);
        var zero = new Tensor<float>([8]);
        ICompiledTrainingPlan<float> plan;
        Tensor<float> observed;
        using (var scope = GraphMode.EnableTraining(new[] { parameter }))
        {
            var loss = engine.ReduceSum(parameter, null);
            observed = engine.TensorDivide(parameter, zero);
            GradientRecording.RetainOutput(observed);
            plan = scope.CompileTraining(new[] { parameter }, loss);
        }
        using (plan)
            for (int step = 1; step <= 3; step++)
            {
                parameter.AsWritableSpan().Fill(step);
                parameter.IncrementVersion();
                Assert.Equal(8f * step, plan.Step().ToArray()[0]);
                foreach (var value in observed.ToArray()) Assert.True(float.IsPositiveInfinity(value));
                foreach (var value in plan.Gradients[0].ToArray()) Assert.Equal(1f, value);
            }
    }

    [Fact]
    public void Mixed_precision_rejects_retained_outputs_before_detaching_the_trace()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var input = new Tensor<float>([8]);
        var prior = GraphMode.Current;
        Assert.Throws<NotSupportedException>(() => MixedPrecisionCompiledPlan.Trace(() =>
        {
            // ReduceSum stays float under autocast, so registration succeeds before the plan rejects it.
            var output = engine.ReduceSum(input, null);
            GradientRecording.RetainOutput(output);
            return output;
        }, engine));
        Assert.Same(prior, GraphMode.Current);
        using var scope = GraphMode.EnableTraining(new[] { input });
        var retained = engine.TensorAdd(input, input);
        GradientRecording.RetainOutput(retained);
        var loss = engine.ReduceSum(retained, null);
        Assert.Throws<NotSupportedException>(() => MixedPrecisionCompiledPlan.Compile(loss, engine));
        Assert.NotNull(loss.LazySource);
        Assert.NotNull(retained.LazySource);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Observation_only_matmul_chain_never_builds_a_fused_backward(bool pool)
    {
        using var options = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Push(
            new AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions { EnableBackwardGradientPooling = pool });
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var input = new Tensor<float>([2, 128]);
        input.AsWritableSpan().Fill(float.PositiveInfinity);
        var weight = new Tensor<float>([128, 128]);
        weight.AsWritableSpan().Fill(1f);
        var zeroWeight = new Tensor<float>([128, 1]);
        ICompiledTrainingPlan<float> plan;
        Tensor<float> observed;
        using (var scope = GraphMode.EnableTraining(new[] { weight }))
        {
            var loss = engine.ReduceSum(weight, null);
            observed = engine.TensorMatMul(engine.ReLU(engine.TensorMatMul(input, weight)), zeroWeight);
            GradientRecording.RetainOutput(observed);
            plan = scope.CompileTraining(new[] { weight }, loss);
        }
        using (plan)
            for (int step = 0; step < 3; step++)
            {
                Assert.Equal(16384f, plan.Step().ToArray()[0]);
                foreach (var value in observed.ToArray()) Assert.True(float.IsNaN(value));
                foreach (var value in plan.Gradients[0].ToArray()) Assert.Equal(1f, value);
            }
    }

    [Fact]
    public void Detached_undefined_derivative_does_not_enter_backward()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<float>([8]);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { parameter }))
        {
            var root = engine.TensorSqrt(parameter);
            GradientRecording.RetainOutput(root);
            var loss = engine.ReduceSum(engine.TensorAdd(parameter, engine.StopGradient(root)), null);
            plan = scope.CompileTraining(new[] { parameter }, loss);
        }
        using (plan)
            for (int step = 0; step < 3; step++)
            {
                Assert.Equal(0f, plan.Step().ToArray()[0]);
                foreach (var value in plan.Gradients[0].ToArray()) Assert.Equal(1f, value);
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
