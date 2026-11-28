using System;
using System.Linq;
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

    // TensorScope.Keep/Detach of a traced activation declares it retained: the training plan must not lease its
    // buffer to a later step or release its value, or the caller reads another tensor's values after Step().
    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void Detached_or_kept_activation_survives_later_forward_and_backward_work(bool gpu, bool detach)
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
            const int length = 65536;   // past the CPU forward-pooling and GPU span-lease thresholds
            var input = new Tensor<float>([256, 256]);
            var parameter = new Tensor<float>([256, 256]);
            parameter.AsWritableSpan().Fill(0.25f);
            if (gpu) parameter.Gpu();
            Tensor<float> retained;
            ICompiledTrainingPlan<float> plan;
            using (var tensorScope = TensorScope.Begin())
            using (var scope = GraphMode.EnableTraining(new[] { parameter }))
            {
                var added = engine.TensorAdd(input, parameter);
                retained = detach ? TensorScope.Detach(added) : tensorScope.Keep(added);
                var doubled = engine.TensorAdd(retained, retained);
                var quadrupled = engine.TensorAdd(doubled, doubled);
                engine.ReduceSum(engine.TensorAdd(quadrupled, quadrupled), null);
                plan = scope.CompileTraining(new[] { parameter });
            }
            using (plan)
            {
                for (int step = 0; step < 6; step++)
                {
                    input.AsWritableSpan().Fill(step + 1);
                    input.IncrementVersion();
                    float expected = step + 1.25f;
                    Assert.Equal(8f * length * expected, plan.Step().ToArray()[0]);
                    var values = retained.ToArray();
                    Assert.Equal(length, values.Length);
                    foreach (float value in values) Assert.Equal(expected, value);
                    foreach (float gradient in plan.Gradients[0].ToArray()) Assert.Equal(8f, gradient);
                }
            }
        }
    }

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

    // A host plan pools forward-only outputs by forward liveness (PoolForwardOnlyOutputs): an add read only by adds
    // shares its array with a later one. The undeclared plan proves this graph pools; the declared output must keep
    // its own array, or it holds a later add's values once the step returns.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Declared_output_is_not_pooled_with_later_forward_only_outputs(bool declare)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        const int length = 1 << 16;
        var input = new Tensor<float>([length]);
        var parameter = new Tensor<float>([length]);
        parameter.AsWritableSpan().Fill(0.25f);
        Tensor<float> first;
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { parameter }))
        {
            first = engine.TensorAdd(input, parameter);
            if (declare) GradientRecording.RetainOutput(first);
            var second = engine.TensorAdd(first, input);
            var third = engine.TensorAdd(second, input);
            var fourth = engine.TensorAdd(third, input);
            engine.ReduceSum(fourth, null);
            plan = scope.CompileTraining(new[] { parameter });
        }
        using (plan)
        {
            if (!declare)
            {
                Assert.True(plan.PooledForwardOutputCount > 0, "this graph no longer pools: the declared case proves nothing");
                return;
            }
            for (int step = 1; step <= 3; step++)
            {
                input.AsWritableSpan().Fill(step);
                input.IncrementVersion();
                Assert.Equal(length * (4f * step + 0.25f), plan.Step().ToArray()[0], length * 1e-3f);
                foreach (float value in first.ToArray()) Assert.Equal(step + 0.25f, value);
            }
        }
    }

    // MatMul -> broadcast add of a 1-D bias -> ReLU is what the linear fusion rewrites into one FusedLinear node, which
    // never writes the matmul's own output. The undeclared plan proves this graph fuses; declaring the matmul result
    // (or the pre-activation sum) must keep that node, with the same loss and gradients.
    /// <summary>
    /// A declared matmul output keeps the linear fusion away, so the bias + ReLU after it fuses instead: the declared
    /// product must still hold each step's values and the gradients must match the unfused chain's.
    /// </summary>
    [Fact]
    public void Declared_matmul_ahead_of_a_fused_bias_relu_is_still_computed()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var input = new Tensor<float>([4, 4]);
        var weight = new Tensor<float>([4, 4]);
        var bias = new Tensor<float>([4]);
        weight.AsWritableSpan().Fill(0.25f);
        bias.AsWritableSpan().Fill(0.5f);
        Tensor<float> product;
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { weight, bias }))
        {
            product = engine.TensorMatMul(input, weight);
            GradientRecording.RetainOutput(product);
            var sum = engine.TensorBroadcastAdd(product, bias);
            engine.ReduceSum(engine.ReLU(sum), null);
            plan = scope.CompileTraining(new[] { weight, bias });
        }
        using (plan)
        {
            Assert.Contains(plan.BackwardStepKinds, kind => kind.IndexOf("FusedBiasActivation", StringComparison.Ordinal) >= 0);
            for (int step = 1; step <= 3; step++)
            {
                input.AsWritableSpan().Fill(step);
                input.IncrementVersion();
                Assert.Equal(16f * (step + 0.5f), plan.Step().ToArray()[0]);
                foreach (float value in product.ToArray()) Assert.Equal((float)step, value);
                foreach (float gradient in plan.Gradients[0].ToArray()) Assert.Equal(4f * step, gradient);
                foreach (float gradient in plan.Gradients[1].ToArray()) Assert.Equal(4f, gradient);
            }
        }
    }

    [Theory]
    [InlineData(0)]   // nothing declared: the control
    [InlineData(1)]   // the matmul result
    [InlineData(2)]   // the pre-activation sum
    public void Declared_interior_of_a_fusable_linear_chain_is_still_computed(int declared)
    {
        // The linear chain's own fusion is under test: the bias+ReLU fusion would absorb the undeclared sum this reads.
        using var suspended0 = FusionPatternRegistry.Suspend("AddAndBiasActivation");
        Assert.NotNull(suspended0);
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var input = new Tensor<float>([4, 4]);
        var weight = new Tensor<float>([4, 4]);
        var bias = new Tensor<float>([4]);
        weight.AsWritableSpan().Fill(0.25f);
        bias.AsWritableSpan().Fill(0.5f);
        Tensor<float> product, sum;
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { weight, bias }))
        {
            product = engine.TensorMatMul(input, weight);
            if (declared == 1) GradientRecording.RetainOutput(product);
            sum = engine.TensorBroadcastAdd(product, bias);
            if (declared == 2) GradientRecording.RetainOutput(sum);
            engine.ReduceSum(engine.ReLU(sum), null);
            plan = scope.CompileTraining(new[] { weight, bias });
        }
        using (plan)
        {
            bool fused = plan.BackwardStepKinds.Any(kind => kind.IndexOf("FusedLinear", StringComparison.Ordinal) >= 0);
            if (declared == 0)
            {
                Assert.True(fused, "this graph no longer fuses (" + string.Join(", ", plan.BackwardStepKinds)
                    + "): the declared cases prove nothing");
                return;
            }
            for (int step = 1; step <= 3; step++)
            {
                input.AsWritableSpan().Fill(step);
                input.IncrementVersion();
                Assert.Equal(16f * (step + 0.5f), plan.Step().ToArray()[0]);
                foreach (float value in product.ToArray().Take(declared == 1 ? 16 : 0)) Assert.Equal((float)step, value);
                foreach (float value in sum.ToArray().Take(declared == 2 ? 16 : 0)) Assert.Equal(step + 0.5f, value);
                foreach (float gradient in plan.Gradients[0].ToArray()) Assert.Equal(4f * step, gradient);
                foreach (float gradient in plan.Gradients[1].ToArray()) Assert.Equal(4f, gradient);
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

    // The embedding gather a device step writes straight into its output (this file's original defect, on every device
    // backend, not only under CUDA graph replay): read on some steps and not others, the declared rows must be the rows
    // of that step's ids.
    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void Declared_embedding_rows_follow_each_steps_indices(bool gpu, bool floatIndices)
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
            const int vocab = 32, width = 8, batch = 16;
            var table = new Tensor<float>([vocab, width]);
            var tableSpan = table.AsWritableSpan();
            for (int i = 0; i < tableSpan.Length; i++) tableSpan[i] = i * 0.125f;
            var hostTable = table.ToArray();
            var weight = new Tensor<float>([width, 4]);
            weight.AsWritableSpan().Fill(0.25f);
            var ids = new Tensor<int>([batch]);
            var floatIds = new Tensor<float>([batch]);
            var parameters = new[] { table, weight };
            if (gpu) foreach (var p in parameters) p.Gpu();
            Tensor<float> rows;
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                rows = floatIndices
                    ? engine.TensorEmbeddingLookupFromFloatIndices(table, floatIds)
                    : engine.TensorEmbeddingLookup<float, int>(table, ids);
                GradientRecording.RetainOutput(rows);
                var hidden = engine.TensorMatMul(rows, weight);
                engine.ReduceSum(engine.TensorMultiply(hidden, hidden), null);
                plan = scope.CompileTraining(parameters);
            }
            using (plan)
            {
                var rng = new Random(9);
                for (int step = 0; step < 8; step++)
                {
                    var picked = new int[batch];
                    for (int i = 0; i < batch; i++) picked[i] = rng.Next(vocab);
                    picked.AsSpan().CopyTo(ids.AsWritableSpan());
                    var floats = floatIds.AsWritableSpan();
                    for (int i = 0; i < batch; i++) floats[i] = picked[i];
                    ids.IncrementVersion();
                    floatIds.IncrementVersion();
                    plan.Step();
                    if (step is 1 or 2 or 5) continue;   // unread steps: the next read must still be its own step's rows
                    var actual = rows.ToArray();
                    Assert.Equal(batch * width, actual.Length);
                    for (int i = 0; i < batch; i++)
                        for (int c = 0; c < width; c++)
                            Assert.Equal(hostTable[picked[i] * width + c], actual[i * width + c]);
                }
            }
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
    public void Observation_only_add_does_not_merge_distinct_loss_gradient_buffers()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<float>([8]);
        ICompiledTrainingPlan<float> plan;
        Tensor<float> observed;
        using (var scope = GraphMode.EnableTraining(new[] { parameter }))
        {
            var left = engine.TensorMultiplyScalar(parameter, 2f);
            var right = engine.TensorMultiplyScalar(parameter, 3f);
            var loss = engine.ReduceSum(engine.TensorMultiply(left, right), null);
            observed = engine.TensorAdd(left, right);
            GradientRecording.RetainOutput(observed);
            plan = scope.CompileTraining(new[] { parameter }, loss);
        }
        using (plan)
            for (int step = 1; step <= 3; step++)
            {
                parameter.AsWritableSpan().Fill(step);
                parameter.IncrementVersion();
                Assert.Equal(48f * step * step, plan.Step().ToArray()[0]);
                foreach (var value in observed.ToArray()) Assert.Equal(5f * step, value);
                foreach (var value in plan.Gradients[0].ToArray()) Assert.Equal(12f * step, value);
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
