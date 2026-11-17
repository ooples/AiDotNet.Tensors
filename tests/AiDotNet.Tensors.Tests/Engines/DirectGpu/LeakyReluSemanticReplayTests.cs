using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class LeakyReluSemanticReplayTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public LeakyReluSemanticReplayTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static void Verify<T>(IEngine engine, double slope, bool compiled)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var input = new Tensor<T>(new[] { 2, 2, 1, 8 });
            var bias = new Tensor<T>(new[] { 2 });
            var seed = new Tensor<T>(input.Shape.ToArray());
            for (int i = 0; i < seed.Length; i++) seed[i] = ops.FromDouble((i % 9 - 4) * .125);
            T alpha = ops.FromDouble(slope);
            Tensor<T>? output = null;
            Tensor<T> Objective()
            {
                output = engine.LeakyReLU(engine.TensorChannelBiasAdd(input, bias), alpha);
                return engine.ReduceSum(engine.TensorMultiply(output, seed));
            }
            void Feed(int step)
            {
                for (int i = 0; i < input.Length; i++) input[i] = ops.FromDouble((i % 11 - 5) * .125 + step * .0625);
                bias[0] = ops.FromDouble(-.25 + step * .0625);
                bias[1] = ops.FromDouble(.375 - step * .03125);
                if (engine is DirectGpuTensorEngine gpu && typeof(T) == typeof(float))
                {
                    gpu.EnsureResidentInput(input);
                    gpu.EnsureResidentInput(bias);
                    gpu.EnsureResidentInput(seed);
                }
            }
            Feed(0);
            ICompiledTrainingPlan<T>? plan = null;
            if (compiled)
            {
                using var scope = GraphMode.Enable();
                var loss = Objective();
                Assert.NotNull(output!.LazySource); // DirectGPU must record the semantic node rather than execute it here.
                scope.RetainOutput(output); // This test observes the activation after backward; its workspace must stay live.
                plan = scope.CompileTraining(new[] { input, bias }, loss);
            }
            using (plan)
            {
                for (int step = 0; step < 3; step++)
                {
                    Feed(step);
                    using var tape = plan is null ? new GradientTape<T>() : null;
                    Tensor<T> gradInput, gradBias;
                    if (plan is not null)
                    {
                        plan.Step();
                        gradInput = plan.Gradients[0];
                        gradBias = plan.Gradients[1];
                    }
                    else
                    {
                        var grads = tape!.ComputeGradients(Objective(), new[] { input, bias });
                        gradInput = grads[input];
                        gradBias = grads[bias];
                    }
                    var actual = output!.ToArray();
                    var dx = gradInput.ToArray();
                    var db = gradBias.ToArray();
                    var expectedBias = new double[2];
                    double tolerance = typeof(T) == typeof(double) ? 1e-10 : 2e-5;
                    double actualSlope = ops.ToDouble(alpha);
                    for (int i = 0; i < input.Length; i++)
                    {
                        int channel = i / 8 % 2;
                        double x = ops.ToDouble(input[i]) + ops.ToDouble(bias[channel]);
                        double expected = x > 0 ? x : actualSlope * x;
                        double derivative = ops.ToDouble(seed[i]) * (x > 0 ? 1 : actualSlope);
                        expectedBias[channel] += derivative;
                        Assert.True(Math.Abs(ops.ToDouble(actual[i]) - expected) <= tolerance,
                            $"step={step} index={i} x={x} alpha={actualSlope}: output={ops.ToDouble(actual[i])}, expected={expected}");
                        Assert.True(Math.Abs(ops.ToDouble(dx[i]) - derivative) <= tolerance,
                            $"step={step} index={i} x={x} alpha={actualSlope}: dx={ops.ToDouble(dx[i])}, expected={derivative}");
                    }
                    for (int c = 0; c < 2; c++) Assert.InRange(Math.Abs(ops.ToDouble(db[c]) - expectedBias[c]), 0, tolerance);
                }
            }
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [Theory]
    [InlineData(.1, false)] [InlineData(.01, false)]
    [InlineData(.1, true)] [InlineData(.01, true)]
    public void Float_cpu_values_and_gradients_match_scalar_reference(double alpha, bool compiled)
        => Verify<float>(new CpuEngine(), alpha, compiled);

    [Theory]
    [InlineData(.1, false)] [InlineData(.01, false)]
    [InlineData(.1, true)] [InlineData(.01, true)]
    public void Double_cpu_values_and_gradients_match_scalar_reference(double alpha, bool compiled)
        => Verify<double>(new CpuEngine(), alpha, compiled);

    [SkippableTheory]
    [InlineData(.1, false)] [InlineData(.01, false)]
    [InlineData(.1, true)] [InlineData(.01, true)]
    public void Gpu_values_and_gradients_match_scalar_reference_through_changed_input_replays(double alpha, bool compiled)
    {
        Skip.IfNot(_fixture.IsAvailable, "No direct GPU backend available.");
        Verify<float>(_fixture.Engine ?? throw new InvalidOperationException("GPU fixture has no engine."), alpha, compiled);
    }

    [SkippableTheory]
    [InlineData(.1f)] [InlineData(.01f)]
    public void Resident_native_activation_keeps_host_storage_deferred_and_reuses_device_output(float alpha)
    {
        Skip.IfNot(_fixture.IsAvailable, "No direct GPU backend available.");
        var gpu = _fixture.Engine ?? throw new InvalidOperationException("GPU fixture has no engine.");
        const int length = 1 << 18;
        var input = new Tensor<float>(new[] { length });
        input.AsWritableSpan().Fill(-2f);
        gpu.EnsureResidentInput(input);
        var pool = gpu.CreateScratchPool();
        var previous = DirectGpuTensorEngine.EnterScratchPool(pool);
        gpu.SuspendActivationEviction();
        try
        {
            using (gpu.EnterCompiledCapturePath())
            {
                gpu.SetCurrentScratchAction(0);
                using var warm = gpu.LeakyReLU(input, alpha);
                var buffer = warm._gpuBuffer;
                long before = GC.GetAllocatedBytesForCurrentThread();
                Tensor<float>? last = null;
                for (int step = 0; step < 4; step++)
                {
                    last?.Dispose();
                    gpu.SetCurrentScratchAction(0);
                    last = gpu.LeakyReLU(input, alpha);
                    Assert.True(last.HasDeferredHostStorage);
                    Assert.False(last._ownsGpuBuffer);
                    Assert.Same(buffer, last._gpuBuffer);
                }
                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                Assert.True(allocated < length * sizeof(float), $"metadata allocated {allocated:N0} bytes");
                Assert.All(last!.ToArray(), value => Assert.Equal(-2f * alpha, value));
                last.Dispose();
            }
        }
        finally
        {
            DirectGpuTensorEngine.RestoreScratchPool(previous);
            pool.Release();
            gpu.ResumeActivationEviction();
        }
    }
}
