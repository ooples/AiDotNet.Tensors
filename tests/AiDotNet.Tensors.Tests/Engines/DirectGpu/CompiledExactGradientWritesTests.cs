using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CompiledExactGradientWritesTests
{
    [SkippableFact]
    public void A_custom_backward_named_Conv2D_keeps_its_gradient_storage()
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable, "Actual native GPU required.");
        var previousEngine = AiDotNetEngine.Current;
        string? previousFlag = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_EXACT_GRAD_WRITES");
        AiDotNetEngine.Current = engine;
        Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_EXACT_GRAD_WRITES", "1");
        try
        {
            using var parameter = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
            using var foreign = new Tensor<float>(new[] { 5f, 6f, 7f, 8f }, new[] { 4 });
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { parameter }))
            {
                var output = scope.RecordBinary(LazyNodeType.Custom, "Conv2D", parameter, foreign, new[] { 4 },
                    (e, dest) => { e.TensorCopy(parameter, dest); e.TensorAddInPlace(dest, foreign); },
                    static (grad, inputs, output, state, e, grads) =>
                    {
                        foreach (var input in inputs)
                            DifferentiableOps.AccumulateGrad(grads, input, e.TensorMultiplyScalar(grad, 1f), e);
                    });
                plan = scope.CompileTraining(new[] { parameter }, engine.ReduceSum(output, null, false));
            }
            using (plan)
            {
                // A custom backward may write every declared input even if it is not a requested source.
                // Its accumulator must stay in physical residency/zero scheduling, independently of its op name.
                var map = (Dictionary<Tensor<float>, Tensor<float>>)typeof(CompiledTrainingPlan<float>)
                    .GetField("_liveGradientMap", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(plan)!;
                var physical = (Tensor<float>[])typeof(CompiledTrainingPlan<float>)
                    .GetField("_preAllocatedGrads", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(plan)!;
                Assert.Contains(physical, gradient => ReferenceEquals(gradient, map[foreign]));
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_EXACT_GRAD_WRITES", previousFlag);
            AiDotNetEngine.Current = previousEngine;
        }
    }

    private sealed record Result(float[] Values, int Buffers, int Pruned, int BackwardSteps, string Metadata);

    private static Result Run(IEngine engine, bool exactWrites, bool requestInputs, bool captured, bool frozenKernel)
    {
        string? previousWrites = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_EXACT_GRAD_WRITES");
        string? previousInputs = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_INPUT_GRADS");
        var previousEngine = AiDotNetEngine.Current;
        Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_EXACT_GRAD_WRITES", exactWrites ? "1" : "0");
        Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_INPUT_GRADS", requestInputs ? "1" : "0");
        AiDotNetEngine.Current = engine;
        try
        {
            using var input = new Tensor<float>(Enumerable.Range(0, 64).Select(i => (i - 32) * .01f).ToArray(), new[] { 1, 4, 4, 4 });
            using var kernel = new Tensor<float>(Enumerable.Range(0, 144).Select(i => (i % 11 - 5) * .02f).ToArray(), new[] { 4, 36 });
            using var bias = new Tensor<float>(new[] { .1f, -.2f, .3f, -.4f }, new[] { 4 });
            var parameters = frozenKernel ? new[] { input } : new[] { kernel };
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var kernelView = engine.Reshape(kernel, new[] { 4, 4, 3, 3 });
                var conv = engine.Conv2D(input, kernelView, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
                var output = engine.TensorChannelBiasAdd(conv, bias);
                var loss = engine.ReduceSum(engine.TensorMultiply(output, output), null, false);
                plan = scope.CompileTraining(parameters, loss);
            }
            using (plan)
            {
                if (!captured) plan.DisableGraphStep();
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: .001f);
                var values = new List<float>();
                for (int step = 0; step < 8; step++)
                {
                    values.AddRange(plan.Step().ToArray());
                    values.AddRange(plan.Gradients[0].ToArray());
                    if (requestInputs)
                        foreach (var leaf in new[] { input, kernel, bias })
                        {
                            Assert.NotNull(leaf.Grad);
                            values.AddRange(leaf.Grad!.ToArray());
                        }
                }
                foreach (var parameter in parameters) values.AddRange(parameter.ToArray());
                if (captured)
                    Assert.NotEqual(IntPtr.Zero, (IntPtr)typeof(CompiledTrainingPlan<float>)
                        .GetField("_stepGraphExec", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                        .GetValue(plan)!);
                var steps = (CompiledStep<float>[])typeof(CompiledTrainingPlan<float>)
                    .GetField("_forwardSteps", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(plan)!;
                string metadata = engine.Unwrap().GetType().Name + "; " + string.Join("; ", steps.Select(s =>
                    $"{s.OpName}:{s.BackwardFn?.Method.DeclaringType?.Name}.{s.BackwardFn?.Method.Name}:target={s.BackwardFn?.Target?.GetType().Name ?? "null"}"));
                return new(values.ToArray(), plan.PreAllocatedGradBufferCount, plan.PrunedGradientBufferCount, plan.BackwardStepCount, metadata);
            }
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_EXACT_GRAD_WRITES", previousWrites);
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_INPUT_GRADS", previousInputs);
        }
    }

    [SkippableTheory]
    [InlineData(false, false, false)]
    [InlineData(false, false, true)]
    [InlineData(false, true, false)]
    [InlineData(false, true, true)]
    [InlineData(true, false, false)]
    [InlineData(true, false, true)]
    [InlineData(true, true, false)]
    [InlineData(true, true, true)]
    public void Exact_builtin_writes_preserve_training_and_requested_gradients(bool captured, bool requestInputs, bool frozenKernel)
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable, "Actual native GPU required.");
        Skip.If(captured && engine.GetBackend() is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
            "Actual CUDA graph capture required.");
        var baseline = Run(engine, false, requestInputs, captured, frozenKernel);
        var candidate = Run(engine, true, requestInputs, captured, frozenKernel);
        Assert.Equal(baseline.Values, candidate.Values);
        Assert.Equal(baseline.BackwardSteps, candidate.BackwardSteps);
        if (requestInputs)
        {
            Assert.Equal(baseline.Buffers, candidate.Buffers);
            Assert.Equal(baseline.Pruned, candidate.Pruned);
        }
        else
        {
            Assert.True(candidate.Buffers < baseline.Buffers, $"buffers: {candidate.Buffers} vs {baseline.Buffers}; {candidate.Metadata}");
            Assert.True(candidate.Pruned > baseline.Pruned, $"pruned: {candidate.Pruned} vs {baseline.Pruned}");
        }
    }
}
