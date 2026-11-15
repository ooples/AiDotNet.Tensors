using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The GPU trace records LayerNorm from shapes alone (no eager pass, so the forward upstream of it stays lazy and is not
/// copied to the host at trace time). The compiled plan's gradients must still match the CPU eager tape on every step:
/// uncaptured steps, the step-graph capture and its replays. LayerNorm's mean and variance reach the backward only
/// through the forward that realizes them.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledGpuLayerNormTraceTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private static Tensor<float>[] MakeParams()
    {
        var gamma = Filled(new[] { 16 }, 43, 0.5f);
        for (int i = 0; i < gamma.Length; i++) gamma[i] += 1f;
        return new[] { Filled(new[] { 8, 16 }, 41, 0.8f), Filled(new[] { 16, 8 }, 42, 0.8f), gamma, Filled(new[] { 16 }, 44, 0.5f) };
    }

    // Two LayerNorms over a rank-3 activation (a transformer's [batch, tokens, features]), one fed by a lazy matmul chain.
    private static Tensor<float> Forward(IEngine engine, Tensor<float> input, Tensor<float>[] p)
    {
        var h = engine.Tanh(engine.TensorMatMul(input, p[0]));                                   // [12, 16]
        var tokens = engine.Reshape(h, new[] { 3, 4, 16 });
        var normed = engine.LayerNorm(tokens, p[2], p[3], 1e-5, out _, out _);
        var again = engine.LayerNorm(engine.Tanh(normed), p[2], p[3], 1e-5, out _, out _);
        var z = engine.Tanh(engine.TensorMatMul(engine.Reshape(again, new[] { 12, 16 }), p[1]));
        return engine.ReduceSum(engine.TensorMultiply(z, z), null);
    }

    [SkippableFact]
    public void GpuPlan_TracedWithoutAnEagerLayerNorm_MatchesTheCpuTapeEveryStep()
    {
        var prior = AiDotNetEngine.Current;
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        bool resolved = gpu is not null && gpu.IsGpuAvailable;
        if (!resolved && string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException("GPU tests required (AIDOTNET_REQUIRE_GPU_TESTS=1) but no GPU backend resolved.");
        Skip.IfNot(resolved, "GPU backend did not resolve.");
        try
        {
            AiDotNetEngine.Current = gpu!;
            var input = Filled(new[] { 12, 8 }, 5, 2f);

            var cpu = new CpuEngine();
            var eagerParams = MakeParams();
            using var tape = new GradientTape<float>();
            var grads = tape.ComputeGradients(Forward(cpu, input, eagerParams), sources: eagerParams);

            var ps = MakeParams();
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(ps))
            {
                Forward(gpu!, input, ps);
                plan = scope.CompileTraining(ps);
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                for (int step = 0; step < 6; step++)
                {
                    plan.Step();
                    for (int i = 0; i < ps.Length; i++)
                    {
                        var expected = grads[eagerParams[i]].ToArray();
                        var actual = ps[i].Grad?.ToArray();
                        Assert.True(actual is not null, $"step {step} parameter {i}: no gradient");
                        for (int j = 0; j < expected.Length; j++)
                            Assert.True(Math.Abs(expected[j] - actual![j]) <= 1e-3f * (1f + Math.Abs(expected[j])),
                                $"step {step} parameter {i}[{j}]: CPU tape {expected[j]}, GPU plan {actual[j]}");
                    }
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
            gpu?.Dispose();
        }
    }
}
