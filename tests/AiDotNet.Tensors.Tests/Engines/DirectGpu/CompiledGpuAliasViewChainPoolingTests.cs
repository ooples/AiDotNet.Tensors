using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Reshape chains and multi-consumer residual adds on a device plan: the gradients the alias views point at are pooled
/// with the other gradients (<see cref="CompiledTrainingPlan{T}.PoolAliasTargetGradients"/>), with every read through a
/// view counted as a use of the buffer it views. Over uncaptured steps, the step-graph capture and its replays, the
/// gradients must be bit for bit those of the same plan with no gradient pooling, and match the CPU eager tape.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledGpuAliasViewChainPoolingTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    // Reshape chains and residual adds: the gradients of their inputs view their outputs' gradient buffers (alias views).
    // On a device plan those viewed buffers are pooled too, with every read through a view counted as a use of the buffer.
    private static Tensor<float> AliasForward(IEngine engine, Tensor<float> input, Tensor<float>[] p)
    {
        var h = engine.Tanh(engine.TensorMatMul(input, p[0]));                                   // [4, 8]
        var folded = engine.Tanh(engine.Reshape(h, new[] { 2, 16 }));                             // reshape alias
        var back = engine.Reshape(engine.Reshape(folded, new[] { 32 }), new[] { 4, 8 });          // chain of reshapes
        var residual = engine.TensorAdd(back, engine.Tanh(engine.TensorMatMul(back, p[1])));     // multi-consumer add
        var residual2 = engine.TensorAdd(residual, engine.Tanh(engine.TensorMatMul(residual, p[2])));
        var z = engine.Tanh(engine.TensorMatMul(engine.Reshape(engine.Reshape(residual2, new[] { 32 }), new[] { 4, 8 }), p[3]));
        return engine.ReduceSum(engine.TensorMultiply(z, z), null);
    }

    private static float[][] RunCompiledOnDevice(DirectGpuTensorEngine engine, Tensor<float> input, bool pooling,
        bool poolAliasTargets, out int pooledCount)
    {
        string? previous = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_VIEWS");
        bool previousAlias = CompiledTrainingPlan<float>.PoolAliasTargetGradients;
        Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_VIEWS", pooling ? null : "0");
        CompiledTrainingPlan<float>.PoolAliasTargetGradients = poolAliasTargets;
        try
        {
            var ps = new[] { Filled(new[] { 8, 8 }, 31, 0.8f), Filled(new[] { 8, 8 }, 32, 0.8f), Filled(new[] { 8, 8 }, 33, 0.8f), Filled(new[] { 8, 8 }, 34, 0.8f) };
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(ps))
            {
                AliasForward(engine, input, ps);
                plan = scope.CompileTraining(ps);
            }
            using (plan)
            {
                var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
                pooledCount = concrete.PooledGradientCount;
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                // Enough steps for the uncaptured runs, the step-graph capture and its replays.
                const int steps = 6;
                var result = new float[ps.Length * steps][];
                for (int step = 0; step < steps; step++)
                {
                    plan.Step();
                    for (int i = 0; i < ps.Length; i++)
                    {
                        var grad = ps[i].Grad;
                        Assert.True(grad is not null, $"parameter {i} step {step}: no gradient");
                        result[step * ps.Length + i] = grad is null ? Array.Empty<float>() : grad.ToArray();
                    }
                }
                return result;
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_VIEWS", previous);
            CompiledTrainingPlan<float>.PoolAliasTargetGradients = previousAlias;
        }
    }

    [SkippableFact]
    public void DevicePlan_PoolsTheBuffersAliasViewsRead_AndKeepsGradientsBitForBit()
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
            var input = Filled(new[] { 4, 8 }, 3, 2f);
            var pooled = RunCompiledOnDevice(gpu!, input, pooling: true, poolAliasTargets: true, out int pooledCount);
            RunCompiledOnDevice(gpu!, input, pooling: true, poolAliasTargets: false, out int withoutAliasCount);
            var unpooled = RunCompiledOnDevice(gpu!, input, pooling: false, poolAliasTargets: true, out int unpooledCount);
            Assert.True(pooledCount > withoutAliasCount,
                $"pooling alias targets pooled {pooledCount} gradients, leaving them out {withoutAliasCount}: no alias target was pooled, so this test proves nothing");
            Assert.Equal(0, unpooledCount);
            for (int k = 0; k < pooled.Length; k++)
                Assert.Equal(unpooled[k], pooled[k]);

            var cpu = new CpuEngine();
            var eagerParams = new[] { Filled(new[] { 8, 8 }, 31, 0.8f), Filled(new[] { 8, 8 }, 32, 0.8f), Filled(new[] { 8, 8 }, 33, 0.8f), Filled(new[] { 8, 8 }, 34, 0.8f) };
            using var tape = new GradientTape<float>();
            var grads = tape.ComputeGradients(AliasForward(cpu, input, eagerParams), sources: eagerParams);
            for (int i = 0; i < eagerParams.Length; i++)
            {
                var expected = grads[eagerParams[i]].ToArray();
                for (int j = 0; j < expected.Length; j++)
                    Assert.True(Math.Abs(expected[j] - pooled[i][j]) <= 1e-3f * (1f + Math.Abs(expected[j])),
                        $"parameter {i}[{j}]: CPU eager {expected[j]}, device pooled {pooled[i][j]}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
            gpu?.Dispose();
        }
    }
}
