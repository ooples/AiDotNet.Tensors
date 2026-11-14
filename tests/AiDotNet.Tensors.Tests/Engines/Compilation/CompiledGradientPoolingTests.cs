using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The CPU compiled plan gives gradients whose lifetimes in the backward do not overlap shared arrays (gradient pooling),
/// zeroing a reused array just before a tenant that accumulates. The graph here has a chain of intermediates (disjoint
/// lifetimes, so arrays are shared), a tensor read by two ops (a pooled tenant that accumulates) and a concatenation.
/// Every parameter gradient must equal, bit for bit and over several steps, the same plan compiled with pooling off
/// (AIDOTNET_COMPILED_GRAD_VIEWS=0), and match the eager tape.
/// </summary>
[Collection("CompilationGlobalState")]
public class CompiledGradientPoolingTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private static Tensor<float>[] MakeParams() => new[]
    {
        Filled(new[] { 4, 8 }, 11, 1f), Filled(new[] { 4, 8 }, 12, 1f), Filled(new[] { 4, 8 }, 13, 1f),
        Filled(new[] { 4, 8 }, 14, 1f), Filled(new[] { 4, 16 }, 15, 1f),
    };

    private static Tensor<float> Forward(IEngine engine, Tensor<float> input, Tensor<float>[] p)
    {
        var h = engine.Tanh(engine.TensorMultiply(input, p[0]));
        h = engine.Tanh(engine.TensorMultiply(h, p[1]));
        var shared = engine.Tanh(engine.TensorMultiply(h, p[2]));                 // read twice below
        var left = engine.Tanh(engine.TensorMultiply(shared, p[3]));
        var joined = engine.TensorConcatenate(new[] { left, shared }, 1);       // [4, 16]
        var z = engine.Tanh(engine.TensorMultiply(joined, p[4]));
        return engine.ReduceSum(engine.TensorMultiply(z, z), null);
    }

    private static float[][] RunCompiled(IEngine engine, Tensor<float> input, bool pooling, out int pooledCount)
    {
        string? previous = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_VIEWS");
        Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_VIEWS", pooling ? null : "0");
        try
        {
            var ps = MakeParams();
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward(engine, input, ps);
                plan = scope.CompileTraining(ps);
            }
            using (plan)
            {
                var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
                pooledCount = concrete.PooledGradientCount;
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                var result = new float[ps.Length * 3][];
                for (int step = 0; step < 3; step++)
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
        }
    }

    private static Tensor<float>[] MakeConvParams() => new[]
    {
        Filled(new[] { 4, 3, 3, 3 }, 21, 0.6f), Filled(new[] { 4, 4, 3, 3 }, 22, 0.6f), Filled(new[] { 4, 4, 3, 3 }, 23, 0.6f),
    };

    // t is read by two convolutions, whose backwards accumulate into its gradient without claiming the first write, so
    // a pooled t must be zeroed before them; the tanh chains after the convolutions die first and free arrays t reuses.
    private static Tensor<float> ConvForward(IEngine engine, Tensor<float> input, Tensor<float>[] p)
    {
        int[] one = { 1, 1 }, pad = { 1, 1 };
        var t = engine.Tanh(engine.Conv2D(input, p[0], one, pad, one));
        var a = engine.Tanh(engine.Tanh(engine.Tanh(engine.Conv2D(t, p[1], one, pad, one))));
        var b = engine.Tanh(engine.Tanh(engine.Tanh(engine.Conv2D(t, p[2], one, pad, one))));
        return engine.TensorAdd(engine.ReduceSum(engine.TensorMultiply(a, a), null), engine.ReduceSum(engine.TensorMultiply(b, b), null));
    }

    [Fact]
    public void PooledTenantReadByTwoConvolutions_IsZeroedBeforeTheyAccumulate()
    {
        var engine = new CpuEngine();
        var input = Filled(new[] { 2, 3, 6, 6 }, 2, 2f);
        float[][] Run(bool pooling, out int pooledCount)
        {
            string? previous = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_VIEWS");
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_VIEWS", pooling ? null : "0");
            try
            {
                var ps = MakeConvParams();
                ICompiledTrainingPlan<float> plan;
                using (var scope = GraphMode.Enable())
                {
                    ConvForward(engine, input, ps);
                    plan = scope.CompileTraining(ps);
                }
                using (plan)
                {
                    pooledCount = Assert.IsType<CompiledTrainingPlan<float>>(plan).PooledGradientCount;
                    plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                    var result = new float[ps.Length * 3][];
                    for (int step = 0; step < 3; step++)
                    {
                        plan.Step();
                        for (int i = 0; i < ps.Length; i++)
                            result[step * ps.Length + i] = ps[i].Grad?.ToArray() ?? Array.Empty<float>();
                    }
                    return result;
                }
            }
            finally
            {
                Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_VIEWS", previous);
            }
        }

        var pooled = Run(pooling: true, out int pooledCount);
        var unpooled = Run(pooling: false, out _);
        Assert.True(pooledCount > 0, "no gradient was pooled, so this test proves nothing");
        for (int k = 0; k < pooled.Length; k++)
        {
            Assert.NotEmpty(pooled[k]);
            Assert.Equal(unpooled[k], pooled[k]);
        }
    }
    [Fact]
    public void PooledGradients_EqualUnpooled_BitForBit_AndMatchEager()
    {
        var engine = new CpuEngine();
        var input = Filled(new[] { 4, 8 }, 1, 2f);

        var pooled = RunCompiled(engine, input, pooling: true, out int pooledCount);
        var unpooled = RunCompiled(engine, input, pooling: false, out int unpooledCount);
        Assert.True(pooledCount > 0, "no gradient was pooled, so this test proves nothing");
        Assert.Equal(0, unpooledCount);
        for (int k = 0; k < pooled.Length; k++)
            Assert.Equal(unpooled[k], pooled[k]);

        var eagerParams = MakeParams();
        using var tape = new GradientTape<float>();
        var grads = tape.ComputeGradients(Forward(engine, input, eagerParams), sources: eagerParams);
        for (int i = 0; i < eagerParams.Length; i++)
        {
            var expected = grads[eagerParams[i]].ToArray();
            for (int j = 0; j < expected.Length; j++)
                Assert.True(Math.Abs(expected[j] - pooled[i][j]) <= 1e-4f * (1f + Math.Abs(expected[j])),
                    $"parameter {i}[{j}]: eager {expected[j]}, pooled {pooled[i][j]}");
        }
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

    private static float[][] RunCompiledOnDevice(DirectGpuTensorEngine engine, Tensor<float> input, bool pooling, out int pooledCount,
        out int aliasViews)
    {
        string? previous = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_VIEWS");
        Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_VIEWS", pooling ? null : "0");
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
                aliasViews = concrete.PoolableAliasViewCount;
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
            var pooled = RunCompiledOnDevice(gpu!, input, pooling: true, out int pooledCount, out int aliasViews);
            var unpooled = RunCompiledOnDevice(gpu!, input, pooling: false, out int unpooledCount, out _);
            Assert.True(pooledCount > 0, "no gradient was pooled, so this test proves nothing");
            Assert.True(aliasViews > 0, "no alias view reads a pooled buffer, so this test proves nothing");
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
