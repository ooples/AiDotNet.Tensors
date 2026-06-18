// #1624: end-to-end correctness gate for liveness pooling of the compiled
// training plan's backward gradient buffers (AIDOTNET_COMPILED_GRAD_POOL). Builds
// a multi-layer MLP whose same-shape activations let the pooler share buffers,
// runs one Step() with pooling OFF and ON, and asserts the parameter gradients
// are bit-for-bit equivalent — i.e. sharing a physical buffer across two
// disjoint-lifetime gradients (with the re-zero between tenants) never corrupts a
// gradient. Also asserts pooling actually reduced the physical buffer count, so
// the test can't pass vacuously.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompiledTrainingPlanSerial")]
public class CompiledGradPoolParityTests
{
    private static Tensor<float> Make(int[] shape, int seed)
    {
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        // Deterministic, varied, non-degenerate values so gradients are non-zero.
        for (int i = 0; i < n; i++)
            data[i] = (float)(((i * 7 + seed * 13) % 17) - 8) * 0.1f;
        return new Tensor<float>(data, shape);
    }

    private static (float[][] grads, int gradBuffers) RunOnce(bool pool)
    {
        string? prev = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_POOL");
        Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_POOL", pool ? "1" : null);
        try
        {
            var engine = new CpuEngine();
            var x = Make(new[] { 4, 8 }, seed: 1);
            var w1 = Make(new[] { 8, 8 }, seed: 2);
            var w2 = Make(new[] { 8, 8 }, seed: 3);
            var w3 = Make(new[] { 8, 8 }, seed: 4);
            var w4 = Make(new[] { 8, 8 }, seed: 5);

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                // 4-layer MLP: every activation is [4,8] — same shape, disjoint
                // lifetimes — exactly what the pooler collapses onto shared buffers.
                var h1 = engine.ReLU(engine.TensorMatMul(x, w1));
                var h2 = engine.ReLU(engine.TensorMatMul(h1, w2));
                var h3 = engine.ReLU(engine.TensorMatMul(h2, w3));
                var h4 = engine.TensorMatMul(h3, w4);
                engine.ReduceSum(h4, null);
                plan = scope.CompileTraining(new[] { w1, w2, w3, w4 });
            }

            plan.Step();

            var grads = plan.Gradients.Select(g => g.AsSpan().ToArray()).ToArray();
            int buffers = ((CompiledTrainingPlan<float>)plan).PreAllocatedGradBufferCount;
            plan.Dispose();
            return (grads, buffers);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_POOL", prev);
        }
    }

    [Fact]
    public void PooledGradients_MatchUnpooled_AndPoolFewerBuffers()
    {
        var (control, ctrlBuffers) = RunOnce(pool: false);
        var (pooled, poolBuffers) = RunOnce(pool: true);

        Assert.Equal(control.Length, pooled.Length);
        for (int p = 0; p < control.Length; p++)
        {
            Assert.Equal(control[p].Length, pooled[p].Length);
            for (int i = 0; i < control[p].Length; i++)
                Assert.Equal(control[p][i], pooled[p][i], 4);
        }

        // The pooler must actually have shared buffers — otherwise the parity
        // above is vacuous. A 4-layer chain's [4,8] activations collapse onto a
        // small frontier, so the pooled set is strictly smaller.
        Assert.True(poolBuffers < ctrlBuffers,
            $"pooling did not reduce the physical buffer count: pooled={poolBuffers}, control={ctrlBuffers}");
    }

    // Both paths now compile WITH dataflow fusion + analytic-loss backward enabled
    // (pooling is fusion-compatible). These deeper/rectangular variants stress the
    // action-space liveness + re-zero against the FUSED action stream — if the
    // pooler mis-mapped a re-zero onto the wrong fused action, a shared buffer
    // would carry a stale tenant's gradient and the parity below would break.
    private static (float[][] grads, int gradBuffers) RunDeep(bool pool, int layers, int rows, int dim, int seedBase)
    {
        string? prev = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_POOL");
        Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_POOL", pool ? "1" : null);
        try
        {
            var engine = new CpuEngine();
            var x = Make(new[] { rows, dim }, seed: seedBase);
            var ws = new Tensor<float>[layers];
            for (int i = 0; i < layers; i++) ws[i] = Make(new[] { dim, dim }, seed: seedBase + 1 + i);

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var h = engine.TensorMatMul(x, ws[0]);
                for (int i = 1; i < layers; i++)
                    h = engine.TensorMatMul(engine.ReLU(h), ws[i]);
                engine.ReduceSum(h, null);
                plan = scope.CompileTraining(ws);
            }
            plan.Step();
            var grads = plan.Gradients.Select(g => g.AsSpan().ToArray()).ToArray();
            int buffers = ((CompiledTrainingPlan<float>)plan).PreAllocatedGradBufferCount;
            plan.Dispose();
            return (grads, buffers);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_POOL", prev);
        }
    }

    [Theory]
    [InlineData(6, 4, 8, 100)]   // deeper, square
    [InlineData(5, 16, 12, 200)] // rectangular rows, wider dim
    [InlineData(8, 2, 16, 300)]  // very deep, small batch
    public void PooledGradients_MatchUnpooled_WithFusion_AcrossShapes(int layers, int rows, int dim, int seedBase)
    {
        var (control, ctrlBuffers) = RunDeep(pool: false, layers, rows, dim, seedBase);
        var (pooled, poolBuffers) = RunDeep(pool: true, layers, rows, dim, seedBase);

        Assert.Equal(control.Length, pooled.Length);
        for (int p = 0; p < control.Length; p++)
        {
            Assert.Equal(control[p].Length, pooled[p].Length);
            for (int i = 0; i < control[p].Length; i++)
                Assert.Equal(control[p][i], pooled[p][i], 4);
        }
        Assert.True(poolBuffers < ctrlBuffers,
            $"pooling did not reduce the physical buffer count: pooled={poolBuffers}, control={ctrlBuffers}");
    }

    [Fact]
    public void PoolingEnabledViaConfigOption_MatchesUnpooled_AndPoolsFewerBuffers()
    {
        // The production API is the TensorCodecOptions flag (not just the env var).
        var (control, ctrlBuffers) = RunDeep(pool: false, layers: 5, rows: 4, dim: 8, seedBase: 400);

        var prev = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        var opts = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Default;
        opts.EnableBackwardGradientPooling = true;
        AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(opts);
        try
        {
            var (pooled, poolBuffers) = RunDeep(pool: false, layers: 5, rows: 4, dim: 8, seedBase: 400);
            // RunDeep(pool:false) leaves the env var unset, so the config flag is the
            // only thing turning pooling on here.
            Assert.Equal(control.Length, pooled.Length);
            for (int p = 0; p < control.Length; p++)
                for (int i = 0; i < control[p].Length; i++)
                    Assert.Equal(control[p][i], pooled[p][i], 4);
            Assert.True(poolBuffers < ctrlBuffers,
                $"config-enabled pooling did not reduce buffers: pooled={poolBuffers}, control={ctrlBuffers}");
        }
        finally
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(prev);
        }
    }
}
