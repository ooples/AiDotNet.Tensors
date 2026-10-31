using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The CPU compiled plan shares arrays between forward outputs no backward reads, by forward liveness. Here three
/// bias-only convolution chains' outputs are read only by a concatenation or an average pool: the first two are both
/// written before either is read (they must not share), the third after both are read (it may). Every parameter's gradient must equal the same plan with the
/// pooling off (AIDOTNET_COMPILED_FORWARD_POOL=0) bit for bit over several steps, and match the eager tape; the loss
/// must be the eager one.
/// </summary>
[Collection("CompilationGlobalState")]
public class CompiledForwardPoolingTests
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
        Filled(new[] { 8, 4, 3, 3 }, 11, 0.5f), Filled(new[] { 8 }, 12, 0.5f),
        Filled(new[] { 8, 4, 3, 3 }, 13, 0.5f), Filled(new[] { 8 }, 14, 0.5f),
        Filled(new[] { 8, 12, 3, 3 }, 15, 0.3f), Filled(new[] { 8 }, 16, 0.5f),
        Filled(new[] { 8, 4, 3, 3 }, 17, 0.5f), Filled(new[] { 8 }, 18, 0.5f),
    };

    private static Tensor<float> Forward(IEngine engine, Tensor<float> x, Tensor<float>[] p)
    {
        int[] one = { 1, 1 }, pad = { 1, 1 };
        // a and c are both written before either is read, so they must not share; d, written after both are read,
        // can share with either. A lifetime that ended at the write would let c overwrite a before the concat reads it.
        var a = engine.TensorChannelBiasAdd(engine.Conv2D(x, p[0], one, pad, one), p[1]);     // read only by the concat
        var c = engine.TensorChannelBiasAdd(engine.Conv2D(x, p[2], one, pad, one), p[3]);     // read only by a pool
        var h = engine.Tanh(engine.TensorConcatenate(new[] { x, a }, 1));
        var pc = engine.AvgPool2D(c, new[] { 2, 2 }, new[] { 2, 2 });
        // e ends in a ReLU, whose backward reads e (its mask): it must keep its own array though only a pool reads it.
        var e = engine.ReLU(engine.TensorChannelBiasAdd(engine.Conv2D(x, p[6], one, pad, one), p[7]));
        var pe = engine.AvgPool2D(e, new[] { 2, 2 }, new[] { 2, 2 });
        var d = engine.TensorChannelBiasAdd(engine.Conv2D(h, p[4], one, pad, one), p[5]);     // read only by a pool
        var pd = engine.AvgPool2D(d, new[] { 2, 2 }, new[] { 2, 2 });
        var sum = engine.TensorAdd(engine.ReduceSum(engine.TensorMultiply(pc, pc), null), engine.ReduceSum(engine.TensorMultiply(pd, pd), null));
        return engine.TensorAdd(sum, engine.ReduceSum(engine.TensorMultiply(pe, pe), null));
    }

    private static (float[][] Grads, float[] Losses, int Pooled) Run(IEngine engine, Tensor<float> x, bool pooling)
    {
        string? previous = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_FORWARD_POOL");
        Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_FORWARD_POOL", pooling ? null : "0");
        try
        {
            var ps = MakeParams();
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward(engine, x, ps);
                plan = scope.CompileTraining(ps);
            }
            using (plan)
            {
                int pooled = Assert.IsType<CompiledTrainingPlan<float>>(plan).PooledForwardOutputCount;
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                var grads = new float[ps.Length * 3][];
                var losses = new float[3];
                for (int step = 0; step < 3; step++)
                {
                    losses[step] = plan.Step()[0];
                    for (int i = 0; i < ps.Length; i++)
                    {
                        var grad = ps[i].Grad;
                        Assert.True(grad is not null, $"parameter {i} step {step}: no gradient");
                        grads[step * ps.Length + i] = grad is null ? Array.Empty<float>() : grad.ToArray();
                    }
                }
                return (grads, losses, pooled);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_FORWARD_POOL", previous);
        }
    }

    [Fact]
    public void PooledForwardOutputs_EqualUnpooled_BitForBit_AndMatchEager()
    {
        var engine = new CpuEngine();
        var x = Filled(new[] { 2, 4, 64, 64 }, 1, 2f);

        var pooled = Run(engine, x, pooling: true);
        var unpooled = Run(engine, x, pooling: false);
        Assert.True(pooled.Pooled >= 2, $"{pooled.Pooled} forward outputs pooled, so this test proves nothing");
        Assert.Equal(0, unpooled.Pooled);
        Assert.Equal(unpooled.Losses, pooled.Losses);
        for (int k = 0; k < pooled.Grads.Length; k++)
            Assert.Equal(unpooled.Grads[k], pooled.Grads[k]);

        var eagerParams = MakeParams();
        using var tape = new GradientTape<float>();
        var loss = Forward(engine, x, eagerParams);
        var grads = tape.ComputeGradients(loss, sources: eagerParams);
        float eagerLoss = loss.ToArray()[0];
        Assert.True(Math.Abs(eagerLoss - pooled.Losses[0]) <= 1e-4f * (1f + Math.Abs(eagerLoss)),
            $"loss: eager {eagerLoss}, pooled {pooled.Losses[0]}");
        for (int i = 0; i < eagerParams.Length; i++)
        {
            var expected = grads[eagerParams[i]].ToArray();
            Assert.Contains(expected, v => v != 0f);
            for (int j = 0; j < expected.Length; j++)
                Assert.True(Math.Abs(expected[j] - pooled.Grads[i][j]) <= 1e-3f * (1f + Math.Abs(expected[j])),
                    $"parameter {i}[{j}]: eager {expected[j]}, pooled {pooled.Grads[i][j]}");
        }
    }

    /// <summary>
    /// A ReLU-ending chain's output is read by its own backward (the ReLU mask), so it keeps its array even when only an
    /// average pool reads it; a later forward-only output must not reuse that array. With it the only other candidate,
    /// nothing is pooled and the gradients equal the unpooled plan's.
    /// </summary>
    [Fact]
    public void AReluChainOutput_IsNeverShared()
    {
        var engine = new CpuEngine();
        var x = Filled(new[] { 8, 4, 64, 64 }, 2, 2f);
        Tensor<float>[] Params() => new[]
        {
            Filled(new[] { 8, 4, 3, 3 }, 21, 0.5f), Filled(new[] { 8 }, 22, 0.5f),
            Filled(new[] { 8, 8, 3, 3 }, 23, 0.3f), Filled(new[] { 8 }, 24, 0.5f),
        };
        Tensor<float> Forward(Tensor<float>[] p)
        {
            int[] one = { 1, 1 }, pad = { 1, 1 };
            var e = engine.ReLU(engine.TensorChannelBiasAdd(engine.Conv2D(x, p[0], one, pad, one), p[1]));
            var pe = engine.AvgPool2D(e, new[] { 2, 2 }, new[] { 2, 2 });
            var le = engine.ReduceSum(engine.TensorMultiply(pe, pe), null);
            // d reads pe, so it is written after e's last forward read: e's array (the only one free) would be d's were
            // e forward-only, and d would overwrite the mask e's backward reads.
            var d = engine.TensorChannelBiasAdd(engine.Conv2D(pe, p[2], one, pad, one), p[3]);
            var pd = engine.AvgPool2D(d, new[] { 2, 2 }, new[] { 2, 2 });
            return engine.TensorAdd(le, engine.ReduceSum(engine.TensorMultiply(pd, pd), null));
        }
        float[][] Run(bool pooling, out int pooledCount)
        {
            string? previous = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_FORWARD_POOL");
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_FORWARD_POOL", pooling ? null : "0");
            try
            {
                var ps = Params();
                ICompiledTrainingPlan<float> plan;
                using (var scope = GraphMode.Enable())
                {
                    Forward(ps);
                    plan = scope.CompileTraining(ps);
                }
                using (plan)
                {
                    var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
                    // The ReLU chain must be fused for the chain rule to be what keeps e out of the pool.
                    Assert.Contains("fused:Conv2D+ChannelBias+ReLU", concrete.BackwardStepKinds);
                    pooledCount = concrete.PooledForwardOutputCount;
                    plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                    var grads = new float[ps.Length * 2][];
                    for (int step = 0; step < 2; step++)
                    {
                        plan.Step();
                        for (int i = 0; i < ps.Length; i++)
                            grads[step * ps.Length + i] = ps[i].Grad?.ToArray() ?? Array.Empty<float>();
                    }
                    return grads;
                }
            }
            finally
            {
                Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_FORWARD_POOL", previous);
            }
        }

        var pooled = Run(pooling: true, out int pooledCount);
        var unpooled = Run(pooling: false, out _);
        Assert.Equal(0, pooledCount);
        for (int k = 0; k < pooled.Length; k++)
        {
            Assert.NotEmpty(pooled[k]);
            Assert.Equal(unpooled[k], pooled[k]);
        }
    }
}