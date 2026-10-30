// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// One step of a HiPPO-style recurrent cell (slice, concat with a zero state, matmul, bias, reshape, concat with
/// zeros, matmul, tanh, gated update, stack) compiles to what eager computes. Found by the AiDotNet model sweep:
/// HippoMemoryCellLayer diverged at its first step.
/// </summary>
public sealed class CompiledRecurrentCellTests
{
    private static Tensor<float> Random(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    private static IEngine EngineFor(int engineCase)
    {
        if (engineCase == 0) return new CpuEngine();
        var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend.");
        return gpu;
    }

    private static void AssertCompiledMatchesEager(Tensor<float> input, Func<Tensor<float>, Tensor<float>> forward, string label)
    {
        var expected = forward(input).AsSpan().ToArray();
        using var cache = new CompiledModelCache<float>();
        var plan = cache.GetOrCompileInference(input, () => forward(input));
        plan.SetInputs(new[] { input });
        var actual = plan.Execute().AsSpan().ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f + 1e-4f * Math.Abs(expected[i]),
                $"{label} [{i}] eager {expected[i]} compiled {actual[i]}");
    }

    /// <summary>
    /// MatMul + bias + activation where the activation has two consumers: the linear fuses without the activation
    /// and the activation stays standalone. Before the fix the fused linear still applied the activation into the
    /// add's output, which the standalone activation then applied again (sigmoid(sigmoid(x))).
    /// </summary>
    [SkippableTheory]
    [InlineData(0, 0)]
    [InlineData(0, 1)]
    [InlineData(0, 2)]
    [InlineData(1, 0)]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    public void LinearWithASharedActivation_CompiledMatchesEager(int engineCase, int activation)
    {
        var engine = EngineFor(engineCase);
        var w = Random(new[] { 5, 8 }, 11);
        var b = Random(new[] { 8 }, 14);
        var input = Random(new[] { 1, 5 }, 13);
        Tensor<float> Forward(Tensor<float> x)
        {
            var linear = engine.TensorAdd(engine.TensorMatMul(x, w), b);
            var g = activation switch
            {
                0 => engine.Sigmoid(linear),
                1 => engine.Tanh(linear),
                _ => engine.ReLU(linear),
            };
            return engine.TensorAdd(engine.TensorMultiply(g, g), g);
        }
        AssertCompiledMatchesEager(input, Forward, $"activation {activation}");
    }

    /// <summary>
    /// An in-place update of a step's output (TensorBroadcastAddInPlace of an attention's output bias) while a dead
    /// buffer of the same shape sits in the memory planner's pool: the planner rebound the in-place target onto the
    /// donor between the producing write and the update, so the update read the donor's stale values. Found by the
    /// AiDotNet model sweep (BranchformerBlock: attention + cgMLP branches).
    /// </summary>
    [Fact]
    public void InPlaceUpdateOfAStepOutput_KeepsTheProducedValues()
    {
        var engine = new CpuEngine();
        var bias = Random(new[] { 8 }, 2);
        var input = Random(new[] { 4, 8 }, 1);
        Tensor<float> Forward(Tensor<float> x)
        {
            var p = engine.TensorMultiplyScalar(x, 2f);
            var a = engine.Tanh(x);
            var r = engine.TensorMultiply(p, p);   // p's last use: a dead [4, 8] buffer in the pool
            engine.TensorBroadcastAddInPlace(a, bias);
            return engine.TensorAdd(a, r);
        }
        AssertCompiledMatchesEager(input, Forward, "in-place");
    }

    /// <summary>
    /// A tensor whose last use passes while a reshape view of it is still live: its storage is not dead, so the
    /// memory planner must not hand it to a later step of the same size.
    /// </summary>
    [Fact]
    public void BaseDyingWhileItsViewLives_KeepsTheViewsValues()
    {
        var engine = new CpuEngine();
        var input = Random(new[] { 4, 8 }, 1);
        Tensor<float> Forward(Tensor<float> x)
        {
            var t = engine.Tanh(x);
            var v = engine.Reshape(t, new[] { 2, 16 });
            var u = engine.Sigmoid(t);              // t's last use as a tensor; v still reads its storage below
            var w = engine.TensorExp(x);               // same size: a candidate for t's buffer
            var vv = engine.Reshape(engine.TensorMultiply(v, v), new[] { 4, 8 });
            return engine.TensorAdd(vv, engine.TensorAdd(u, w));
        }
        AssertCompiledMatchesEager(input, Forward, "view");
    }

    [Theory]
    [InlineData(1024)]
    [InlineData(3072)]
    public void AttentionWithInPlaceBias_BesideADepthwiseBranch_CompiledMatchesEager(int hidden)
    {
        var engine = new CpuEngine();
        const int d = 512, seq = 64;
        int half = hidden / 2;
        var wq = Random(new[] { d, d }, 1); var wk = Random(new[] { d, d }, 2); var wv = Random(new[] { d, d }, 3); var wo = Random(new[] { d, d }, 4);
        var we = Random(new[] { d, hidden }, 5);
        var ob = Random(new[] { d }, 8);
        var kernel = Random(new[] { half, 1, 31 }, 6);
        var input = Random(new[] { 1, seq, d }, 7);
        Tensor<float> Forward(Tensor<float> x)
        {
            var att = engine.MultiHeadAttentionForward(x, wq, wk, wv, wo, 8);
            engine.TensorBroadcastAddInPlace(att, ob);
            var e = engine.Reshape(engine.TensorMatMul(engine.Reshape(x, new[] { seq, d }), we), new[] { 1, seq, hidden });
            var conv = engine.DepthwiseConv1D(engine.TensorPermute(engine.TensorNarrow(e, 2, half, half), new[] { 0, 2, 1 }), kernel, 1, 15);
            return engine.TensorAdd(att, engine.TensorNarrow(engine.TensorPermute(conv, new[] { 0, 2, 1 }), 2, 0, d));
        }
        AssertCompiledMatchesEager(input, Forward, $"hidden {hidden}");
    }

    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(10)]
    [InlineData(11)]
    [InlineData(12)]
    [InlineData(13)]
    [InlineData(14)]
    [InlineData(15)]
    [InlineData(16)]
    public void HippoFirstStep_CompiledMatchesEager(int upToAndEngine)
    {
        // 0-6: CpuEngine; 10-16: the GPU engine, when there is one.
        int upTo = upToAndEngine % 10;
        var engine = EngineFor(upToAndEngine / 10);
        const int hidden = 8, order = 4;
        var w1 = Random(new[] { 1 + hidden, 1 }, 1);
        var b1 = Random(new[] { 1 }, 2);
        var w2 = Random(new[] { 1 + order, hidden }, 3);
        var b2 = Random(new[] { hidden }, 4);
        var wg = Random(new[] { 1 + order, hidden }, 6);
        var bg = Random(new[] { hidden }, 7);
        var input = Random(new[] { 1, 1, 1 }, 5);

        Tensor<float> Forward(Tensor<float> x)
        {
            var sequence = engine.Reshape(x, new[] { 1, 1, 1 });
            var h = new Tensor<float>(new[] { 1, hidden });
            var current = engine.TensorSliceAxis(sequence, 1, 0);
            if (upTo == 0) return engine.TensorAddScalar(current, 0f);
            var xh = engine.TensorConcatenate(new[] { current, h }, axis: 1);
            if (upTo == 1) return engine.TensorAddScalar(xh, 0f);
            var memoryInput = engine.TensorAdd(engine.TensorMatMul(xh, w1), b1);
            if (upTo == 2) return memoryInput;
            var first = engine.Reshape(memoryInput, new[] { 1, 1, 1 });
            var memory = engine.TensorConcatenate(new[] { first, new Tensor<float>(new[] { 1, 1, order - 1 }) }, axis: 2);
            if (upTo == 3) return engine.TensorAddScalar(memory, 0f);
            var xm = engine.TensorConcatenate(new[] { current, engine.Reshape(memory, new[] { 1, order }) }, axis: 1);
            if (upTo == 4) return engine.TensorAddScalar(xm, 0f);
            var candidate = engine.Tanh(engine.TensorAdd(engine.TensorMatMul(xm, w2), b2));
            if (upTo == 6)
            {
                // The gated update: hidden = (1 - g) * hidden + g * candidate, g = sigmoid(xm Wg + bg).
                var gate = engine.Sigmoid(engine.TensorAdd(engine.TensorMatMul(xm, wg), bg));
                var keep = engine.TensorAddScalar(engine.TensorMultiplyScalar(gate, -1f), 1f);
                candidate = engine.TensorAdd(engine.TensorMultiply(keep, h), engine.TensorMultiply(gate, candidate));
            }
            return engine.Reshape(engine.TensorStack(new[] { candidate }, axis: 1), new[] { 1, 1, hidden });
        }

        AssertCompiledMatchesEager(input, Forward, $"upTo {upTo}");
    }
}
