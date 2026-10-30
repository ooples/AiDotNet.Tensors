// Copyright (c) AiDotNet. All rights reserved.

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
