using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Training steps inside one <see cref="TensorArena"/> whose parameters have equal element counts and transposed
/// shapes, and whose graphs differ from step to step. Issue #1022 saw [256, 64] and [64, 256] gradients accumulate
/// into each other under an arena; issue #1031 saw a weight overwritten through <c>TensorCopy</c> of an arena
/// temporary corrupt the next backward. Both reports came from consumers; these pin the engine-level patterns.
/// </summary>
public class ArenaTrainingStepTests
{
    private readonly CpuEngine _engine = new();

    private static Tensor<float> Random(int[] shape, Random rng, float scale)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    /// <summary>
    /// #1022: two weights of 16384 elements, [64, 256] and [256, 64]. Each step builds a different graph (an extra
    /// scaled branch on some steps, a different scalar on each) so successive compiled backward plans reach different
    /// sets of tensors. Every gradient must keep its parameter's shape and equal the gradient computed outside any arena.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TransposedEqualSizeWeights_KeepTheirOwnGradients_AcrossVaryingGraphs(bool resetEachStep)
    {
        var rng = new Random(1022);
        var x = Random(new[] { 4, 64 }, rng, 1f);
        var w1 = Random(new[] { 64, 256 }, rng, 0.05f);
        var w2 = Random(new[] { 256, 64 }, rng, 0.05f);

        Dictionary<Tensor<float>, Tensor<float>> Step(int step)
        {
            using var tape = new GradientTape<float>();
            var h = _engine.ReLU(_engine.TensorMatMul(x, w1));
            var y = _engine.TensorMatMul(h, w2);
            if (step % 3 == 1)
                y = _engine.TensorAdd(y, _engine.TensorMultiplyScalar(_engine.TensorMatMul(x, _engine.TensorMatMul(w1, w2)), 0.5f));
            var loss = _engine.ReduceMean(_engine.TensorMultiply(_engine.TensorMultiplyScalar(y, 1f + 0.25f * step), y), new[] { 0, 1 }, keepDims: false);
            return tape.ComputeGradients(loss, new[] { w1, w2 });
        }

        var expected = new List<(float[] G1, float[] G2)>();
        for (int step = 0; step < 8; step++)
        {
            var g = Step(step);
            expected.Add((g[w1].ToArray(), g[w2].ToArray()));
        }

        using var arena = TensorArena.Create();
        for (int step = 0; step < 8; step++)
        {
            if (resetEachStep) arena.Reset();
            var g = Step(step);
            Assert.Equal(w1.Shape.ToArray(), g[w1].Shape.ToArray());
            Assert.Equal(w2.Shape.ToArray(), g[w2].Shape.ToArray());
            var g1 = g[w1].ToArray();
            var g2 = g[w2].ToArray();
            for (int i = 0; i < g1.Length; i++) Assert.Equal(expected[step].G1[i], g1[i], 4);
            for (int i = 0; i < g2.Length; i++) Assert.Equal(expected[step].G2[i], g2[i], 4);
        }
    }

    /// <summary>
    /// #1031: a WGAN-critic step. Gradient step on [1, 64] and [64, 1] weights, then weight clipping written back with
    /// <c>TensorCopy(TensorClamp(w, lo, hi), w)</c>, repeated inside one arena. The weights must equal the same sequence
    /// run outside any arena, and every backward must succeed.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ClampWrittenBackThroughTensorCopy_DoesNotCorruptTheNextBackward(bool resetEachStep)
    {
        float[] Run(bool inArena)
        {
            var rng = new Random(1031);
            var x = Random(new[] { 8, 1 }, rng, 1f);
            var w1 = Random(new[] { 1, 64 }, rng, 0.5f);
            var w2 = Random(new[] { 64, 1 }, rng, 0.5f);
            using var arena = inArena ? TensorArena.Create() : null;
            for (int step = 0; step < 6; step++)
            {
                if (resetEachStep) arena?.Reset();
                Dictionary<Tensor<float>, Tensor<float>> grads;
                using (var tape = new GradientTape<float>())
                {
                    var y = _engine.TensorMatMul(_engine.ReLU(_engine.TensorMatMul(x, w1)), w2);
                    var loss = _engine.ReduceMean(_engine.TensorMultiply(y, y), new[] { 0, 1 }, keepDims: false);
                    grads = tape.ComputeGradients(loss, new[] { w1, w2 });
                }

                foreach (var w in new[] { w1, w2 })
                {
                    var updated = _engine.TensorSubtract(w, _engine.TensorMultiplyScalar(grads[w], 0.05f));
                    _engine.TensorCopy(updated, w);
                    _engine.TensorCopy(_engine.TensorClamp(w, -0.3f, 0.3f), w);
                }
            }

            var result = new float[w1.Length + w2.Length];
            w1.ToArray().CopyTo(result, 0);
            w2.ToArray().CopyTo(result, w1.Length);
            return result;
        }

        var outside = Run(inArena: false);
        var inside = Run(inArena: true);
        for (int i = 0; i < outside.Length; i++) Assert.Equal(outside[i], inside[i], 5);
    }
}
