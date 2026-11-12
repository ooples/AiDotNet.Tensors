using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class GradientCheckpointingSegmentsTests
{
    private static Tensor<float> CreateLinearWeight(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)((rng.NextDouble() - 0.5) * 0.5);
        return new Tensor<float>(data, new[] { rows, cols });
    }

    [Fact]
    public void CheckpointSegments_MatchesEagerGradients_AndSkipsFrozenParameters()
    {
        var engine = AiDotNetEngine.Current;
        var x = CreateLinearWeight(3, 4, seed: 11);
        var w0 = CreateLinearWeight(4, 4, seed: 21);
        var b0 = CreateLinearWeight(1, 4, seed: 22);
        var wFrozen = CreateLinearWeight(4, 4, seed: 31);
        var w2 = CreateLinearWeight(4, 4, seed: 41);

        Func<Tensor<float>, Tensor<float>> f0 = h => engine.Tanh(engine.TensorAdd(engine.TensorMatMul(h, w0), b0));
        Func<Tensor<float>, Tensor<float>> f1 = h => engine.Tanh(engine.TensorMatMul(h, wFrozen));
        Func<Tensor<float>, Tensor<float>> f2 = h => engine.Tanh(engine.TensorMatMul(h, w2));
        var funcs = new[] { f0, f1, f2 };

        Dictionary<Tensor<float>, Tensor<float>> eagerGrads;
        Tensor<float> eagerOut;
        using (var tape = new GradientTape<float>())
        {
            eagerOut = f2(f1(f0(x)));
            var loss = engine.ReduceSum(engine.TensorMultiply(eagerOut, eagerOut));
            eagerGrads = tape.ComputeGradients(loss, new[] { x, w0, b0, w2 });
        }

        // 3 functions with segmentSize = 2 -> 2 segments: [f0, f1] and [f2].
        // Omit wFrozen from segment 0's parameter list to verify selective differentiation.
        var segmentParams = new IReadOnlyList<Tensor<float>>[]
        {
            new[] { w0, b0 },
            new[] { w2 },
        };

        Dictionary<Tensor<float>, Tensor<float>> ckptGrads;
        Tensor<float> ckptOut;
        using (var tape = new GradientTape<float>())
        {
            ckptOut = GradientCheckpointing<float>.CheckpointSegments(
                funcs, x, segmentParams, segmentSize: 2, tryConsumeParameterGradient: null);
            var loss = engine.ReduceSum(engine.TensorMultiply(ckptOut, ckptOut));
            ckptGrads = tape.ComputeGradients(loss, new[] { x, w0, b0, w2 });
        }

        for (int i = 0; i < eagerOut.Length; i++)
            Assert.Equal(eagerOut[i], ckptOut[i], 5);

        Assert.False(ckptGrads.ContainsKey(wFrozen));
        foreach (var param in new[] { x, w0, b0, w2 })
        {
            Assert.True(ckptGrads.ContainsKey(param));
            for (int i = 0; i < eagerGrads[param].Length; i++)
                Assert.Equal(eagerGrads[param][i], ckptGrads[param][i], 4);
        }
    }

    [Fact]
    public void CheckpointSegments_PostAccumulateConsumer_StepsSingleTouchInPlace_AndAccumulatesSharedWeights()
    {
        var engine = AiDotNetEngine.Current;
        const float lr = 0.05f;

        var x = CreateLinearWeight(3, 4, seed: 101);
        var wSharedEager = CreateLinearWeight(4, 4, seed: 201);
        var wUnique1Eager = CreateLinearWeight(4, 4, seed: 202);
        var wUnique2Eager = CreateLinearWeight(4, 4, seed: 203);

        var wSharedCkpt = new Tensor<float>(wSharedEager.AsSpan().ToArray(), wSharedEager._shape);
        var wUnique1Ckpt = new Tensor<float>(wUnique1Eager.AsSpan().ToArray(), wUnique1Eager._shape);
        var wUnique2Ckpt = new Tensor<float>(wUnique2Eager.AsSpan().ToArray(), wUnique2Eager._shape);

        // Eager reference: 3 layers where wShared is used in both layer 0 and layer 2 (multi-segment shared root),
        // while wUnique1 (layer 1) and wUnique2 (layer 2) each appear in exactly one segment.
        Dictionary<Tensor<float>, Tensor<float>> eagerGrads;
        using (var tape = new GradientTape<float>())
        {
            var h0 = engine.Tanh(engine.TensorMatMul(x, wSharedEager));
            var h1 = engine.Tanh(engine.TensorMatMul(h0, wUnique1Eager));
            var h2 = engine.Tanh(engine.TensorAdd(engine.TensorMatMul(h1, wUnique2Eager), engine.TensorMatMul(h1, wSharedEager)));
            var loss = engine.ReduceSum(engine.TensorMultiply(h2, h2));
            eagerGrads = tape.ComputeGradients(loss, new[] { x, wSharedEager, wUnique1Eager, wUnique2Eager });
        }

        // Eager SGD reference update
        var expectedUnique1 = engine.TensorSubtract(wUnique1Eager, engine.TensorMultiplyScalar(eagerGrads[wUnique1Eager], lr));
        var expectedUnique2 = engine.TensorSubtract(wUnique2Eager, engine.TensorMultiplyScalar(eagerGrads[wUnique2Eager], lr));
        var expectedShared = engine.TensorSubtract(wSharedEager, engine.TensorMultiplyScalar(eagerGrads[wSharedEager], lr));

        // Checkpointed run with segmentSize = 1 -> 3 segments:
        // wSharedCkpt has touchCount = 2 -> consumer returns false so outer tape accumulates both segments.
        // wUnique1Ckpt and wUnique2Ckpt have touchCount = 1 -> consumer updates in-place during backward and returns true!
        var funcs = new Func<Tensor<float>, Tensor<float>>[]
        {
            h => engine.Tanh(engine.TensorMatMul(h, wSharedCkpt)),
            h => engine.Tanh(engine.TensorMatMul(h, wUnique1Ckpt)),
            h => engine.Tanh(engine.TensorAdd(engine.TensorMatMul(h, wUnique2Ckpt), engine.TensorMatMul(h, wSharedCkpt))),
        };
        var segmentParams = new IReadOnlyList<Tensor<float>>[]
        {
            new[] { wSharedCkpt },
            new[] { wUnique1Ckpt },
            new[] { wUnique2Ckpt, wSharedCkpt },
        };

        int consumedCount = 0;
        Dictionary<Tensor<float>, Tensor<float>> ckptGrads;
        using (var tape = new GradientTape<float>())
        {
            var outTensor = GradientCheckpointing<float>.CheckpointSegments(
                funcs,
                x,
                segmentParams,
                segmentSize: 1,
                tryConsumeParameterGradient: (param, grad) =>
                {
                    if (ReferenceEquals(param, wSharedCkpt))
                        return false; // shared across segments -> defer to outer accumulation

                    using (GradientTape<float>.NoGrad())
                    {
                        engine.TensorSubtractInPlace(param, engine.TensorMultiplyScalar(grad, lr));
                    }
                    consumedCount++;
                    return true;
                });

            var loss = engine.ReduceSum(engine.TensorMultiply(outTensor, outTensor));
            ckptGrads = tape.ComputeGradients(loss, new[] { x, wSharedCkpt, wUnique1Ckpt, wUnique2Ckpt });
        }

        Assert.Equal(2, consumedCount);
        Assert.False(ckptGrads.ContainsKey(wUnique1Ckpt), "Consumed single-segment parameter gradient must not remain allocated in outer dictionary.");
        Assert.False(ckptGrads.ContainsKey(wUnique2Ckpt), "Consumed single-segment parameter gradient must not remain allocated in outer dictionary.");
        Assert.True(ckptGrads.ContainsKey(wSharedCkpt), "Multi-segment shared parameter gradient must accumulate in outer dictionary.");

        // Step shared parameter after backward completes
        engine.TensorSubtractInPlace(wSharedCkpt, engine.TensorMultiplyScalar(ckptGrads[wSharedCkpt], lr));

        for (int i = 0; i < expectedUnique1.Length; i++)
            Assert.Equal(expectedUnique1[i], wUnique1Ckpt[i], 4);
        for (int i = 0; i < expectedUnique2.Length; i++)
            Assert.Equal(expectedUnique2[i], wUnique2Ckpt[i], 4);
        for (int i = 0; i < expectedShared.Length; i++)
            Assert.Equal(expectedShared[i], wSharedCkpt[i], 4);
        for (int i = 0; i < eagerGrads[x].Length; i++)
            Assert.Equal(eagerGrads[x][i], ckptGrads[x][i], 4);
    }

    [Fact]
    public void CheckpointSegments_ThrowsWhenSegmentParameterCountMismatchesSegments()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f }, new[] { 1, 2 });
        var funcs = new Func<Tensor<float>, Tensor<float>>[] { h => h, h => h, h => h };
        var wrongParams = new IReadOnlyList<Tensor<float>>[] { Array.Empty<Tensor<float>>() }; // 1 instead of 2 for segmentSize=2

        Assert.Throws<ArgumentException>(() =>
            GradientCheckpointing<float>.CheckpointSegments(funcs, x, wrongParams, segmentSize: 2));
    }
}
