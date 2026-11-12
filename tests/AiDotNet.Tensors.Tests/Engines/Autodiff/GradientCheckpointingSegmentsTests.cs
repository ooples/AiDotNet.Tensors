using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class GradientCheckpointingSegmentsTests
{
    internal static Tensor<float> CreateLinearWeight(int rows, int cols, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)((rng.NextDouble() - 0.5) * 0.5);
        return new Tensor<float>(data, new[] { rows, cols });
    }

    internal static Tensor<float> Copy(Tensor<float> source) =>
        new Tensor<float>(source.AsSpan().ToArray(), source._shape);

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
            eagerGrads = tape.ComputeGradients(loss, new[] { x, w0, b0, wFrozen, w2 });
        }
        // The frozen weight does receive a gradient when it is differentiated, so its absence below is a real check.
        Assert.Contains(eagerGrads[wFrozen].AsSpan().ToArray(), g => g != 0f);

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
            // wFrozen is requested too, so a gradient the checkpoint backward produced for it would be returned.
            ckptGrads = tape.ComputeGradients(loss, new[] { x, w0, b0, wFrozen, w2 });
        }

        for (int i = 0; i < eagerOut.Length; i++)
            Assert.Equal(eagerOut[i], ckptOut[i], 5);

        if (ckptGrads.TryGetValue(wFrozen, out var frozenGrad))
            Assert.All(frozenGrad.AsSpan().ToArray(), g => Assert.Equal(0f, g));
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
        // Large enough that every weight's SGD update is observable: at 0.05 the largest unique-weight update was
        // 6.8e-6, below what a weight near 0.25 can be compared at.
        const float lr = 1.0f;

        var x = CreateLinearWeight(3, 4, seed: 101);
        var wSharedEager = CreateLinearWeight(4, 4, seed: 201);
        var wUnique1Eager = CreateLinearWeight(4, 4, seed: 202);
        var wUnique2Eager = CreateLinearWeight(4, 4, seed: 203);

        var wSharedCkpt = Copy(wSharedEager);
        var wUnique1Ckpt = Copy(wUnique1Eager);
        var wUnique2Ckpt = Copy(wUnique2Eager);

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

        // Checkpointed run with segmentSize = 1 -> 3 segments:
        // wSharedCkpt has touchCount = 2 -> consumer returns false so outer tape accumulates both segments.
        // wUnique1Ckpt and wUnique2Ckpt have touchCount = 1 -> consumer updates in-place during backward and returns true.
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
        // A consumed gradient lived in the recompute scope, so the parameter must not keep pointing at it.
        Assert.Null(wUnique1Ckpt.Grad);
        Assert.Null(wUnique2Ckpt.Grad);

        // Step shared parameter after backward completes
        engine.TensorSubtractInPlace(wSharedCkpt, engine.TensorMultiplyScalar(ckptGrads[wSharedCkpt], lr));

        // Compare the applied update (new - old) with -lr * eager gradient, at a tolerance well below the update
        // itself: comparing the weights at 4 decimals would pass an update that was skipped altogether.
        AssertAppliedUpdate(wUnique1Eager, wUnique1Ckpt, eagerGrads[wUnique1Eager], lr);
        AssertAppliedUpdate(wUnique2Eager, wUnique2Ckpt, eagerGrads[wUnique2Eager], lr);
        AssertAppliedUpdate(wSharedEager, wSharedCkpt, eagerGrads[wSharedEager], lr);
        for (int i = 0; i < eagerGrads[x].Length; i++)
            Assert.Equal(eagerGrads[x][i], ckptGrads[x][i], 4);
    }

    internal static void AssertAppliedUpdate(Tensor<float> before, Tensor<float> after, Tensor<float> eagerGradient, float lr)
    {
        double largest = 0;
        for (int i = 0; i < before.Length; i++)
        {
            double expected = -(double)lr * eagerGradient[i];
            double applied = (double)after[i] - before[i];
            largest = Math.Max(largest, Math.Abs(expected));
            // float weights near 0.25 resolve about 3e-8; the updates are around 1e-5 to 1e-3.
            Assert.True(Math.Abs(applied - expected) <= 1e-6 + 1e-3 * Math.Abs(expected),
                $"element {i}: applied update {applied:G6}, expected {expected:G6}");
        }
        Assert.True(largest > 1e-5, $"the expected update is too small to observe ({largest:G3})");
    }

    [Fact]
    public void CheckpointSegments_UnderStreamingBackward_SharedEmbeddingMatchesEager()
    {
        // The embedding table is read outside the checkpoint and inside its segment, so the outer streaming pass
        // indexes it (and collects its sparse rows) before the segment's nested recompute backward runs. The nested
        // pass must neither take the outer index slots nor clear them.
        var engine = AiDotNetEngine.Current;
        var table = CreateLinearWeight(10, 4, seed: 301);
        var w = CreateLinearWeight(4, 4, seed: 302);
        var idsA = new Tensor<int>(new[] { 3 });
        idsA[0] = 1; idsA[1] = 3; idsA[2] = 3;
        var idsB = new Tensor<int>(new[] { 3 });
        idsB[0] = 2; idsB[1] = 3; idsB[2] = 5;

        Func<Tensor<float>, Tensor<float>> segment =
            h => engine.Tanh(engine.TensorMatMul(engine.TensorAdd(h, engine.TensorEmbeddingLookup<float, int>(table, idsB)), w));

        Dictionary<Tensor<float>, Tensor<float>> eagerGrads;
        using (var tape = new GradientTape<float>())
        {
            var outTensor = segment(engine.TensorEmbeddingLookup<float, int>(table, idsA));
            var loss = engine.ReduceSum(engine.TensorMultiply(outTensor, outTensor));
            eagerGrads = tape.ComputeGradients(loss, new[] { table, w });
        }

        var streamed = new Dictionary<Tensor<float>, float[]>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        using (var tape = new GradientTape<float>())
        {
            var h0 = engine.TensorEmbeddingLookup<float, int>(table, idsA);
            var outTensor = GradientCheckpointing<float>.CheckpointSegments(
                new[] { segment }, h0, new IReadOnlyList<Tensor<float>>[] { new[] { table, w } }, segmentSize: 1);
            var loss = engine.ReduceSum(engine.TensorMultiply(outTensor, outTensor));
            // The callback may not keep the gradient, so copy it out.
            tape.ComputeGradientsStreaming(loss, new[] { table, w },
                (source, gradient) => streamed[source] = gradient.AsSpan().ToArray());
        }

        foreach (var param in new[] { table, w })
        {
            Assert.True(streamed.ContainsKey(param), "every source must be emitted");
            var expected = eagerGrads[param];
            Assert.Equal(expected.Length, streamed[param].Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], streamed[param][i], 4);
        }
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

    [Fact]
    public void CheckpointSegments_ThrowsOnANullSegmentParameterList()
    {
        var x = new Tensor<float>(new float[] { 1f, 2f }, new[] { 1, 2 });
        var funcs = new Func<Tensor<float>, Tensor<float>>[] { h => h, h => h };
        // A null entry would otherwise mean "differentiate every leaf" for that segment. The list is built with
        // nullable annotations off because a caller outside nullable analysis is exactly who can pass one.
#nullable disable
        var withNull = new IReadOnlyList<Tensor<float>>[] { Array.Empty<Tensor<float>>(), null };
#nullable restore

        Assert.Throws<ArgumentException>(() =>
            GradientCheckpointing<float>.CheckpointSegments(funcs, x, withNull, segmentSize: 1));
    }

    [Fact]
    public void CheckpointSegments_SegmentSizeNearIntMax_RunsEveryFunctionAsOneSegment()
    {
        var engine = AiDotNetEngine.Current;
        var x = new Tensor<float>(new float[] { 1f, 2f }, new[] { 1, 2 });
        var funcs = new Func<Tensor<float>, Tensor<float>>[]
        {
            h => engine.TensorMultiplyScalar(h, 2f),
            h => engine.TensorMultiplyScalar(h, 3f),
        };

        using var tape = new GradientTape<float>();
        // (count + size - 1) / size overflowed here, giving no segments and returning the input unchanged.
        var output = GradientCheckpointing<float>.CheckpointSegments(
            funcs, x, new IReadOnlyList<Tensor<float>>[] { Array.Empty<Tensor<float>>() }, segmentSize: int.MaxValue);

        Assert.Equal(6f, output[0]);
        Assert.Equal(12f, output[1]);
    }
}

/// <summary>
/// The in-backward consumer on a device-resident engine: consumed gradients come from the recompute's GPU scope and
/// must be usable inside the callback, and both the consumed and the declined gradients must match the eager values
/// after that scope has been released.
/// </summary>
[Collection("DirectGpuSerial")]
public class GradientCheckpointingSegmentsGpuTests : IDisposable
{
    private readonly DirectGpuTensorEngine _gpu;
    private readonly bool _available;

    public GradientCheckpointingSegmentsGpuTests()
    {
        // A machine without a device backend constructs the engine with IsGpuAvailable false, and the test skips. Any
        // exception is a real initialization failure on a GPU runner, so it fails the test instead of skipping it.
        _gpu = new DirectGpuTensorEngine();
        _available = _gpu.IsGpuAvailable;
    }

    // The flat indexer reads a device-resident tensor back through the engine.
    private static float[] ReadBack(Tensor<float> tensor)
    {
        var values = new float[tensor.Length];
        for (int i = 0; i < values.Length; i++) values[i] = tensor[i];
        return values;
    }

    public void Dispose()
    {
        _gpu.Dispose();
        GC.SuppressFinalize(this);
    }

    [SkippableFact]
    public void CheckpointSegments_GpuConsumer_ConsumedAndDeclinedGradientsMatchEager()
    {
        Skip.If(!_available, "GPU backend not available");
        var gpu = _gpu;
        var cpu = new CpuEngine();

        var x = GradientCheckpointingSegmentsTests.CreateLinearWeight(3, 4, seed: 401);
        var wShared = GradientCheckpointingSegmentsTests.CreateLinearWeight(4, 4, seed: 402);
        var wUnique = GradientCheckpointingSegmentsTests.CreateLinearWeight(4, 4, seed: 403);

        Dictionary<Tensor<float>, Tensor<float>> eagerGrads;
        using (var tape = new GradientTape<float>())
        {
            tape.BindEngineIfUnset(cpu);
            var h0 = cpu.Tanh(cpu.TensorMatMul(x, wShared));
            var h1 = cpu.Tanh(cpu.TensorAdd(cpu.TensorMatMul(h0, wUnique), cpu.TensorMatMul(h0, wShared)));
            var loss = cpu.ReduceSum(cpu.TensorMultiply(h1, h1));
            eagerGrads = tape.ComputeGradients(loss, new[] { wShared, wUnique });
        }

        var xGpu = GradientCheckpointingSegmentsTests.Copy(x);
        var wSharedGpu = GradientCheckpointingSegmentsTests.Copy(wShared);
        var wUniqueGpu = GradientCheckpointingSegmentsTests.Copy(wUnique);
        var funcs = new Func<Tensor<float>, Tensor<float>>[]
        {
            h => gpu.Tanh(gpu.TensorMatMul(h, wSharedGpu)),
            h => gpu.Tanh(gpu.TensorAdd(gpu.TensorMatMul(h, wUniqueGpu), gpu.TensorMatMul(h, wSharedGpu))),
        };
        var segmentParams = new IReadOnlyList<Tensor<float>>[] { new[] { wSharedGpu }, new[] { wUniqueGpu, wSharedGpu } };

        float[]? consumed = null;
        Dictionary<Tensor<float>, Tensor<float>> ckptGrads;
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = gpu;
            using var tape = new GradientTape<float>();
            tape.BindEngineIfUnset(gpu);
            var outTensor = GradientCheckpointing<float>.CheckpointSegments(
                funcs, xGpu, segmentParams, segmentSize: 1,
                tryConsumeParameterGradient: (param, grad) =>
                {
                    if (!ReferenceEquals(param, wUniqueGpu)) return false;
                    consumed = ReadBack(grad);
                    return true;
                });
            var loss = gpu.ReduceSum(gpu.TensorMultiply(outTensor, outTensor));
            ckptGrads = tape.ComputeGradients(loss, new[] { wSharedGpu, wUniqueGpu });
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }

        Assert.NotNull(consumed);
        Assert.False(ckptGrads.ContainsKey(wUniqueGpu), "a consumed gradient must not reach the outer dictionary");
        Assert.Null(wUniqueGpu.Grad);
        Assert.True(ckptGrads.ContainsKey(wSharedGpu), "a declined gradient must accumulate in the outer dictionary");

        var consumedValues = consumed ?? Array.Empty<float>();
        for (int i = 0; i < eagerGrads[wUnique].Length; i++)
            Assert.Equal(eagerGrads[wUnique][i], consumedValues[i], 3);
        var declined = ReadBack(ckptGrads[wSharedGpu]);
        for (int i = 0; i < eagerGrads[wShared].Length; i++)
            Assert.Equal(eagerGrads[wShared][i], declined[i], 3);
    }
}
