// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A symbolic-batch compile serves every batch: the second batch's trace makes the plan batch-polymorphic and
/// later batches are built from it without tracing, each executing exactly what eager computes.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class BatchPolymorphicPlanTests
{
    private static IEngine? TryEngine(string name, out string skipReason)
    {
        skipReason = string.Empty;
        if (name == "cpu") return new CpuEngine();
        try
        {
            var gpu = new DirectGpuTensorEngine();
            if (gpu.IsGpuAvailable) return gpu;
            gpu.Dispose();
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            skipReason = $"No DirectGpu backend: {ex.Message}";
            return null;
        }
        skipReason = "No DirectGpu backend.";
        return null;
    }

    private static Tensor<float> Random(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, string what)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var e = expected.AsSpan().ToArray();
        var a = actual.AsSpan().ToArray();
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= 1e-4f + 1e-4f * Math.Abs(e[i]), $"{what}: [{i}] eager {e[i]} plan {a[i]}");
    }

    /// <summary>Runs every batch through the symbolic cache and checks it against eager.</summary>
    private static (int Traces, int Instantiated) RunBatches(
        IEngine engine, int[] perSampleShape, int[] batches, Func<Tensor<float>, Tensor<float>> forward, string? expectFailure = null)
    {
        using var cache = new CompiledModelCache<float>();
        int traces = 0;
        foreach (var batch in batches)
        {
            var shape = new int[perSampleShape.Length + 1];
            shape[0] = batch;
            Array.Copy(perSampleShape, 0, shape, 1, perSampleShape.Length);
            var input = Random(shape, 1000 + batch);
            var expected = forward(input);
            var symbolic = SymbolicShape.BatchDynamic(shape);
            var plan = cache.GetOrCompileInference(shape, () => { traces++; return forward(input); }, symbolic);
            // A fresh tensor per call, as a server hands requests in.
            var request = Random(shape, 1000 + batch);
            plan.SetInputs(new[] { request });
            AssertClose(expected, plan.Execute(), $"batch {batch}");
            // Replays see the new data, not the first call's.
            var second = Random(shape, 7 + batch);
            plan.SetInputs(new[] { second });
            AssertClose(forward(second), plan.Execute(), $"batch {batch} replay");
            if (expectFailure is not null)
                Assert.Contains(expectFailure, cache.GetBatchPolymorphismFailure(symbolic) ?? "(polymorphic)");
        }
        return (traces, cache.InstantiatedPlanCount);
    }

    public static TheoryData<string> Engines => new() { "cpu", "gpu" };

    [SkippableTheory]
    [MemberData(nameof(Engines))]
    public void MlpForward_TracesTwice_ThenBuildsEveryOtherBatch(string engineName)
    {
        var engine = TryEngine(engineName, out var skip);
        Skip.If(engine is null, skip);
        var w1 = Random(new[] { 8, 16 }, 1);
        var w2 = Random(new[] { 16, 4 }, 2);
        var (traces, built) = RunBatches(engine!, new[] { 8 }, new[] { 1, 4, 2, 3, 7, 16, 4, 1 },
            x => engine!.Softmax(engine.TensorMatMul(engine.ReLU(engine.TensorMatMul(x, w1)), w2)));
        // Batch 1 is specialized (traced, as in torch); 4 is the base, 2 shows what scales; 3, 7, 16 are built.
        Assert.Equal(3, traces);
        Assert.Equal(3, built);
        (engine as IDisposable)?.Dispose();
    }

    [SkippableTheory]
    [MemberData(nameof(Engines))]
    public void ConvFlattenForward_RescalesTheCapturedReshapeTarget(string engineName)
    {
        var engine = TryEngine(engineName, out var skip);
        Skip.If(engine is null, skip);
        var kernel = Random(new[] { 4, 3, 3, 3 }, 3);
        var head = Random(new[] { 4 * 6 * 6, 5 }, 4);
        // The reshape target names the batch, so it is the captured int[] that has to scale.
        var (traces, built) = RunBatches(engine!, new[] { 3, 8, 8 }, new[] { 1, 2, 5, 9, 4 },
            x => engine!.TensorMatMul(engine.Reshape(engine.ReLU(engine.Conv2D(x, kernel)), new[] { x.Shape[0], 4 * 6 * 6 }), head));
        Assert.Equal(3, traces);
        Assert.Equal(2, built);
        (engine as IDisposable)?.Dispose();
    }

    /// <summary>
    /// An op whose delegate captured the batch as an int and as an int[] shape: both must be rescaled for the
    /// built plan, or it writes the template batch's row count.
    /// </summary>
    private static Tensor<float> RowIndexOp(Tensor<float> x)
    {
        int rows = x.Shape[0];
        int[] shape = { x.Shape[0], x.Shape[1] };
        Action<Tensor<float>> fill = output =>
        {
            var span = output.AsWritableSpan();
            for (int r = 0; r < shape[0]; r++)
                for (int c = 0; c < shape[1]; c++)
                    span[r * shape[1] + c] = rows * 100 + r;
        };
        var scope = GraphMode.Current;
        if (scope is null)
        {
            var eager = new Tensor<float>(new[] { rows, x.Shape[1] });
            fill(eager);
            return eager;
        }
        return scope.RecordUnary(LazyNodeType.Custom, "RowIndex", x, new[] { rows, x.Shape[1] }, (_, output) => fill(output));
    }

    [Fact]
    public void ADelegateThatCapturedTheBatch_GetsItRescaled()
    {
        var engine = new CpuEngine();
        var (traces, built) = RunBatches(engine, new[] { 3 }, new[] { 1, 2, 5, 8, 3 },
            x => engine.TensorAdd(RowIndexOp(x), x));
        Assert.Equal(3, traces);
        Assert.Equal(2, built);
    }

    /// <summary>An op whose delegate captured its input's backing array rather than the tensor.</summary>
    private static Tensor<float> DoubleFromBackingOp(Tensor<float> x)
    {
        var src = x.GetBackingArrayForCacheLookupUnsafe()!;
        int length = x.Length;
        Action<Tensor<float>> fill = output =>
        {
            var span = output.AsWritableSpan();
            for (int i = 0; i < length; i++) span[i] = 2 * src[i];
        };
        var scope = GraphMode.Current;
        if (scope is null)
        {
            var eager = new Tensor<float>(x.Shape.ToArray());
            fill(eager);
            return eager;
        }
        return scope.RecordUnary(LazyNodeType.Custom, "DoubleFromBacking", x, x.Shape.ToArray(), (_, output) => fill(output));
    }

    [Fact]
    public void ADelegateThatCapturedABackingArray_ReadsTheBuiltPlansBuffer()
    {
        var engine = new CpuEngine();
        // Built plans must read their own input buffer, not the base trace's.
        var (traces, built) = RunBatches(engine, new[] { 4 }, new[] { 2, 3, 6, 9 },
            x => engine.TensorAdd(DoubleFromBackingOp(x), x));
        Assert.Equal(2, traces);
        Assert.Equal(2, built);
    }

    [Fact]
    public void ACapturedScaleThatDependsOnTheBatch_KeepsTracingEachBatch()
    {
        var engine = new CpuEngine();
        var w = Random(new[] { 6, 6 }, 5);
        // 1/batch is not a batch multiple: no template, every new batch is traced and still correct.
        var (traces, built) = RunBatches(engine, new[] { 6 }, new[] { 1, 2, 3, 4 },
            x => engine.TensorMultiplyScalar(engine.TensorMatMul(x, w), 1f / x.Shape[0]),
            expectFailure: null);
        Assert.Equal(4, traces);
        Assert.Equal(0, built);
    }
}
