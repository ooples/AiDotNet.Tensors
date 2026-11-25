// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A gather or index-copy whose indices are built by integer ops inside the traced forward (an arange plus an offset,
/// a broadcast sum of row and column offsets) indexes with those values on every step of the compiled plan, the first
/// included. The plan schedules only the float operands, so the first step used to read the indices before anything
/// had computed them: all zeros, a loss for the wrong elements and a gradient into the wrong weights.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class CapturedIndexProducerTests
{
    private const int Batch = 2, Seq = 16, Dim = 3;

    public enum IndexRecipe
    {
        HostTensor,
        Arange,
        ArangePlusScalar,
        ReshapedArange,
        BroadcastRowsPlusColumns,
    }

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

    private static Tensor<double> Wave(int[] shape, double scale)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = scale * Math.Sin(0.7 * i + 0.3);
        return t;
    }

    private static Tensor<int> BuildIndices(IEngine engine, IndexRecipe recipe)
    {
        switch (recipe)
        {
            case IndexRecipe.HostTensor:
                return new Tensor<int>(new[] { 1, 3, 9, 11 }, new[] { 4 });
            case IndexRecipe.Arange:
                return engine.TensorArange<int>(1, 9, 2);
            case IndexRecipe.ArangePlusScalar:
                return engine.TensorAddScalar(engine.TensorArange<int>(0, 8, 2), 1);
            case IndexRecipe.ReshapedArange:
                return engine.Reshape(engine.Reshape(engine.TensorArange<int>(1, 9, 2), new[] { 2, 2 }), new[] { 4 });
            case IndexRecipe.BroadcastRowsPlusColumns:
                return engine.Reshape(
                    engine.TensorAdd(
                        engine.Reshape(engine.TensorArange<int>(1, 17, 8), new[] { 2, 1 }),
                        engine.Reshape(engine.TensorArange<int>(0, 4, 2), new[] { 1, 2 })),
                    new[] { 4 });
            default:
                throw new ArgumentOutOfRangeException(nameof(recipe));
        }
    }

    public static TheoryData<string, IndexRecipe> Cases()
    {
        var data = new TheoryData<string, IndexRecipe>();
        foreach (var engine in new[] { "cpu", "gpu" })
            foreach (IndexRecipe recipe in Enum.GetValues(typeof(IndexRecipe)))
                data.Add(engine, recipe);
        return data;
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Gather_indexes_with_traced_integer_indices_on_every_step(string engineName, IndexRecipe recipe)
    {
        var engine = TryEngine(engineName, out var skipReason);
        Skip.If(engine is null, skipReason);
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var weight = Wave(new[] { Batch, Seq, Dim }, 1.0);
            var input = Wave(new[] { Batch, Seq, Dim }, 0.5);
            Tensor<double> Loss()
            {
                var gathered = engine.TensorGather(engine.TensorMultiply(input, weight), BuildIndices(engine, recipe), axis: 1);
                return engine.Reshape(engine.ReduceSum(engine.TensorMultiply(gathered, gathered), null), new[] { 1 });
            }

            double eager = Loss()[0];
            // An all-zero index tensor gives a different loss, so a first step that reads zeros cannot pass.
            var zeros = engine.TensorGather(engine.TensorMultiply(input, weight), new Tensor<int>(new int[4], new[] { 4 }), axis: 1);
            Assert.NotEqual(eager, engine.ReduceSum(engine.TensorMultiply(zeros, zeros), null)[0]);

            var cache = new CompiledModelCache<double>();
            var plan = cache.GetOrCompileTraining(new[] { Batch, Seq, Dim, (int)recipe }, Loss, new[] { weight });
            double first = plan.Step()[0];
            double second = plan.Step()[0];

            Assert.Equal(eager, first, 12);
            Assert.Equal(eager, second, 12);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            if (engine is IDisposable disposable && engineName != "cpu") disposable.Dispose();
        }
    }

    [SkippableTheory]
    [InlineData("cpu")]
    [InlineData("gpu")]
    public void Index_copy_writes_at_traced_integer_indices_on_every_step(string engineName)
    {
        var engine = TryEngine(engineName, out var skipReason);
        Skip.If(engine is null, skipReason);
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var weight = Wave(new[] { 4, 3 }, 1.0);
            var input = Wave(new[] { 4, 3 }, 0.5);
            var rows = Wave(new[] { 2, 3 }, 2.0);
            var ramp = Wave(new[] { 4, 3 }, 3.0);
            Tensor<double> Loss()
            {
                var at = engine.TensorAddScalar(engine.TensorArange<int>(0, 2, 1), 2);
                var copied = engine.TensorIndexCopy(engine.TensorMultiply(input, weight), 0, at, rows);
                return engine.Reshape(engine.ReduceSum(engine.TensorMultiply(copied, ramp), null), new[] { 1 });
            }

            double eager = Loss()[0];
            var plan = new CompiledModelCache<double>().GetOrCompileTraining(new[] { 4, 3 }, Loss, new[] { weight });
            double first = plan.Step()[0];
            double second = plan.Step()[0];

            Assert.Equal(eager, first, 12);
            Assert.Equal(eager, second, 12);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            if (engine is IDisposable disposable && engineName != "cpu") disposable.Dispose();
        }
    }

    public enum IndexedOp
    {
        Take,
        TakeAlongDim,
        IndexAdd,
        TensorScatterAdd,
        ScatterAddInto,
        ScatterAddSegments,
    }

    /// <summary>
    /// Every other op that reads an integer index tensor, each with indices an integer op of the trace produces.
    /// </summary>
    [Theory]
    [InlineData(IndexedOp.Take)]
    [InlineData(IndexedOp.TakeAlongDim)]
    [InlineData(IndexedOp.IndexAdd)]
    [InlineData(IndexedOp.TensorScatterAdd)]
    [InlineData(IndexedOp.ScatterAddInto)]
    [InlineData(IndexedOp.ScatterAddSegments)]
    public void Indexed_ops_use_traced_integer_indices_on_every_step(IndexedOp op)
    {
        var engine = new CpuEngine();
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var weight = Wave(new[] { 4, 3 }, 1.0);
            var input = Wave(new[] { 4, 3 }, 0.5);
            var rows = Wave(new[] { 2, 3 }, 2.0);
            var ramp = Wave(new[] { 4, 3 }, 3.0);
            Tensor<double> Loss()
            {
                var x = engine.TensorMultiply(input, weight);
                // Rows 2 and 3, never row 0: indices read as zeros give another loss.
                var at = engine.TensorAddScalar(engine.TensorArange<int>(0, 2, 1), 2);
                Tensor<double> y;
                switch (op)
                {
                    case IndexedOp.Take:
                        y = engine.TensorTake(x, engine.TensorAddScalar(engine.TensorArange<int>(0, 4, 1), 7));
                        return engine.Reshape(engine.ReduceSum(engine.TensorMultiply(y, y), null), new[] { 1 });
                    case IndexedOp.TakeAlongDim:
                        y = engine.TensorTakeAlongDim(x,
                            engine.Reshape(engine.TensorAddScalar(engine.TensorArange<int>(0, 3, 1), 1), new[] { 1, 3 }), 0);
                        return engine.Reshape(engine.ReduceSum(engine.TensorMultiply(y, y), null), new[] { 1 });
                    case IndexedOp.IndexAdd:
                        y = engine.TensorIndexAdd(x, 0, at, rows);
                        break;
                    case IndexedOp.TensorScatterAdd:
                        y = engine.TensorScatterAdd(x, at, rows, 0);
                        break;
                    case IndexedOp.ScatterAddInto:
                        y = engine.ScatterAdd(x, at, rows, 0);
                        break;
                    case IndexedOp.ScatterAddSegments:
                        // Rows 0..3 of x summed into segments 1, 2, 1, 2 of 4.
                        y = engine.ScatterAdd(x, engine.TensorAddScalar(new Tensor<int>(new[] { 0, 1, 0, 1 }, new[] { 4 }), 1), 0, 4);
                        break;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(op));
                }
                return engine.Reshape(engine.ReduceSum(engine.TensorMultiply(engine.TensorMultiply(y, y), ramp), null), new[] { 1 });
            }

            double eager = Loss()[0];
            var plan = new CompiledModelCache<double>().GetOrCompileTraining(new[] { 4, 3, (int)op }, Loss, new[] { weight });
            double first = plan.Step()[0];
            double second = plan.Step()[0];

            Assert.Equal(eager, first, 12);
            Assert.Equal(eager, second, 12);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    /// <summary>
    /// The indices are recomputed on each replay, as each eager call recomputes them: an integer tensor the caller
    /// edits between steps moves the gather with it.
    /// </summary>
    [Fact]
    public void Traced_indices_follow_an_integer_tensor_edited_between_steps()
    {
        var engine = new CpuEngine();
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var weight = Wave(new[] { Batch, Seq, Dim }, 1.0);
            var input = Wave(new[] { Batch, Seq, Dim }, 0.5);
            var positions = new Tensor<int>(new[] { 0, 2, 8, 10 }, new[] { 4 });
            Tensor<double> Loss()
            {
                var at = engine.TensorAddScalar(positions, 1);
                var gathered = engine.TensorGather(engine.TensorMultiply(input, weight), at, axis: 1);
                return engine.Reshape(engine.ReduceSum(engine.TensorMultiply(gathered, gathered), null), new[] { 1 });
            }

            double eagerBefore = Loss()[0];
            var plan = new CompiledModelCache<double>().GetOrCompileTraining(new[] { Batch, Seq, Dim }, Loss, new[] { weight });
            Assert.Equal(eagerBefore, plan.Step()[0], 12);

            positions[0] = 4;
            positions[3] = 14;
            double eagerAfter = Loss()[0];
            Assert.NotEqual(eagerBefore, eagerAfter);
            Assert.Equal(eagerAfter, plan.Step()[0], 12);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
