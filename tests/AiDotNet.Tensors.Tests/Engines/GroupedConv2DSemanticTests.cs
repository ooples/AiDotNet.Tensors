using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("CompilationGlobalState")]
public sealed class GroupedConv2DSemanticTests
{
    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void Parallel_groups_preserve_double_reference_offsets_accumulation_and_thread_invariance(bool twoDimensional, bool accumulate)
    {
        var g = twoDimensional
            ? new GroupedConv2DGeometry(2, 16, 9, 67, 24, 3, 5, 1, 2, 1, 3, 1, 2, 4)
            : new GroupedConv2DGeometry(2, 32, 1, 521, 48, 1, 5, 1, 2, 0, 3, 1, 2, 4);
        CheckGroupedResults(g, accumulate);
    }

    [Theory]
    [InlineData(1, 270, 1, false)]
    [InlineData(4, 35, 3, true)]
    public void Waveform_panel_clips_padding_across_tile_boundaries(int stride, int padding, int dilation, bool accumulate)
    {
        var g = new GroupedConv2DGeometry(1, 8, 1, 1027, 8, 1, 5, 1, stride, 0, padding, 1, dilation, 4);
        CheckGroupedResults(g, accumulate);
    }

    private static void CheckGroupedResults(GroupedConv2DGeometry g, bool accumulate)
    {
        var engine = new CpuEngine();
        using var input = Values<float>(g.InputShape);
        using var kernel = Values<float>(g.KernelShape, .71);
        using var dy = Values<float>(g.OutputShape, .37);
        using var xd = new Tensor<double>(input.ToArray().Select(x => (double)x).ToArray(), g.InputShape);
        using var wd = new Tensor<double>(kernel.ToArray().Select(x => (double)x).ToArray(), g.KernelShape);
        using var dyd = new Tensor<double>(dy.ToArray().Select(x => (double)x).ToArray(), g.OutputShape);
        using var expectedY = GroupedConvolution.Forward(engine, xd, wd, g.Stride, g.Padding, g.Dilation, g.Groups);
        using var expectedDx = new Tensor<double>(Enumerable.Repeat(.25, g.InputCount).ToArray(), g.InputShape);
        using var expectedDw = new Tensor<double>(Enumerable.Repeat(-.125, g.KernelCount).ToArray(), g.KernelShape);
        GroupedConvolution.BackwardInto(engine, dyd, xd, wd, expectedDx, expectedDw, g, accumulate, accumulate);
        int savedThreads = CpuParallelSettings.MaxDegreeOfParallelism;
        bool savedDeterministic = BlasProvider.IsDeterministicMode;
        float[][]? serial = null;
        try
        {
            BlasProvider.SetDeterministicMode(true);
            foreach (int threads in new[] { 1, 4 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                const float sentinel = 123.75f;
                var yData = Enumerable.Repeat(sentinel, g.OutputCount + 14).ToArray();
                var dxData = Enumerable.Repeat(sentinel, g.InputCount + 14).ToArray();
                var dwData = Enumerable.Repeat(sentinel, g.KernelCount + 14).ToArray();
                dxData.AsSpan(7, g.InputCount).Fill(.25f);
                dwData.AsSpan(7, g.KernelCount).Fill(-.125f);
                using var y = Tensor<float>.FromMemory(yData.AsMemory(7, g.OutputCount), g.OutputShape);
                using var dx = Tensor<float>.FromMemory(dxData.AsMemory(7, g.InputCount), g.InputShape);
                using var dw = Tensor<float>.FromMemory(dwData.AsMemory(7, g.KernelCount), g.KernelShape);
                GroupedConvolution.ForwardInto(engine, y, input, kernel, g);
                GroupedConvolution.BackwardInto(engine, dy, input, kernel, dx, dw, g, accumulate, accumulate);
                var actual = new[] { y.ToArray(), dx.ToArray(), dw.ToArray() };
                var expected = new[] { expectedY.ToArray(), expectedDx.ToArray(), expectedDw.ToArray() };
                for (int result = 0; result < actual.Length; result++)
                {
                    Close(expected[result], actual[result].Select(value => (double)value), 2e-5);
                    if (serial is not null) Assert.Equal(serial[result], actual[result]);
                }
                foreach (var data in new[] { yData, dxData, dwData })
                {
                    Assert.All(data.Take(7), value => Assert.Equal(sentinel, value));
                    Assert.All(data.Skip(data.Length - 7), value => Assert.Equal(sentinel, value));
                }
                serial = actual;
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = savedThreads;
            BlasProvider.SetDeterministicMode(savedDeterministic);
        }
    }

    internal static Tensor<T> Values<T>(int[] shape, double phase = 0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int count = shape.Aggregate(1, (a, b) => checked(a * b));
        return new Tensor<T>(Enumerable.Range(0, count).Select(i => ops.FromDouble(.1 * Math.Sin(i * .173 + phase))).ToArray(), shape);
    }

    internal static void Close<T>(IEnumerable<T> expected, IEnumerable<T> actual, double tolerance)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double[] e = expected.Select(ops.ToDouble).ToArray(), a = actual.Select(ops.ToDouble).ToArray();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; ++i)
            Assert.True(!double.IsNaN(a[i]) && !double.IsInfinity(a[i]) && Math.Abs(e[i] - a[i]) <= tolerance * (1 + Math.Abs(e[i])),
                $"Element {i}: expected {e[i]:R}, actual {a[i]:R}.");
    }

    internal static Tensor<T> Forward<T>(IEngine engine, Tensor<T> input, Tensor<T> kernel,
        GroupedConv2DGeometry g, bool grouped)
        => grouped ? GroupedConvolution.Forward(engine, input, kernel, g.Stride, g.Padding, g.Dilation, g.Groups)
            : GroupedConvolution.SplitForward(engine, input, kernel, g);

    internal static T[][] Trajectory<T>(IEngine engine, GroupedConv2DGeometry g, bool grouped, bool compile)
    {
        using var input = Values<T>(g.InputShape);
        using var kernel = Values<T>(g.KernelShape, .71);
        var snapshots = new List<T[]>();
        Tensor<T> Objective()
        {
            var first = Forward(engine, input, kernel, g, grouped);
            var second = Forward(engine, input, kernel, g, grouped);
            return engine.ReduceSum(engine.TensorMultiply(first, second), null);
        }
        if (!compile)
        {
            using var tape = new GradientTape<T>();
            tape.BindEngineIfUnset(engine);
            var loss = Objective();
            var gradients = tape.ComputeGradients(loss, new[] { input, kernel });
            snapshots.Add(loss.ToArray()); snapshots.Add(gradients[input].ToArray()); snapshots.Add(gradients[kernel].ToArray());
            return snapshots.ToArray();
        }
        ICompiledTrainingPlan<T> plan;
        using (var scope = GraphMode.EnableTraining(new[] { input, kernel }))
            plan = scope.CompileTraining(new[] { input, kernel }, Objective());
        using (plan)
        {
            if (grouped)
            {
                var field = typeof(CompiledTrainingPlan<T>).GetField("_forwardSteps", BindingFlags.Instance | BindingFlags.NonPublic)
                    ?? throw new MissingFieldException("_forwardSteps");
                var steps = field.GetValue(plan) as CompiledStep<T>[] ?? throw new InvalidOperationException("Missing compiled grouped steps.");
                Assert.Equal(2, steps.Count(step => step.OpType == OpType.GroupedConv2D));
                Assert.DoesNotContain(steps, step => step.OpName is "TensorSlice" or "Concatenate" or "Concat");
                foreach (var step in steps.Where(step => step.OpType == OpType.GroupedConv2D))
                    Assert.All(step.SavedState, saved => Assert.True(saved is int[] or int));
            }
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: .0005);
            for (int step = 0; step < 4; ++step)
            {
                snapshots.Add(plan.Step().ToArray());
                snapshots.Add(input.Grad?.ToArray() ?? throw new InvalidOperationException("Input gradient unpublished."));
                snapshots.Add(kernel.Grad?.ToArray() ?? throw new InvalidOperationException("Kernel gradient unpublished."));
                snapshots.Add(input.ToArray()); snapshots.Add(kernel.ToArray());
            }
        }
        return snapshots.ToArray();
    }

    private static GroupedConv2DGeometry Geometry(int variant) => variant switch
    {
        0 => new(2, 4, 5, 7, 6, 3, 3, 1, 2, 1, 2, 1, 2, 2),
        1 => new(3, 8, 3, 11, 8, 1, 3, 1, 2, 0, 2, 1, 2, 4),
        2 => new(1, 128, 1, 65, 256, 1, 41, 1, 2, 0, 20, 1, 1, 16),
        _ => new(2, 4, 3, 5, 4, 1, 1, 1, 1, 0, 0, 1, 1, 1),
    };

    [Theory]
    [InlineData(0, false)] [InlineData(0, true)] [InlineData(1, false)] [InlineData(1, true)]
    [InlineData(2, false)] [InlineData(2, true)] [InlineData(3, false)] [InlineData(3, true)]
    public void Float_outputs_gradients_and_updates_match_split_control(int variant, bool compile)
    {
        var engine = new CpuEngine(); var g = Geometry(variant);
        var expected = Trajectory<float>(engine, g, false, compile);
        var actual = Trajectory<float>(engine, g, true, compile);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i) Close(expected[i], actual[i], 2e-5);
    }

    [Theory] [InlineData(false)] [InlineData(true)]
    public void Double_multi_batch_gradients_and_updates_match_split_control(bool compile)
    {
        var engine = new CpuEngine(); var g = Geometry(0);
        var expected = Trajectory<double>(engine, g, false, compile);
        var actual = Trajectory<double>(engine, g, true, compile);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i) Close(expected[i], actual[i], 1e-11);
    }

    [Theory] [InlineData(false)] [InlineData(true)]
    public void Nonzero_memory_offsets_preserve_readonly_sources_and_destination_sentinels(bool useDouble)
    {
        if (useDouble) OffsetCase<double>(1e-11); else OffsetCase<float>(2e-5);
    }

    private static void OffsetCase<T>(double tolerance)
    {
        var g = Geometry(0); var engine = new CpuEngine(); var ops = MathHelper.GetNumericOperations<T>();
        using var input = Values<T>(g.InputShape); using var kernel = Values<T>(g.KernelShape, .71);
        using var dy = Values<T>(g.OutputShape, .37);
        var savedInput = input.ToArray(); var savedKernel = kernel.ToArray(); var savedDy = dy.ToArray();
        T sentinel = ops.FromDouble(123.75);
        T[] outputValues = Enumerable.Repeat(sentinel, g.OutputCount + 14).ToArray();
        T[] dxValues = Enumerable.Repeat(sentinel, g.InputCount + 14).ToArray();
        T[] dwValues = Enumerable.Repeat(sentinel, g.KernelCount + 14).ToArray();
        using var output = Tensor<T>.FromMemory(outputValues.AsMemory(7, g.OutputCount), g.OutputShape);
        using var dx = Tensor<T>.FromMemory(dxValues.AsMemory(7, g.InputCount), g.InputShape);
        using var dw = Tensor<T>.FromMemory(dwValues.AsMemory(7, g.KernelCount), g.KernelShape);
        GroupedConvolution.ForwardInto(engine, output, input, kernel, g);
        Close(Forward(engine, input, kernel, g, false).ToArray(), output.ToArray(), tolerance);
        GroupedConvolution.BackwardInto(engine, dy, input, kernel, dx, dw, g, false, false);
        using var tape = new GradientTape<T>(); tape.BindEngineIfUnset(engine);
        var expectedY = Forward(engine, input, kernel, g, false);
        var loss = engine.ReduceSum(engine.TensorMultiply(expectedY, dy), null);
        var gradients = tape.ComputeGradients(loss, new[] { input, kernel });
        Close(gradients[input].ToArray(), dx.ToArray(), tolerance); Close(gradients[kernel].ToArray(), dw.ToArray(), tolerance);
        GroupedConvolution.BackwardInto(engine, dy, input, kernel, dx, dw, g, true, true);
        Close(gradients[input].ToArray().Select(value => ops.Multiply(value, ops.FromDouble(2))), dx.ToArray(), tolerance);
        Close(gradients[kernel].ToArray().Select(value => ops.Multiply(value, ops.FromDouble(2))), dw.ToArray(), tolerance);
        foreach (var values in new[] { outputValues, dxValues, dwValues })
        {
            Assert.All(values.Take(7), value => Assert.Equal(sentinel, value));
            Assert.All(values.Skip(values.Length - 7), value => Assert.Equal(sentinel, value));
        }
        Close(output.ToArray(), outputValues.Skip(7).Take(g.OutputCount), tolerance);
        Close(dx.ToArray(), dxValues.Skip(7).Take(g.InputCount), tolerance);
        Close(dw.ToArray(), dwValues.Skip(7).Take(g.KernelCount), tolerance);
        Assert.Equal(savedInput, input.ToArray()); Assert.Equal(savedKernel, kernel.ToArray()); Assert.Equal(savedDy, dy.ToArray());
    }

    [Fact]
    public void Custom_engine_preserves_ordinary_virtual_convolution_calls()
    {
        var engine = new CountingEngine(); var g = Geometry(1);
        using var input = Values<float>(g.InputShape); using var kernel = Values<float>(g.KernelShape);
        GroupedConvolution.Forward(engine, input, kernel, g.Stride, g.Padding, g.Dilation, g.Groups);
        Assert.Equal(g.Groups, engine.Calls);
    }

    private sealed class CountingEngine : CpuEngine
    {
        internal int Calls;
        public override Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] dilation)
        { Calls++; return base.Conv2D(input, kernel, stride, padding, dilation); }
    }

    [Fact]
    public void Higher_order_input_and_kernel_derivatives_match_the_split_expression()
    {
        double[][] Run(bool grouped)
        {
            var engine = new CpuEngine(); var g = new GroupedConv2DGeometry(1, 4, 3, 5, 4, 1, 3, 1, 1, 0, 1, 1, 1, 2);
            using var input = Values<double>(g.InputShape); using var kernel = Values<double>(g.KernelShape, .71);
            using var v = Values<double>(g.InputShape, .2); using var w = Values<double>(g.KernelShape, .4);
            using var tape = new GradientTape<double>(); tape.BindEngineIfUnset(engine);
            var y = Forward(engine, input, kernel, g, grouped);
            var loss = engine.ReduceSum(engine.TensorMultiply(y, y), null);
            var first = tape.ComputeGradients(loss, new[] { input, kernel }, createGraph: true);
            var h = engine.TensorAdd(engine.ReduceSum(engine.TensorMultiply(first[input], v), null),
                engine.ReduceSum(engine.TensorMultiply(first[kernel], w), null));
            var second = tape.ComputeGradients(h, new[] { input, kernel });
            return new[] { first[input].ToArray(), first[kernel].ToArray(), second[input].ToArray(), second[kernel].ToArray() };
        }
        var expected = Run(false); var actual = Run(true);
        for (int i = 0; i < expected.Length; ++i) Close(expected[i], actual[i], 1e-10);
    }

    [Fact]
    public void Empty_batch_overwrites_kernel_gradient_with_zero_and_accumulation_preserves_it()
    {
        var engine = new CpuEngine(); var g = new GroupedConv2DGeometry(0, 4, 3, 5, 4, 1, 3, 1, 1, 0, 1, 1, 1, 2);
        using var input = new Tensor<float>(g.InputShape); using var kernel = Values<float>(g.KernelShape);
        using var output = GroupedConvolution.Forward(engine, input, kernel, g.Stride, g.Padding, g.Dilation, g.Groups);
        Assert.Equal(0, output.Length);
        using var dx = new Tensor<float>(g.InputShape); using var dw = new Tensor<float>(g.KernelShape); dw.Fill(7);
        GroupedConvolution.BackwardInto(engine, output, input, kernel, dx, dw, g, false, false);
        Assert.All(dw.ToArray(), value => Assert.Equal(0, value));
        dw.Fill(3);
        GroupedConvolution.BackwardInto(engine, output, input, kernel, dx, dw, g, true, true);
        Assert.All(dw.ToArray(), value => Assert.Equal(3, value));
    }

    [Fact]
    public void Overlapping_primal_destinations_snapshot_before_dx_and_dw()
    {
        var engine = new CpuEngine(); var g = new GroupedConv2DGeometry(2, 4, 3, 5, 4, 1, 1, 1, 1, 0, 0, 1, 1, 2);
        using var input = Values<float>(g.InputShape); using var kernel = Values<float>(g.KernelShape, .71);
        using var dy = Values<float>(g.OutputShape, .37);
        float[] expectedInput, expectedKernel;
        using (var tape = new GradientTape<float>())
        {
            tape.BindEngineIfUnset(engine);
            var y = Forward(engine, input, kernel, g, false);
            var gradients = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(y, dy), null), new[] { input, kernel });
            expectedInput = gradients[input].ToArray(); expectedKernel = gradients[kernel].ToArray();
        }
        var savedDy = dy.ToArray();
        GroupedConvolution.BackwardInto(engine, dy, input, kernel, input, kernel, g, false, false);
        Close(expectedInput, input.ToArray(), 2e-5); Close(expectedKernel, kernel.ToArray(), 2e-5);
        Assert.Equal(savedDy, dy.ToArray());
        var expectedForward = Forward(engine, input, kernel, g, false).ToArray();
        GroupedConvolution.ForwardInto(engine, input, input, kernel, g);
        Close(expectedForward, input.ToArray(), 2e-5);
    }

    [Fact]
    public void Strided_destination_commits_logical_values()
    {
        var engine = new CpuEngine(); var g = Geometry(0);
        using var input = Values<double>(g.InputShape); using var kernel = Values<double>(g.KernelShape, .71);
        using var storage = new Tensor<double>(new[] { g.Batch, g.OutChannels, g.OutWidth, g.OutHeight });
        using var output = storage.Transpose(new[] { 0, 1, 3, 2 });
        Assert.False(output.IsContiguous);
        var expected = Forward(engine, input, kernel, g, false).ToArray();
        GroupedConvolution.ForwardInto(engine, output, input, kernel, g);
        Close(expected, output.ToArray(), 1e-11);
    }

    [Fact]
    public void Compiled_grouped_operation_removes_actual_retained_slice_backings_and_keeps_public_output()
    {
        var g = Geometry(2);
        (long Bytes, float[] Output) Run(bool grouped)
        {
            var engine = new CpuEngine();
            using var input = Values<float>(g.InputShape); using var kernel = Values<float>(g.KernelShape, .71);
            Tensor<float> y; CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { kernel }))
            {
                y = Forward(engine, input, kernel, g, grouped);
                scope.RetainOutput(y);
                var loss = engine.ReduceSum(engine.TensorMultiply(y, y), null);
                plan = Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(new[] { kernel }, loss));
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
                plan.Step();
                var field = typeof(CompiledTrainingPlan<float>).GetField("_forwardSteps", BindingFlags.Instance | BindingFlags.NonPublic)
                    ?? throw new MissingFieldException("_forwardSteps");
                var steps = field.GetValue(plan) as CompiledStep<float>[] ?? throw new InvalidOperationException();
                var arrays = new HashSet<float[]>(ReferenceEqualityComparer<float[]>.Instance);
                foreach (var step in steps)
                {
                    var array = step.OutputBuffer.GraphBackingArrayIdentity as float[]
                        ?? throw new InvalidOperationException("Forward backing could not be inspected without materialization.");
                    if (!ReferenceEquals(array, input.GraphBackingArrayIdentity) && !ReferenceEquals(array, kernel.GraphBackingArrayIdentity))
                        arrays.Add(array);
                }
                return (arrays.Sum(array => array.LongLength * sizeof(float)), y.ToArray());
            }
        }
        var baseline = Run(false); var candidate = Run(true);
        Close(baseline.Output, candidate.Output, 2e-5);
        long removedCopies = (long)(g.InputCount + g.KernelCount) * sizeof(float);
        Assert.True(baseline.Bytes - candidate.Bytes >= removedCopies,
            $"Retained FP32 backing capacity {baseline.Bytes} -> {candidate.Bytes}; expected at least {removedCopies}B of input/kernel copies removed.");
    }
}
