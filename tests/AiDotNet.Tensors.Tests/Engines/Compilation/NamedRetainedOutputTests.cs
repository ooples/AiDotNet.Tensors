using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("DirectGpuSerial")]
public sealed class NamedRetainedOutputTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    [Theory]
    [InlineData("offset", 64)]
    [InlineData("strided", 64)]
    [InlineData("offset-strided", 32768)]
    public void Partial_and_strided_named_views_transfer_the_whole_storage_slab(string layout, int rows)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        int count = rows * 3, start = rows / 2;
        var parameter = new Tensor<double>(Enumerable.Range(0, count).Select(value => (double)value).ToArray(), new[] { rows, 3 });
        var expected = new List<double>();
        if (layout == "offset")
            for (int i = start * 3; i < count; i++) expected.Add(2.0 * i);
        else
            for (int column = 0; column < 3; column++)
                for (int row = layout == "offset-strided" ? start : 0; row < rows; row++)
                    expected.Add(2.0 * (row * 3 + column));
        int[] shape = layout == "offset" ? new[] { start, 3 }
            : layout == "strided" ? new[] { 3, rows } : new[] { 3, start };
        using var outputs = RetainedOutputCapture<double>.Begin();
        Tensor<double> detached;
        Tensor<double> retained;
        object storage;
        using (var arena = TensorArena.Create())
        {
            using (var tape = new GradientTape<double>())
            {
                detached = engine.StopGradient(engine.TensorMultiplyScalar(parameter, 2.0));
                retained = layout == "strided" ? detached.Transpose(new[] { 1, 0 })
                    : engine.TensorNarrow(detached, 0, start, start);
                if (layout == "offset-strided") retained = retained.Transpose(new[] { 1, 0 });
                Assert.Null(retained.GetLiveBackingArrayOrNull()); // The old logical-layout accessor cannot name this slab.
                Assert.Same(detached._storage, retained._storage);
                storage = retained._storage;
                GradientRecording.RetainOutput("generation", retained);
                RetainedOutputCapture<double>.PublishEager(tape);
            }
            Assert.True(((IBackingArraySource)retained).TryGetBackingArray(out var backing));
            var nonNullBacking = Assert.IsAssignableFrom<Array>(backing);
            arena.ReturnIssued(new TensorArena.IssuedBuffer(nonNullBacking, -1, 0));
            for (int step = 0; step < 3; step++)
            {
                arena.Reset();
                for (int rent = 0; rent < 3; rent++)
                {
                    var scratch = Assert.IsType<Tensor<double>>(arena.TryRentTensor<double>(count, new[] { count }));
                    scratch.AsWritableSpan().Fill(-9.0);
                    Assert.NotSame(detached, scratch);
                }
                Assert.Equal(new[] { rows, 3 }, detached.Shape.ToArray());
                Assert.Equal(shape, retained.Shape.ToArray());
                Assert.Same(storage, retained._storage);
                Assert.Equal(expected, retained.ToArray());
                Assert.Equal(0, arena.PinnedArrayCount);
            }
        }
        outputs.Complete();
        Assert.Same(retained, outputs.Get("generation"));
        using (var next = TensorArena.Create())
            for (int i = 0; i < 8; i++)
                Assert.IsType<Tensor<double>>(next.TryRentTensor<double>(count, new[] { count })).AsWritableSpan().Fill(-77.0);
        Assert.Equal(shape, retained.Shape.ToArray());
        Assert.Equal(expected, retained.ToArray());
    }

    [Fact]
    public void Finite_forward_with_nonfinite_gradients_cannot_publish_a_discarded_update()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var weight = new Tensor<float>(new[] { 1.0f }, new[] { 1 });
        var parameters = new[] { weight };
        CompiledTrainingPlan<float> plan;
        using (var trace = GraphMode.EnableTraining(parameters))
        {
            GradientRecording.RetainOutput("generation", engine.StopGradient(engine.TensorAddScalar(weight, 2.0f)));
            plan = trace.CompileTraining(parameters, engine.ReduceSum(engine.TensorSqrt(weight), null));
        }
        using (plan)
        using (var successful = RetainedOutputCapture<float>.Begin())
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.125);
            Assert.Equal(1.0f, plan.Step().ToArray()[0]);
            successful.Complete();
            Assert.Equal(3.0f, successful.Get("generation").ToArray()[0]);
            weight[0] = 0; // sqrt(0) is finite, while its derivative is positive infinity.
            using var skipped = RetainedOutputCapture<float>.Begin();
            Assert.Equal(0.0f, plan.Step().ToArray()[0]);
            Assert.True(plan.LastStepSkippedNonFiniteGradients);
            Assert.Equal(1, plan.OptimizerStep);
            Assert.Equal(0.0f, weight.ToArray()[0]);
            Assert.Throws<InvalidOperationException>(() => skipped.Complete());
            Assert.Throws<InvalidOperationException>(() => skipped.Get("generation"));
            Assert.Throws<InvalidOperationException>(() => successful.Get("generation"));
        }
    }

    [Theory]
    [InlineData(32, 3)]
    [InlineData(256, 256)]
    public void Named_stopgradient_view_survives_caller_arena_reset_disposal_and_pool_reuse(int rows, int columns)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        int count = rows * columns;
        var shape = new[] { rows, columns };
        var parameter = new Tensor<double>(shape);
        parameter.AsWritableSpan().Fill(1.0);
        using var outputs = RetainedOutputCapture<double>.Begin();
        Tensor<double> detached;
        Tensor<double> retained;
        object storage;
        using (var arena = TensorArena.Create())
        {
            using (var tape = new GradientTape<double>())
            {
                detached = engine.StopGradient(engine.TensorMultiplyScalar(parameter, 2.0));
                retained = engine.Reshape(engine.Reshape(detached, new[] { count }), shape);
                Assert.Same(detached._storage, retained._storage); // Actual view alias of a detached arena result.
                storage = retained._storage;
                GradientRecording.RetainOutput("generation", retained);
                RetainedOutputCapture<double>.PublishEager(tape);
            }
            // A later last-use return must not put transferred storage back in a free queue.
            var liveBacking = Assert.IsType<double[]>(retained.GetLiveBackingArrayOrNull());
            arena.ReturnIssued(new TensorArena.IssuedBuffer(liveBacking, -1, 0));
            for (int step = 0; step < 3; step++)
            {
                arena.Reset();
                for (int rent = 0; rent < 3; rent++)
                {
                    var scratch = Assert.IsType<Tensor<double>>(arena.TryRentTensor<double>(count, new[] { count }));
                    scratch.AsWritableSpan().Fill(9.0);
                    Assert.NotSame(detached, scratch);
                }
                Assert.Equal(shape, detached.Shape.ToArray());
                Assert.Equal(shape, retained.Shape.ToArray());
                Assert.Same(storage, retained._storage);
                Assert.All(retained.ToArray(), value => Assert.Equal(2.0, value));
                Assert.Equal(0, arena.PinnedArrayCount);
            }
            // Ordinary, unretained scratch still follows the arena's reuse contract.
            var ordinary = Assert.IsType<Tensor<double>>(arena.TryRentTensor<double>(17, new[] { 17 }));
            arena.Reset();
            Assert.Same(ordinary, arena.TryRentTensor<double>(17, new[] { 17 }));
        }
        outputs.Complete();
        Assert.Same(retained, outputs.Get("generation"));
        using (var next = TensorArena.Create())
            for (int i = 0; i < 8; i++)
                Assert.IsType<Tensor<double>>(next.TryRentTensor<double>(count, new[] { count })).AsWritableSpan().Fill(-7.0);
        Assert.Equal(shape, retained.Shape.ToArray());
        Assert.Same(storage, retained._storage);
        Assert.All(retained.ToArray(), value => Assert.Equal(2.0, value));
    }

    [Fact]
    public void Repeated_named_handoffs_do_not_accumulate_strong_arena_pins()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<double>(new[] { 1.0, 2.0 }, new[] { 2 });
        using var arena = TensorArena.Create();
        for (int step = 0; step < 16; step++)
        {
            using var outputs = RetainedOutputCapture<double>.Begin();
            using (var tape = new GradientTape<double>())
            {
                GradientRecording.RetainOutput("generation", engine.StopGradient(engine.TensorMultiplyScalar(parameter, 2.0)));
                RetainedOutputCapture<double>.PublishEager(tape);
            }
            outputs.Complete();
            Assert.Equal(new[] { 2.0, 4.0 }, outputs.Get("generation").ToArray());
            Assert.Equal(0, arena.PinnedArrayCount);
        }
    }

    [Theory]
    [InlineData("missing")]
    [InlineData("renamed")]
    [InlineData("moved")]
    [InlineData("duplicate")]
    public void Probe_checks_the_retained_output_schema_and_graph_producer(string change)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var weight = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
        var parameters = new[] { weight };
        string mode = "initial";
        Tensor<double> Objective()
        {
            var first = engine.TensorMultiplyScalar(weight, 2.0);
            var second = engine.TensorMultiplyScalar(weight, 3.0);
            if (mode != "missing")
                GradientRecording.RetainOutput(mode == "renamed" ? "other" : "generation", mode == "moved" ? second : first);
            if (mode == "duplicate") GradientRecording.RetainOutput("generation", second);
            return engine.ReduceSum(engine.TensorAdd(first, second), null);
        }
        using var recording = ClosureObjectiveGuard<double>.BeginRecording();
        CompiledTrainingPlan<double> plan;
        using (var trace = GraphMode.EnableTraining(parameters))
            plan = trace.CompileTraining(parameters, Objective());
        using (plan)
        {
            var recorded = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
            Assert.True(recorded is not null, reason);
            var guard = Assert.IsType<ClosureObjectiveGuard<double>>(recorded);
            mode = change;
            Assert.Equal(change == "duplicate" ? ClosureGuardResult.Aborted : ClosureGuardResult.Mismatched,
                guard.TryRebind(Objective, parameters, out _));
        }
    }

    [Fact]
    public void Compiled_handoff_requires_outer_success_and_expires_at_the_next_step_or_disposal()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var weight = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
        var parameters = new[] { weight };
        CompiledTrainingPlan<double> plan;
        using (var trace = GraphMode.EnableTraining(parameters))
        {
            var generated = engine.TensorMultiplyScalar(weight, 3.0);
            GradientRecording.RetainOutput("generation", generated);
            plan = trace.CompileTraining(parameters, engine.ReduceSum(generated, null));
        }
        using (plan)
        using (var outputs = RetainedOutputCapture<double>.Begin())
        {
            plan.Step();
            Assert.Throws<InvalidOperationException>(() => outputs.Get("generation"));
            // A post-step validation exception would leave this capture uncompleted and unreadable.
            outputs.Complete();
            var borrowed = outputs.Get("generation");
            Assert.Equal(6.0, borrowed.ToArray()[0]);
            Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() => outputs.Get("missing"));
            plan.Step();
            Assert.Throws<InvalidOperationException>(() => outputs.Get("generation"));
            outputs.Dispose();
            Assert.Throws<ObjectDisposedException>(() => outputs.Get("generation"));
            // Capture disposal never disposes a plan-owned tensor.
            Assert.Equal(6.0, borrowed.ToArray()[0]);
        }

        using var disposedPlanOutputs = RetainedOutputCapture<double>.Begin();
        CompiledTrainingPlan<double> secondPlan;
        using (var trace = GraphMode.EnableTraining(parameters))
        {
            var generated = engine.TensorMultiplyScalar(weight, 4.0);
            GradientRecording.RetainOutput("generation", generated);
            secondPlan = trace.CompileTraining(parameters, engine.ReduceSum(generated, null));
        }
        secondPlan.Step();
        disposedPlanOutputs.Complete();
        secondPlan.Dispose();
        Assert.Throws<ObjectDisposedException>(() => disposedPlanOutputs.Get("generation"));
    }

    [Fact]
    public void Warmup_only_and_failed_outer_invocations_cannot_publish_outputs()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        using var outputs = RetainedOutputCapture<double>.Begin();
        using (var warmup = new GradientTape<double>())
        {
            var value = engine.TensorMultiplyScalar(new Tensor<double>(new[] { 2.0 }, new[] { 1 }), 3.0);
            GradientRecording.RetainOutput("generation", value);
        }
        Assert.Throws<InvalidOperationException>(() => outputs.Complete());
        Assert.Throws<InvalidOperationException>(() => outputs.Get("generation"));
        using (var actual = new GradientTape<double>())
        {
            // A successful step with no declaration must replace, not inherit, the warm-up's map.
            RetainedOutputCapture<double>.PublishEager(actual);
        }
        outputs.Complete();
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() => outputs.Get("generation"));

        using var failed = RetainedOutputCapture<double>.Begin();
        using (var tape = new GradientTape<double>())
        {
            var value = engine.TensorMultiplyScalar(new Tensor<double>(new[] { 4.0 }, new[] { 1 }), 3.0);
            GradientRecording.RetainOutput("generation", value);
            RetainedOutputCapture<double>.PublishEager(tape);
        }
        // Simulate outer validation failing after the update: Complete is deliberately not called.
        Assert.Throws<InvalidOperationException>(() => failed.Get("generation"));
        failed.Dispose();
        Assert.Throws<ObjectDisposedException>(() => failed.Get("generation"));
    }

    [Fact]
    public void Eager_keys_are_nonempty_unique_and_ordinal_even_without_a_capture()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        using var tape = new GradientTape<double>();
        var value = engine.TensorMultiplyScalar(new Tensor<double>(new[] { 2.0 }, new[] { 1 }), 3.0);
        Assert.Throws<ArgumentException>(() => GradientRecording.RetainOutput("", value));
        GradientRecording.RetainOutput("generation", value);
        GradientRecording.RetainOutput("Generation", value);
        Assert.Throws<InvalidOperationException>(() => GradientRecording.RetainOutput("generation", value));
    }

    [Fact]
    public async Task Nested_and_parallel_captures_do_not_publish_each_others_outputs()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        static double Capture(double value)
        {
            using var outputs = RetainedOutputCapture<double>.Begin();
            using (var tape = new GradientTape<double>())
            {
                var generated = AiDotNetEngine.Current.TensorMultiplyScalar(new Tensor<double>(new[] { value }, new[] { 1 }), 2.0);
                GradientRecording.RetainOutput("generation", generated);
                RetainedOutputCapture<double>.PublishEager(tape);
            }
            outputs.Complete();
            return outputs.Get("generation").ToArray()[0];
        }
        using (var parent = RetainedOutputCapture<double>.Begin())
        {
            Assert.Equal(6.0, Capture(3.0));
            Assert.Throws<InvalidOperationException>(() => parent.Complete());
        }
        var first = Task.Run(() => Capture(5.0));
        var second = Task.Run(() => Capture(7.0));
        var values = await Task.WhenAll(first, second);
        Assert.Equal(10.0, values[0]);
        Assert.Equal(14.0, values[1]);
    }
}
