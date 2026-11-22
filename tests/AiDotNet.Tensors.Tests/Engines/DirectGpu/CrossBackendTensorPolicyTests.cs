using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A tensor whose GPU buffer another backend placed (Tensor.Gpu() while a different engine was
/// AiDotNetEngine.Current) must never reach this engine's kernels as a raw address: an OpenCL handle read as a CUDA
/// device pointer was a sticky CUDA-700 that failed every later GPU test in the process. The engine applies
/// AiDotNetEngine.CrossBackendPolicy instead: migrate once (warning once per tensor and backend pair), migrate
/// silently, or throw.
/// </summary>
// AiDotNetEngine.CrossBackendPolicy and Logger are process-wide: these tests switch them (to Throw, among others), so
// they run with no other test in parallel rather than only serialized against the other GPU tests.
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class CrossBackendPolicyCollection
{
    public const string Name = "CrossBackendPolicySerial";
}

[Collection(CrossBackendPolicyCollection.Name)]
public sealed class CrossBackendTensorPolicyTests
{
    private sealed class EnginePair : IDisposable
    {
        private readonly DirectGpuEngine _placerDevice;
        public DirectGpuTensorEngine Consumer { get; }
        public DirectGpuTensorEngine Placer { get; }

        public EnginePair(DirectGpuTensorEngine consumer, DirectGpuTensorEngine placer, DirectGpuEngine placerDevice)
        {
            Consumer = consumer;
            Placer = placer;
            _placerDevice = placerDevice;
        }

        // The tensor engine borrows the device engine it was given: the backend (its context and compiled kernels)
        // is released by disposing that one, after the tensor engine that uses it.
        public void Dispose()
        {
            Placer.Dispose();
            _placerDevice.Dispose();
            Consumer.Dispose();
        }
    }

    // The default engine plus an OpenCL engine on its own backend instance (its own context, so its buffers are
    // foreign to the default engine's even when that one is OpenCL too); skips when the machine cannot make both.
    private static EnginePair CreateEngines()
    {
        var consumer = new DirectGpuTensorEngine();
        if (!consumer.IsGpuAvailable) { consumer.Dispose(); Skip.If(true, "No native GPU backend."); }
        DirectGpuEngine? device = null;
        DirectGpuTensorEngine? placer = null;
        try
        {
            device = new DirectGpuEngine(new OpenClBackend());
            placer = new DirectGpuTensorEngine(device);
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException or InvalidOperationException
                                       or NotSupportedException)
        {
            // No OpenCL runtime or device here: the test is skipped below.
        }
        if (device is not null && placer is not null && placer.IsGpuAvailable && placer.QueryBackend is { } placed
            && consumer.QueryBackend is { } consuming && !CrossBackendTensors.SameAddressSpace(placed, consuming))
            return new EnginePair(consumer, placer, device);
        placer?.Dispose();
        device?.Dispose();
        consumer.Dispose();
        throw new SkipException("Needs a second GPU backend instance (OpenCL) next to the default one.");
    }

    private static Tensor<float> Seeded(int rows, int cols, int seed)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        return new Tensor<float>(
            Enumerable.Range(0, rows * cols).Select(_ => (float)(random.NextDouble() * 0.2 - 0.1)).ToArray(),
            new[] { rows, cols });
    }

    // Places the tensors on the placer's backend, the way Tensor.Gpu() does while that engine is current.
    private static void PlaceOn(DirectGpuTensorEngine placer, params Tensor<float>[] tensors)
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = placer;
        try { foreach (var tensor in tensors) tensor.Gpu(); }
        finally { AiDotNetEngine.Current = previous; }
        foreach (var tensor in tensors)
            Assert.Same(placer.QueryBackend, tensor._gpuBackend);
    }

    private static CompiledTrainingPlan<float> Compile(DirectGpuTensorEngine engine, Tensor<float> input,
        Tensor<float> w1, Tensor<float> w2)
    {
        try
        {
            using var scope = GraphMode.Enable();
            var hidden = engine.TensorMatMul(input, w1);
            var prediction = engine.TensorMatMul(hidden, w2);
            engine.ReduceSum(engine.TensorMultiply(prediction, prediction), null);
            return scope.CompileTraining(new[] { w1, w2 });
        }
        finally { GraphMode.SetCurrent(null); }
    }

    private sealed class PolicyScope : IDisposable
    {
        private readonly Action<string>? _previousLogger = AiDotNetEngine.Logger;
        private readonly CrossBackendTensorPolicy _previousPolicy = AiDotNetEngine.CrossBackendPolicy;
        public List<string> MigrationLines { get; } = new();

        public PolicyScope(CrossBackendTensorPolicy policy)
        {
            AiDotNetEngine.CrossBackendPolicy = policy;
            AiDotNetEngine.Logger = line =>
            {
                if (line.Contains("Migrated a tensor")) lock (MigrationLines) MigrationLines.Add(line);
            };
        }

        public void Dispose()
        {
            AiDotNetEngine.Logger = _previousLogger;
            AiDotNetEngine.CrossBackendPolicy = _previousPolicy;
        }
    }

    [SkippableFact]
    public void MigrateWarn_moves_parameters_once_at_compile_and_warns_once_per_tensor()
    {
        using var engines = CreateEngines();
        var input = Seeded(32, 64, 1);
        var w1 = Seeded(64, 64, 2);
        var w2 = Seeded(64, 64, 3);
        PlaceOn(engines.Placer, w1, w2);
        using var policy = new PolicyScope(CrossBackendTensorPolicy.MigrateWarn);

        long before = CrossBackendTensors.MigrationCount;
        using var plan = Compile(engines.Consumer, input, w1, w2);
        // Moved before anything ran: the step never meets a foreign buffer, so whole-step capture stays possible.
        Assert.Equal(2, CrossBackendTensors.MigrationCount - before);
        Assert.NotSame(engines.Placer.QueryBackend, w1._gpuBackend);
        Assert.NotSame(engines.Placer.QueryBackend, w2._gpuBackend);

        float first = plan.Step().GetFlat(0);
        plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-3f);
        float last = first;
        for (int i = 0; i < 10; i++) last = plan.Step().GetFlat(0);

        Assert.Equal(2, CrossBackendTensors.MigrationCount - before);   // once: never re-read per step
        Assert.Equal(2, policy.MigrationLines.Count);                     // one line per tensor
        Assert.All(policy.MigrationLines, line => Assert.Contains("[64, 64]", line));
        Assert.True(last < first * 0.5f, $"training must proceed on the migrated parameters: first={first}, last={last}.");
    }

    [SkippableFact]
    public void Migrate_trains_identically_to_parameters_that_were_never_placed_elsewhere()
    {
        using var engines = CreateEngines();
        float[] Train(bool placeElsewhere)
        {
            var input = Seeded(32, 64, 1);
            var w1 = Seeded(64, 64, 2);
            var w2 = Seeded(64, 64, 3);
            if (placeElsewhere) PlaceOn(engines.Placer, w1, w2);
            using var plan = Compile(engines.Consumer, input, w1, w2);
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-3f);
            var losses = new float[6];
            for (int i = 0; i < losses.Length; i++) losses[i] = plan.Step().GetFlat(0);
            return losses;
        }

        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        var reference = Train(placeElsewhere: false);
        var migrated = Train(placeElsewhere: true);
        Assert.Equal(reference, migrated);       // the migration copies the same values, bit for bit
        Assert.Empty(policy.MigrationLines);     // Migrate is silent
    }

    [SkippableFact]
    public void Throw_reports_the_mismatch_at_compile_and_leaves_the_gpu_usable()
    {
        using var engines = CreateEngines();
        var input = Seeded(32, 64, 1);
        var w1 = Seeded(64, 64, 2);
        var w2 = Seeded(64, 64, 3);
        PlaceOn(engines.Placer, w1, w2);
        using (new PolicyScope(CrossBackendTensorPolicy.Throw))
        {
            // The dedicated rejection type (an InvalidOperationException to callers that catch that).
            InvalidOperationException error =
                Assert.Throws<CrossBackendTensorException>(() => Compile(engines.Consumer, input, w1, w2));
            Assert.Contains("CrossBackendPolicy", error.Message);
            Assert.Same(engines.Placer.QueryBackend, w1._gpuBackend);   // nothing was moved
        }

        // No foreign pointer reached a kernel, so the consumer's context is healthy: an ordinary op still runs.
        var a = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var b = new Tensor<float>(new[] { 4f, 5f, 6f }, new[] { 3 });
        Assert.Equal(new[] { 5f, 7f, 9f }, engines.Consumer.TensorAdd(a, b).ToArray());
    }

    [SkippableFact]
    public void Eager_op_on_tensors_placed_elsewhere_reads_their_values()
    {
        using var engines = CreateEngines();
        var a = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var b = new Tensor<float>(new[] { 10f, 20f, 30f, 40f }, new[] { 4 });
        PlaceOn(engines.Placer, a, b);
        using var policy = new PolicyScope(CrossBackendTensorPolicy.MigrateWarn);
        Assert.Equal(new[] { 11f, 22f, 33f, 44f }, engines.Consumer.TensorAdd(a, b).ToArray());
        Assert.Equal(2, policy.MigrationLines.Count);
    }

    [SkippableFact]
    public void Deferred_result_of_another_backend_is_materialized_before_its_buffer_is_dropped()
    {
        using var engines = CreateEngines();
        var a = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var b = new Tensor<float>(new[] { 10f, 20f, 30f, 40f }, new[] { 4 });
        var c = new Tensor<float>(new[] { 100f, 200f, 300f, 400f }, new[] { 4 });
        Tensor<float> produced;
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engines.Placer;
        try { produced = engines.Placer.TensorAdd(a, b); }   // a device result whose host copy is a pending download
        finally { AiDotNetEngine.Current = previous; }
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        Assert.Equal(new[] { 111f, 222f, 333f, 444f }, engines.Consumer.TensorAdd(produced, c).ToArray());
        Assert.Equal(new[] { 11f, 22f, 33f, 44f }, produced.ToArray());
    }

    [SkippableFact]
    public void External_input_keeps_host_writes_made_after_another_backend_placed_it()
    {
        using var engines = CreateEngines();
        var input = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        PlaceOn(engines.Placer, input);
        // The caller writes the next batch through the host, as a training loop does for its input tensor.
        var span = input.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = 50f + i;
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        using (engines.Consumer.EnterGpuOp()) engines.Consumer.EnsureResidentInput(input);
        var zeros = new Tensor<float>(new float[4], new[] { 4 });
        Assert.Equal(new[] { 50f, 51f, 52f, 53f }, engines.Consumer.TensorAdd(input, zeros).ToArray());
    }

    [SkippableFact]
    public void Partial_view_placed_elsewhere_migrates_instead_of_recursing()
    {
        using var engines = CreateEngines();
        var whole = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, new[] { 2, 4 });
        var view = whole.Slice(1);
        Assert.False(view.CoversWholeVector);   // the view's binding is its own, not its storage's
        PlaceOn(engines.Placer, view);
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        var zeros = new Tensor<float>(new float[4], new[] { 4 });
        // Before the fix the kept view binding was judged foreign again on every lookup: a stack overflow.
        Assert.Equal(new[] { 5f, 6f, 7f, 8f }, engines.Consumer.TensorAdd(view, zeros).ToArray());
        Assert.NotSame(engines.Placer.QueryBackend, view._gpuBackend);
    }

    [SkippableFact]
    public void Compile_keeps_host_writes_to_an_input_another_backend_placed()
    {
        using var engines = CreateEngines();
        var input = Seeded(4, 3, 1);
        var w1 = Seeded(3, 5, 2);
        var w2 = Seeded(5, 2, 3);
        PlaceOn(engines.Placer, input);
        // The next batch, written through the host after the placement: the stale device copy must not come back.
        var span = input.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = 0.5f + i;
        var expected = span.ToArray();
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        using var plan = Compile(engines.Consumer, input, w1, w2);
        Assert.Equal(expected, input.ToArray());
        Assert.NotSame(engines.Placer.QueryBackend, input._gpuBackend);
    }

    [SkippableFact]
    public void A_wrapper_around_the_op_backend_is_not_foreign()
    {
        using var engines = CreateEngines();
        var backend = engines.Consumer.QueryBackend!;
        var wrapper = new AiDotNet.Tensors.Engines.Gpu.DelegatingGpuBackend(backend);
        Assert.True(CrossBackendTensors.SameAddressSpace(backend, wrapper));
        Assert.True(CrossBackendTensors.SameAddressSpace(wrapper, backend));
        Assert.False(CrossBackendTensors.SameAddressSpace(engines.Placer.QueryBackend!, wrapper));
    }

    [SkippableFact]
    public void A_lease_returned_from_another_thread_leaves_no_op_backend_behind()
    {
        using var engines = CreateEngines();
        var gate = new GpuOpGate();
        var lease = gate.Enter(engines.Consumer.QueryBackend);
        Assert.Same(engines.Consumer.QueryBackend, GpuOpGate.CurrentOpBackend);
        var other = new System.Threading.Thread(() => lease.Dispose());   // an async op finishing elsewhere
        other.Start();
        other.Join();
        Assert.Null(GpuOpGate.CurrentOpBackend);
    }

    // BatchMatMul answers any exception from its GPU route with the CPU engine: the policy rejection must not be one.
    [SkippableFact]
    public void Throw_reaches_the_caller_through_an_op_that_falls_back_to_the_cpu()
    {
        using var engines = CreateEngines();
        var a = Seeded(4, 4, 1).Reshape(1, 4, 4);
        var b = Seeded(4, 4, 2).Reshape(1, 4, 4);
        PlaceOn(engines.Placer, a);
        using (new PolicyScope(CrossBackendTensorPolicy.Throw))
        {
            var error = Assert.Throws<CrossBackendTensorException>(() => engines.Consumer.BatchMatMul(a, b));
            Assert.Contains("CrossBackendPolicy", error.Message);
            Assert.Same(engines.Placer.QueryBackend, a._gpuBackend);   // nothing was moved
        }
        // The rejection does not outlive its op: the next op on this thread runs.
        var x = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var y = new Tensor<float>(new[] { 3f, 4f }, new[] { 2 });
        Assert.Equal(new[] { 4f, 6f }, engines.Consumer.TensorAdd(x, y).ToArray());
    }

    // The same gate again after its lease was returned from another thread: the entry left on this thread is stale, and
    // the op starting here is the outermost one. A rejection raised and handled inside it is rethrown when IT ends -
    // not held for an enclosing op that does not exist, to surface in whatever op the thread runs next.
    [SkippableFact]
    public void A_stale_entry_on_the_same_gate_is_not_an_enclosing_op()
    {
        using var engines = CreateEngines();
        var backend = engines.Consumer.QueryBackend;
        var gate = new GpuOpGate();
        var first = gate.Enter(backend);
        var other = new System.Threading.Thread(() => first.Dispose());
        other.Start();
        other.Join();

        Assert.Throws<CrossBackendTensorException>(() =>
        {
            using (gate.Enter(backend))
            {
                Assert.Same(backend, GpuOpGate.CurrentOpBackend);
                _ = CrossBackendTensors.Reject("raised and handled inside the op");
            }
        });
        Assert.Null(GpuOpGate.CurrentOpBackend);
        var x = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var y = new Tensor<float>(new[] { 3f, 4f }, new[] { 2 });
        Assert.Equal(new[] { 4f, 6f }, engines.Consumer.TensorAdd(x, y).ToArray());
    }

    // A result a recording-only deferred scope has not run is unwritten memory: another engine may not migrate it.
    [SkippableFact]
    public void An_unexecuted_deferred_result_is_not_migrated()
    {
        using var engines = CreateEngines();
        var input = new Tensor<float>(new[] { 0f, 0f }, new[] { 2 });
        var independent = new Tensor<float>(new[] { 0f, 0f }, new[] { 2 });
        PlaceOn(engines.Placer, independent);
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engines.Placer;
        try
        {
            var scope = engines.Placer.BeginDeferredScope();
            Skip.If(scope is null, "Backend has no deferred execution");
            using (scope)
            using (new PolicyScope(CrossBackendTensorPolicy.Migrate))
            {
                var recorded = engines.Placer.TensorSigmoid(input);
                Skip.If(recorded._gpuBuffer is null, "The op did not record a device result.");
                var error = Assert.Throws<CrossBackendTensorException>(() => engines.Consumer.TensorSigmoid(recorded));
                Assert.Contains("not executed", error.Message);
                // A tensor the scope did not produce still migrates.
                Assert.Equal(new[] { 0.5f, 0.5f }, engines.Consumer.TensorSigmoid(independent).ToArray());
                scope.Execute();
                Assert.Equal(new[] { 0.5f, 0.5f }, recorded.ToArray());
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    // The recording scope is an ENCLOSING one: a nested scope on the consumer's backend is current when the consumer
    // meets the outer scope's unexecuted result.
    [SkippableFact]
    public void An_enclosing_scopes_unexecuted_result_is_not_migrated()
    {
        using var engines = CreateEngines();
        var input = new Tensor<float>(new[] { 0f, 0f }, new[] { 2 });
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engines.Placer;
        try
        {
            var outer = engines.Placer.BeginDeferredScope();
            Skip.If(outer is null, "Backend has no deferred execution");
            using (outer)
            using (new PolicyScope(CrossBackendTensorPolicy.Migrate))
            {
                var recorded = engines.Placer.TensorSigmoid(input);
                Skip.If(recorded._gpuBuffer is null, "The op did not record a device result.");
                var inner = engines.Consumer.BeginDeferredScope();
                Skip.If(inner is null, "The consumer's backend has no deferred execution");
                using (inner)
                {
                    var error = Assert.Throws<CrossBackendTensorException>(() => engines.Consumer.TensorSigmoid(recorded));
                    Assert.Contains("not executed", error.Message);
                }
                outer.Execute();
                Assert.Equal(new[] { 0.5f, 0.5f }, recorded.ToArray());
                // Executed: the result now migrates like any other tensor, with its values (sigmoid(0.5)).
                var migrated = engines.Consumer.TensorSigmoid(recorded).ToArray();
                Assert.Equal(2, migrated.Length);
                Assert.All(migrated, value => Assert.Equal(0.62245935f, value, 5));
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    // The graph Compile returns can be run by the caller: a successful run writes the recorded results too.
    [SkippableFact]
    public void A_compiled_graph_run_by_the_caller_marks_its_results_executed()
    {
        using var engines = CreateEngines();
        var input = new Tensor<float>(new[] { 0f, 0f }, new[] { 2 });
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engines.Placer;
        try
        {
            var scope = engines.Placer.BeginDeferredScope();
            Skip.If(scope is null, "Backend has no deferred execution");
            using (scope)
            {
                var recorded = engines.Placer.TensorSigmoid(input);
                var buffer = recorded._gpuBuffer;
                Skip.If(buffer is null, "The op did not record a device result.");
                var graph = scope.Compile();
                Assert.True(AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder.IsUnexecutedWrite(buffer));
                graph.Execute(engines.Placer.QueryBackend!);
                Assert.False(AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder.IsUnexecutedWrite(buffer));
                Assert.Equal(new[] { 0.5f, 0.5f }, recorded.ToArray());
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    // Clearing the builder discards what was recorded: those results never run, whatever is recorded and run after.
    [SkippableFact]
    public void A_result_discarded_by_clearing_the_builder_stays_unexecuted()
    {
        using var engines = CreateEngines();
        var input = new Tensor<float>(new[] { 0f, 0f }, new[] { 2 });
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engines.Placer;
        try
        {
            var scope = engines.Placer.BeginDeferredScope() as AiDotNet.Tensors.Engines.Gpu.DeferredScope;
            Skip.If(scope is null, "Backend has no deferred execution");
            using (scope)
            {
                var discarded = engines.Placer.TensorSigmoid(input);
                var discardedBuffer = discarded._gpuBuffer;
                Skip.If(discardedBuffer is null, "The op did not record a device result.");
                scope.GraphBuilder.Clear();
                var kept = engines.Placer.TensorSigmoid(input);
                var keptBuffer = kept._gpuBuffer;
                Skip.If(keptBuffer is null, "The op did not record a device result.");
                scope.Execute();
                Assert.False(AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder.IsUnexecutedWrite(keptBuffer));
                Assert.True(AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder.IsUnexecutedWrite(discardedBuffer));
                Assert.Equal(new[] { 0.5f, 0.5f }, kept.ToArray());
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    // A slice of a slice that was uploaded on its own must not read the packed parent buffer as its own upload.
    [SkippableFact]
    public void A_view_of_a_packed_partial_view_migrates_its_own_elements()
    {
        using var engines = CreateEngines();
        var whole = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var middle = whole.Slice(0, 1, 3);                  // [2, 3]
        Assert.False(middle.CoversWholeVector);
        PlaceOn(engines.Placer, middle);                    // packed upload of [2, 3]
        var last = middle.Slice(0, 1, 2);                   // [3]
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        var zero = new Tensor<float>(new float[1], new[] { 1 });
        Assert.Equal(new[] { 3f }, engines.Consumer.TensorAdd(last, zero).ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, whole.ToArray());
    }

    // A rejection whose op returned its lease from another thread never reached that op's rethrow.
    [SkippableFact]
    public void A_rejection_left_pending_by_an_op_that_ended_elsewhere_does_not_surface_in_the_next_op()
    {
        using var engines = CreateEngines();
        var gate = new GpuOpGate();
        var lease = gate.Enter(engines.Consumer.QueryBackend);
        _ = CrossBackendTensors.Reject("left pending");
        var other = new System.Threading.Thread(() => lease.Dispose());
        other.Start();
        other.Join();
        var x = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var y = new Tensor<float>(new[] { 3f, 4f }, new[] { 2 });
        Assert.Equal(new[] { 4f, 6f }, engines.Consumer.TensorAdd(x, y).ToArray());
    }

    // A raw int32 buffer's words are integers: read as floats and converted, 7 would come home as 9.8e-45 -> 0.
    [SkippableFact]
    public void Raw_integer_tensor_placed_elsewhere_comes_home_with_its_integers()
    {
        using var engines = CreateEngines();
        var placer = engines.Placer.QueryBackend!;
        var values = new[] { 7, -3, 1_000_000, 42 };
        Tensor<int> indices;
        using (engines.Placer.EnterGpuOp())
            indices = Tensor<int>.FromGpuBuffer(placer, placer.AllocateIntBuffer(values), new[] { 4 },
                bufferContainsRawInt32: true);
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        CrossBackendTensors.ResolveIfForeign(indices, engines.Consumer.QueryBackend!);
        Assert.NotSame(placer, indices._gpuBackend);
        Assert.Equal(values, indices.ToArray());
    }

    // A raw int64 buffer holds integer words. Re-placing it goes through the float upload, where 16777217 is not
    // representable: the tensor comes home exact and stays on the host.
    [SkippableFact]
    public void Raw_int64_tensor_placed_elsewhere_keeps_its_exact_values()
    {
        using var engines = CreateEngines();
        Skip.If(engines.Placer.QueryBackend is null || engines.Consumer.QueryBackend is null, "No backend.");
        var placer = engines.Placer.QueryBackend;
        var consumer = engines.Consumer.QueryBackend;
        if (placer is null || consumer is null) return;
        var values = new[] { 16777217L, -16777219L, 9_007_199_254_740_993L, 42L };
        var words = new float[values.Length * 2];
        Buffer.BlockCopy(values, 0, words, 0, values.Length * sizeof(long));
        Tensor<long> ids;
        using (engines.Placer.EnterGpuOp())
            ids = Tensor<long>.FromGpuBuffer(placer, placer.AllocateBuffer(words), new[] { values.Length },
                bufferContainsRawInt64: true);
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        CrossBackendTensors.ResolveIfForeign(ids, consumer);
        Assert.Null(ids._gpuBuffer);   // not re-placed through the float upload
        Assert.Equal(values, ids.ToArray());
        Assert.Equal(values, ids.Cpu().ToArray());
    }

    // A transposed view placed on its own holds its elements in LOGICAL order in a buffer as long as the storage: it
    // must be read back as the view, not copied over the storage as if it were the parent's buffer.
    [SkippableFact]
    public void Transposed_view_placed_on_its_own_migrates_in_logical_order()
    {
        using var engines = CreateEngines();
        var whole = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 });
        var transposed = whole.Transpose();
        Skip.If(transposed.CoversWholeVector, "The transpose is not a strided view here.");
        PlaceOn(engines.Placer, transposed);
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        var zeros = new Tensor<float>(new float[6], new[] { 3, 2 });
        Assert.Equal(new[] { 1f, 4f, 2f, 5f, 3f, 6f }, engines.Consumer.TensorAdd(transposed, zeros).ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, whole.ToArray());
    }

    // The parent's host was written after the view borrowed its buffer: the view's superseded device copy must not
    // come back over the newer host values when the view migrates.
    [SkippableFact]
    public void A_view_whose_borrowed_buffer_is_superseded_migrates_the_current_host_values()
    {
        using var engines = CreateEngines();
        var whole = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, new[] { 2, 4 });
        PlaceOn(engines.Placer, whole);
        var view = whole.Slice(1);
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        var zeros8 = new Tensor<float>(new float[8], new[] { 2, 4 });
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, engines.Consumer.TensorAdd(whole, zeros8).ToArray());
        whole[1, 0] = 100f;   // host write after the parent left the placer
        var zeros = new Tensor<float>(new float[4], new[] { 4 });
        Assert.Equal(new[] { 100f, 6f, 7f, 8f }, engines.Consumer.TensorAdd(view, zeros).ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 100f, 6f, 7f, 8f }, whole.ToArray());
    }

    // A view taken of a placed tensor borrows the WHOLE storage's buffer: its own elements are not that buffer's first.
    [SkippableFact]
    public void View_borrowing_its_parents_buffer_migrates_its_own_elements()
    {
        using var engines = CreateEngines();
        var whole = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, new[] { 2, 4 });
        Tensor<float> produced;
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engines.Placer;
        try { produced = engines.Placer.TensorAdd(whole, whole); }   // device result, host copy pending
        finally { AiDotNetEngine.Current = previous; }
        var view = produced.Slice(1);
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        var zeros = new Tensor<float>(new float[4], new[] { 4 });
        Assert.Equal(new[] { 10f, 12f, 14f, 16f }, engines.Consumer.TensorAdd(view, zeros).ToArray());
        Assert.Equal(new[] { 2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f }, produced.ToArray());
    }

    [SkippableFact]
    public void An_engine_is_found_by_its_unwrapped_backend()
    {
        using var engines = CreateEngines();
        var backend = CrossBackendTensors.Unwrap(engines.Placer.QueryBackend!);
        Assert.Same(engines.Placer, DirectGpuTensorEngine.EngineOwning(backend));
        Assert.Same(engines.Consumer,
            DirectGpuTensorEngine.EngineOwning(CrossBackendTensors.Unwrap(engines.Consumer.QueryBackend!)));
    }

    // An input its owner updates on the device (a running statistic) is asked for as host-authoritative by a plan that
    // merely reads it: the device value must still come home.
    [SkippableFact]
    public void Device_authoritative_input_is_read_back_even_when_asked_for_as_host_authoritative()
    {
        using var engines = CreateEngines();
        var state = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        PlaceOn(engines.Placer, state);
        var placer = engines.Placer.QueryBackend!;
        using (engines.Placer.EnterGpuOp())
        {
            placer.UploadBufferInPlace(new[] { 9f, 8f, 7f, 6f }, state._gpuBuffer!);   // the owner's in-place update
            engines.Placer.NoteBoundDeviceAuthoritative(state, state._gpuBuffer!);
        }
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        CrossBackendTensors.ResolveIfForeign(state, engines.Consumer.QueryBackend!, hostAuthoritative: true);
        Assert.Equal(new[] { 9f, 8f, 7f, 6f }, state.ToArray());
    }

    // Another backend's op on a plan's parameter releases the buffer the plan's optimizer was configured with.
    [SkippableFact]
    public void Plan_keeps_training_after_another_backend_used_its_parameter()
    {
        using var engines = CreateEngines();
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        float[] Train(bool interfere)
        {
            var input = Seeded(32, 64, 1);
            var w1 = Seeded(64, 64, 2);
            var w2 = Seeded(64, 64, 3);
            using var plan = Compile(engines.Consumer, input, w1, w2);
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-3f);
            var losses = new List<float>();
            for (int i = 0; i < 8; i++)
            {
                if (interfere && i == 4)
                {
                    var zeros = new Tensor<float>(new float[64 * 64], new[] { 64, 64 });
                    var bound = w1._gpuBuffer;
                    _ = engines.Placer.TensorAdd(w1, zeros).ToArray();   // the placer's op takes w1 off this backend
                    Assert.NotSame(bound, w1._gpuBuffer);
                }
                losses.Add(plan.Step().GetFlat(0));
            }
            Assert.NotSame(engines.Placer.QueryBackend, w1._gpuBackend);
            return losses.ToArray();
        }

        float[] undisturbed = Train(interfere: false);
        float[] disturbed = Train(interfere: true);
        for (int i = 0; i < undisturbed.Length; i++)
            Assert.True(Math.Abs(undisturbed[i] - disturbed[i]) <= 1e-4f * Math.Max(1f, Math.Abs(undisturbed[i])),
                $"step {i}: {disturbed[i]} after the parameter moved, {undisturbed[i]} without.");
        Assert.True(disturbed[7] < disturbed[0], "training must keep reducing the loss.");
    }

    [Fact]
    public void Policy_rejects_values_outside_the_enum()
    {
        var previous = AiDotNetEngine.CrossBackendPolicy;
        try
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => AiDotNetEngine.CrossBackendPolicy = (CrossBackendTensorPolicy)7);
            Assert.Equal(previous, AiDotNetEngine.CrossBackendPolicy);
            Assert.Equal(CrossBackendTensorPolicy.MigrateWarn, default(CrossBackendTensorPolicy));
        }
        finally { AiDotNetEngine.CrossBackendPolicy = previous; }
    }

#if NETCOREAPP
    // The policy on two backends that are not devices at all (the mock is a DispatchProxy): the paths below need no
    // GPU, so they are measured where the unit lane runs.
    private static Tensor<float> PlacedOnMock(IDirectGpuBackend owner, params float[] values)
        => Tensor<float>.FromGpuBuffer(owner, new MockGpuBuffer(values), new[] { values.Length });

    [Fact]
    public void Throw_names_a_backend_that_cannot_name_itself_by_its_type()
    {
        var ownerState = new MockBackendState();
        ownerState.ThrowOnCall["get_BackendName"] = new InvalidOperationException("the device is gone");
        var owner = MockDirectGpuBackend.Create(ownerState);
        var consumer = MockDirectGpuBackend.Create(new MockBackendState());
        var tensor = PlacedOnMock(owner, 1f, 2f);
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Throw);

        var error = Assert.Throws<CrossBackendTensorException>(() => CrossBackendTensors.Resolve(tensor, owner, consumer));
        Assert.Contains("A tensor of shape [2] is on the ", error.Message);
        Assert.DoesNotContain("the device is gone", error.Message);
        Assert.Same(owner, tensor._gpuBackend);
    }

    [Fact]
    public void A_tensor_asked_for_inside_another_backends_op_leaves_its_owner()
    {
        var owner = MockDirectGpuBackend.Create(new MockBackendState());
        var consumer = MockDirectGpuBackend.Create(new MockBackendState());
        var tensor = PlacedOnMock(owner, 1f, 2f);
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        var gate = new GpuOpGate();
        using (gate.Enter(consumer))
        {
            var buffer = tensor.TryGetGpuBuffer();
            Assert.NotSame(owner, tensor._gpuBackend);
            Assert.True(buffer is null || ReferenceEquals(tensor._gpuBackend, consumer));
        }
        Assert.Equal(new[] { 1f, 2f }, tensor.ToArray());
    }

    [Fact]
    public void A_read_back_the_owner_fails_is_reported_as_the_migration_failing()
    {
        var ownerState = new MockBackendState();
        ownerState.ThrowOnCall["DownloadBuffer"] = new InvalidOperationException("device lost");
        var owner = MockDirectGpuBackend.Create(ownerState);
        var consumer = MockDirectGpuBackend.Create(new MockBackendState());
        var tensor = PlacedOnMock(owner, 1f, 2f);
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);

        var error = Assert.Throws<InvalidOperationException>(() => CrossBackendTensors.Resolve(tensor, owner, consumer));
        Assert.Contains("Could not read this tensor back", error.Message);
        Assert.IsType<InvalidOperationException>(error.InnerException);
    }
#endif
}
