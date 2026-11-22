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
        public DirectGpuTensorEngine Consumer { get; }
        public DirectGpuTensorEngine Placer { get; }
        public EnginePair(DirectGpuTensorEngine consumer, DirectGpuTensorEngine placer) { Consumer = consumer; Placer = placer; }
        public void Dispose() { Placer.Dispose(); Consumer.Dispose(); }
    }

    // The default engine plus an OpenCL engine on a DIFFERENT backend type; skips when the machine cannot make both.
    private static EnginePair CreateEngines()
    {
        var consumer = new DirectGpuTensorEngine();
        if (!consumer.IsGpuAvailable) { consumer.Dispose(); Skip.If(true, "No native GPU backend."); }
        DirectGpuTensorEngine? placer = null;
        try { placer = new DirectGpuTensorEngine(new DirectGpuEngine(new OpenClBackend())); }
        catch (Exception) { placer = null; }
        if (placer is null || !placer.IsGpuAvailable
            || placer.QueryBackend?.GetType() == consumer.QueryBackend?.GetType())
        {
            placer?.Dispose();
            consumer.Dispose();
            Skip.If(true, "Needs a second, different GPU backend (OpenCL) next to the default one.");
        }
        return new EnginePair(consumer, placer!);
    }

    private static Tensor<float> Seeded(int rows, int cols, int seed)
    {
        var random = new Random(seed);
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
            var error = Assert.Throws<InvalidOperationException>(() => Compile(engines.Consumer, input, w1, w2));
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

    [SkippableFact]
    public void Raw_int32_indices_placed_elsewhere_reach_the_consumer_as_indices()
    {
        using var engines = CreateEngines();
        var placerBackend = engines.Placer.QueryBackend!;
        // Raw int32 device storage, as an argmax on the placer's backend produces it.
        var indices = Tensor<int>.FromGpuBuffer(placerBackend, placerBackend.AllocateIntBuffer(new[] { 3, 0, 2, 1 }),
            new[] { 4 }, bufferContainsRawInt32: true);
        var source = new Tensor<float>(new[] { 10f, 20f, 30f, 40f }, new[] { 4 });
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        // Before the fix the raw-int32 marker was read before the migration, which re-uploads the indices as floats:
        // index 3 arrived as 0x40400000.
        Assert.Equal(new[] { 40f, 10f, 30f, 20f }, engines.Consumer.TensorTake(source, indices).ToArray());
        Assert.Equal(new[] { 3, 0, 2, 1 }, indices.ToArray());
    }

    [SkippableFact]
    public void A_tensor_whose_engine_was_disposed_moves_from_its_host_copy()
    {
        using var engines = CreateEngines();
        var retiredEngine = new DirectGpuTensorEngine(new DirectGpuEngine(new OpenClBackend()), ownsDirectGpu: true);
        var retiredBackend = retiredEngine.QueryBackend;
        var a = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        PlaceOn(retiredEngine, a);
        retiredEngine.Dispose();
        Skip.IfNot(DirectGpuTensorEngine.IsRetiredBackend(retiredBackend), "That engine did not own its backend.");
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        // Its device copy is gone with the backend; the host copy is current, so the op reads it and nothing touches
        // the freed backend (before the fix: a read-back and a free through it).
        var zeros = new Tensor<float>(new float[4], new[] { 4 });
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, engines.Consumer.TensorAdd(a, zeros).ToArray());
    }

    [SkippableFact]
    public void Two_threads_migrating_one_tensor_move_it_once_and_both_read_it()
    {
        using var engines = CreateEngines();
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        for (int round = 0; round < 20; round++)
        {
            var shared = new Tensor<float>(new[] { 1f + round, 2f, 3f, 4f }, new[] { 4 });
            PlaceOn(engines.Placer, shared);
            long before = CrossBackendTensors.MigrationCount;
            var results = new float[2][];
            using var start = new System.Threading.Barrier(2);
            var threads = Enumerable.Range(0, 2).Select(t => new System.Threading.Thread(() =>
            {
                var zeros = new Tensor<float>(new float[4], new[] { 4 });
                start.SignalAndWait();
                results[t] = engines.Consumer.TensorAdd(shared, zeros).ToArray();
            })).ToArray();
            foreach (var thread in threads) thread.Start();
            foreach (var thread in threads) thread.Join();
            Assert.Equal(1, CrossBackendTensors.MigrationCount - before);
            Assert.All(results, r => Assert.Equal(new[] { 1f + round, 2f, 3f, 4f }, r));
            Assert.NotSame(engines.Placer.QueryBackend, shared._gpuBackend);
        }
    }

    // A result on an engine that owns its backend (it registers its results as owning their buffers).
    private static Tensor<float> OwnedResult(DirectGpuTensorEngine owner)
    {
        var a = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var b = new Tensor<float>(new[] { 10f, 20f, 30f, 40f }, new[] { 4 });
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = owner;
        try
        {
            var produced = owner.TensorAdd(a, b);
            _ = produced.ToArray();
            return produced;
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [SkippableFact]
    public void Migrating_through_a_non_owning_view_frees_the_result_buffer_it_leaves()
    {
        using var engines = CreateEngines();
        using var owner = new DirectGpuTensorEngine(new DirectGpuEngine(new OpenClBackend()), ownsDirectGpu: true);
        var produced = OwnedResult(owner);
        var old = produced._gpuBuffer;
        Assert.NotNull(old);
        Assert.True(owner.OwnsDeviceResult(produced.DataVector, owner.QueryBackend!));
        var view = produced.Reshape(2, 2);   // shares the result's storage, owns nothing itself
        Assert.False(view._ownsGpuBuffer);
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        var zeros = new Tensor<float>(new float[4], new[] { 2, 2 });
        Assert.Equal(new[] { 11f, 22f, 33f, 44f }, engines.Consumer.TensorAdd(view, zeros).ToArray());
        // Whole-vector aliases share one binding: the result moved with the view, so nothing still reads the old
        // buffer, and it is freed rather than leaked to the finalizer.
        Assert.Same(view._gpuBuffer, produced._gpuBuffer);
        Assert.Equal(System.IntPtr.Zero, old!.Handle);
        Assert.Equal(new[] { 11f, 22f, 33f, 44f }, produced.ToArray());
        produced.Dispose();   // the storage owns the new buffer: this alias must not free it under the view
        Assert.Equal(new[] { 11f, 22f, 33f, 44f }, view.ToArray());
    }

    [SkippableFact]
    public void Migrating_the_last_alias_of_a_result_frees_its_buffer()
    {
        using var engines = CreateEngines();
        using var owner = new DirectGpuTensorEngine(new DirectGpuEngine(new OpenClBackend()), ownsDirectGpu: true);
        var produced = OwnedResult(owner);
        var old = produced._gpuBuffer;
        Assert.NotNull(old);
        Assert.True(owner.OwnsDeviceResult(produced.DataVector, owner.QueryBackend!));
        var view = produced.Reshape(2, 2);
        Assert.False(view._ownsGpuBuffer);
        produced.Dispose();   // the view is now the storage's only alias, still bound to the result's buffer
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        var zeros = new Tensor<float>(new float[4], new[] { 2, 2 });
        Assert.Equal(new[] { 11f, 22f, 33f, 44f }, engines.Consumer.TensorAdd(view, zeros).ToArray());
        Assert.Equal(System.IntPtr.Zero, old!.Handle);   // freed, not leaked to the finalizer
    }

    [SkippableFact]
    public void Placing_again_after_coming_home_frees_the_buffer_it_replaces()
    {
        using var engines = CreateEngines();
        var t = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        PlaceOn(engines.Placer, t);
        var first = t._gpuBuffer;
        Assert.NotNull(first);
        t.Cpu();
        PlaceOn(engines.Placer, t);
        Assert.NotSame(first, t._gpuBuffer);
        Assert.Equal(System.IntPtr.Zero, first!.Handle);   // before the fix: still owned, every round trip leaked one
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, t.ToArray());
    }

    [SkippableFact]
    public void Indices_rewritten_through_the_host_reach_the_consumer_as_written()
    {
        using var engines = CreateEngines();
        var indices = new Tensor<int>(new[] { 0, 1, 2, 3 }, new[] { 4 });
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engines.Placer;
        try { indices.Gpu(); }
        finally { AiDotNetEngine.Current = previous; }
        Assert.Same(engines.Placer.QueryBackend, indices._gpuBackend);
        // The next batch, written through the host span as a data loader does (the version does not move).
        var span = indices.AsWritableSpan();
        span[0] = 3; span[1] = 2; span[2] = 1; span[3] = 0;
        var source = new Tensor<float>(new[] { 10f, 20f, 30f, 40f }, new[] { 4 });
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        Assert.Equal(new[] { 40f, 30f, 20f, 10f }, engines.Consumer.TensorTake(source, indices).ToArray());
    }

    [SkippableFact]
    public void A_complex_result_placed_elsewhere_keeps_its_imaginary_parts()
    {
        using var engines = CreateEngines();
        var signal = new Tensor<float>(new[] { 1f, 2f, 0f, -1f, 3f, 0.5f, -2f, 1f }, new[] { 8 });
        var reference = new CpuEngine().NativeComplexFFT(signal).ToArray();
        Tensor<Complex<float>> spectrum;
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engines.Placer;
        try { spectrum = engines.Placer.NativeComplexFFT(signal); }
        finally { AiDotNetEngine.Current = previous; }
        Skip.IfNot(spectrum._gpuBufferIsSplitComplex, "The placer did not keep the spectrum resident.");
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        // The consumer reads it as split planes (an inverse would do; the forward of the spectrum is as good a reader).
        var again = engines.Consumer.NativeComplexFFTComplex(spectrum).ToArray();
        var expected = new CpuEngine().NativeComplexFFTComplex(new Tensor<Complex<float>>(reference, new[] { 8 })).ToArray();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Real, again[i].Real, 3);
            Assert.Equal(expected[i].Imaginary, again[i].Imaginary, 3);
        }
        spectrum.Cpu();   // a read-back from wherever it is resident now (before the fix: real-only floats over the host)
        var back = spectrum.ToArray();
        for (int i = 0; i < reference.Length; i++) Assert.Equal(reference[i].Imaginary, back[i].Imaginary, 4);
    }

    [SkippableFact]
    public void An_inner_lease_returned_elsewhere_does_not_cost_the_outer_op_its_backend()
    {
        using var engines = CreateEngines();
        var outerGate = new GpuOpGate();
        var innerGate = new GpuOpGate();
        using var outer = outerGate.Enter(engines.Consumer.QueryBackend);
        var middle = outerGate.Enter(engines.Consumer.QueryBackend);
        var inner = innerGate.Enter(engines.Placer.QueryBackend);
        var other = new System.Threading.Thread(() => inner.Dispose());   // returned from another thread
        other.Start();
        other.Join();
        middle.Dispose();
        // Before the fix the middle lease saw the inner entry installed, skipped its restore, and the rest of the outer
        // op ran with no op backend: the foreign-buffer guard off.
        Assert.Same(engines.Consumer.QueryBackend, GpuOpGate.CurrentOpBackend);
    }

    [SkippableFact]
    public void Compile_re_places_a_leaf_whose_device_copy_is_current()
    {
        using var engines = CreateEngines();
        var input = Seeded(4, 3, 4);
        var w1 = Seeded(3, 5, 5);
        var w2 = Seeded(5, 2, 6);
        var expected = input.ToArray();
        PlaceOn(engines.Placer, input);   // a frozen leaf: nothing writes it after the placement
        using var policy = new PolicyScope(CrossBackendTensorPolicy.Migrate);
        using var plan = Compile(engines.Consumer, input, w1, w2);
        Assert.Same(engines.Consumer.QueryBackend, input._gpuBackend);   // still resident, now on the plan's backend
        Assert.Equal(expected, input.ToArray());
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
}
