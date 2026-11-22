using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class InferenceWorkspaceHandoffTests
{
    [Fact]
    public void Broker_returns_only_current_queue_idle_cells_and_preserves_outstanding_loans()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        var other = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var broker = engine.CompiledWorkspaceBroker;
        var active = new Cell(32); broker.Return(backend, active);
        Assert.Same(active, broker.TryRent(backend, 32));
        var idle = new Cell(16); var foreign = new Cell(64);
        broker.Return(backend, idle); broker.Return(other, foreign);
        long foreignReturn = -1;
        var thread = new Thread(() => { lock (engine.CompiledStepLock) foreignReturn = broker.ReturnIdleCellsToBackendCache(backend); });
        thread.Start(); Assert.True(thread.Join(TimeSpan.FromSeconds(10)));
        Assert.Equal(0, foreignReturn); Assert.Equal(0, idle.Disposes);
        Assert.Equal(64, broker.ReturnIdleCellsToBackendCache(backend));
        Assert.Equal(1, idle.Disposes); Assert.Equal(0, active.Disposes); Assert.Equal(0, foreign.Disposes);
        Assert.False(StepScratchPool.IsPooled(idle));
        Assert.Equal(256, broker.RetainedBytes);
        Assert.Equal(0, broker.ReturnIdleCellsToBackendCache(backend));
        broker.Return(backend, active);
        engine.ClearActionScratchPool();
        Assert.Equal(1, idle.Disposes); Assert.Equal(1, active.Disposes); Assert.Equal(1, foreign.Disposes);
    }

    [Fact]
    public void Completed_training_returns_only_unused_private_backward_leases()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var pool = engine.CreateScratchPool();
        var obsolete = pool.Lease(0, backend, 16);
        var reused = pool.Lease(1, backend, 64);
        var pinned = pool.Lease(2, backend, 128);
        pool.EndLeases(new HashSet<IGpuBuffer> { pinned });
        pool.TrimUnleased(backend);
        Assert.Equal(0, Assert.IsType<MockGpuBuffer>(obsolete).DisposeCount);

        Assert.Same(reused, pool.Lease(1, backend, 64));
        pool.EndLeases(null);
        pool.TrimUnleased(backend);
        Assert.Equal(1, Assert.IsType<MockGpuBuffer>(obsolete).DisposeCount);
        Assert.Equal(0, Assert.IsType<MockGpuBuffer>(reused).DisposeCount);
        Assert.Equal(0, Assert.IsType<MockGpuBuffer>(pinned).DisposeCount);
        Assert.Equal(0, pool.CacheReturnedForInferenceBytes);
        pool.Release();
    }

    [Fact]
    public void Step_pool_returns_free_leases_and_keeps_pinned_slots_and_transient_storage()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var pool = engine.CreateScratchPool();
        var free = pool.Lease(0, backend, 16); pool.EndLeases(null);
        var pinned = pool.Lease(1, backend, 32); pool.EndLeases(new HashSet<IGpuBuffer> { pinned });
        var transient = backend.AllocateBuffer(8); pool.Track(transient); pool.Transient[(0, 8)] = transient;
        Assert.Equal(64, pool.ReturnIdleLeasesToBackendCache(backend));
        Assert.False(StepScratchPool.IsPooled(free));
        Assert.True(StepScratchPool.IsPooled(pinned)); Assert.True(StepScratchPool.IsPooled(transient));
        Assert.Same(pinned, pool.Slots[1]); Assert.Same(transient, pool.Transient[(0, 8)]);
        Assert.Equal(0, pool.ReturnIdleLeasesToBackendCache(backend));
        pool.Release();
    }

    [SkippableFact]
    public void Cuda_prediction_reuses_handed_off_native_backing_without_allocating_again()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "Native CUDA is required to check the device allocation counters.");
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var broker = engine.CompiledWorkspaceBroker;
        var buffer = backend.AllocateBufferUninitialized(32); IntPtr address = buffer.Handle;
        broker.Return(backend, buffer);
        bool previous = InferenceWorkspaceDemandScope.HandoffIdleWorkspaceEnabled;
        try
        {
            InferenceWorkspaceDemandScope.HandoffIdleWorkspaceEnabled = true;
            var before = GpuMemoryStats.Snapshot(backend);
            using var inference = InferenceWorkspaceDemandScope.Enter(engine);
            Assert.Equal(0, broker.RetainedBytes); Assert.Equal(128, broker.CacheReturnedForInferenceBytes);
            Assert.Equal(IntPtr.Zero, buffer.Handle);
            using var next = backend.AllocateBufferUninitialized(32);
            Assert.Equal(address, next.Handle);
            var after = GpuMemoryStats.Snapshot(backend);
            Assert.Equal(before.DeviceAllocations, after.DeviceAllocations);
            Assert.Equal(before.DeviceFrees, after.DeviceFrees);
        }
        finally { InferenceWorkspaceDemandScope.HandoffIdleWorkspaceEnabled = previous; }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Train_predict_train_preserves_unread_gradients_retained_views_and_optimizer_updates(bool inputGradients)
    {
        using var gpu = new DirectGpuTensorEngine();
        // A device trajectory: the hosted build runners have no GPU, the GPU lanes run it.
        Skip.IfNot(gpu.IsGpuAvailable, "No GPU device.");
        var previous = AiDotNetEngine.Current;
        bool handoff = InferenceWorkspaceDemandScope.HandoffIdleWorkspaceEnabled;
        bool? gradients = CompiledTrainingPlan<float>.CompiledInputGradientsForTests;
        var trajectory = typeof(InferenceWorkspaceDemandTests).GetMethod("Trajectory", BindingFlags.Static | BindingFlags.NonPublic)!;
        try
        {
            InferenceWorkspaceDemandScope.HandoffIdleWorkspaceEnabled = true;
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = inputGradients;
            var expected = (List<float[]>)trajectory.Invoke(null, new object[] { new CpuEngine(), inputGradients, 0L })!;
            var actual = (List<float[]>)trajectory.Invoke(null, new object[] { gpu, inputGradients, 0L })!;
            Assert.True(gpu.CompiledWorkspaceBroker.CacheReturnedForInferenceBytes > 0);
            Assert.Equal(expected.Count, actual.Count);
            for (int sample = 0; sample < expected.Count; sample++)
            {
                Assert.Equal(expected[sample].Length, actual[sample].Length);
                for (int i = 0; i < expected[sample].Length; i++)
                    Assert.True(Math.Abs(expected[sample][i] - actual[sample][i]) <= 2e-6f + 3e-3f * Math.Abs(expected[sample][i]));
            }
        }
        finally
        {
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = gradients;
            InferenceWorkspaceDemandScope.HandoffIdleWorkspaceEnabled = handoff;
            AiDotNetEngine.Current = previous;
        }
    }

    private sealed class Cell : IGpuBuffer, IPoolableGpuBuffer
    {
        private static int s_next;
        internal Cell(int size) { Size = size; Handle = new IntPtr(Interlocked.Increment(ref s_next)); }
        public AiDotNet.Tensors.Engines.DirectGpu.GpuElementType ElementType => AiDotNet.Tensors.Engines.DirectGpu.GpuElementType.Float32;
        public int Size { get; }
        public int Capacity => Size;
        public long SizeInBytes => (long)Size * sizeof(float);
        public IntPtr Handle { get; private set; }
        internal int Disposes { get; private set; }
        public void MarkRented(int size) => throw new InvalidOperationException();
        public void Release() => throw new InvalidOperationException("Handoff must use the existing cache-return path.");
        public void Dispose() { Disposes++; Handle = IntPtr.Zero; }
    }
}
