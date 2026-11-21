using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class InferenceBufferCacheRegimeTests
{
    private sealed class Cell : IGpuBuffer, IPoolableGpuBuffer
    {
        private static int s_next;
        private readonly int _capacity;
        internal Action? OnRelease;
        internal Action? OnCapacityRead;
        internal int Releases;
        public Cell(int capacity) { _capacity = capacity; Size = capacity; Handle = (IntPtr)Interlocked.Increment(ref s_next); }
        public int Capacity { get { Interlocked.Exchange(ref OnCapacityRead, null)?.Invoke(); return _capacity; } }
        public AiDotNet.Tensors.Engines.DirectGpu.GpuElementType ElementType => AiDotNet.Tensors.Engines.DirectGpu.GpuElementType.Float32;
        public int Size { get; private set; }
        public long SizeInBytes => (long)Size * sizeof(float);
        public IntPtr Handle { get; private set; }
        public void MarkRented(int size) { Assert.InRange(size, 1, _capacity); Assert.NotEqual(IntPtr.Zero, Handle); Size = size; }
        public void Release() { Interlocked.Increment(ref Releases); Handle = IntPtr.Zero; OnRelease?.Invoke(); }
        public void Dispose() => Release();
    }

    [Fact]
    public void LargerMissChangesIdleSizeClassAndMakesTheRequestedClassReusable()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var pool = new GpuBufferPool<Cell>(16, 1024) { MaxPooledElements = 256 };
        var cells = new[] { new Cell(64), new Cell(64), new Cell(64), new Cell(64) };
        foreach (var cell in cells) pool.Return(cell);
        Assert.False(pool.TryRentForBackend(128, backend, out _));
        Assert.All(cells, cell => Assert.Equal(0, cell.Releases));
        using (InferenceWorkspaceDemandScope.Enter(engine))
        {
            Assert.False(pool.TryRentForBackend(128, backend, out _));
            Assert.All(cells, cell => Assert.Equal(1, cell.Releases));
            Assert.Equal(4, pool.InferenceRegimeNativeReleaseRequestedBuffers);
            Assert.Equal(1024, pool.InferenceRegimeNativeReleaseRequestedCapacityBytes);
            var current = new Cell(128);
            pool.Return(current);
            Assert.True(pool.TryRentForBackend(128, backend, out var rented));
            Assert.Same(current, rented);
            Assert.Equal(0, current.Releases);
            rented!.Release();
        }
    }

    [Fact]
    public void ACompatibleOrLargerIdleCellDoesNotAuthorizeClearingOtherClasses()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var pool = new GpuBufferPool<Cell>(16, 1024);
        var small = new Cell(32); var large = new Cell(256);
        pool.Return(small); pool.Return(large);
        using (InferenceWorkspaceDemandScope.Enter(engine)) Assert.False(pool.TryRentForBackend(128, backend, out _));
        Assert.Equal(0, small.Releases); Assert.Equal(0, large.Releases);
        Assert.True(pool.TryRent(256, out var retained)); Assert.Same(large, retained);
        retained!.Release();
    }

    [Fact]
    public void AnotherQueueAndAnOutstandingRentalRemainProtected()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var pool = new GpuBufferPool<Cell>(16, 1024);
        var a = GpuBufferPoolAffinity.ForNativeQueue((IntPtr)31);
        var b = GpuBufferPoolAffinity.ForNativeQueue((IntPtr)32);
        var active = new Cell(64); var idle = new Cell(32); var foreign = new Cell(16);
        pool.Return(active, a); Assert.True(pool.TryRent(64, a, out var loan));
        pool.Return(idle, a); pool.Return(foreign, b);
        using (InferenceWorkspaceDemandScope.Enter(engine)) Assert.False(pool.TryRentForBackend(128, a, backend, out _));
        Assert.Equal(1, idle.Releases); Assert.Equal(0, active.Releases); Assert.Equal(0, foreign.Releases);
        Assert.True(pool.TryRent(16, b, out var other)); Assert.Same(foreign, other);
        other!.Release(); loan!.Release();
    }

    [Fact]
    public void WrongBackendThreadMaskedChildTapeAndResidentExecutionKeepIdleCells()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        var foreign = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var pool = new GpuBufferPool<Cell>(16, 1024);
        var cell = new Cell(32); pool.Return(cell);
        using var boundary = InferenceWorkspaceDemandScope.Enter(engine);
        Assert.False(pool.TryRentForBackend(64, foreign, out _));
        using (InferenceWorkspaceDemandScope.Enter(null)) Assert.False(pool.TryRentForBackend(64, backend, out _));
        Exception? threadFailure = null;
        var thread = new Thread(() =>
        {
            try { Assert.False(pool.TryRentForBackend(64, backend, out _)); }
            catch (Exception error) { threadFailure = error; }
        });
        thread.Start(); Assert.True(thread.Join(TimeSpan.FromSeconds(5)));
        Assert.Null(threadFailure);
        using (var tape = new GradientTape<float>()) Assert.False(pool.TryRentForBackend(64, backend, out _));
        engine.SuspendActivationEviction();
        try { using (engine.EnterCompiledCapturePath()) Assert.False(pool.TryRentForBackend(64, backend, out _)); }
        finally { engine.ResumeActivationEviction(); }
        Assert.Equal(0, cell.Releases);
        Assert.False(pool.TryRentForBackend(64, backend, out _)); Assert.Equal(1, cell.Releases);
    }

    [Theory] [InlineData(false)] [InlineData(true)]
    public void NativeReleaseIsOutsideThePoolGateAndNestedClosingOrTerminalDisposalCannotReturnDetachedCells(bool terminal)
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var pool = new GpuBufferPool<Cell>(16, 1024);
        var boundary = InferenceWorkspaceDemandScope.Enter(engine);
        try
        {
            var first = new Cell(16); var second = new Cell(32); var reentrant = new Cell(8);
            int entered = 0;
            Action callback = () =>
            {
                if (Interlocked.Exchange(ref entered, 1) != 0) return;
                // A different thread can take the lifecycle gate while native release runs.
                Assert.True(Task.Run(() => pool.Return(reentrant)).Wait(TimeSpan.FromSeconds(5)));
                boundary.Dispose();
                if (terminal) pool.Dispose();
            };
            first.OnRelease = callback; second.OnRelease = callback;
            pool.Return(first); pool.Return(second);
            Assert.False(pool.TryRentForBackend(64, backend, out _));
            Assert.Equal(1, first.Releases); Assert.Equal(1, second.Releases);
            Assert.Equal(terminal ? 1 : 0, reentrant.Releases);
            Assert.False(pool.TryRentForBackend(64, backend, out _));
            Assert.Equal(2, pool.InferenceRegimeNativeReleaseRequestedBuffers);
        }
        finally { boundary.Dispose(); }
    }

    [Fact]
    public void AConcurrentSuccessfulRentalCannotBeFreedByTheLargerMiss()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var pool = new GpuBufferPool<Cell>(16, 1024);
        var active = new Cell(64); var idle = new Cell(32);
        pool.Return(active); pool.Return(idle);
        Cell? loan = null;
        active.OnCapacityRead = () => Assert.True(Task.Run(() => Assert.True(pool.TryRent(64, out loan))).Wait(TimeSpan.FromSeconds(5)));
        using (InferenceWorkspaceDemandScope.Enter(engine)) Assert.False(pool.TryRentForBackend(128, backend, out _));
        Assert.Same(active, loan); Assert.Equal(0, active.Releases); Assert.NotEqual(IntPtr.Zero, active.Handle);
        Assert.Equal(1, idle.Releases); loan!.Release();
    }

    [Fact]
    public void AReleaseFailureStillProcessesEveryDetachedIdleCellAndPreservesTheOriginalError()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var pool = new GpuBufferPool<Cell>(16, 1024);
        var failed = new Cell(16) { OnRelease = () => throw new InvalidOperationException("native release fixture") };
        var other = new Cell(32); pool.Return(failed); pool.Return(other);
        using (InferenceWorkspaceDemandScope.Enter(engine))
            Assert.Equal("native release fixture", Assert.Throws<InvalidOperationException>(() => pool.TryRentForBackend(64, backend, out _)).Message);
        Assert.Equal(1, failed.Releases); Assert.Equal(1, other.Releases); Assert.Equal(0, pool.PooledCount);
    }
}
