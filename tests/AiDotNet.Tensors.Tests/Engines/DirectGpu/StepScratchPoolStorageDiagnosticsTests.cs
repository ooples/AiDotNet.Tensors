using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class StepScratchPoolStorageDiagnosticsTests
{
    [Fact]
    public void SnapshotClassifiesUniqueHandlesWithoutDownloadingOrChangingOwnership()
    {
        var state = new MockBackendState();
        state.ThrowOnCall["DownloadBuffer"] = new InvalidOperationException("A metadata snapshot cannot download.");
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var pool = engine.CreateScratchPool();
        var forward = new MockGpuBuffer(new float[10]);
        var backward = new MockGpuBuffer(new float[20]);
        var transient = new MockGpuBuffer(new float[30]);
        var spanning = new MockGpuBuffer(new float[50]);
        pool.Slots[0] = forward;
        pool.Slots[5L << 32] = backward;
        pool.Transient[(0, 30)] = transient;
        pool.Transient[(1, 10)] = new BorrowedAlias(forward); // Distinct wrapper, same native allocation.
        pool.Slots[2L << 32] = spanning;
        Assert.True(pool.SpanLease(2L << 32));
        pool.Begin(5);
        var free = pool.Lease(7L << 32, backend, 40);
        pool.EndLeases(null);
        var activeLease = pool.Lease(8L << 32, backend, 80);
        int allocationsBefore = state.AllocatedBuffers.Count;

        var snapshot = pool.SnapshotStorage(forwardActionCount: 4);

        Assert.Equal((1, 40L), snapshot["slots_forward"]);
        Assert.Equal((1, 80L), snapshot["slots_backward"]);
        Assert.Equal((1, 120L), snapshot["transient"]);
        Assert.Equal((1, 160L), snapshot["free"]);
        Assert.Equal((1, 320L), snapshot["leases"]);
        Assert.Equal((1, 200L), snapshot["spanning"]);
        Assert.Equal(6, snapshot.Values.Sum(row => row.Count));
        Assert.Equal(920L, snapshot.Values.Sum(row => row.Bytes));
        Assert.Equal(allocationsBefore, state.AllocatedBuffers.Count);
        Assert.Same(forward, pool.Slots[0]);
        Assert.Same(backward, pool.Slots[5L << 32]);
        Assert.True(pool.IsSpanLeased(spanning));
        Assert.Equal(0, forward.DisposeCount);
        Assert.Equal(0, backward.DisposeCount);
        Assert.Equal(0, transient.DisposeCount);
        Assert.Equal(0, spanning.DisposeCount);
        Assert.NotEqual(IntPtr.Zero, free.Handle);
        Assert.NotEqual(IntPtr.Zero, activeLease.Handle);
    }

    private sealed class BorrowedAlias : IGpuBuffer
    {
        private readonly IGpuBuffer _owner;
        internal BorrowedAlias(IGpuBuffer owner) => _owner = owner;
        public int Size => _owner.Size;
        public long SizeInBytes => _owner.SizeInBytes;
        public IntPtr Handle => _owner.Handle;
        public void Dispose() { }
    }
}
