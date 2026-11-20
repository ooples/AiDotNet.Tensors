using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class SharedBackwardScratchTests
{
    [Fact]
    public void CompletedBackwardLoansAreReusedAcrossPlansWithoutAnotherAllocation()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var broker = engine.CompiledWorkspaceBroker;
        var a = engine.CreateScratchPool();
        var b = engine.CreateScratchPool();
        a.WorkspaceBroker = b.WorkspaceBroker = broker;
        a.ShareBackwardScratch = b.ShareBackwardScratch = true;
        var buffer = a.Lease(10, backend, 256);
        a.EndLeases(null);
        Assert.Equal(buffer.SizeInBytes, a.ReturnIdleBackwardScratch(backend, broker));
        int allocated = state.AllocatedBuffers.Count;
        for (int step = 0; step < 8; step++)
        {
            var pool = (step & 1) == 0 ? b : a;
            Assert.Same(buffer, pool.Lease(10, backend, 256));
            pool.EndLeases(null);
            pool.ReturnIdleBackwardScratch(backend, broker);
        }
        Assert.Equal(allocated, state.AllocatedBuffers.Count);
        Assert.True(StepScratchPool.IsPooled(buffer));
        engine.ClearActionScratchPool();
        Assert.All(state.AllocatedBuffers, item => Assert.Equal(1, item.DisposeCount));
    }

    [Fact]
    public void OutstandingLeasesPinnedSlotsAndTransientBackingStayPrivate()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var broker = engine.CompiledWorkspaceBroker;
        var pool = engine.CreateScratchPool();
        var pinned = pool.Lease(1, backend, 64);
        Assert.Equal(0, pool.ReturnIdleBackwardScratch(backend, broker));
        pool.EndLeases(new HashSet<IGpuBuffer> { pinned });
        var transient = backend.AllocateBuffer(64);
        pool.Track(transient);
        pool.Transient[(0, 64)] = transient;
        Assert.Equal(0, pool.ReturnIdleBackwardScratch(backend, broker));
        Assert.Null(broker.TryRent(backend, 64));
        Assert.Same(pinned, pool.Slots[1]);
        Assert.Same(transient, pool.Transient[(0, 64)]);
    }

    [Fact]
    public void DisabledBorrowDoesNotTakeAnIdleBrokersBacking()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var broker = engine.CompiledWorkspaceBroker;
        var parked = backend.AllocateBuffer(64);
        broker.Return(backend, parked);
        var pool = engine.CreateScratchPool();
        pool.WorkspaceBroker = broker;
        Assert.NotSame(parked, pool.Lease(1, backend, 64));
        Assert.Same(parked, broker.TryRent(backend, 64));
    }
}
