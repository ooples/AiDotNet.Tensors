using System;
using System.Collections.Generic;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class CompiledTransientWorkspaceBrokerTests
{
    [Fact]
    public void CompletedPlansReuseBackingWithoutNativeRequestsAndNestedCheckoutsStayDistinct()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var broker = engine.CompiledWorkspaceBroker;
        var a = engine.CreateScratchPool();
        var b = engine.CreateScratchPool();
        var previous = DirectGpuTensorEngine.EnterScratchPool(a);
        engine.SuspendActivationEviction();
        try
        {
            using (engine.EnterCompiledCapturePath())
            {
                Assert.True(engine.ResidentStepActive);
                a.WorkspaceBroker = broker;
                b.WorkspaceBroker = broker;
                engine.SetCurrentScratchAction(0);
                var outer = engine.RentActionScratchOrAllocate(backend, 32, fullyWritten: true);
                // A nested call cannot take backing the outer caller has not returned.
                DirectGpuTensorEngine.EnterScratchPool(b);
                engine.SetCurrentScratchAction(0);
                var inner = engine.RentActionScratchOrAllocate(backend, 32, fullyWritten: true);
                Assert.NotSame(outer, inner);
                Return(b, backend, broker);
                DirectGpuTensorEngine.EnterScratchPool(a);
                engine.SetCurrentScratchAction(0);
                Assert.Same(outer, engine.RentActionScratchOrAllocate(backend, 32, fullyWritten: true));
                Return(a, backend, broker);
                int allocations = state.AllocatedBuffers.Count;
                for (int step = 0; step < 8; step++)
                {
                    var pool = (step & 1) == 0 ? a : b;
                    DirectGpuTensorEngine.EnterScratchPool(pool);
                    engine.SetCurrentScratchAction(0);
                    engine.RentActionScratchOrAllocate(backend, 32, fullyWritten: true);
                    Return(pool, backend, broker);
                }
                Assert.Equal(allocations, state.AllocatedBuffers.Count);
                Assert.All(state.AllocatedBuffers, buffer => Assert.Equal(0, buffer.DisposeCount));
                Assert.True(broker.ReuseCount >= 8);
            }
            DirectGpuTensorEngine.RestoreScratchPool(null);
            engine.ClearActionScratchPool();
            Assert.All(state.AllocatedBuffers, buffer => Assert.Equal(1, buffer.DisposeCount));
            Assert.Null(broker.TryRent(backend, 32)); // A released HIP/mock wrapper may retain a nonzero handle.
        }
        finally
        {
            DirectGpuTensorEngine.RestoreScratchPool(previous);
            engine.ResumeActivationEviction();
        }
    }

    [Fact]
    public void AThreadHopDoesNotReuseAnotherThreadsUnfinishedNativeQueue()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var broker = engine.CompiledWorkspaceBroker;
        var buffer = backend.AllocateBuffer(16);
        broker.Return(backend, buffer);
        IGpuBuffer? foreign = null;
        var thread = new Thread(() => { lock (engine.CompiledStepLock) foreign = broker.TryRent(backend, 16); });
        thread.Start();
        Assert.True(thread.Join(TimeSpan.FromSeconds(10)));
        Assert.Null(foreign);
        Assert.Same(buffer, broker.TryRent(backend, 16));
        broker.Return(backend, buffer);
        engine.ClearActionScratchPool();
    }

    [Fact]
    public void AnIncompleteBorrowRemainsPrivateAfterTheBodyUnwinds()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var broker = engine.CompiledWorkspaceBroker;
        var first = backend.AllocateBuffer(32);
        broker.Return(backend, first);
        var failedPool = engine.CreateScratchPool();
        var nextPool = engine.CreateScratchPool();
        var previous = DirectGpuTensorEngine.EnterScratchPool(failedPool);
        engine.SuspendActivationEviction();
        try
        {
            using (engine.EnterCompiledCapturePath())
            {
                Assert.True(engine.ResidentStepActive);
                failedPool.WorkspaceBroker = broker;
                engine.SetCurrentScratchAction(0);
                Assert.Same(first, engine.RentActionScratchOrAllocate(backend, 32, fullyWritten: true));
                // The plan's failure path ends its action/context without returning its unfinished backing.
                failedPool.WorkspaceBroker = null;
                engine.EndScratchAction();
                DirectGpuTensorEngine.EnterScratchPool(nextPool);
                nextPool.WorkspaceBroker = broker;
                engine.SetCurrentScratchAction(0);
                Assert.NotSame(first, engine.RentActionScratchOrAllocate(backend, 32, fullyWritten: true));
                Assert.Equal(2, state.AllocatedBuffers.Count);
            }
        }
        finally
        {
            DirectGpuTensorEngine.RestoreScratchPool(previous);
            engine.ClearActionScratchPool();
            engine.ResumeActivationEviction();
        }
    }

    [Fact]
    public void DetachingADeferredInternalOutputInvalidatesAllViewsAndNextBindingPublishesFreshData()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var tensor = Tensor<float>.CreateGpuResident(new[] { 4 }, backend.DeviceType);
        var alias = tensor.Reshape(new[] { 2, 2 });
        var first = backend.AllocateBuffer(new[] { 1f, 2f, 3f, 4f });
        engine.BindResidentBuffer(tensor, first, backend);
        engine.DetachCompiledWorkspaceTensor(tensor);
        Assert.Throws<InvalidOperationException>(() => alias.ToArray());
        var next = backend.AllocateBuffer(new[] { 5f, 6f, 7f, 8f });
        engine.BindResidentBuffer(tensor, next, backend);
        Assert.Equal(new[] { 5f, 6f, 7f, 8f }, alias.ToArray());
        Assert.Equal(1, state.DownloadBufferCalls);
    }

    private static void Return(StepScratchPool pool, IDirectGpuBackend backend, CompiledTransientWorkspaceBroker broker)
    {
        var candidates = new HashSet<IGpuBuffer>();
        pool.CollectTransferableForwardSlots(1, _ => true, candidates);
        Assert.Equal(128, pool.TransferForwardSlots(candidates, backend, broker, out int count));
        Assert.Equal(1, count);
    }
}
