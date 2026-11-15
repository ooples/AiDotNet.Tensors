using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class StepScratchPoolLifetimeTests
{
    [Fact]
    public void ReplacingActivationEntries_DoesNotReleaseBorrowedScratch()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var pool = engine.CreateScratchPool();
        var previous = DirectGpuTensorEngine.EnterScratchPool(pool);
        engine.SuspendActivationEviction();
        try
        {
            using (engine.EnterCompiledCapturePath())
            {
                engine.SetCurrentScratchAction(0);
                var buffer = engine.RentActionScratchOrAllocate(backend, 5, fullyWritten: true);
                var entry = new ActivationCacheEntry(buffer, new[] { 5 }, 1, backend, 20);
                entry.Dispose();
                Assert.Equal(0, state.AllocatedBuffers[0].DisposeCount);
                engine.SetCurrentScratchAction(0);
                Assert.Same(buffer, engine.RentActionScratchOrAllocate(backend, 5, fullyWritten: true));

                var ordinary = backend.AllocateBuffer(5);
                new ActivationCacheEntry(ordinary, new[] { 5 }, 2, backend, 20).Dispose();
                Assert.Equal(1, state.AllocatedBuffers[1].DisposeCount);
            }
            engine.ClearActionScratchPool();
            Assert.Equal(1, state.AllocatedBuffers[0].DisposeCount);
        }
        finally
        {
            DirectGpuTensorEngine.RestoreScratchPool(previous);
            engine.ClearActionScratchPool();
            engine.ResumeActivationEviction();
        }
    }

    [Fact]
    public void TerminalCleanup_ReleasesEveryPlansBuffersOutsideAStep()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var pools = new[] { engine.CreateScratchPool(), engine.CreateScratchPool() };
        var previous = DirectGpuTensorEngine.EnterScratchPool(null);
        engine.SuspendActivationEviction();
        try
        {
            using (engine.EnterCompiledCapturePath())
            {
                foreach (var pool in pools)
                {
                    DirectGpuTensorEngine.EnterScratchPool(pool);
                    engine.SetCurrentScratchAction(0);
                    var buffer = engine.RentActionScratchOrAllocate(backend, 5, fullyWritten: true);
                    Assert.True(StepScratchPool.IsPooled(buffer));
                    engine.SetCurrentScratchAction(0);
                    Assert.Same(buffer, engine.RentActionScratchOrAllocate(backend, 5, fullyWritten: true));
                }
            }
            Assert.Equal(2, state.AllocatedBuffers.Count);
            Assert.NotSame(state.AllocatedBuffers[0], state.AllocatedBuffers[1]);
            DirectGpuTensorEngine.RestoreScratchPool(null);

            engine.ClearActionScratchPool();
            engine.ClearActionScratchPool();

            Assert.All(state.AllocatedBuffers, buffer =>
            {
                Assert.Equal(1, buffer.DisposeCount);
                Assert.False(StepScratchPool.IsPooled(buffer));
            });
            Assert.All(pools, pool => Assert.Empty(pool.Slots));
        }
        finally
        {
            DirectGpuTensorEngine.RestoreScratchPool(previous);
            engine.ClearActionScratchPool();
            engine.ResumeActivationEviction();
        }
    }
}
