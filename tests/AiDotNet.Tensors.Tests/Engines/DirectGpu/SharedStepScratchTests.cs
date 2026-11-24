using System.Linq;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The co-owned step scratch registry (SharedStepScratch) and the pool side of it: who may take a buffer, that no owner
/// frees one another still uses, that a revoked buffer leaves every other owner without being freed (and makes them
/// recapture), and that a plan stepping nested inside another leaves co-ownership.
/// </summary>
public sealed class SharedStepScratchTests
{
    private static (MockBackendState State, IDirectGpuBackend Backend, DirectGpuEngine Direct, DirectGpuTensorEngine Engine) Create()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        var direct = new DirectGpuEngine(backend);
        return (state, backend, direct, new DirectGpuTensorEngine(direct));
    }

    [Fact]
    public void A_published_buffer_is_taken_by_another_pool_and_freed_only_by_its_last_owner()
    {
        var (state, backend, direct, engine) = Create();
        using (direct)
        using (engine)
        {
            var registry = engine.SharedStepScratch;
            var a = engine.CreateScratchPool();
            var b = engine.CreateScratchPool();
            a.Shared = registry;
            var x = a.AllocateNew(backend, 100, fullyWritten: true);   // nothing offered yet: a fresh buffer
            registry.Publish(a, backend, new[] { x });
            Assert.True(SharedStepScratch.IsShared(x));

            b.Shared = registry;
            Assert.Same(x, b.AllocateNew(backend, 80, fullyWritten: true));   // within twice the length
            Assert.Equal(1, registry.Takes);
            Assert.Same(x, Assert.Single(b.TakeTakenThisRun()));
            Assert.NotSame(x, b.AllocateNew(backend, 40, fullyWritten: true));   // less than half: not this one
            Assert.NotSame(x, a.AllocateNew(backend, 100, fullyWritten: true));  // its own owner never takes it twice

            Assert.True(b.ReleaseOwnership(x));    // a still owns it: b must not free it
            Assert.True(SharedStepScratch.IsShared(x));
            Assert.False(a.ReleaseOwnership(x));   // the last owner: the caller frees it as before
            Assert.False(SharedStepScratch.IsShared(x));
            Assert.False(StepScratchPool.IsPooled(x));
        }
    }

    [Fact]
    public void A_buffer_is_shared_only_on_its_publishing_thread()
    {
        var (_, backend, direct, engine) = Create();
        using (direct)
        using (engine)
        {
            var registry = engine.SharedStepScratch;
            var a = engine.CreateScratchPool();
            var x = backend.AllocateBuffer(64);
            registry.Publish(a, backend, new[] { x });
            IGpuBuffer? taken = null;
            var thread = new Thread(() => taken = registry.TryTake(engine.CreateScratchPool(), backend, 64));
            thread.Start();
            thread.Join();
            Assert.Null(taken);
        }
    }

    [Fact]
    public void A_revoked_buffer_leaves_the_other_owners_unfreed_and_makes_them_recapture()
    {
        var (state, backend, direct, engine) = Create();
        using (direct)
        using (engine)
        {
            var registry = engine.SharedStepScratch;
            var a = engine.CreateScratchPool();
            var b = engine.CreateScratchPool();
            var x = backend.AllocateBuffer(64);
            a.Track(x);
            a.Slots[7] = x;
            a.Keys[x] = 7;
            registry.Publish(a, backend, new[] { x });
            b.Shared = registry;
            Assert.Same(x, b.AllocateNew(backend, 64, fullyWritten: true));

            // b's run kept a value in x past its step: x becomes b's alone.
            registry.Revoke(x, b);
            Assert.Equal(1, registry.Revocations);
            Assert.False(SharedStepScratch.IsShared(x));
            Assert.DoesNotContain(x, a.AllBuffers());
            Assert.True(a.TakeRevoked());
            Assert.False(a.TakeRevoked());
            Assert.False(b.TakeRevoked());
            Assert.True(StepScratchPool.IsPooled(x));   // still b's scratch
            Assert.All(state.AllocatedBuffers, buffer => Assert.Equal(0, buffer.DisposeCount));
        }
    }

    [Fact]
    public void A_pool_stepping_nested_drops_what_others_co_own_and_never_co_owns_again()
    {
        var (state, backend, direct, engine) = Create();
        using (direct)
        using (engine)
        {
            var registry = engine.SharedStepScratch;
            var a = engine.CreateScratchPool();
            var b = engine.CreateScratchPool();
            var shared = backend.AllocateBuffer(64);
            registry.Publish(a, backend, new[] { shared });
            b.Shared = registry;
            Assert.Same(shared, b.AllocateNew(backend, 64, fullyWritten: true));
            b.Slots[1] = shared;
            b.Keys[shared] = 1;
            var own = backend.AllocateBuffer(64);   // offered by b, taken by no one
            b.Track(own);
            b.Slots[2] = own;
            b.Keys[own] = 2;
            registry.Publish(b, backend, new[] { own });

            b.LeaveSharing();
            Assert.True(b.NeverCoOwn);
            Assert.Null(b.Shared);
            Assert.DoesNotContain(shared, b.AllBuffers());   // a's to use: b allocates afresh
            Assert.Contains(own, b.AllBuffers());            // nobody else owned it: b keeps it, unshared
            Assert.False(SharedStepScratch.IsShared(own));
            Assert.True(SharedStepScratch.IsShared(shared));   // still a's co-owned scratch
            Assert.True(b.TakeRevoked());
            b.Shared = registry;
            Assert.Null(b.Shared);
            Assert.All(state.AllocatedBuffers, buffer => Assert.Equal(0, buffer.DisposeCount));
        }
    }
}
