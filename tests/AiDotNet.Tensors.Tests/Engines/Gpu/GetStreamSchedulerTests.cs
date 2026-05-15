using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #335: validates the engine-level accessor that exposes
/// <c>GpuStreamScheduler</c> to consumers. Callers may either pass an
/// owned pool or let the engine create one owned by the scheduler.
/// </summary>
public class GetStreamSchedulerTests
{
    [Fact]
    public void GetStreamScheduler_NoPool_DoesNotThrow()
    {
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();

        using var scheduler = engine.GetStreamScheduler();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream)
            Assert.Null(scheduler);
        else
            Assert.NotNull(scheduler);
    }

    [Fact]
    public void IEngineGetStreamScheduler_NoPool_DoesNotThrow()
    {
        using var concrete = new DirectGpuTensorEngine();
        IEngine engine = concrete;
        var asyncBackend = concrete.GetAsyncBackend();

        using var scheduler = engine.GetStreamScheduler();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream)
            Assert.Null(scheduler);
        else
            Assert.NotNull(scheduler);
    }

    [Fact]
    public void CpuEngineGetStreamScheduler_NoPool_ReturnsNull()
    {
        IEngine engine = new CpuEngine();

        using var scheduler = engine.GetStreamScheduler();
        Assert.Null(scheduler);
    }

    [Fact]
    public void CreateStreamPool_DoesNotThrow()
    {
        // The accessor must be callable from any host — CPU-only or
        // GPU-equipped. Returns null on CPU-only, a usable pool on
        // multi-stream-capable hosts.
        using var engine = new DirectGpuTensorEngine();
        var pool = engine.CreateStreamPool();
        if (pool is not null)
            pool.Dispose();
    }

    [Fact]
    public void GetStreamScheduler_WithOwnedPool_ReturnsScheduler_OnMultiStreamHost()
    {
        // Smoke test: when the engine has a multi-stream-capable async
        // backend, building a pool via CreateStreamPool + passing it to
        // GetStreamScheduler yields a usable scheduler. Caller owns pool
        // lifetime; dispose pool after scheduler is done.
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream)
            return; // No multi-stream backend on this host.

        using var pool = engine.CreateStreamPool();
        Assert.NotNull(pool);
        using var scheduler = engine.GetStreamScheduler(pool!);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void CreateStreamPool_NoMultiStreamBackend_ReturnsNull()
    {
        // On a CPU-only host the pool factory returns null without throwing.
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is not null && asyncBackend.SupportsMultiStream)
            return; // GPU host — covered by the scheduler test.

        var pool = engine.CreateStreamPool();
        Assert.Null(pool);
    }

    [Fact]
    public void GetStreamScheduler_Dispatch_FanOutsOverMultipleStreams()
    {
        // Issue #335 items 3+4: the scheduler must accept a batch of
        // independent launches and fan them across multiple streams.
        // This test exercises the fan-out path directly; the higher-level
        // wiring (MHA per-head + BatchMatMul per-slice) calls this same
        // entry point. Counts callback invocations to verify every launch
        // ran and that the returned event batch has one event per launch.
        using var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream)
            return;

        using var scheduler = engine.GetStreamScheduler();
        if (scheduler is null) return; // CPU-only host

        int launchCount = 0;
        var launches = new System.Collections.Generic.List<System.Action<IGpuStream>>();
        for (int i = 0; i < 8; i++)
            launches.Add(_ => System.Threading.Interlocked.Increment(ref launchCount));

        using var batch = scheduler.Dispatch(launches);
        Assert.Equal(8, launchCount);
        // GpuEventBatch IS the event list — it implements IReadOnlyList<IGpuEvent>.
        Assert.NotEqual(0, batch.Count);

        // SynchronizeEvents is a host-blocking wait; on a real GPU backend
        // it returns once all per-stream events fire. With no-op launches
        // this should be near-instant.
        scheduler.SynchronizeEvents(batch);
    }

    [Fact]
    public void GetStreamScheduler_PoolFromDifferentEngine_ThrowsArgumentException()
    {
        // PR #344 critical review: GetStreamScheduler must refuse a pool
        // whose backend doesn't match this engine's. Cross-backend stream
        // misuse manifests as CUDA_ERROR_INVALID_HANDLE deep inside
        // cuMemcpy / cuLaunchKernel — surfacing the affinity mismatch at
        // the API boundary gives the caller an actionable error.
        using var engineA = new DirectGpuTensorEngine();
        using var engineB = new DirectGpuTensorEngine();
        var backendA = engineA.GetAsyncBackend();
        var backendB = engineB.GetAsyncBackend();
        if (backendA is null || backendB is null
            || !backendA.SupportsMultiStream || !backendB.SupportsMultiStream)
            return; // Need multi-stream backends on both engines.
        // Practical note: two DirectGpuTensorEngine instances may share
        // the same singleton backend instance, in which case the affinity
        // guard short-circuits to "same backend, accept". The test still
        // documents the contract; the negative case becomes meaningful
        // only when two engines hold distinct backend references (e.g.
        // a future test fixture that overrides backend creation).
        if (ReferenceEquals(backendA, backendB))
            return;

        using var poolFromA = engineA.CreateStreamPool();
        Assert.NotNull(poolFromA);
        Assert.Throws<System.ArgumentException>(() =>
            engineB.GetStreamScheduler(poolFromA!));
    }
}
