using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #335: validates the engine-level accessor that exposes
/// <c>GpuStreamScheduler</c> to consumers. Pool ownership is explicit —
/// caller builds via <c>CreateStreamPool()</c> and disposes it
/// alongside the scheduler.
/// </summary>
public class GetStreamSchedulerTests
{
    [Fact]
    public void GetStreamScheduler_NullPoolArg_Throws()
    {
        // Explicit pool ownership: passing null is a usage error, not a
        // valid fallback. Throws ArgumentNullException so the caller
        // discovers the contract immediately. DirectGpuTensorEngine is
        // IDisposable — wrap in `using` so the test doesn't leak native
        // backend handles on failure paths.
        using var engine = new DirectGpuTensorEngine();
        Assert.Throws<System.ArgumentNullException>(() =>
            engine.GetStreamScheduler(streamPool: null!));
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
        var scheduler = engine.GetStreamScheduler(pool!);
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
