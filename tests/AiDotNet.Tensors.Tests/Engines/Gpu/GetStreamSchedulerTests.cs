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
        // discovers the contract immediately.
        var engine = new DirectGpuTensorEngine();
        Assert.Throws<System.ArgumentNullException>(() =>
            engine.GetStreamScheduler(streamPool: null!));
    }

    [Fact]
    public void CreateStreamPool_DoesNotThrow()
    {
        // The accessor must be callable from any host — CPU-only or
        // GPU-equipped. Returns null on CPU-only, a usable pool on
        // multi-stream-capable hosts.
        var engine = new DirectGpuTensorEngine();
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
        var engine = new DirectGpuTensorEngine();
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
        var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is not null && asyncBackend.SupportsMultiStream)
            return; // GPU host — covered by the scheduler test.

        var pool = engine.CreateStreamPool();
        Assert.Null(pool);
    }
}
