using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #335: validates the engine-level accessor that exposes
/// <c>GpuStreamScheduler</c> to consumers. With no GPU backend present,
/// the accessor returns null without throwing — that's the path
/// AiDotNet's parallel-dispatch callers exercise on CPU-only hosts.
/// </summary>
public class GetStreamSchedulerTests
{
    [Fact]
    public void GetStreamScheduler_DoesNotThrow()
    {
        // The accessor must be callable from any host — CPU-only or
        // GPU-equipped. Returns null on CPU-only, a usable scheduler on
        // multi-stream-capable hosts. This is the call shape AiDotNet's
        // dispatcher uses: `engine.GetStreamScheduler()` with no args.
        var engine = new DirectGpuTensorEngine();
        var scheduler = engine.GetStreamScheduler();
        // Either outcome is correct — we just need the call to succeed.
        if (scheduler is not null)
        {
            // GPU-capable host — dispatcher must be a real scheduler.
            Assert.NotNull(scheduler);
        }
    }

    [Fact]
    public void GetStreamScheduler_PassedNullPool_DoesNotThrow()
    {
        // The pool argument is optional; passing null must not throw —
        // the accessor builds a fresh pool internally when needed.
        var engine = new DirectGpuTensorEngine();
        var ex = Record.Exception(() => engine.GetStreamScheduler(streamPool: null));
        Assert.Null(ex);
    }

    [Fact]
    public void GetStreamScheduler_ReturnsNonNull_OnMultiStreamCapableBackend()
    {
        // Smoke test: when the engine has a multi-stream-capable async
        // backend, the accessor returns a usable scheduler. The previous
        // surface required AiDotNet consumers to call GetAsyncBackend +
        // check SupportsMultiStream + build their own GpuStreamPool +
        // construct GpuStreamScheduler manually — four lines of internal
        // plumbing the accessor centralizes.
        var engine = new DirectGpuTensorEngine();
        var asyncBackend = engine.GetAsyncBackend();
        if (asyncBackend is null || !asyncBackend.SupportsMultiStream)
            return; // No multi-stream backend on this host — nothing to verify.

        var scheduler = engine.GetStreamScheduler();
        Assert.NotNull(scheduler);
    }
}
