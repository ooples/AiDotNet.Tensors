using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class GraphAttentionDeviceLifetimeTests
{
    private static object Pattern(DirectGpuTensorEngine engine, IDirectGpuBackend backend, CsrGraph graph)
        => typeof(DirectGpuTensorEngine).GetMethod("DeviceGraph", BindingFlags.NonPublic | BindingFlags.Instance)!
            .Invoke(engine, new object[] { backend, graph })!;

    [Fact]
    public void SharedGraph_HasIndependentEngineOwners_AndReleasesAtEngineDisposal()
    {
        var a = new MockBackendState();
        var b = new MockBackendState();
        var ab = MockDirectGpuBackend.Create(a);
        var bb = MockDirectGpuBackend.Create(b);
        using var ad = new DirectGpuEngine(ab);
        using var bd = new DirectGpuEngine(bb);
        using var ae = new DirectGpuTensorEngine(ad);
        using var be = new DirectGpuTensorEngine(bd);
        var graph = new CsrGraph(1, new[] { 0, 1 }, new[] { 0 });
        var first = Pattern(ae, ab, graph);
        Assert.Same(first, Pattern(ae, ab, graph));
        Assert.NotSame(first, Pattern(be, bb, graph));
        Assert.Equal(5, a.AllocatedBuffers.Count);
        Assert.Equal(5, b.AllocatedBuffers.Count);
        ae.Dispose();
        Assert.All(a.AllocatedBuffers, buffer => Assert.Equal(1, buffer.DisposeCount));
        Assert.All(b.AllocatedBuffers, buffer => Assert.Equal(0, buffer.DisposeCount));
        be.Dispose();
        Assert.All(b.AllocatedBuffers, buffer => Assert.Equal(1, buffer.DisposeCount));
        GC.KeepAlive(graph);
    }

    [Fact]
    public void PartialAllocationFailure_ReleasesBuffers_AndDoesNotPublishCacheEntry()
    {
        var state = new MockBackendState { FailIntAllocationAt = 3 };
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var graph = new CsrGraph(1, new[] { 0, 1 }, new[] { 0 });
        var ex = Assert.Throws<TargetInvocationException>(() => Pattern(engine, backend, graph));
        Assert.IsType<InvalidOperationException>(ex.InnerException);
        Assert.Equal(2, state.AllocatedBuffers.Count);
        Assert.All(state.AllocatedBuffers, buffer => Assert.Equal(1, buffer.DisposeCount));
        Pattern(engine, backend, graph);
        Assert.Equal(7, state.AllocatedBuffers.Count);
        engine.Dispose();
        Assert.All(state.AllocatedBuffers, buffer => Assert.Equal(1, buffer.DisposeCount));
        GC.KeepAlive(graph);
    }
}
