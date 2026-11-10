// Gated to non-Framework targets: the mock backend is a DispatchProxy (.NET Core 5+).
#if NETCOREAPP
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="DelegatingGpuBackend.InnerBackend"/> is how the engine's per-backend caches (the IRFFT tables) reach the
/// physical backend under a DeferredScope, whose backend is a <see cref="RecordingGpuBackend"/>. Keyed on the wrapper,
/// each scope added its own entries and engine disposal skipped the physical backend's queued release.
/// </summary>
public class DelegatingBackendUnwrapTests
{
    [Fact]
    public void InnerBackend_returns_the_wrapped_backend_through_nested_wrappers()
    {
        var physical = MockDirectGpuBackend.Create(new MockBackendState());
        var recording = new RecordingGpuBackend(physical);
        var outer = new DelegatingGpuBackend(recording);

        Assert.Same(physical, recording.InnerBackend);
        Assert.Same(recording, outer.InnerBackend);

        // The engine's unwrap loop: any depth of wrapping ends at the one physical backend.
        AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend backend = outer;
        while (backend is DelegatingGpuBackend wrapper) backend = wrapper.InnerBackend;
        Assert.Same(physical, backend);
    }

    // Every wrapper implements the index conversions by forwarding, so a recording wrapper over a backend without them
    // used to record the conversion and fail only when the graph replayed, outside the engine's capability fallback.
    [Fact]
    public void Recording_an_index_conversion_the_physical_backend_lacks_fails_at_record_time()
    {
        var physical = MockDirectGpuBackend.Create(new MockBackendState());
        var recording = new RecordingGpuBackend(new DelegatingGpuBackend(physical)) { ExecuteWhileRecording = false };
        using var graph = new AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder();
        var raw = new MockGpuBuffer(new float[4]);
        var output = new MockGpuBuffer(new float[4]);
        recording.BeginRecording(graph);
        try
        {
            Assert.Throws<AiDotNet.Tensors.Engines.DirectGpu.GpuCapabilityUnavailableException>(
                () => recording.Int32ToFloat(raw, output, 4));
            Assert.Throws<AiDotNet.Tensors.Engines.DirectGpu.GpuCapabilityUnavailableException>(
                () => recording.ScatterPositions(output, 2, 2, 4));
            Assert.Equal(0, graph.NodeCount);
        }
        finally
        {
            recording.EndRecording();
        }
    }
}
#endif
