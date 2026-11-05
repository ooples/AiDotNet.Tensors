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
}
#endif
