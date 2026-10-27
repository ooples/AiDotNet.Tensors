using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class ResidentHostMirrorRangeTests
{
    [Fact]
    public void CoveringBuffer_RequiresTheSameStorageAndBackend_AndHonorsDetachment()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var source = new Tensor<float>(new float[6], new[] { 6 });
        using var view = new CpuEngine().TensorNarrow(source, 0, 2, 3);
        var array = source.GetBackingArrayForCacheLookupUnsafe()!;
        using var unrelated = new Tensor<float>(array, new[] { 6 });
        using var buffer = new MockGpuBuffer(new float[6]);
        source._gpuBackend = backend;
        source._gpuBuffer = buffer;
        Assert.Same(array, ResidentHostMirror.Attach(source));

        Assert.True(ResidentHostMirror.TryFindCovering(view, array, 2, 3, backend, out var found, out var offset));
        Assert.Same(buffer, found);
        Assert.Equal(2, offset);
        Assert.False(ResidentHostMirror.TryFindCovering(unrelated, array, 2, 3, backend, out _, out _));
        var otherBackend = MockDirectGpuBackend.Create(new MockBackendState());
        Assert.False(ResidentHostMirror.TryFindCovering(view, array, 2, 3, otherBackend, out _, out _));
        Assert.False(ResidentHostMirror.TryFindCovering(view, array, 4, 3, backend, out _, out _));
        Assert.False(ResidentHostMirror.TryFindCovering(view, array, -1, 3, backend, out _, out _));

        ResidentHostMirror.Detach(source);

        Assert.False(ResidentHostMirror.TryFindCovering(view, array, 2, 3, backend, out found, out offset));
        Assert.Null(found);
        Assert.Equal(0, offset);
        Assert.Equal(0, state.DownloadBufferCalls);
    }

    [Fact]
    public void AnUnregisteredHostTensor_HasNoCoveringDeviceBuffer()
    {
        using var tensor = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        var array = tensor.GetBackingArrayForCacheLookupUnsafe()!;

        Assert.Null(ResidentHostMirror.Attach(tensor));
        Assert.False(ResidentHostMirror.TryFindCovering(tensor, array, 0, 2, backend, out var buffer, out var offset));
        Assert.Null(buffer);
        Assert.Equal(0, offset);
        ResidentHostMirror.Detach(tensor);
        Assert.Equal(new[] { 1f, 2f }, tensor.ToArray());
    }
}
