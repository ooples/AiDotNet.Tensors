using System;
using System.Collections.Concurrent;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
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
    public void RepeatedAttach_AllocatesNothing_AndDownloadsTheReboundBufferOnce()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var tensor = new Tensor<float>(new float[2], new[] { 2 });
        using var original = new MockGpuBuffer(new[] { 1f, 2f });
        using var rebound = new MockGpuBuffer(new[] { 3f, 4f });
        tensor._gpuBackend = backend;
        tensor._gpuBuffer = original;
        var array = tensor.GetBackingArrayForCacheLookupUnsafe()!;
        ResidentHostMirror.Attach(tensor);
        for (int i = 0; i < 100; i++) ResidentHostMirror.Attach(tensor);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++) ResidentHostMirror.Attach(tensor);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);

        // Membership follows the tensor, not its old device buffer. Reattaching must neither duplicate
        // its download nor leave the prior buffer authoritative after an optimizer swaps buffers.
        tensor._gpuBuffer = rebound;
        Assert.True(ResidentHostMirror.ArmDownload(tensor));
        Assert.Equal(new[] { 3f, 4f }, tensor.ToArray());
        Assert.Equal(1, state.DownloadBufferCalls);
        Assert.Equal(new[] { 3f, 4f }, tensor.ToArray());
        Assert.Equal(1, state.DownloadBufferCalls);
        ResidentHostMirror.Detach(tensor);
        Assert.False(ResidentHostMirror.TryFindCovering(tensor, array, 0, 2, backend, out _, out _));
    }

    [Fact]
    public void RepeatedParameterCommit_AllocatesNothing_AndPreservesFreshHostReads()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var parameter = new Tensor<float>(new float[2], new[] { 2 });
        using var buffer = new MockGpuBuffer(new[] { 7f, 9f });
        for (int i = 0; i < 100; i++)
            engine.CommitParameterUpdatedOnDevice(parameter, buffer, backend, syncPoint: null);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++)
            engine.CommitParameterUpdatedOnDevice(parameter, buffer, backend, syncPoint: null);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
        Assert.Equal(parameter.GpuCacheVersion, parameter._gpuBufferVersion);
        Assert.Equal(new[] { 7f, 9f }, parameter.ToArray());
        Assert.Equal(1, state.DownloadBufferCalls);
        engine.CommitParameterUpdatedOnDevice(parameter, buffer, backend, syncPoint: null);
        Assert.Equal(new[] { 7f, 9f }, parameter.ToArray());
        Assert.Equal(2, state.DownloadBufferCalls);
    }

    [Fact]
    public void RepeatedParameterBufferMoves_PreserveCacheMetadataAndOwners_WithoutAllocating()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var parameter = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        engine.RegisterResidentParamBuffer(parameter);
        var original = Assert.IsType<MockGpuBuffer>(parameter.TryGetGpuBuffer());
        using var alternate = new MockGpuBuffer(new[] { 3f, 4f });
        var cacheField = typeof(DirectGpuTensorEngine).GetField("_persistentBufferCache", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var cache = (ConcurrentDictionary<object, GpuBufferCacheEntry>)cacheField.GetValue(engine)!;
        var key = parameter.GetBackingArrayForCacheLookupUnsafe()!;
        var role = cache[key].Role;
        cache[key].Version = 17;
        for (int i = 0; i < 100; i++)
        {
            engine.MovePersistentParameterBuffer(parameter, original, alternate);
            engine.MovePersistentParameterBuffer(parameter, alternate, original);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++)
        {
            var from = i % 2 == 0 ? original : alternate;
            var to = i % 2 == 0 ? alternate : original;
            engine.MovePersistentParameterBuffer(parameter, from, to);
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Same(original, cache[key].Buffer);
        Assert.Equal(role, cache[key].Role);
        Assert.Equal(17, cache[key].Version);
        Assert.Equal(0, original.DisposeCount);
        Assert.Equal(0, alternate.DisposeCount);
        engine.MovePersistentParameterBuffer(parameter, original, alternate);
        Assert.Same(alternate, cache[key].Buffer);
        engine.MovePersistentParameterBuffer(parameter, alternate, original);
        Assert.Equal(0, allocated);
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
