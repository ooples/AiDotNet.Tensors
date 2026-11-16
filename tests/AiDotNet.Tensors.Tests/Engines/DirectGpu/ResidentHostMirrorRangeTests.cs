using System;
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
    public void RepeatedAttach_DownloadsReboundBufferOnce_AndHonorsDetachment()
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

        // .NET Framework lacks this allocation metric; replay and functional checks run on every target.
#if NETCOREAPP
        long before = GC.GetAllocatedBytesForCurrentThread();
#endif
        for (int i = 0; i < 1000; i++) ResidentHostMirror.Attach(tensor);
#if NETCOREAPP
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
#endif

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
    public void RepeatedParameterCommit_PreservesVersionsAndFreshHostReads()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var parameter = new Tensor<float>(new float[2], new[] { 2 });
        using var buffer = new MockGpuBuffer(new[] { 7f, 9f });
        for (int i = 0; i < 100; i++)
            engine.CommitParameterUpdatedOnDevice(parameter, buffer, backend, syncPoint: null);

        // .NET Framework lacks this allocation metric; replay and functional checks run on every target.
#if NETCOREAPP
        long before = GC.GetAllocatedBytesForCurrentThread();
#endif
        for (int i = 0; i < 1000; i++)
            engine.CommitParameterUpdatedOnDevice(parameter, buffer, backend, syncPoint: null);
#if NETCOREAPP
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
#endif
        Assert.Equal(parameter.GpuCacheVersion, parameter._gpuBufferVersion);
        Assert.Equal(new[] { 7f, 9f }, parameter.ToArray());
        Assert.Equal(1, state.DownloadBufferCalls);
        engine.CommitParameterUpdatedOnDevice(parameter, buffer, backend, syncPoint: null);
        Assert.Equal(new[] { 7f, 9f }, parameter.ToArray());
        Assert.Equal(2, state.DownloadBufferCalls);
    }

    [Fact]
    public void UnmaterializedParameter_ReplayRebindsLatestBuffer_AndRekeysAfterHostRead()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var parameter = Tensor<float>.CreateGpuResident(new[] { 2 }, TensorDevice.OpenCL);
        using var first = new MockGpuBuffer(new[] { 1f, 2f });
        using var second = new MockGpuBuffer(new[] { 3f, 4f });
        engine.BindResidentBuffer(parameter, first, backend);
        var callback = parameter._gpuMaterializerCallback;
        void Replay(int step)
        {
            var source = step % 2 == 0 ? second : first;
            var target = step % 2 == 0 ? first : second;
            engine.MovePersistentParameterBuffer(parameter, source, target);
            engine.BindResidentBuffer(parameter, target, backend);
            engine.CommitParameterUpdatedOnDevice(parameter, target, backend, syncPoint: null);
        }
        for (int i = 0; i < 100; i++) Replay(i);

        // .NET Framework lacks this allocation metric; replay and functional checks run on every target.
#if NETCOREAPP
        long before = GC.GetAllocatedBytesForCurrentThread();
#endif
        for (int i = 0; i < 1000; i++) Replay(i);
#if NETCOREAPP
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
#endif
        Assert.Null(parameter.GetBackingArrayForCacheLookupUnsafe());
        Assert.Same(callback, parameter._gpuMaterializerCallback);
        Assert.Equal(new[] { 3f, 4f }, parameter.ToArray());
        Assert.Equal(1, state.DownloadBufferCalls);

        // After materialization the normal array mirror replaces the vector callback and sees the new binding.
        engine.MovePersistentParameterBuffer(parameter, second, first);
        engine.BindResidentBuffer(parameter, first, backend);
        engine.CommitParameterUpdatedOnDevice(parameter, first, backend, syncPoint: null);
        Assert.Equal(new[] { 1f, 2f }, parameter.ToArray());
        Assert.Equal(2, state.DownloadBufferCalls);
        Assert.Equal(new[] { 1f, 2f }, parameter.ToArray());
        Assert.Equal(2, state.DownloadBufferCalls);
        Assert.Equal(0, first.DisposeCount);
        Assert.Equal(0, second.DisposeCount);
    }

    [Fact]
    public void RebindingMetadataView_DoesNotRedirectTheSourcesMaterializer()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var source = Tensor<float>.CreateGpuResident(new[] { 2 }, TensorDevice.OpenCL);
        using var sourceBuffer = new MockGpuBuffer(new[] { 1f, 2f });
        using var viewBuffer = new MockGpuBuffer(new[] { 9f });
        engine.BindResidentBuffer(source, sourceBuffer, backend);
        var sourceCallback = source._gpuMaterializerCallback;
        using var view = source.CreateNarrowView(0, 1, 2, recordAutodiff: false);
        Assert.Same(sourceCallback, view._gpuMaterializerCallback);
        Assert.Null(source.GetBackingArrayForCacheLookupUnsafe());

        engine.BindResidentBuffer(view, viewBuffer, backend);

        Assert.NotSame(sourceCallback, view._gpuMaterializerCallback);
        Assert.Same(sourceCallback, source._gpuMaterializerCallback);
        Assert.Same(sourceBuffer, source._gpuBuffer);
        // Re-arm the source's own saved callback after the shared vector's registration was replaced by the view.
        source.MarkModified(syncPoint: null);
        Assert.Equal(new[] { 1f, 2f }, source.ToArray());
        Assert.Equal(1, state.DownloadBufferCalls);
        Assert.Equal(0, sourceBuffer.DisposeCount);
        Assert.Equal(0, viewBuffer.DisposeCount);
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
