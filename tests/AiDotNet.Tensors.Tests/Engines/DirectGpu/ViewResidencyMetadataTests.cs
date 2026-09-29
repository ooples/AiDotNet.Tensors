using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A reshape of a tensor whose device buffer is bound while its device tag is still CPU takes the shape-only-view path
/// (the storage-view path returns early on a CPU tag). That path carried the buffer but none of the metadata the
/// storage-view path carries - materializer key/callback, role, non-ownership, dirty flag, layout - so two views of the
/// same resident tensor behaved differently (the reshape joined the host mirror separately, and claimed nothing about
/// ownership).
/// </summary>
[Collection("DirectGpuSerial")]
public class ViewResidencyMetadataTests
{
    [SkippableFact]
    public void A_shape_only_view_carries_the_same_gpu_metadata_as_a_storage_view()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend.");
        var backend = gpu.GetBackend()!;
        var source = new Tensor<float>(new[] { 4, 4 });
        using var buffer = backend.AllocateBuffer(new float[16]);
        object key = new object();
        Action<object> callback = _ => { };
        source._gpuBuffer = buffer;
        source._gpuBackend = backend;
        source._gpuBufferVersion = source.GpuCacheVersion;
        source._ownsGpuBuffer = true;
        source._gpuMaterializerKey = key;
        source._gpuMaterializerCallback = callback;
        source.IsDirty = true;

        var view = source.Reshape(16);

        Assert.Same(buffer, view._gpuBuffer);
        Assert.False(view._ownsGpuBuffer, "the view claimed ownership of its source's buffer");
        Assert.Same(key, view._gpuMaterializerKey);
        Assert.Same(callback, view._gpuMaterializerCallback);
        Assert.Equal(source._gpuRole, view._gpuRole);
        Assert.True(view.IsDirty);
        Assert.Equal(source.Layout, view.Layout);
        source._gpuBuffer = null;
        view._gpuBuffer = null;
    }
}
