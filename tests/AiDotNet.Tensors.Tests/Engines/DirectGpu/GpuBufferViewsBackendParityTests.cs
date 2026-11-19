#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IGpuBufferViews"/> (#1142): a kernel that writes through a view changes exactly that range of the
/// parent, a download of the view reads that range, and out-of-range requests are refused.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuBufferViewsBackendParityTests
{
    private const int ParentLength = 4096;
    // 1024 floats = 4096 bytes, a multiple of every OpenCL device's sub-buffer base-address alignment.
    private const int Offset = 1024;
    private const int Count = 777;

    private static void Check(IDirectGpuBackend backend)
    {
        var views = Assert.IsAssignableFrom<IGpuBufferViews>(backend);
        var data = new float[ParentLength];
        for (int i = 0; i < data.Length; i++) data[i] = i * 0.5f;
        using var parent = backend.AllocateBuffer(data);

        Assert.Null(views.TryCreateView(parent, -1, 4));
        Assert.Null(views.TryCreateView(parent, ParentLength - 2, 4));
        Assert.Null(views.TryCreateView(parent, 0, 0));

        using var view = views.TryCreateView(parent, Offset, Count);
        Assert.NotNull(view);
        Assert.Equal(Count, view!.Size);
        var read = backend.DownloadBuffer(view);
        for (int i = 0; i < Count; i++) Assert.Equal(data[Offset + i], read[i]);

        backend.Scale(view, view, -2f, Count);
        var after = backend.DownloadBuffer(parent);
        for (int i = 0; i < ParentLength; i++)
        {
            float expected = i >= Offset && i < Offset + Count ? data[i] * -2f : data[i];
            Assert.True(after[i] == expected, $"parent[{i}]: {after[i]}, expected {expected}");
        }
    }

    [SkippableFact]
    public void OpenCl_BufferViews_AddressTheParentRange()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    /// <summary>
    /// A view must keep working after its parent is released, and releasing it afterwards must not fault: releases run
    /// in no fixed order (the retirement reap goes newest first), and NVIDIA's runtime crashed the test host
    /// (0xC0000005 in clReleaseMemObject) releasing a sub-buffer whose parent had already gone.
    /// </summary>
    [SkippableFact]
    public void OpenCl_AView_OutlivesItsReleasedParent()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        var views = Assert.IsAssignableFrom<IGpuBufferViews>(backend);
        var data = new float[ParentLength];
        for (int i = 0; i < data.Length; i++) data[i] = i * 0.5f;
        var parent = Assert.IsType<DirectOpenClGpuBuffer>(backend.AllocateBuffer(data));
        var view = views.TryCreateView(parent, Offset, Count)
            ?? throw new InvalidOperationException($"OpenCL refused a view at element offset {Offset}.");

        backend.Scale(view, view, -2f, Count);
        // Released, not returned to the pool: the native parent is let go while the view still has work queued on it.
        parent.Release();
        backend.Scale(view, view, 3f, Count);
        var read = backend.DownloadBuffer(view);
        for (int i = 0; i < Count; i++) Assert.Equal(data[Offset + i] * -6f, read[i]);

        view.Dispose();
        // A reap releases whatever retired with it; the next allocation runs one.
        using var next = backend.AllocateBuffer(new float[Count]);
        Assert.Equal(Count, next.Size);
    }

    [SkippableFact]
    public void Cuda_BufferViews_AddressTheParentRange()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_BufferViews_AddressTheParentRange()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }
}
#endif
