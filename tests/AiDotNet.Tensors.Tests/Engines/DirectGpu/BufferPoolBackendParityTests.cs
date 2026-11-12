#if NET6_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
#if NET7_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
#endif
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Verifies symmetric buffer pooling, <see cref="IDeviceMemoryPressure.ReleaseCachedBuffers"/>,
/// <see cref="IDeviceMemoryPressure.ReclaimsOnAllocationFailure"/>, zero-initialization on pooled
/// <see cref="IDirectGpuBackend.AllocateBuffer(int)"/> where guaranteed, and uninitialized pool reuse
/// across all 6 DirectGpu backends (CUDA, HIP, OpenCL, Vulkan, Metal, WebGPU).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class BufferPoolBackendParityTests
{
    private static void CheckPoolAndMemoryPressure(IDirectGpuBackend backend, bool zeroInitializesAllocateBuffer)
    {
        var pressure = Assert.IsAssignableFrom<IDeviceMemoryPressure>(backend);
        Assert.True(pressure.ReclaimsOnAllocationFailure);

        // Drain any prior cached buffers so counts start clean.
        _ = pressure.ReleaseCachedBuffers();

        const int count = 4096;
        using (var first = backend.AllocateBuffer(count))
        {
            backend.Fill(first, 7.25f, count);
            backend.Synchronize();
        }
        backend.Synchronize();

        // Returning the buffer and immediately re-allocating the same size hits the pool.
        using (var reused = backend.AllocateBuffer(count))
        {
            if (zeroInitializesAllocateBuffer)
            {
                backend.Synchronize();
                float[] readback = backend.DownloadBuffer(reused);
                for (int i = 0; i < readback.Length; i++)
                {
                    Assert.Equal(0f, readback[i]);
                }
            }
            backend.Fill(reused, -3.5f, count);
            backend.Synchronize();
        }
        backend.Synchronize();

        int freed = pressure.ReleaseCachedBuffers();
        Assert.True(freed >= 1, $"Expected ReleaseCachedBuffers() >= 1 on {backend.BackendName}, got {freed}.");
        Assert.Equal(0, pressure.ReleaseCachedBuffers());

        var uninit = Assert.IsAssignableFrom<IUninitializedGpuAllocation>(backend);
        using (var buf = uninit.AllocateBufferUninitialized(count))
        {
            backend.Fill(buf, 1.5f, count);
            backend.Synchronize();
            float[] values = backend.DownloadBuffer(buf);
            Assert.Equal(1.5f, values[0]);
            Assert.Equal(1.5f, values[count - 1]);
        }
        backend.Synchronize();
        Assert.True(pressure.ReleaseCachedBuffers() >= 1);
    }

    [SkippableFact]
    public void Cuda_BufferPoolAndMemoryPressure_Parity()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        CheckPoolAndMemoryPressure(backend, zeroInitializesAllocateBuffer: true);
    }

    [SkippableFact]
    public void Hip_BufferPoolAndMemoryPressure_Parity()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        CheckPoolAndMemoryPressure(backend, zeroInitializesAllocateBuffer: true);
    }

    [SkippableFact]
    public void OpenCl_BufferPoolAndMemoryPressure_Parity()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        CheckPoolAndMemoryPressure(backend, zeroInitializesAllocateBuffer: false);
    }

    [SkippableFact]
    public void Vulkan_BufferPoolAndMemoryPressure_Parity()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize(), "Vulkan is not available.");
        CheckPoolAndMemoryPressure(backend, zeroInitializesAllocateBuffer: false);
    }

    [SkippableFact]
    public void Metal_BufferPoolAndMemoryPressure_Parity()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        long allocBefore = MetalGpuBuffer.DriverAllocations;
        CheckPoolAndMemoryPressure(backend, zeroInitializesAllocateBuffer: true);
        Assert.True(MetalGpuBuffer.DriverAllocations >= allocBefore + 1);
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_BufferPoolAndMemoryPressure_Parity()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        var pressure = Assert.IsAssignableFrom<IDeviceMemoryPressure>(backend);
        _ = pressure.ReleaseCachedBuffers();

        long allocBefore = WebGpuBuffer.DriverAllocations;
        const int count = 2048;
        using (var first = backend.AllocateBuffer(count))
        {
            backend.Fill(first, 9.5f, count);
            backend.Synchronize();
        }
        long allocAfterFirst = WebGpuBuffer.DriverAllocations;
        Assert.Equal(allocBefore + 1, allocAfterFirst);

        // Pool hit: zero new driver allocations AND zero-initialized contents.
        using (var second = backend.AllocateBuffer(count))
        {
            Assert.Equal(allocAfterFirst, WebGpuBuffer.DriverAllocations);
            float[] zeros = backend.DownloadBuffer(second);
            for (int i = 0; i < zeros.Length; i++) Assert.Equal(0f, zeros[i]);
        }

        CheckPoolAndMemoryPressure(backend, zeroInitializesAllocateBuffer: true);
    }
#endif
}
#endif
