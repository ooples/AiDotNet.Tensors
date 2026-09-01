// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Issue #276 sub-feature 4: every GPU backend has an
/// <see cref="IGpuOffloadAllocator"/>. Each must:
/// 1. Surface IsAvailable correctly (no exception in static init).
/// 2. Throw a clean NotSupportedException on Allocate when not available.
/// 3. Round-trip allocate → free without crashing on hosts that have the runtime.
///
/// We can't test happy-path allocation without the matching hardware/runtime,
/// but the IsAvailable + clean-throw contract is testable everywhere.
/// </summary>
[Collection("DirectGpuSerial")]
public class OffloadAllocatorTests
{
    private static IGpuOffloadAllocator[] AllAllocators() => new IGpuOffloadAllocator[]
    {
        new CudaOffloadAllocator(),
        new HipOffloadAllocator(),
        new MetalOffloadAllocator(),
        new OpenClOffloadAllocator(),
        new VulkanOffloadAllocator(),
        new WebGpuOffloadAllocator(),
    };

    [Theory]
    [InlineData(false, false, false)]
    [InlineData(false, true, false)]
    [InlineData(true, false, true)]
    [InlineData(true, true, false)]
    public async Task CudaAvailability_FailsClosedAfterCircuitBreaker(
        bool driverAvailable, bool circuitBroken, bool expected)
    {
        await Task.Yield();

        Assert.Equal(expected, CudaOffloadAllocator.IsCudaUsable(driverAvailable, circuitBroken));
    }

    [Theory]
    [InlineData(false, false, false)]
    [InlineData(false, true, false)]
    [InlineData(true, false, true)]
    [InlineData(true, true, false)]
    public async Task CudaContextCleanup_NeverPopsAfterCircuitBreaker(
        bool pushed, bool circuitBroken, bool expected)
    {
        await Task.Yield();

        Assert.Equal(expected, CudaOffloadAllocator.ShouldPopContext(pushed, circuitBroken));
    }

    [SkippableFact]
    public async Task SharedCudaAllocator_FailsClosedAfterBackendDestroysContext()
    {
        await Task.Yield();

        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver is unavailable.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend failed to initialize.");
        using var allocator = new CudaOffloadAllocator(backend.CudaContextHandle);

        Assert.True(allocator.IsAvailable);
        backend.Dispose();

        Assert.False(allocator.IsAvailable);
        var error = Assert.Throws<NotSupportedException>(
            () => allocator.Allocate(1024, OffloadScheme.Pinned));
        Assert.Contains("no longer live", error.Message, StringComparison.Ordinal);
    }

    [SkippableFact]
    public async Task SharedCudaAllocator_DisposeWithLiveBackend_FreesOutstandingAllocations()
    {
        await Task.Yield();

        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver is unavailable.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend failed to initialize.");
        using var allocator = new CudaOffloadAllocator(backend.CudaContextHandle);

        var pinned = allocator.Allocate(1024, OffloadScheme.Pinned);
        var managed = allocator.Allocate(1024, OffloadScheme.Managed);
        Assert.NotEqual(IntPtr.Zero, pinned.HostPointer);
        Assert.NotEqual(IntPtr.Zero, managed.DevicePointer);

        // Dispose owns both outstanding handles. Checked native frees make either cleanup failure
        // surface here, while the probe below proves a shared allocator never tears down its owner.
        allocator.Dispose();
        Assert.False(allocator.IsAvailable);

        using var probe = backend.AllocateBuffer(new[] { 1.25f, -2.5f, 4.75f });
        Assert.Equal(new[] { 1.25f, -2.5f, 4.75f }, backend.DownloadBuffer(probe));
    }

    [Fact]
    public void IsAvailable_ProbeNeverThrows_OnAnyBackend()
    {
        foreach (var alloc in AllAllocators())
        {
            using (alloc)
            {
                // Just reading the property must not throw, regardless of host.
                _ = alloc.IsAvailable;
            }
        }
    }

    [Fact]
    public void Allocate_WhenUnavailable_ThrowsCleanly()
    {
        foreach (var alloc in AllAllocators())
        {
            using (alloc)
            {
                if (alloc.IsAvailable) continue; // skip — happy path needs hardware
                Assert.Throws<NotSupportedException>(() => alloc.Allocate(1024, OffloadScheme.Pinned));
            }
        }
    }

    [Fact]
    public void Allocate_HappyPath_AllAvailableBackends()
    {
        foreach (var alloc in AllAllocators())
        {
            using (alloc)
            {
                if (!alloc.IsAvailable) continue;
                var h = alloc.Allocate(1024, OffloadScheme.Pinned);
                Assert.NotEqual(IntPtr.Zero, h.HostPointer);
                Assert.Equal(1024, h.Bytes);
                alloc.Free(h);
            }
        }
    }
}
