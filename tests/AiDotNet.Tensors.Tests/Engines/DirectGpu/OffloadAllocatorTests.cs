// Copyright (c) AiDotNet. All rights reserved.

using System;
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
