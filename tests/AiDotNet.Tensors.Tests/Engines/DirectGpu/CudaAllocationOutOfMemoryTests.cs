// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[CollectionDefinition("CudaAllocationOverride", DisableParallelization = true)]
public sealed class CudaAllocationOverrideCollection { }

/// <summary>
/// A CUDA allocation still out of memory after reclamation raised a bare <see cref="InvalidOperationException"/>, so a
/// caller with a lighter path (the fused-CE kernel when the chunked forward's scratch does not fit, a tuned-kernel slot's
/// direct kernel) could not tell it from a real CUDA fault. Every allocator now raises
/// <see cref="GpuOutOfMemoryException"/>. The test makes each allocation request a pebibyte, so the driver itself
/// refuses it and the allocators' own failure handling runs, whatever else holds device memory at the time. Twice the
/// device's memory was not enough: on the Windows RTX 3080 runner the driver's sysmem fallback let the synchronous
/// <c>cuMemAlloc</c> behind the int path succeed in shared system memory.
/// </summary>
[Collection("CudaAllocationOverride")]
public class CudaAllocationOutOfMemoryTests
{
    // More than any GPU's memory plus the shared system memory a sysmem fallback could add.
    private const ulong UnbackableBytes = 1UL << 50;

    [SkippableFact]
    public void Every_cuda_allocation_path_reports_exhausted_memory_as_GpuOutOfMemoryException()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "The CUDA backend did not initialize.");

        Assert.True((ulong)backend.GlobalMemoryBytes < UnbackableBytes, "The device has more memory than the test requests.");
        CudaBackend.AllocationBytesOverrideForTests = UnbackableBytes;
        try
        {
            var floatFailure = Record.Exception(() => backend.AllocateBuffer(1024).Dispose());
            var byteFailure = Record.Exception(() => backend.AllocateByteBuffer(4096).Dispose());
            var intFailure = Record.Exception(() => backend.AllocateIntBuffer(new int[1024]).Dispose());

            Assert.IsType<GpuOutOfMemoryException>(floatFailure);
            Assert.IsType<GpuOutOfMemoryException>(byteFailure);
            Assert.IsType<GpuOutOfMemoryException>(intFailure);
        }
        finally
        {
            CudaBackend.AllocationBytesOverrideForTests = null;
        }

        // The override is gone, so the same paths allocate again.
        backend.AllocateBuffer(1024).Dispose();
    }
}
