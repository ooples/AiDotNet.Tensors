// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[CollectionDefinition("CudaDeviceMemoryExhaustion", DisableParallelization = true)]
public sealed class CudaDeviceMemoryExhaustionCollection { }

/// <summary>
/// A CUDA allocation still out of memory after reclamation raised a bare <see cref="InvalidOperationException"/>, so a
/// caller with a lighter path (the fused-CE kernel when the chunked forward's scratch does not fit, a tuned-kernel slot's
/// direct kernel) could not tell it from a real CUDA fault. Every allocator now raises
/// <see cref="GpuOutOfMemoryException"/>. The test fills the device through the backend and asserts the type each path
/// raises, so an allocator that stops classifying its failure is caught here, not only the helper.
/// </summary>
[Collection("CudaDeviceMemoryExhaustion")]
public class CudaAllocationOutOfMemoryTests
{
    private const int GiBFloats = 1 << 28, Chunk64MiBFloats = 1 << 24;

    [SkippableFact]
    public void Every_cuda_allocation_path_reports_exhausted_memory_as_GpuOutOfMemoryException()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "The CUDA backend did not initialize.");

        var held = new List<IGpuBuffer>();
        try
        {
            // Fill in 1 GiB blocks, then 64 MiB ones, so less than 64 MiB is left when the paths are tried.
            var first = FillUntilOutOfMemory(backend, held, GiBFloats);
            Assert.IsType<GpuOutOfMemoryException>(first);
            Assert.IsType<GpuOutOfMemoryException>(FillUntilOutOfMemory(backend, held, Chunk64MiBFloats));

            Assert.IsType<GpuOutOfMemoryException>(
                Record.Exception(() => held.Add(backend.AllocateBuffer(Chunk64MiBFloats))));
            Assert.IsType<GpuOutOfMemoryException>(
                Record.Exception(() => held.Add(backend.AllocateByteBuffer(Chunk64MiBFloats * sizeof(float)))));
            Assert.IsType<GpuOutOfMemoryException>(
                Record.Exception(() => held.Add(backend.AllocateIntBuffer(new int[Chunk64MiBFloats]))));
        }
        finally
        {
            foreach (var buffer in held) buffer.Dispose();
            // Return the stream-ordered pool's reservation so later tests' synchronous allocations find the memory.
            ((IDeviceMemoryPressure)backend).ReleaseCachedBuffers();
        }
    }

    private static Exception FillUntilOutOfMemory(CudaBackend backend, List<IGpuBuffer> held, int floats)
    {
        // 64 GiB of 1 GiB blocks, or 16 GiB of 64 MiB ones, bounds the loop well past any device on the lane.
        for (int i = 0; i < 256; i++)
        {
            var failure = Record.Exception(() => held.Add(backend.AllocateBuffer(floats)));
            if (failure is not null) return failure;
        }
        throw new InvalidOperationException($"Allocated {held.Count} buffers without running out of device memory.");
    }
}
