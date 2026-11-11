// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : ISparseRoutingBackend
{
    internal uint? MaxSparseRoutingWorkgroupsXForTests { get; set; }

    private void DispatchSparseRouting(string glsl, IGpuBuffer[] buffers, uint[] scalars, long workItems)
    {
        if (workItems == 0) return;
        if (workItems > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(workItems), "Sparse routing exceeds the Vulkan launch limit.");
        EnsureInitialized();
        if (!IsGlslCompilerAvailable)
            throw new GpuCapabilityUnavailableException(
                "Vulkan sparse-routing shaders require runtime GLSL compilation (shaderc), which is unavailable.");

        uint deviceMaxGroups = _device.Limits.maxComputeWorkGroupCountX;
        if (MaxSparseRoutingWorkgroupsXForTests is { } cap && cap > 0 && cap < deviceMaxGroups)
            deviceMaxGroups = cap;
        long maxItemsPerLaunch = System.Math.Min(
            (long)(int.MaxValue / VulkanKernels.WorkgroupSize) * VulkanKernels.WorkgroupSize,
            (long)deviceMaxGroups * VulkanKernels.WorkgroupSize);
        if (maxItemsPerLaunch <= 0)
            maxItemsPerLaunch = VulkanKernels.WorkgroupSize;

        var push = new uint[scalars.Length + 1];
        System.Array.Copy(scalars, push, scalars.Length);

        try
        {
            for (long offset = 0; offset < workItems; )
            {
                int chunk = (int)System.Math.Min(maxItemsPerLaunch, workItems - offset);
                push[scalars.Length] = (uint)offset;
                GlslNaryOp(glsl, buffers, chunk, push);
                offset += chunk;
            }
        }
        catch (InvalidOperationException ex)
        {
            throw new GpuCapabilityUnavailableException(
                $"Vulkan sparse-routing compute pipeline is unavailable: {ex.Message}", ex);
        }
    }

    void ISparseRoutingBackend.GatherRowValuesForward(IGpuBuffer values, IGpuBuffer int32Indices, IGpuBuffer output,
        int rows, int cols, int k)
        => DispatchSparseRouting(VulkanRecurrenceKernels.GatherRowValuesForward,
            new[] { values, int32Indices, output },
            new[] { (uint)rows, (uint)cols, (uint)k },
            (long)rows * k);

    void ISparseRoutingBackend.GatherRowValuesBackward(IGpuBuffer gradOutput, IGpuBuffer int32Indices, IGpuBuffer gradValues,
        int rows, int cols, int k)
        => DispatchSparseRouting(VulkanRecurrenceKernels.GatherRowValuesBackward,
            new[] { gradOutput, int32Indices, gradValues },
            new[] { (uint)rows, (uint)cols, (uint)k },
            (long)rows * cols);

    void ISparseRoutingBackend.TopKWeightedRowMixForward(IGpuBuffer table, IGpuBuffer int32Indices, IGpuBuffer weights,
        IGpuBuffer output, int rows, int numEntries, int k, int channels)
        => DispatchSparseRouting(VulkanRecurrenceKernels.TopKWeightedRowMixForward,
            new[] { table, int32Indices, weights, output },
            new[] { (uint)rows, (uint)numEntries, (uint)k, (uint)channels },
            (long)rows * channels);

    void ISparseRoutingBackend.TopKWeightedRowMixBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer table,
        IGpuBuffer int32Indices, IGpuBuffer gradWeights, int rows, int numEntries, int k, int channels)
        => DispatchSparseRouting(VulkanRecurrenceKernels.TopKWeightedRowMixBackwardWeights,
            new[] { gradOutput, table, int32Indices, gradWeights },
            new[] { (uint)rows, (uint)numEntries, (uint)k, (uint)channels },
            (long)rows * k);

    void ISparseRoutingBackend.TopKWeightedRowMixBackwardTable(IGpuBuffer gradOutput, IGpuBuffer int32Indices,
        IGpuBuffer weights, IGpuBuffer gradTable, int rows, int numEntries, int k, int channels)
    {
        long totalOut = checked((long)numEntries * channels);
        if (totalOut == 0) return;
        EnsureInitialized();
        if (!IsGlslCompilerAvailable)
            throw new GpuCapabilityUnavailableException(
                "Vulkan sparse-routing shaders require runtime GLSL compilation (shaderc), which is unavailable.");

        SparseRoutingCsrPlan.Compute(rows, k, numEntries,
            out long totalSlotsLong, out int numTiles, out int offSorted, out int offTile, out int offChunk, out int totalScratchInts);
        int totalSlots = (int)totalSlotsLong;
        int tileCountInts = checked(numTiles * numEntries);

        var csrBuf = AllocateBuffer(totalScratchInts);
        try
        {
            DispatchSparseRouting(VulkanRecurrenceKernels.SparseZeroInt32,
                new[] { csrBuf }, new[] { (uint)offTile, (uint)tileCountInts }, tileCountInts);

            if (totalSlots > 0)
            {
                DispatchSparseRouting(VulkanRecurrenceKernels.TopKCsrTileCounts,
                    new[] { int32Indices, csrBuf },
                    new[] { (uint)totalSlots, (uint)numEntries, (uint)numTiles, (uint)offTile },
                    numTiles);
            }

            DispatchSparseRouting(VulkanRecurrenceKernels.TopKCsrEntryPrefix,
                new[] { csrBuf }, new[] { (uint)numEntries, (uint)numTiles, (uint)offTile }, numEntries);

            DispatchSparseRouting(VulkanRecurrenceKernels.TopKCsrChunkSums,
                new[] { csrBuf }, new[] { (uint)numEntries, (uint)offChunk }, SparseRoutingCsrPlan.ChunkScratchInts);

            DispatchSparseRouting(VulkanRecurrenceKernels.TopKCsrChunkFinalize,
                new[] { csrBuf }, new[] { (uint)numEntries, (uint)offChunk }, (long)numEntries + 1);

            if (totalSlots > 0)
            {
                DispatchSparseRouting(VulkanRecurrenceKernels.TopKCsrScatterSlots,
                    new[] { int32Indices, csrBuf },
                    new[] { (uint)totalSlots, (uint)numEntries, (uint)numTiles, (uint)offSorted, (uint)offTile },
                    numTiles);
            }

            DispatchSparseRouting(VulkanRecurrenceKernels.TopKWeightedRowMixBackwardTable,
                new[] { gradOutput, weights, csrBuf, gradTable },
                new[] { (uint)numEntries, (uint)k, (uint)channels, (uint)offSorted },
                totalOut);
        }
        finally
        {
            ((IQueuedBufferRelease)this).ReleaseAfterQueuedWork(csrBuf);
        }
    }

    void ISparseRoutingBackend.CountIndicesInt32(IGpuBuffer int32Indices, IGpuBuffer counts, int length, int numBins)
    {
        if (numBins <= 0) return;
        DispatchSparseRouting(VulkanRecurrenceKernels.SparseZeroInt32,
            new[] { counts }, new[] { 0u, (uint)numBins }, numBins);
        if (length > 0)
        {
            int totalThreads = System.Math.Min(length, SparseRoutingLimits.MaxReductionThreads);
            DispatchSparseRouting(VulkanRecurrenceKernels.CountIndicesAccumulateInt32,
                new[] { int32Indices, counts },
                new[] { (uint)length, (uint)numBins, (uint)totalThreads },
                totalThreads);
        }
        DispatchSparseRouting(VulkanRecurrenceKernels.CountIndicesFinalizeFloat,
            new[] { counts }, new[] { (uint)numBins }, numBins);
    }

    void ISparseRoutingBackend.MinMaxInt32(IGpuBuffer int32Data, IGpuBuffer outMinMaxInt32, int length)
    {
        if (length <= 0) return;
        DispatchSparseRouting(VulkanRecurrenceKernels.MinMaxInitInt32,
            new[] { outMinMaxInt32 }, new[] { 2u }, 2);
        int totalThreads = System.Math.Min(length, SparseRoutingLimits.MaxReductionThreads);
        DispatchSparseRouting(VulkanRecurrenceKernels.MinMaxReduceInt32,
            new[] { int32Data, outMinMaxInt32 },
            new[] { (uint)length, (uint)totalThreads },
            totalThreads);
    }
}
