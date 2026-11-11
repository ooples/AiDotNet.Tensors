// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : ISparseRoutingBackend
{
    private void LaunchSparseRouting3Buf(string name, IGpuBuffer b0, IGpuBuffer b1, IGpuBuffer b2,
        int s0, int s1, int s2, long workItems)
    {
        if (_context == null)
            throw new GpuCapabilityUnavailableException($"OpenCL context is not initialized. Cannot execute {name}.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new GpuCapabilityUnavailableException($"OpenCL kernel not found: {name}");
        if (workItems == 0) return;
        if (workItems > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(s0), "Sparse routing exceeds the OpenCL launch limit.");
        int total = (int)workItems;
        int localSize = CalculateOptimalWorkGroupSize1D(total);
        int globalSize = ((total + localSize - 1) / localSize) * localSize;
        uint arg = 0;
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)b0).Buffer.Handle);
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)b1).Buffer.Handle);
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)b2).Buffer.Handle);
        kernel.SetArg(arg++, s0);
        kernel.SetArg(arg++, s1);
        kernel.SetArg(arg, s2);
        kernel.Execute1D(globalSize, localSize);
    }

    private void LaunchSparseRouting4Buf(string name, IGpuBuffer b0, IGpuBuffer b1, IGpuBuffer b2, IGpuBuffer b3,
        int s0, int s1, int s2, int s3, long workItems)
    {
        if (_context == null)
            throw new GpuCapabilityUnavailableException($"OpenCL context is not initialized. Cannot execute {name}.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new GpuCapabilityUnavailableException($"OpenCL kernel not found: {name}");
        if (workItems == 0) return;
        if (workItems > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(s0), "Sparse routing exceeds the OpenCL launch limit.");
        int total = (int)workItems;
        int localSize = CalculateOptimalWorkGroupSize1D(total);
        int globalSize = ((total + localSize - 1) / localSize) * localSize;
        uint arg = 0;
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)b0).Buffer.Handle);
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)b1).Buffer.Handle);
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)b2).Buffer.Handle);
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)b3).Buffer.Handle);
        kernel.SetArg(arg++, s0);
        kernel.SetArg(arg++, s1);
        kernel.SetArg(arg++, s2);
        kernel.SetArg(arg, s3);
        kernel.Execute1D(globalSize, localSize);
    }

    void ISparseRoutingBackend.GatherRowValuesForward(IGpuBuffer values, IGpuBuffer int32Indices, IGpuBuffer output,
        int rows, int cols, int k)
        => LaunchSparseRouting3Buf("gather_row_values_forward", values, int32Indices, output,
            rows, cols, k, (long)rows * k);

    void ISparseRoutingBackend.GatherRowValuesBackward(IGpuBuffer gradOutput, IGpuBuffer int32Indices, IGpuBuffer gradValues,
        int rows, int cols, int k)
        => LaunchSparseRouting3Buf("gather_row_values_backward", gradOutput, int32Indices, gradValues,
            rows, cols, k, (long)rows * cols);

    void ISparseRoutingBackend.TopKWeightedRowMixForward(IGpuBuffer table, IGpuBuffer int32Indices, IGpuBuffer weights,
        IGpuBuffer output, int rows, int numEntries, int k, int channels)
        => LaunchSparseRouting4Buf("topk_weighted_row_mix_forward", table, int32Indices, weights, output,
            rows, numEntries, k, channels, (long)rows * channels);

    void ISparseRoutingBackend.TopKWeightedRowMixBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer table,
        IGpuBuffer int32Indices, IGpuBuffer gradWeights, int rows, int numEntries, int k, int channels)
        => LaunchSparseRouting4Buf("topk_weighted_row_mix_backward_weights", gradOutput, table, int32Indices, gradWeights,
            rows, numEntries, k, channels, (long)rows * k);

    private DirectOpenClKernel RequireSparseRoutingKernel(string name)
    {
        if (_context == null)
            throw new GpuCapabilityUnavailableException($"OpenCL context is not initialized. Cannot execute {name}.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new GpuCapabilityUnavailableException($"OpenCL kernel not found: {name}");
        return kernel;
    }

    private void ExecuteSparseRouting1D(DirectOpenClKernel kernel, long workItems)
    {
        if (workItems == 0) return;
        if (workItems > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(workItems), "Sparse routing exceeds the OpenCL launch limit.");
        int total = (int)workItems;
        int localSize = CalculateOptimalWorkGroupSize1D(total);
        int globalSize = ((total + localSize - 1) / localSize) * localSize;
        kernel.Execute1D(globalSize, localSize);
    }

    void ISparseRoutingBackend.TopKWeightedRowMixBackwardTable(IGpuBuffer gradOutput, IGpuBuffer int32Indices,
        IGpuBuffer weights, IGpuBuffer gradTable, int rows, int numEntries, int k, int channels)
    {
        long totalOut = checked((long)numEntries * channels);
        if (totalOut == 0) return;
        SparseRoutingCsrPlan.Compute(rows, k, numEntries,
            out long totalSlotsLong, out int numTiles, out int offSorted, out int offTile, out int offChunk, out int totalScratchInts);
        int totalSlots = (int)totalSlotsLong;
        int tileCountInts = checked(numTiles * numEntries);

        var kZero = RequireSparseRoutingKernel("sparse_zero_int32");
        var kTileCounts = RequireSparseRoutingKernel("topk_csr_tile_counts");
        var kEntryPrefix = RequireSparseRoutingKernel("topk_csr_entry_prefix");
        var kChunkSums = RequireSparseRoutingKernel("topk_csr_chunk_sums");
        var kChunkFinalize = RequireSparseRoutingKernel("topk_csr_chunk_finalize");
        var kScatter = RequireSparseRoutingKernel("topk_csr_scatter_slots");
        var kSum = RequireSparseRoutingKernel("topk_weighted_row_mix_backward_table");

        using var csrBuf = AllocateBuffer(totalScratchInts);
        var hCsr = ((DirectOpenClGpuBuffer)csrBuf).Buffer.Handle;
        var hIndices = ((DirectOpenClGpuBuffer)int32Indices).Buffer.Handle;

        kZero.SetArg(0, hCsr);
        kZero.SetArg(1, offTile);
        kZero.SetArg(2, tileCountInts);
        ExecuteSparseRouting1D(kZero, tileCountInts);

        if (totalSlots > 0)
        {
            kTileCounts.SetArg(0, hIndices);
            kTileCounts.SetArg(1, hCsr);
            kTileCounts.SetArg(2, totalSlots);
            kTileCounts.SetArg(3, numEntries);
            kTileCounts.SetArg(4, numTiles);
            kTileCounts.SetArg(5, offTile);
            ExecuteSparseRouting1D(kTileCounts, numTiles);
        }

        kEntryPrefix.SetArg(0, hCsr);
        kEntryPrefix.SetArg(1, numEntries);
        kEntryPrefix.SetArg(2, numTiles);
        kEntryPrefix.SetArg(3, offTile);
        ExecuteSparseRouting1D(kEntryPrefix, numEntries);

        kChunkSums.SetArg(0, hCsr);
        kChunkSums.SetArg(1, numEntries);
        kChunkSums.SetArg(2, offChunk);
        ExecuteSparseRouting1D(kChunkSums, 256);

        kChunkFinalize.SetArg(0, hCsr);
        kChunkFinalize.SetArg(1, numEntries);
        kChunkFinalize.SetArg(2, offChunk);
        ExecuteSparseRouting1D(kChunkFinalize, (long)numEntries + 1);

        if (totalSlots > 0)
        {
            kScatter.SetArg(0, hIndices);
            kScatter.SetArg(1, hCsr);
            kScatter.SetArg(2, totalSlots);
            kScatter.SetArg(3, numEntries);
            kScatter.SetArg(4, numTiles);
            kScatter.SetArg(5, offSorted);
            kScatter.SetArg(6, offTile);
            ExecuteSparseRouting1D(kScatter, numTiles);
        }

        kSum.SetArg(0, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
        kSum.SetArg(1, ((DirectOpenClGpuBuffer)weights).Buffer.Handle);
        kSum.SetArg(2, hCsr);
        kSum.SetArg(3, ((DirectOpenClGpuBuffer)gradTable).Buffer.Handle);
        kSum.SetArg(4, numEntries);
        kSum.SetArg(5, k);
        kSum.SetArg(6, channels);
        kSum.SetArg(7, offSorted);
        ExecuteSparseRouting1D(kSum, totalOut);
    }

    void ISparseRoutingBackend.CountIndicesInt32(IGpuBuffer int32Indices, IGpuBuffer counts, int length, int numBins)
    {
        if (numBins <= 0) return;
        var kZero = RequireSparseRoutingKernel("sparse_zero_int32");
        var kAcc = RequireSparseRoutingKernel("count_indices_accumulate_int32");
        var kFin = RequireSparseRoutingKernel("count_indices_finalize_float");
        var hCounts = ((DirectOpenClGpuBuffer)counts).Buffer.Handle;

        kZero.SetArg(0, hCounts);
        kZero.SetArg(1, 0);
        kZero.SetArg(2, numBins);
        ExecuteSparseRouting1D(kZero, numBins);

        if (length > 0)
        {
            int totalThreads = System.Math.Min(length, 4096 * 256);
            kAcc.SetArg(0, ((DirectOpenClGpuBuffer)int32Indices).Buffer.Handle);
            kAcc.SetArg(1, hCounts);
            kAcc.SetArg(2, length);
            kAcc.SetArg(3, numBins);
            kAcc.SetArg(4, totalThreads);
            ExecuteSparseRouting1D(kAcc, totalThreads);
        }

        kFin.SetArg(0, hCounts);
        kFin.SetArg(1, numBins);
        ExecuteSparseRouting1D(kFin, numBins);
    }

    void ISparseRoutingBackend.MinMaxInt32(IGpuBuffer int32Data, IGpuBuffer outMinMaxInt32, int length)
    {
        if (length <= 0) return;
        var kInit = RequireSparseRoutingKernel("minmax_init_int32");
        var kRed = RequireSparseRoutingKernel("minmax_reduce_int32");
        var hOut = ((DirectOpenClGpuBuffer)outMinMaxInt32).Buffer.Handle;

        kInit.SetArg(0, hOut);
        ExecuteSparseRouting1D(kInit, 2);

        int totalThreads = System.Math.Min(length, 4096 * 256);
        kRed.SetArg(0, ((DirectOpenClGpuBuffer)int32Data).Buffer.Handle);
        kRed.SetArg(1, hOut);
        kRed.SetArg(2, length);
        kRed.SetArg(3, totalThreads);
        ExecuteSparseRouting1D(kRed, totalThreads);
    }
}
