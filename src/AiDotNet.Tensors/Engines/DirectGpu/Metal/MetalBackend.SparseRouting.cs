// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : ISparseRoutingBackend
{
    private void DispatchSparseRouting(string name, MetalGpuBuffer[] buffers, int[] scalars, long workItems)
    {
        if (workItems == 0) return;
        if (workItems > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(workItems), "Sparse routing exceeds the Metal launch limit.");
        DispatchRecurrence(name, (int)workItems, buffers, scalars);
    }

    void ISparseRoutingBackend.GatherRowValuesForward(IGpuBuffer values, IGpuBuffer int32Indices, IGpuBuffer output,
        int rows, int cols, int k)
        => DispatchSparseRouting("gather_row_values_forward",
            new[] { M(values), M(int32Indices), M(output) },
            new[] { rows, cols, k },
            (long)rows * k);

    void ISparseRoutingBackend.GatherRowValuesBackward(IGpuBuffer gradOutput, IGpuBuffer int32Indices, IGpuBuffer gradValues,
        int rows, int cols, int k)
        => DispatchSparseRouting("gather_row_values_backward",
            new[] { M(gradOutput), M(int32Indices), M(gradValues) },
            new[] { rows, cols, k },
            (long)rows * cols);

    void ISparseRoutingBackend.TopKWeightedRowMixForward(IGpuBuffer table, IGpuBuffer int32Indices, IGpuBuffer weights,
        IGpuBuffer output, int rows, int numEntries, int k, int channels)
        => DispatchSparseRouting("topk_weighted_row_mix_forward",
            new[] { M(table), M(int32Indices), M(weights), M(output) },
            new[] { rows, numEntries, k, channels },
            (long)rows * channels);

    void ISparseRoutingBackend.TopKWeightedRowMixBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer table,
        IGpuBuffer int32Indices, IGpuBuffer gradWeights, int rows, int numEntries, int k, int channels)
        => DispatchSparseRouting("topk_weighted_row_mix_backward_weights",
            new[] { M(gradOutput), M(table), M(int32Indices), M(gradWeights) },
            new[] { rows, numEntries, k, channels },
            (long)rows * k);

    void ISparseRoutingBackend.TopKWeightedRowMixBackwardTable(IGpuBuffer gradOutput, IGpuBuffer int32Indices,
        IGpuBuffer weights, IGpuBuffer gradTable, int rows, int numEntries, int k, int channels)
    {
        long totalOut = checked((long)numEntries * channels);
        if (totalOut == 0) return;
        SparseRoutingCsrPlan.Compute(rows, k, numEntries,
            out long totalSlotsLong, out int numTiles, out int offSorted, out int offTile, out int offChunk, out int totalScratchInts);
        int totalSlots = (int)totalSlotsLong;
        int tileCountInts = checked(numTiles * numEntries);

        using var csrBuf = AllocateBuffer(totalScratchInts);
        var mCsr = M(csrBuf);
        var mIdx = M(int32Indices);

        DispatchSparseRouting("sparse_zero_int32",
            new[] { mCsr }, new[] { offTile, tileCountInts }, tileCountInts);

        if (totalSlots > 0)
        {
            DispatchSparseRouting("topk_csr_tile_counts",
                new[] { mIdx, mCsr }, new[] { totalSlots, numEntries, numTiles, offTile }, numTiles);
        }

        DispatchSparseRouting("topk_csr_entry_prefix",
            new[] { mCsr }, new[] { numEntries, numTiles, offTile }, numEntries);

        DispatchSparseRouting("topk_csr_chunk_sums",
            new[] { mCsr }, new[] { numEntries, offChunk }, 256);

        DispatchSparseRouting("topk_csr_chunk_finalize",
            new[] { mCsr }, new[] { numEntries, offChunk }, (long)numEntries + 1);

        if (totalSlots > 0)
        {
            DispatchSparseRouting("topk_csr_scatter_slots",
                new[] { mIdx, mCsr }, new[] { totalSlots, numEntries, numTiles, offSorted, offTile }, numTiles);
        }

        DispatchSparseRouting("topk_weighted_row_mix_backward_table",
            new[] { M(gradOutput), M(weights), mCsr, M(gradTable) },
            new[] { numEntries, k, channels, offSorted },
            totalOut);
    }

    void ISparseRoutingBackend.CountIndicesInt32(IGpuBuffer int32Indices, IGpuBuffer counts, int length, int numBins)
    {
        if (numBins <= 0) return;
        var mCounts = M(counts);
        DispatchSparseRouting("sparse_zero_int32", new[] { mCounts }, new[] { 0, numBins }, numBins);
        if (length > 0)
        {
            int totalThreads = System.Math.Min(length, 4096 * 256);
            DispatchSparseRouting("count_indices_accumulate_int32",
                new[] { M(int32Indices), mCounts },
                new[] { length, numBins, totalThreads },
                totalThreads);
        }
        DispatchSparseRouting("count_indices_finalize_float", new[] { mCounts }, new[] { numBins }, numBins);
    }

    void ISparseRoutingBackend.MinMaxInt32(IGpuBuffer int32Data, IGpuBuffer outMinMaxInt32, int length)
    {
        if (length <= 0) return;
        var mOut = M(outMinMaxInt32);
        DispatchSparseRouting("minmax_init_int32", new[] { mOut }, new[] { 2 }, 2);
        int totalThreads = System.Math.Min(length, 4096 * 256);
        DispatchSparseRouting("minmax_reduce_int32",
            new[] { M(int32Data), mOut },
            new[] { length, totalThreads },
            totalThreads);
    }
}
