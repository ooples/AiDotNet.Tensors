// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET7_0_OR_GREATER

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : ISparseRoutingBackend
{
    private void DispatchSparseRouting(string key, string wgsl, IGpuBuffer[] buffers, int[] scalars, long workItems)
    {
        if (workItems == 0) return;
        if (workItems > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(workItems), "Sparse routing exceeds the WebGPU launch limit.");
        DispatchRecurrence(key, wgsl, (int)workItems, buffers, scalars);
    }

    void ISparseRoutingBackend.GatherRowValuesForward(IGpuBuffer values, IGpuBuffer int32Indices, IGpuBuffer output,
        int rows, int cols, int k)
        => DispatchSparseRouting("sr_gather_row_fwd", WebGpuRecurrenceKernels.GatherRowValuesForward,
            new[] { values, int32Indices, output },
            new[] { rows, cols, k },
            (long)rows * k);

    void ISparseRoutingBackend.GatherRowValuesBackward(IGpuBuffer gradOutput, IGpuBuffer int32Indices, IGpuBuffer gradValues,
        int rows, int cols, int k)
        => DispatchSparseRouting("sr_gather_row_bwd", WebGpuRecurrenceKernels.GatherRowValuesBackward,
            new[] { gradOutput, int32Indices, gradValues },
            new[] { rows, cols, k },
            (long)rows * cols);

    void ISparseRoutingBackend.TopKWeightedRowMixForward(IGpuBuffer table, IGpuBuffer int32Indices, IGpuBuffer weights,
        IGpuBuffer output, int rows, int numEntries, int k, int channels)
        => DispatchSparseRouting("sr_topk_mix_fwd", WebGpuRecurrenceKernels.TopKWeightedRowMixForward,
            new[] { table, int32Indices, weights, output },
            new[] { rows, numEntries, k, channels },
            (long)rows * channels);

    void ISparseRoutingBackend.TopKWeightedRowMixBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer table,
        IGpuBuffer int32Indices, IGpuBuffer gradWeights, int rows, int numEntries, int k, int channels)
        => DispatchSparseRouting("sr_topk_mix_bwd_w", WebGpuRecurrenceKernels.TopKWeightedRowMixBackwardWeights,
            new[] { gradOutput, table, int32Indices, gradWeights },
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

        DispatchSparseRouting("sr_zero_i32", WebGpuRecurrenceKernels.SparseZeroInt32,
            new[] { csrBuf }, new[] { offTile, tileCountInts }, tileCountInts);

        if (totalSlots > 0)
        {
            DispatchSparseRouting("sr_csr_tile_cnt", WebGpuRecurrenceKernels.TopKCsrTileCounts,
                new[] { int32Indices, csrBuf }, new[] { totalSlots, numEntries, numTiles, offTile }, numTiles);
        }

        DispatchSparseRouting("sr_csr_entry_pfx", WebGpuRecurrenceKernels.TopKCsrEntryPrefix,
            new[] { csrBuf }, new[] { numEntries, numTiles, offTile }, numEntries);

        DispatchSparseRouting("sr_csr_chunk_sum", WebGpuRecurrenceKernels.TopKCsrChunkSums,
            new[] { csrBuf }, new[] { numEntries, offChunk }, 256);

        DispatchSparseRouting("sr_csr_chunk_fin", WebGpuRecurrenceKernels.TopKCsrChunkFinalize,
            new[] { csrBuf }, new[] { numEntries, offChunk }, (long)numEntries + 1);

        if (totalSlots > 0)
        {
            DispatchSparseRouting("sr_csr_scatter", WebGpuRecurrenceKernels.TopKCsrScatterSlots,
                new[] { int32Indices, csrBuf }, new[] { totalSlots, numEntries, numTiles, offSorted, offTile }, numTiles);
        }

        DispatchSparseRouting("sr_topk_mix_bwd_tbl_csr", WebGpuRecurrenceKernels.TopKWeightedRowMixBackwardTable,
            new[] { gradOutput, weights, csrBuf, gradTable },
            new[] { numEntries, k, channels, offSorted },
            totalOut);
    }

    void ISparseRoutingBackend.CountIndicesInt32(IGpuBuffer int32Indices, IGpuBuffer counts, int length, int numBins)
    {
        if (numBins <= 0) return;
        DispatchSparseRouting("sr_zero_i32", WebGpuRecurrenceKernels.SparseZeroInt32,
            new[] { counts }, new[] { 0, numBins }, numBins);
        if (length > 0)
        {
            int totalThreads = System.Math.Min(length, 4096 * 256);
            DispatchSparseRouting("sr_count_acc_i32", WebGpuRecurrenceKernels.CountIndicesAccumulateInt32,
                new[] { int32Indices, counts }, new[] { length, numBins, totalThreads }, totalThreads);
        }
        DispatchSparseRouting("sr_count_fin_f32", WebGpuRecurrenceKernels.CountIndicesFinalizeFloat,
            new[] { counts }, new[] { numBins }, numBins);
    }

    void ISparseRoutingBackend.MinMaxInt32(IGpuBuffer int32Data, IGpuBuffer outMinMaxInt32, int length)
    {
        if (length <= 0) return;
        DispatchSparseRouting("sr_minmax_init_i32", WebGpuRecurrenceKernels.MinMaxInitInt32,
            new[] { outMinMaxInt32 }, new[] { 2 }, 2);
        int totalThreads = System.Math.Min(length, 4096 * 256);
        DispatchSparseRouting("sr_minmax_red_i32", WebGpuRecurrenceKernels.MinMaxReduceInt32,
            new[] { int32Data, outMinMaxInt32 }, new[] { length, totalThreads }, totalThreads);
    }
}
#endif
