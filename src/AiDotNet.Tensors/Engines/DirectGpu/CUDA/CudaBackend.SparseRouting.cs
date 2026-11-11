// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : ISparseRoutingBackend
{
    private IntPtr SparseRoutingKernel(string name)
        => _kernelCache.TryGetValue(name, out var k) ? k
            : throw new GpuCapabilityUnavailableException($"CUDA kernel not found: {name}");

    private static uint SparseRoutingGrid(long total)
    {
        long grid = (total + DefaultBlockSize - 1) / DefaultBlockSize;
        if (grid > uint.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(total), "Sparse routing grid exceeds the CUDA launch limit.");
        return (uint)grid;
    }

    unsafe void ISparseRoutingBackend.GatherRowValuesForward(IGpuBuffer values, IGpuBuffer int32Indices, IGpuBuffer output,
        int rows, int cols, int k)
    {
        var fn = SparseRoutingKernel("gather_row_values_forward");
        long total = checked((long)rows * k);
        if (total == 0) return;
        using var _ = PushContext();
        IntPtr pv = values.Handle, pi = int32Indices.Handle, po = output.Handle;
        void** a = stackalloc void*[6];
        a[0] = &pv; a[1] = &pi; a[2] = &po; a[3] = &rows; a[4] = &cols; a[5] = &k;
        LaunchKernel(fn, SparseRoutingGrid(total), (uint)DefaultBlockSize, a);
    }

    unsafe void ISparseRoutingBackend.GatherRowValuesBackward(IGpuBuffer gradOutput, IGpuBuffer int32Indices, IGpuBuffer gradValues,
        int rows, int cols, int k)
    {
        var fn = SparseRoutingKernel("gather_row_values_backward");
        long total = checked((long)rows * cols);
        if (total == 0) return;
        using var _ = PushContext();
        IntPtr pg = gradOutput.Handle, pi = int32Indices.Handle, pdv = gradValues.Handle;
        void** a = stackalloc void*[6];
        a[0] = &pg; a[1] = &pi; a[2] = &pdv; a[3] = &rows; a[4] = &cols; a[5] = &k;
        LaunchKernel(fn, SparseRoutingGrid(total), (uint)DefaultBlockSize, a);
    }

    unsafe void ISparseRoutingBackend.TopKWeightedRowMixForward(IGpuBuffer table, IGpuBuffer int32Indices, IGpuBuffer weights,
        IGpuBuffer output, int rows, int numEntries, int k, int channels)
    {
        var fn = SparseRoutingKernel("topk_weighted_row_mix_forward");
        long total = checked((long)rows * channels);
        if (total == 0) return;
        using var _ = PushContext();
        IntPtr pt = table.Handle, pi = int32Indices.Handle, pw = weights.Handle, po = output.Handle;
        void** a = stackalloc void*[8];
        a[0] = &pt; a[1] = &pi; a[2] = &pw; a[3] = &po;
        a[4] = &rows; a[5] = &numEntries; a[6] = &k; a[7] = &channels;
        LaunchKernel(fn, SparseRoutingGrid(total), (uint)DefaultBlockSize, a);
    }

    unsafe void ISparseRoutingBackend.TopKWeightedRowMixBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer table,
        IGpuBuffer int32Indices, IGpuBuffer gradWeights, int rows, int numEntries, int k, int channels)
    {
        var fn = SparseRoutingKernel("topk_weighted_row_mix_backward_weights");
        long total = checked((long)rows * k);
        if (total == 0) return;
        using var _ = PushContext();
        IntPtr pg = gradOutput.Handle, pt = table.Handle, pi = int32Indices.Handle, pdw = gradWeights.Handle;
        void** a = stackalloc void*[8];
        a[0] = &pg; a[1] = &pt; a[2] = &pi; a[3] = &pdw;
        a[4] = &rows; a[5] = &numEntries; a[6] = &k; a[7] = &channels;
        LaunchKernel(fn, SparseRoutingGrid(total), (uint)DefaultBlockSize, a);
    }

    unsafe void ISparseRoutingBackend.TopKWeightedRowMixBackwardTable(IGpuBuffer gradOutput, IGpuBuffer int32Indices,
        IGpuBuffer weights, IGpuBuffer gradTable, int rows, int numEntries, int k, int channels)
    {
        long totalOut = checked((long)numEntries * channels);
        if (totalOut == 0) return;
        SparseRoutingCsrPlan.Compute(rows, k, numEntries,
            out long totalSlotsLong, out int numTiles, out int offSorted, out int offTile, out int offChunk, out int totalScratchInts);
        int totalSlots = (int)totalSlotsLong;
        int tileCountInts = checked(numTiles * numEntries);

        var fnZero = SparseRoutingKernel("sparse_zero_int32");
        var fnTileCounts = SparseRoutingKernel("topk_csr_tile_counts");
        var fnEntryPrefix = SparseRoutingKernel("topk_csr_entry_prefix");
        var fnChunkSums = SparseRoutingKernel("topk_csr_chunk_sums");
        var fnChunkFinalize = SparseRoutingKernel("topk_csr_chunk_finalize");
        var fnScatter = SparseRoutingKernel("topk_csr_scatter_slots");
        var fnSum = SparseRoutingKernel("topk_weighted_row_mix_backward_table");

        using var csrBuf = AllocateBuffer(totalScratchInts);
        using var _ = PushContext();
        IntPtr pg = gradOutput.Handle, pi = int32Indices.Handle, pw = weights.Handle, pdt = gradTable.Handle, pcsr = csrBuf.Handle;

        void** aZero = stackalloc void*[3];
        aZero[0] = &pcsr; aZero[1] = &offTile; aZero[2] = &tileCountInts;
        LaunchKernel(fnZero, SparseRoutingGrid(tileCountInts), (uint)DefaultBlockSize, aZero);

        if (totalSlots > 0)
        {
            void** aTc = stackalloc void*[6];
            aTc[0] = &pi; aTc[1] = &pcsr; aTc[2] = &totalSlots; aTc[3] = &numEntries; aTc[4] = &numTiles; aTc[5] = &offTile;
            LaunchKernel(fnTileCounts, SparseRoutingGrid(numTiles), (uint)DefaultBlockSize, aTc);
        }

        void** aEp = stackalloc void*[4];
        aEp[0] = &pcsr; aEp[1] = &numEntries; aEp[2] = &numTiles; aEp[3] = &offTile;
        LaunchKernel(fnEntryPrefix, SparseRoutingGrid(numEntries), (uint)DefaultBlockSize, aEp);

        void** aCs = stackalloc void*[3];
        aCs[0] = &pcsr; aCs[1] = &numEntries; aCs[2] = &offChunk;
        LaunchKernel(fnChunkSums, SparseRoutingGrid(256), (uint)DefaultBlockSize, aCs);

        void** aCf = stackalloc void*[3];
        aCf[0] = &pcsr; aCf[1] = &numEntries; aCf[2] = &offChunk;
        LaunchKernel(fnChunkFinalize, SparseRoutingGrid((long)numEntries + 1), (uint)DefaultBlockSize, aCf);

        if (totalSlots > 0)
        {
            void** aSc = stackalloc void*[7];
            aSc[0] = &pi; aSc[1] = &pcsr; aSc[2] = &totalSlots; aSc[3] = &numEntries;
            aSc[4] = &numTiles; aSc[5] = &offSorted; aSc[6] = &offTile;
            LaunchKernel(fnScatter, SparseRoutingGrid(numTiles), (uint)DefaultBlockSize, aSc);
        }

        void** aSum = stackalloc void*[8];
        aSum[0] = &pg; aSum[1] = &pw; aSum[2] = &pcsr; aSum[3] = &pdt;
        aSum[4] = &numEntries; aSum[5] = &k; aSum[6] = &channels; aSum[7] = &offSorted;
        LaunchKernel(fnSum, SparseRoutingGrid(totalOut), (uint)DefaultBlockSize, aSum);
    }

    unsafe void ISparseRoutingBackend.CountIndicesInt32(IGpuBuffer int32Indices, IGpuBuffer counts, int length, int numBins)
    {
        if (numBins <= 0) return;
        var fnZero = SparseRoutingKernel("sparse_zero_int32");
        var fnAcc = SparseRoutingKernel("count_indices_accumulate_int32");
        var fnFin = SparseRoutingKernel("count_indices_finalize_float");
        using var _ = PushContext();
        IntPtr pi = int32Indices.Handle, pc = counts.Handle;
        int zeroOffset = 0;

        void** aZero = stackalloc void*[3];
        aZero[0] = &pc; aZero[1] = &zeroOffset; aZero[2] = &numBins;
        LaunchKernel(fnZero, SparseRoutingGrid(numBins), (uint)DefaultBlockSize, aZero);

        if (length > 0)
        {
            int totalThreads = System.Math.Min(length, SparseRoutingLimits.MaxReductionThreads);
            void** aAcc = stackalloc void*[5];
            aAcc[0] = &pi; aAcc[1] = &pc; aAcc[2] = &length; aAcc[3] = &numBins; aAcc[4] = &totalThreads;
            LaunchKernel(fnAcc, SparseRoutingGrid(totalThreads), (uint)DefaultBlockSize, aAcc);
        }

        void** aFin = stackalloc void*[2];
        aFin[0] = &pc; aFin[1] = &numBins;
        LaunchKernel(fnFin, SparseRoutingGrid(numBins), (uint)DefaultBlockSize, aFin);
    }

    unsafe void ISparseRoutingBackend.MinMaxInt32(IGpuBuffer int32Data, IGpuBuffer outMinMaxInt32, int length)
    {
        if (length <= 0) return;
        var fnInit = SparseRoutingKernel("minmax_init_int32");
        var fnRed = SparseRoutingKernel("minmax_reduce_int32");
        using var _ = PushContext();
        IntPtr pd = int32Data.Handle, po = outMinMaxInt32.Handle;

        void** aInit = stackalloc void*[1];
        aInit[0] = &po;
        LaunchKernel(fnInit, 1u, (uint)DefaultBlockSize, aInit);

        int totalThreads = System.Math.Min(length, SparseRoutingLimits.MaxReductionThreads);
        void** aRed = stackalloc void*[4];
        aRed[0] = &pd; aRed[1] = &po; aRed[2] = &length; aRed[3] = &totalThreads;
        LaunchKernel(fnRed, SparseRoutingGrid(totalThreads), (uint)DefaultBlockSize, aRed);
    }
}
