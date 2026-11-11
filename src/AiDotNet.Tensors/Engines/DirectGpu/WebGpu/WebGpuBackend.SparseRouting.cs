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
        => DispatchSparseRouting("sr_topk_mix_bwd_tbl", WebGpuRecurrenceKernels.TopKWeightedRowMixBackwardTable,
            new[] { gradOutput, int32Indices, weights, gradTable },
            new[] { rows, numEntries, k, channels },
            (long)numEntries * channels);

    void ISparseRoutingBackend.CountIndicesInt32(IGpuBuffer int32Indices, IGpuBuffer counts, int length, int numBins)
        => DispatchSparseRouting("sr_count_idx_i32", WebGpuRecurrenceKernels.CountIndicesInt32,
            new[] { int32Indices, counts },
            new[] { length, numBins },
            numBins);
}
#endif
