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
        => DispatchSparseRouting("topk_weighted_row_mix_backward_table",
            new[] { M(gradOutput), M(int32Indices), M(weights), M(gradTable) },
            new[] { rows, numEntries, k, channels },
            (long)numEntries * channels);

    void ISparseRoutingBackend.CountIndicesInt32(IGpuBuffer int32Indices, IGpuBuffer counts, int length, int numBins)
        => DispatchSparseRouting("count_indices_int32",
            new[] { M(int32Indices), M(counts) },
            new[] { length, numBins },
            numBins);
}
