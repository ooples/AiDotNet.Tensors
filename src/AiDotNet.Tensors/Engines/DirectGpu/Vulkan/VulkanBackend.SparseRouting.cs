// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : ISparseRoutingBackend
{
    private void DispatchSparseRouting(string glsl, IGpuBuffer[] buffers, uint[] scalars, long workItems)
    {
        if (workItems == 0) return;
        if (workItems > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(workItems), "Sparse routing exceeds the Vulkan launch limit.");
        GlslNaryOp(glsl, buffers, (int)workItems, scalars);
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
        => DispatchSparseRouting(VulkanRecurrenceKernels.TopKWeightedRowMixBackwardTable,
            new[] { gradOutput, int32Indices, weights, gradTable },
            new[] { (uint)rows, (uint)numEntries, (uint)k, (uint)channels },
            (long)numEntries * channels);

    void ISparseRoutingBackend.CountIndicesInt32(IGpuBuffer int32Indices, IGpuBuffer counts, int length, int numBins)
        => DispatchSparseRouting(VulkanRecurrenceKernels.CountIndicesInt32,
            new[] { int32Indices, counts },
            new[] { (uint)length, (uint)numBins },
            numBins);
}
