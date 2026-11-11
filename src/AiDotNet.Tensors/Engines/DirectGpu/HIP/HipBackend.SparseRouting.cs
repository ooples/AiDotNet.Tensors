// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : ISparseRoutingBackend
{
    private IntPtr SparseRoutingKernel(string name)
        => _kernelCache.TryGetValue(name, out var k) ? k
            : throw new GpuCapabilityUnavailableException($"HIP kernel not found: {name}");

    private static uint SparseRoutingGrid(long total)
    {
        long grid = (total + DefaultBlockSize - 1) / DefaultBlockSize;
        if (grid > uint.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(total), "Sparse routing grid exceeds the HIP launch limit.");
        return (uint)grid;
    }

    unsafe void ISparseRoutingBackend.GatherRowValuesForward(IGpuBuffer values, IGpuBuffer int32Indices, IGpuBuffer output,
        int rows, int cols, int k)
    {
        var fn = SparseRoutingKernel("gather_row_values_forward");
        long total = checked((long)rows * k);
        if (total == 0) return;
        IntPtr pv = values.Handle, pi = int32Indices.Handle, po = output.Handle;
        void** a = stackalloc void*[6];
        a[0] = &pv; a[1] = &pi; a[2] = &po; a[3] = &rows; a[4] = &cols; a[5] = &k;
        LaunchKernel(fn, SparseRoutingGrid(total), DefaultBlockSize, a);
    }

    unsafe void ISparseRoutingBackend.GatherRowValuesBackward(IGpuBuffer gradOutput, IGpuBuffer int32Indices, IGpuBuffer gradValues,
        int rows, int cols, int k)
    {
        var fn = SparseRoutingKernel("gather_row_values_backward");
        long total = checked((long)rows * cols);
        if (total == 0) return;
        IntPtr pg = gradOutput.Handle, pi = int32Indices.Handle, pdv = gradValues.Handle;
        void** a = stackalloc void*[6];
        a[0] = &pg; a[1] = &pi; a[2] = &pdv; a[3] = &rows; a[4] = &cols; a[5] = &k;
        LaunchKernel(fn, SparseRoutingGrid(total), DefaultBlockSize, a);
    }

    unsafe void ISparseRoutingBackend.TopKWeightedRowMixForward(IGpuBuffer table, IGpuBuffer int32Indices, IGpuBuffer weights,
        IGpuBuffer output, int rows, int numEntries, int k, int channels)
    {
        var fn = SparseRoutingKernel("topk_weighted_row_mix_forward");
        long total = checked((long)rows * channels);
        if (total == 0) return;
        IntPtr pt = table.Handle, pi = int32Indices.Handle, pw = weights.Handle, po = output.Handle;
        void** a = stackalloc void*[8];
        a[0] = &pt; a[1] = &pi; a[2] = &pw; a[3] = &po;
        a[4] = &rows; a[5] = &numEntries; a[6] = &k; a[7] = &channels;
        LaunchKernel(fn, SparseRoutingGrid(total), DefaultBlockSize, a);
    }

    unsafe void ISparseRoutingBackend.TopKWeightedRowMixBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer table,
        IGpuBuffer int32Indices, IGpuBuffer gradWeights, int rows, int numEntries, int k, int channels)
    {
        var fn = SparseRoutingKernel("topk_weighted_row_mix_backward_weights");
        long total = checked((long)rows * k);
        if (total == 0) return;
        IntPtr pg = gradOutput.Handle, pt = table.Handle, pi = int32Indices.Handle, pdw = gradWeights.Handle;
        void** a = stackalloc void*[8];
        a[0] = &pg; a[1] = &pt; a[2] = &pi; a[3] = &pdw;
        a[4] = &rows; a[5] = &numEntries; a[6] = &k; a[7] = &channels;
        LaunchKernel(fn, SparseRoutingGrid(total), DefaultBlockSize, a);
    }

    unsafe void ISparseRoutingBackend.TopKWeightedRowMixBackwardTable(IGpuBuffer gradOutput, IGpuBuffer int32Indices,
        IGpuBuffer weights, IGpuBuffer gradTable, int rows, int numEntries, int k, int channels)
    {
        var fn = SparseRoutingKernel("topk_weighted_row_mix_backward_table");
        if ((long)numEntries * channels == 0) return;
        IntPtr pg = gradOutput.Handle, pi = int32Indices.Handle, pw = weights.Handle, pdt = gradTable.Handle;
        void** a = stackalloc void*[8];
        a[0] = &pg; a[1] = &pi; a[2] = &pw; a[3] = &pdt;
        a[4] = &rows; a[5] = &numEntries; a[6] = &k; a[7] = &channels;
        LaunchKernel2D(fn, (uint)((channels + 31) / 32), (uint)numEntries, 32, 8, a);
    }

    unsafe void ISparseRoutingBackend.CountIndicesInt32(IGpuBuffer int32Indices, IGpuBuffer counts, int length, int numBins)
    {
        var fn = SparseRoutingKernel("count_indices_int32");
        if (numBins <= 0) return;
        IntPtr pi = int32Indices.Handle, pc = counts.Handle;
        void** a = stackalloc void*[4];
        a[0] = &pi; a[1] = &pc; a[2] = &length; a[3] = &numBins;
        LaunchKernel(fn, SparseRoutingGrid(numBins), DefaultBlockSize, a);
    }
}
