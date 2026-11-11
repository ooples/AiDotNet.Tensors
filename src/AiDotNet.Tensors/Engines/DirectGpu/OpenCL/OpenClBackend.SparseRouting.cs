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

    void ISparseRoutingBackend.TopKWeightedRowMixBackwardTable(IGpuBuffer gradOutput, IGpuBuffer int32Indices,
        IGpuBuffer weights, IGpuBuffer gradTable, int rows, int numEntries, int k, int channels)
        => LaunchSparseRouting4Buf("topk_weighted_row_mix_backward_table", gradOutput, int32Indices, weights, gradTable,
            rows, numEntries, k, channels, (long)numEntries * channels);

    void ISparseRoutingBackend.CountIndicesInt32(IGpuBuffer int32Indices, IGpuBuffer counts, int length, int numBins)
    {
        if (_context == null)
            throw new GpuCapabilityUnavailableException("OpenCL context is not initialized. Cannot execute count_indices_int32.");
        if (!_kernelCache.TryGetValue("count_indices_int32", out var kernel))
            throw new GpuCapabilityUnavailableException("OpenCL kernel not found: count_indices_int32");
        if (numBins <= 0) return;
        int localSize = CalculateOptimalWorkGroupSize1D(numBins);
        int globalSize = ((numBins + localSize - 1) / localSize) * localSize;
        uint arg = 0;
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)int32Indices).Buffer.Handle);
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)counts).Buffer.Handle);
        kernel.SetArg(arg++, length);
        kernel.SetArg(arg, numBins);
        kernel.Execute1D(globalSize, localSize);
    }
}
