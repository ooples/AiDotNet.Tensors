// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Internal capability interface for sparse top-K MoE / codebook routing primitives:
/// <list type="bullet">
///   <item><see cref="CpuEngine.TensorGatherRowValues{T}"/> and its backward</item>
///   <item><see cref="CpuEngine.TopKWeightedRowMix{T}"/> and its backwards</item>
///   <item><see cref="CpuEngine.TensorCountIndices{T}"/> (fixed-bin deterministic index histogram)</item>
/// </list>
/// Every kernel writes all output elements directly without atomics, scanning reduction axes in fixed ascending order
/// so forward and backward execution are bit-deterministic across runs. Internal so external
/// <see cref="IDirectGpuBackend"/> implementers are not broken.
/// </summary>
internal interface ISparseRoutingBackend
{
    /// <summary>output[r, j] = (0 &lt;= idx &lt; cols) ? values[r, idx] : 0, where idx = int32Indices[r, j].</summary>
    void GatherRowValuesForward(IGpuBuffer values, IGpuBuffer int32Indices, IGpuBuffer output,
        int rows, int cols, int k);

    /// <summary>gradValues[r, c] = sum_{j : int32Indices[r, j] == c} gradOutput[r, j].</summary>
    void GatherRowValuesBackward(IGpuBuffer gradOutput, IGpuBuffer int32Indices, IGpuBuffer gradValues,
        int rows, int cols, int k);

    /// <summary>output[r, c] = sum_{j=0..k-1} weights[r, j] * table[int32Indices[r, j], c].</summary>
    void TopKWeightedRowMixForward(IGpuBuffer table, IGpuBuffer int32Indices, IGpuBuffer weights, IGpuBuffer output,
        int rows, int numEntries, int k, int channels);

    /// <summary>gradWeights[r, j] = sum_{c=0..channels-1} gradOutput[r, c] * table[int32Indices[r, j], c].</summary>
    void TopKWeightedRowMixBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer table, IGpuBuffer int32Indices,
        IGpuBuffer gradWeights, int rows, int numEntries, int k, int channels);

    /// <summary>gradTable[g, c] = sum_{r, j : int32Indices[r, j] == g} weights[r, j] * gradOutput[r, c].</summary>
    void TopKWeightedRowMixBackwardTable(IGpuBuffer gradOutput, IGpuBuffer int32Indices, IGpuBuffer weights,
        IGpuBuffer gradTable, int rows, int numEntries, int k, int channels);

    /// <summary>counts[g] = (float) |{ i in [0, length) : int32Indices[i] == g }| for g in [0, numBins).</summary>
    void CountIndicesInt32(IGpuBuffer int32Indices, IGpuBuffer counts, int length, int numBins);
}

internal static class SparseRoutingResolver
{
    /// <summary>The physical backend's sparse-routing kernels, or null when unavailable or deferred-recording.</summary>
    public static ISparseRoutingBackend? Resolve(IDirectGpuBackend backend)
    {
        while (backend is Gpu.DelegatingGpuBackend wrapper)
        {
            if (wrapper is Gpu.RecordingGpuBackend { IsRecording: true, ExecuteWhileRecording: false })
                return null;
            backend = wrapper.InnerBackend;
        }
        return backend as ISparseRoutingBackend;
    }
}
