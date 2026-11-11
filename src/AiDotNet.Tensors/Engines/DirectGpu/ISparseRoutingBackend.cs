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

    /// <summary>outMinMaxInt32[0] = min_i int32Data[i], outMinMaxInt32[1] = max_i int32Data[i] (raw int32 bits, length &gt; 0).</summary>
    void MinMaxInt32(IGpuBuffer int32Data, IGpuBuffer outMinMaxInt32, int length);
}

internal static class SparseRoutingResolver
{
    /// <summary>
    /// Resolves sparse-routing execution for <paramref name="backend"/>: returns an outer <see cref="Gpu.RecordingGpuBackend"/>
    /// during eager recording so trace-by-execution graphs capture sparse-routing ops, returns <see langword="null"/>
    /// during non-eager deferred recording or when the underlying physical backend lacks sparse-routing kernels, and
    /// otherwise returns the physical <see cref="ISparseRoutingBackend"/>.
    /// </summary>
    public static ISparseRoutingBackend? Resolve(IDirectGpuBackend backend)
    {
        var terminal = backend;
        while (terminal is Gpu.DelegatingGpuBackend w)
        {
            if (w is Gpu.RecordingGpuBackend { IsRecording: true, ExecuteWhileRecording: false })
                return null;
            terminal = w.InnerBackend;
        }
        if (terminal is not ISparseRoutingBackend)
            return null;

        var current = backend;
        while (current is Gpu.DelegatingGpuBackend wrapper)
        {
            if (current is ISparseRoutingBackend capable)
                return capable;
            current = wrapper.InnerBackend;
        }
        return current as ISparseRoutingBackend;
    }

    public static ISparseRoutingBackend RequireInner(IDirectGpuBackend backend)
    {
        var terminal = backend;
        while (terminal is Gpu.DelegatingGpuBackend w)
            terminal = w.InnerBackend;
        if (terminal is not ISparseRoutingBackend)
        {
            throw new GpuCapabilityUnavailableException(
                $"{terminal.GetType().Name} does not implement sparse-routing kernels.");
        }

        var current = backend;
        while (current is Gpu.DelegatingGpuBackend wrapper)
        {
            if (current is ISparseRoutingBackend capable)
                return capable;
            current = wrapper.InnerBackend;
        }
        return (ISparseRoutingBackend)current;
    }
}

internal static class SparseRoutingCsrPlan
{
    private const long MaxTileMatrixInts = 1L << 22; // 4M ints (16 MB) cap on tileCounts[numTiles, numEntries]

    public static void Compute(int rows, int k, int numEntries,
        out long totalSlots, out int numTiles, out int offSorted, out int offTile, out int offChunk, out int totalScratchInts)
    {
        totalSlots = checked((long)rows * k);
        if (totalSlots > int.MaxValue - numEntries - 512)
            throw new ArgumentOutOfRangeException(nameof(rows), "Sparse routing slot count exceeds CSR scratch indexing limits.");
        int maxTilesByMem = (int)System.Math.Max(1L, System.Math.Min(256L, MaxTileMatrixInts / System.Math.Max(1, numEntries)));
        int desiredTiles = (int)System.Math.Min(256L, (totalSlots + 255L) / 256L);
        numTiles = System.Math.Max(1, System.Math.Min(desiredTiles, maxTilesByMem));
        offSorted = checked(numEntries + 1);
        offTile = checked(offSorted + (int)totalSlots);
        offChunk = checked(offTile + checked(numTiles * numEntries));
        totalScratchInts = checked(offChunk + 256);
    }
}
