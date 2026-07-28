// Copyright (c) AiDotNet. All rights reserved.
// Metal launcher shims for the fused ANN kernels (IAnnBackend). Mirrors
// MetalBackend.Detection.cs — pipeline resolved via shader library handle,
// dispatched with a 1-D thread grid sized to total output cells. One thread
// per output cell, matching the AnnPrimitives oracle loop nests.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IAnnBackend
{
    private const string AnnLibName = "Ann";

    private MetalPipelineState GetAnnPipeline(string kernelName)
    {
        if (_annLibrary == IntPtr.Zero)
            throw new InvalidOperationException(
                "Metal ANN library was not compiled (shader compile failed at init). " +
                "Callers should catch and fall back to the AnnPrimitives CPU reference.");
        return GetPipeline(AnnLibName, _annLibrary, kernelName);
    }

    private static int CheckedTotal(long total, string op)
    {
        if (total > int.MaxValue)
            throw new OverflowException($"{op} total {total} exceeds Int32.MaxValue.");
        return (int)total;
    }

    /// <inheritdoc />
    public void ComputeDistances(IGpuBuffer queries, IGpuBuffer database, IGpuBuffer distances,
        int numQueries, int numDatabase, int dim, AnnMetric metric)
    {
        if (numQueries <= 0 || numDatabase <= 0) return;
        ThrowIfDisposed();
        if (queries is not MetalGpuBuffer qBuf || database is not MetalGpuBuffer dbBuf
            || distances is not MetalGpuBuffer distBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        int total = CheckedTotal((long)numQueries * numDatabase, "ComputeDistances");
        var pipeline = GetAnnPipeline("ann_compute_distances");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(qBuf, 0);
        encoder.SetBuffer(dbBuf, 1);
        encoder.SetBuffer(distBuf, 2);
        encoder.SetBytes(numQueries, 3);
        encoder.SetBytes(numDatabase, 4);
        encoder.SetBytes(dim, 5);
        encoder.SetBytes((int)metric, 6);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <inheritdoc />
    public void IvfAssign(IGpuBuffer vectors, IGpuBuffer centroids, IGpuBuffer assignments,
        int numVectors, int numCentroids, int dim, AnnMetric metric)
    {
        if (numVectors <= 0) return;
        if (numCentroids <= 0)
            throw new ArgumentException("numCentroids must be positive", nameof(numCentroids));
        ThrowIfDisposed();
        if (vectors is not MetalGpuBuffer vBuf || centroids is not MetalGpuBuffer cBuf
            || assignments is not MetalGpuBuffer aBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        var pipeline = GetAnnPipeline("ann_ivf_assign");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(numVectors);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(vBuf, 0);
        encoder.SetBuffer(cBuf, 1);
        encoder.SetBuffer(aBuf, 2);
        encoder.SetBytes(numVectors, 3);
        encoder.SetBytes(numCentroids, 4);
        encoder.SetBytes(dim, 5);
        encoder.SetBytes((int)metric, 6);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <inheritdoc />
    public void PqComputeDistanceTables(IGpuBuffer queries, IGpuBuffer codebooks, IGpuBuffer tables,
        int numQueries, int m, int ksub, int dsub, AnnMetric metric)
    {
        if (numQueries <= 0 || m <= 0 || ksub <= 0) return;
        ThrowIfDisposed();
        if (queries is not MetalGpuBuffer qBuf || codebooks is not MetalGpuBuffer cbBuf
            || tables is not MetalGpuBuffer tblBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        int total = CheckedTotal((long)numQueries * m * ksub, "PqComputeDistanceTables");
        var pipeline = GetAnnPipeline("ann_pq_distance_tables");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(qBuf, 0);
        encoder.SetBuffer(cbBuf, 1);
        encoder.SetBuffer(tblBuf, 2);
        encoder.SetBytes(numQueries, 3);
        encoder.SetBytes(m, 4);
        encoder.SetBytes(ksub, 5);
        encoder.SetBytes(dsub, 6);
        encoder.SetBytes((int)metric, 7);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <inheritdoc />
    public void PqAdcScan(IGpuBuffer codes, IGpuBuffer tables, IGpuBuffer distances,
        int numQueries, int numCodes, int m, int ksub)
    {
        if (numQueries <= 0 || numCodes <= 0) return;
        ThrowIfDisposed();
        if (codes is not MetalGpuBuffer codesBuf || tables is not MetalGpuBuffer tblBuf
            || distances is not MetalGpuBuffer distBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        int total = CheckedTotal((long)numQueries * numCodes, "PqAdcScan");
        var pipeline = GetAnnPipeline("ann_pq_adc_scan");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(codesBuf, 0);
        encoder.SetBuffer(tblBuf, 1);
        encoder.SetBuffer(distBuf, 2);
        encoder.SetBytes(numQueries, 3);
        encoder.SetBytes(numCodes, 4);
        encoder.SetBytes(m, 5);
        encoder.SetBytes(ksub, 6);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }
}
