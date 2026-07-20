// Copyright (c) AiDotNet. All rights reserved.
// WebGPU launcher shims for the native ANN kernels backing IAnnBackend
// (IVF / PQ / IVFPQ / HNSW). Each WGSL source in WebGpuAnnKernels is a
// self-contained compute shader (one pipeline per source), dispatched 1-D
// over the output element count and cached via GetOrCreatePipelineAsync —
// the same best-effort idiom as WebGpuBackend.Detection / WebGpuBackend.Fft.
//
// The *Async methods do the work; the synchronous IAnnBackend members block
// on them (GetAwaiter().GetResult()) so the engine's sync dispatch runs these
// on WebGPU instead of CPU-falling-back. Caveat: safe under native wgpu (no
// captured SynchronizationContext); a single-threaded UI/JS event-loop host
// should call the *Async methods directly.

#if NET7_0_OR_GREATER
using System;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IAnnBackend
{
    private const string AnnModuleKey = "Ann";

    // -----------------------------------------------------------------------
    // Synchronous IAnnBackend surface — blocks on the async workers.
    // -----------------------------------------------------------------------

    void IAnnBackend.ComputeDistances(IGpuBuffer queries, IGpuBuffer database, IGpuBuffer distances,
        int numQueries, int numDatabase, int dim, AnnMetric metric)
        => ComputeDistancesAsync(queries, database, distances, numQueries, numDatabase, dim, metric)
            .GetAwaiter().GetResult();

    void IAnnBackend.IvfAssign(IGpuBuffer vectors, IGpuBuffer centroids, IGpuBuffer assignments,
        int numVectors, int numCentroids, int dim, AnnMetric metric)
        => IvfAssignAsync(vectors, centroids, assignments, numVectors, numCentroids, dim, metric)
            .GetAwaiter().GetResult();

    void IAnnBackend.PqComputeDistanceTables(IGpuBuffer queries, IGpuBuffer codebooks, IGpuBuffer tables,
        int numQueries, int m, int ksub, int dsub, AnnMetric metric)
        => PqComputeDistanceTablesAsync(queries, codebooks, tables, numQueries, m, ksub, dsub, metric)
            .GetAwaiter().GetResult();

    void IAnnBackend.PqAdcScan(IGpuBuffer codes, IGpuBuffer tables, IGpuBuffer distances,
        int numQueries, int numCodes, int m, int ksub)
        => PqAdcScanAsync(codes, tables, distances, numQueries, numCodes, m, ksub)
            .GetAwaiter().GetResult();

    // -----------------------------------------------------------------------
    // Async workers.
    // -----------------------------------------------------------------------

    /// <summary>
    /// Dense query×database distance matrix, row-major <c>[numQueries, numDatabase]</c>.
    /// One shader invocation per (query, db) cell.
    /// </summary>
    public async Task ComputeDistancesAsync(IGpuBuffer queries, IGpuBuffer database, IGpuBuffer distances,
        int numQueries, int numDatabase, int dim, AnnMetric metric)
    {
        ThrowIfNotInitialized();
        if (numQueries <= 0 || numDatabase <= 0 || dim <= 0) return;
        int total = CheckedTotal(numQueries, numDatabase, "ComputeDistances");

        var pipelineId = await GetOrCreatePipelineAsync(
            AnnModuleKey + ":ComputeDistances", WebGpuAnnKernels.ComputeDistances, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(numQueries, numDatabase, dim, (int)metric),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId,
            AsWgpu(queries), AsWgpu(database), AsWgpu(distances));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    /// <summary>
    /// Nearest-centroid assignment (argmin/argmax by metric, ties→lowest index).
    /// One shader invocation per vector; <paramref name="assignments"/> is an int32 buffer.
    /// </summary>
    public async Task IvfAssignAsync(IGpuBuffer vectors, IGpuBuffer centroids, IGpuBuffer assignments,
        int numVectors, int numCentroids, int dim, AnnMetric metric)
    {
        ThrowIfNotInitialized();
        if (numCentroids <= 0)
            throw new ArgumentException("numCentroids must be positive", nameof(numCentroids));
        if (numVectors <= 0 || dim <= 0) return;

        var pipelineId = await GetOrCreatePipelineAsync(
            AnnModuleKey + ":IvfAssign", WebGpuAnnKernels.IvfAssign, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(numVectors, numCentroids, dim, (int)metric),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId,
            AsWgpu(vectors), AsWgpu(centroids), AsWgpu(assignments));
        var (wg, _) = _device.CalculateWorkgroups1D(numVectors);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    /// <summary>
    /// PQ asymmetric distance tables, row-major <c>[query][subspace][ksub]</c>.
    /// One shader invocation per (query, subspace, sub-centroid).
    /// </summary>
    public async Task PqComputeDistanceTablesAsync(IGpuBuffer queries, IGpuBuffer codebooks, IGpuBuffer tables,
        int numQueries, int m, int ksub, int dsub, AnnMetric metric)
    {
        ThrowIfNotInitialized();
        if (numQueries <= 0 || m <= 0 || ksub <= 0 || dsub <= 0) return;
        long mksubLong = (long)m * ksub;
        int total = CheckedTotal(numQueries, (int)Math.Min(mksubLong, int.MaxValue), "PqComputeDistanceTables");
        if (mksubLong > int.MaxValue)
            throw new OverflowException($"PqComputeDistanceTables m*ksub {mksubLong} exceeds Int32.MaxValue.");

        var pipelineId = await GetOrCreatePipelineAsync(
            AnnModuleKey + ":PqComputeDistanceTables", WebGpuAnnKernels.PqComputeDistanceTables, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(numQueries, m, ksub, dsub, (int)metric),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId,
            AsWgpu(queries), AsWgpu(codebooks), AsWgpu(tables));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    /// <summary>
    /// PQ ADC scan, output row-major <c>[numQueries, numCodes]</c>. One shader
    /// invocation per (query, code). <paramref name="codes"/> is a u32 buffer of
    /// PQ code bytes packed 4-per-word (little-endian) — see WebGpuAnnKernels.
    /// </summary>
    public async Task PqAdcScanAsync(IGpuBuffer codes, IGpuBuffer tables, IGpuBuffer distances,
        int numQueries, int numCodes, int m, int ksub)
    {
        ThrowIfNotInitialized();
        if (numQueries <= 0 || numCodes <= 0 || m <= 0 || ksub <= 0) return;
        int total = CheckedTotal(numQueries, numCodes, "PqAdcScan");

        var pipelineId = await GetOrCreatePipelineAsync(
            AnnModuleKey + ":PqAdcScan", WebGpuAnnKernels.PqAdcScan, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(numQueries, numCodes, m, ksub),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId,
            AsWgpu(codes), AsWgpu(tables), AsWgpu(distances));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    private static int CheckedTotal(int a, int b, string op)
    {
        long total = (long)a * b;
        if (total > int.MaxValue)
            throw new OverflowException($"ANN {op} total {total} exceeds Int32.MaxValue.");
        return (int)total;
    }
}
#endif
