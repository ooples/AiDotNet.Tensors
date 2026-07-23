// Copyright (c) AiDotNet. All rights reserved.
// CUDA launcher shims for the native ANN kernels (IAnnBackend).
// Pattern matches CudaBackend.Detection.cs: kernel resolved from _kernelCache,
// dispatched via 256-thread block / grid-ceil. Numerically mirrors AnnPrimitives.

using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IAnnBackend
{
    private IntPtr ResolveAnnKernel(string name)
    {
        if (_annModule == IntPtr.Zero)
            throw new InvalidOperationException(
                "ANN CUDA module was not compiled (older toolkit?). Falling back to CPU reference.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {name}");
        return kernel;
    }

    public unsafe void ComputeDistances(IGpuBuffer queries, IGpuBuffer database, IGpuBuffer distances,
        int numQueries, int numDatabase, int dim, AnnMetric metric)
    {
        if (numQueries <= 0 || numDatabase <= 0) return;
        long totalLong = (long)numQueries * numDatabase;
        if (totalLong > int.MaxValue)
            throw new OverflowException($"ANN distance matrix total {totalLong} exceeds Int32.MaxValue.");
        if (TryDirectPtxAnnComputeDistances(queries, database, distances, numQueries, numDatabase, dim, metric)) return;
        var kernel = ResolveAnnKernel("ann_compute_distances");
        using var _ = PushContext();
        int total = (int)totalLong;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr qPtr = queries.Handle, dbPtr = database.Handle, distPtr = distances.Handle;
        int nq = numQueries, nd = numDatabase, dd = dim, mm = (int)metric;
        void** args = stackalloc void*[7];
        args[0] = &qPtr; args[1] = &dbPtr; args[2] = &distPtr;
        args[3] = &nq; args[4] = &nd; args[5] = &dd; args[6] = &mm;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void IvfAssign(IGpuBuffer vectors, IGpuBuffer centroids, IGpuBuffer assignments,
        int numVectors, int numCentroids, int dim, AnnMetric metric)
    {
        if (numVectors <= 0 || numCentroids <= 0) return;
        if (TryDirectPtxAnnIvfAssign(vectors, centroids, assignments, numVectors, numCentroids, dim, metric)) return;
        var kernel = ResolveAnnKernel("ann_ivf_assign");
        using var _ = PushContext();
        uint grid = (uint)((numVectors + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr vPtr = vectors.Handle, cPtr = centroids.Handle, aPtr = assignments.Handle;
        int nv = numVectors, nc = numCentroids, dd = dim, mm = (int)metric;
        void** args = stackalloc void*[7];
        args[0] = &vPtr; args[1] = &cPtr; args[2] = &aPtr;
        args[3] = &nv; args[4] = &nc; args[5] = &dd; args[6] = &mm;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void PqComputeDistanceTables(IGpuBuffer queries, IGpuBuffer codebooks, IGpuBuffer tables,
        int numQueries, int m, int ksub, int dsub, AnnMetric metric)
    {
        if (numQueries <= 0 || m <= 0 || ksub <= 0) return;
        long totalLong = (long)numQueries * m * ksub;
        if (totalLong > int.MaxValue)
            throw new OverflowException($"ANN PQ table total {totalLong} exceeds Int32.MaxValue.");
        if (TryDirectPtxAnnPqDistanceTables(queries, codebooks, tables, numQueries, m, ksub, dsub, metric)) return;
        var kernel = ResolveAnnKernel("ann_pq_distance_tables");
        using var _ = PushContext();
        int total = (int)totalLong;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr qPtr = queries.Handle, cbPtr = codebooks.Handle, tPtr = tables.Handle;
        int nq = numQueries, mmm = m, kk = ksub, dd = dsub, met = (int)metric;
        void** args = stackalloc void*[8];
        args[0] = &qPtr; args[1] = &cbPtr; args[2] = &tPtr;
        args[3] = &nq; args[4] = &mmm; args[5] = &kk; args[6] = &dd; args[7] = &met;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void PqAdcScan(IGpuBuffer codes, IGpuBuffer tables, IGpuBuffer distances,
        int numQueries, int numCodes, int m, int ksub)
    {
        if (numQueries <= 0 || numCodes <= 0) return;
        long totalLong = (long)numQueries * numCodes;
        if (totalLong > int.MaxValue)
            throw new OverflowException($"ANN PQ ADC total {totalLong} exceeds Int32.MaxValue.");
        var kernel = ResolveAnnKernel("ann_pq_adc_scan");
        using var _ = PushContext();
        int total = (int)totalLong;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr codesPtr = codes.Handle, tPtr = tables.Handle, distPtr = distances.Handle;
        int nq = numQueries, ncodes = numCodes, mmm = m, kk = ksub;
        void** args = stackalloc void*[7];
        args[0] = &codesPtr; args[1] = &tPtr; args[2] = &distPtr;
        args[3] = &nq; args[4] = &ncodes; args[5] = &mmm; args[6] = &kk;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
}
