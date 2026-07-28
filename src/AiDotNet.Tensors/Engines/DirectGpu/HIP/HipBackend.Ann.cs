// Copyright (c) AiDotNet. All rights reserved.
// HIP launcher shims for the fused ANN kernels (IAnnBackend). Kernels resolved
// from _kernelCache, dispatched via 256-thread block / grid-ceil, mirroring the
// detection idiom. Numerically identical to AnnPrimitives (the CPU oracle).
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IAnnBackend
{
    // ANN kernel module handle. Compiled best-effort in HipBackend.cs; zero when
    // hipRTC rejected the source (callers then route to the CpuEngine reference).
    private IntPtr _annModule;

    private IntPtr ResolveAnnKernel(string name)
    {
        if (_annModule == IntPtr.Zero)
            throw new InvalidOperationException(
                "ANN HIP module was not compiled (hipRTC rejected source?). " +
                "DirectGpuTensorEngine catches this and routes to the AnnPrimitives reference; " +
                "direct callers to HipBackend see this exception.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"HIP kernel not found: {name}");
        return kernel;
    }

    public unsafe void ComputeDistances(IGpuBuffer queries, IGpuBuffer database, IGpuBuffer distances,
        int numQueries, int numDatabase, int dim, AnnMetric metric)
    {
        if (numQueries <= 0 || numDatabase <= 0) return;
        long totalLong = (long)numQueries * numDatabase;
        if (totalLong > int.MaxValue)
            throw new OverflowException($"ComputeDistances total {totalLong} exceeds Int32.MaxValue.");
        var kernel = ResolveAnnKernel("ann_compute_distances");
        int total = (int)totalLong;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr qPtr = queries.Handle, dbPtr = database.Handle, oPtr = distances.Handle;
        int nq = numQueries, nd = numDatabase, dd = dim, mm = (int)metric;
        void** args = stackalloc void*[7];
        args[0] = &qPtr; args[1] = &dbPtr; args[2] = &oPtr;
        args[3] = &nq; args[4] = &nd; args[5] = &dd; args[6] = &mm;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void IvfAssign(IGpuBuffer vectors, IGpuBuffer centroids, IGpuBuffer assignments,
        int numVectors, int numCentroids, int dim, AnnMetric metric)
    {
        if (numVectors <= 0) return;
        var kernel = ResolveAnnKernel("ann_ivf_assign");
        uint grid = (uint)((numVectors + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr vPtr = vectors.Handle, cPtr = centroids.Handle, aPtr = assignments.Handle;
        int nv = numVectors, nc = numCentroids, dd = dim, mm = (int)metric;
        void** args = stackalloc void*[7];
        args[0] = &vPtr; args[1] = &cPtr; args[2] = &aPtr;
        args[3] = &nv; args[4] = &nc; args[5] = &dd; args[6] = &mm;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void PqComputeDistanceTables(IGpuBuffer queries, IGpuBuffer codebooks, IGpuBuffer tables,
        int numQueries, int m, int ksub, int dsub, AnnMetric metric)
    {
        if (numQueries <= 0 || m <= 0 || ksub <= 0) return;
        long totalLong = (long)numQueries * m * ksub;
        if (totalLong > int.MaxValue)
            throw new OverflowException($"PqComputeDistanceTables total {totalLong} exceeds Int32.MaxValue.");
        var kernel = ResolveAnnKernel("ann_pq_distance_tables");
        int total = (int)totalLong;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr qPtr = queries.Handle, cbPtr = codebooks.Handle, tPtr = tables.Handle;
        int nq = numQueries, mm = m, kk = ksub, ds = dsub, met = (int)metric;
        void** args = stackalloc void*[8];
        args[0] = &qPtr; args[1] = &cbPtr; args[2] = &tPtr;
        args[3] = &nq; args[4] = &mm; args[5] = &kk; args[6] = &ds; args[7] = &met;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void PqAdcScan(IGpuBuffer codes, IGpuBuffer tables, IGpuBuffer distances,
        int numQueries, int numCodes, int m, int ksub)
    {
        if (numQueries <= 0 || numCodes <= 0) return;
        long totalLong = (long)numQueries * numCodes;
        if (totalLong > int.MaxValue)
            throw new OverflowException($"PqAdcScan total {totalLong} exceeds Int32.MaxValue.");
        var kernel = ResolveAnnKernel("ann_pq_adc_scan");
        int total = (int)totalLong;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr cPtr = codes.Handle, tPtr = tables.Handle, oPtr = distances.Handle;
        int nq = numQueries, ncodes = numCodes, mm = m, kk = ksub;
        void** args = stackalloc void*[7];
        args[0] = &cPtr; args[1] = &tPtr; args[2] = &oPtr;
        args[3] = &nq; args[4] = &ncodes; args[5] = &mm; args[6] = &kk;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }
}
