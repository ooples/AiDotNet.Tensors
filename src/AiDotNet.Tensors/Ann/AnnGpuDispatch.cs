// Copyright (c) AiDotNet. All rights reserved.
using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Ann
{
    /// <summary>
    /// Dispatch layer for the ANN primitives: runs each op on the fused GPU kernels when the active backend
    /// implements <see cref="IAnnBackend"/> and the problem is large enough to amortise host↔device transfer,
    /// otherwise on the managed <see cref="AnnPrimitives"/> CPU reference. Every GPU path is wrapped so any
    /// runtime failure (allocation, launch, download) transparently falls back to the CPU oracle — the GPU
    /// kernels are validated to be numerically identical to <see cref="AnnPrimitives"/>, so the result is the
    /// same either way. This is what lets <see cref="AnnIndex"/> run entirely on the AiDotNet stack (CPU or any
    /// of the six GPU backends) with no external FAISS / MKL dependency.
    /// </summary>
    internal static class AnnGpuDispatch
    {
        // Minimum number of output elements before a GPU launch is worth the transfer overhead. Below this the
        // managed path is faster (and the GPU path would just add PCIe latency). Deliberately conservative.
        private const long GpuMinElements = 4096;

        private static bool TryGetBackend(IDirectGpuBackend? gpu, long workElements, out IAnnBackend ann)
        {
            ann = null!;
            if (gpu == null || !gpu.IsAvailable) return false;
            if (workElements < GpuMinElements) return false;
            if (gpu is IAnnBackend a) { ann = a; return true; }
            return false;
        }

        internal static void ComputeDistances(IDirectGpuBackend? gpu, float[] queries, float[] database,
            float[] distances, int numQueries, int numDatabase, int dim, int metric)
        {
            if (TryGetBackend(gpu, (long)numQueries * numDatabase, out var ann))
            {
                IGpuBuffer? q = null, db = null, d = null;
                try
                {
                    q = gpu!.AllocateBuffer(queries);
                    db = gpu.AllocateBuffer(database);
                    d = gpu.AllocateBuffer(numQueries * numDatabase);
                    ann.ComputeDistances(q, db, d, numQueries, numDatabase, dim, (AnnMetric)metric);
                    gpu.DownloadBuffer(d, distances);
                    return;
                }
                catch { /* fall through to CPU */ }
                finally { d?.Dispose(); db?.Dispose(); q?.Dispose(); }
            }
            AnnPrimitives.ComputeDistances(queries, database, distances, numQueries, numDatabase, dim, metric);
        }

        internal static void IvfAssign(IDirectGpuBackend? gpu, float[] vectors, float[] centroids,
            int[] assignments, int numVectors, int numCentroids, int dim, int metric)
        {
            if (TryGetBackend(gpu, (long)numVectors * numCentroids, out var ann))
            {
                IGpuBuffer? v = null, c = null, a = null;
                try
                {
                    v = gpu!.AllocateBuffer(vectors);
                    c = gpu.AllocateBuffer(centroids);
                    a = gpu.AllocateByteBuffer(numVectors * sizeof(int)); // int32 assignments
                    ann.IvfAssign(v, c, a, numVectors, numCentroids, dim, (AnnMetric)metric);
                    var bytes = gpu.DownloadByteBuffer(a, numVectors * sizeof(int));
                    Buffer.BlockCopy(bytes, 0, assignments, 0, numVectors * sizeof(int));
                    return;
                }
                catch { /* fall through to CPU */ }
                finally { a?.Dispose(); c?.Dispose(); v?.Dispose(); }
            }
            AnnPrimitives.IvfAssign(vectors, centroids, assignments, numVectors, numCentroids, dim, metric);
        }

        internal static void PqComputeDistanceTables(IDirectGpuBackend? gpu, float[] queries, float[] codebooks,
            float[] tables, int numQueries, int m, int ksub, int dsub, int metric)
        {
            if (TryGetBackend(gpu, (long)numQueries * m * ksub, out var ann))
            {
                IGpuBuffer? q = null, cb = null, t = null;
                try
                {
                    q = gpu!.AllocateBuffer(queries);
                    cb = gpu.AllocateBuffer(codebooks);
                    t = gpu.AllocateBuffer(numQueries * m * ksub);
                    ann.PqComputeDistanceTables(q, cb, t, numQueries, m, ksub, dsub, (AnnMetric)metric);
                    gpu.DownloadBuffer(t, tables);
                    return;
                }
                catch { /* fall through to CPU */ }
                finally { t?.Dispose(); cb?.Dispose(); q?.Dispose(); }
            }
            AnnPrimitives.PqComputeDistanceTables(queries, codebooks, tables, numQueries, m, ksub, dsub, metric);
        }

        internal static void PqAdcScan(IDirectGpuBackend? gpu, byte[] codes, float[] tables, float[] distances,
            int numQueries, int numCodes, int m, int ksub)
        {
            if (TryGetBackend(gpu, (long)numQueries * numCodes, out var ann))
            {
                IGpuBuffer? cBuf = null, t = null, d = null;
                try
                {
                    int codeLen = numCodes * m;
                    // _codes may be over-allocated (geometric growth); upload exactly numCodes*m bytes.
                    byte[] codeSlice = codes;
                    if (codes.Length != codeLen)
                    {
                        codeSlice = new byte[codeLen];
                        Array.Copy(codes, codeSlice, codeLen);
                    }
                    cBuf = gpu!.AllocateByteBuffer(codeLen);
                    gpu.UploadByteBuffer(cBuf, codeSlice);
                    t = gpu.AllocateBuffer(tables);
                    d = gpu.AllocateBuffer(numQueries * numCodes);
                    ann.PqAdcScan(cBuf, t, d, numQueries, numCodes, m, ksub);
                    gpu.DownloadBuffer(d, distances);
                    return;
                }
                catch { /* fall through to CPU */ }
                finally { d?.Dispose(); t?.Dispose(); cBuf?.Dispose(); }
            }
            AnnPrimitives.PqAdcScan(codes, tables, distances, numQueries, numCodes, m, ksub);
        }
    }
}
