// Copyright (c) AiDotNet. All rights reserved.
// OpenCL launcher shims for the fused ANN kernels (IVF / PQ / IVFPQ / HNSW).
// Mirrors OpenClBackend.Detection.cs's pattern: each method pulls the compiled
// DirectOpenClKernel from _kernelCache and dispatches via kernel.Execute1D with
// 256-thread workgroups. Numerically mirrors AiDotNet.Tensors.Ann.AnnPrimitives.
#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : IAnnBackend
    {
        private const int AnnLocalSize = 256;

        private DirectOpenClKernel GetAnnKernel(string name)
        {
            if (!_kernelCache.TryGetValue(name, out var kernel))
                throw new InvalidOperationException(
                    $"OpenCL ANN kernel not found: {name}. Module may have failed to compile.");
            return kernel;
        }

        private static int RoundUpToAnnGroup(int v) =>
            ((v + AnnLocalSize - 1) / AnnLocalSize) * AnnLocalSize;

        private static IntPtr AnnBufHandle(IGpuBuffer b) =>
            ((DirectOpenClGpuBuffer)b).Buffer.Handle;

        // --------------------------------------------------------------
        // Dense query x database distance matrix — one thread per (q, j).
        // --------------------------------------------------------------

        public void ComputeDistances(IGpuBuffer queries, IGpuBuffer database, IGpuBuffer distances,
            int numQueries, int numDatabase, int dim, AnnMetric metric)
        {
            long totalLong = (long)numQueries * numDatabase;
            if (totalLong > int.MaxValue)
                throw new OverflowException($"ANN distance matrix total {totalLong} exceeds Int32.MaxValue.");
            int total = (int)totalLong;
            if (total <= 0) return;
            var k = GetAnnKernel("ann_compute_distances");
            k.SetArg(0, AnnBufHandle(queries));
            k.SetArg(1, AnnBufHandle(database));
            k.SetArg(2, AnnBufHandle(distances));
            k.SetArg(3, numQueries);
            k.SetArg(4, numDatabase);
            k.SetArg(5, dim);
            k.SetArg(6, (int)metric);
            k.Execute1D(RoundUpToAnnGroup(total), AnnLocalSize);
        }

        // --------------------------------------------------------------
        // Nearest-centroid assignment — one thread per vector.
        // --------------------------------------------------------------

        public void IvfAssign(IGpuBuffer vectors, IGpuBuffer centroids, IGpuBuffer assignments,
            int numVectors, int numCentroids, int dim, AnnMetric metric)
        {
            if (numVectors <= 0) return;
            var k = GetAnnKernel("ann_ivf_assign");
            k.SetArg(0, AnnBufHandle(vectors));
            k.SetArg(1, AnnBufHandle(centroids));
            k.SetArg(2, AnnBufHandle(assignments));
            k.SetArg(3, numVectors);
            k.SetArg(4, numCentroids);
            k.SetArg(5, dim);
            k.SetArg(6, (int)metric);
            k.Execute1D(RoundUpToAnnGroup(numVectors), AnnLocalSize);
        }

        // --------------------------------------------------------------
        // PQ asymmetric distance tables — one thread per (q, s, c).
        // --------------------------------------------------------------

        public void PqComputeDistanceTables(IGpuBuffer queries, IGpuBuffer codebooks, IGpuBuffer tables,
            int numQueries, int m, int ksub, int dsub, AnnMetric metric)
        {
            long totalLong = (long)numQueries * m * ksub;
            if (totalLong > int.MaxValue)
                throw new OverflowException($"ANN PQ table total {totalLong} exceeds Int32.MaxValue.");
            int total = (int)totalLong;
            if (total <= 0) return;
            var k = GetAnnKernel("ann_pq_distance_tables");
            k.SetArg(0, AnnBufHandle(queries));
            k.SetArg(1, AnnBufHandle(codebooks));
            k.SetArg(2, AnnBufHandle(tables));
            k.SetArg(3, numQueries);
            k.SetArg(4, m);
            k.SetArg(5, ksub);
            k.SetArg(6, dsub);
            k.SetArg(7, (int)metric);
            k.Execute1D(RoundUpToAnnGroup(total), AnnLocalSize);
        }

        // --------------------------------------------------------------
        // PQ ADC scan — one thread per (q, i).
        // --------------------------------------------------------------

        public void PqAdcScan(IGpuBuffer codes, IGpuBuffer tables, IGpuBuffer distances,
            int numQueries, int numCodes, int m, int ksub)
        {
            long totalLong = (long)numQueries * numCodes;
            if (totalLong > int.MaxValue)
                throw new OverflowException($"ANN ADC scan total {totalLong} exceeds Int32.MaxValue.");
            int total = (int)totalLong;
            if (total <= 0) return;
            var k = GetAnnKernel("ann_pq_adc_scan");
            k.SetArg(0, AnnBufHandle(codes));
            k.SetArg(1, AnnBufHandle(tables));
            k.SetArg(2, AnnBufHandle(distances));
            k.SetArg(3, numQueries);
            k.SetArg(4, numCodes);
            k.SetArg(5, m);
            k.SetArg(6, ksub);
            k.Execute1D(RoundUpToAnnGroup(total), AnnLocalSize);
        }
    }
}
#endif
