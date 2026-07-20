// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for the fused ANN primitives declared by IAnnBackend.
// Each function is a single-pass kernel launched over output element count.
// Mirrors AiDotNet.Tensors.Ann.AnnPrimitives function-for-function so the GPU
// path is numerically identical to the managed CPU oracle/fallback.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// Source bundle for the native ANN CUDA kernels (IVF / PQ / IVFPQ / HNSW families).
    /// fp32 only for the vector inputs — the only non-float buffers are PQ <c>codes</c>
    /// (uint8) and IVF <c>assignments</c> (int32). Both metrics (L2² = 0, InnerProduct = 1)
    /// are supported through an <c>int metric</c> kernel parameter, exactly matching
    /// <see cref="AiDotNet.Tensors.Ann.AnnPrimitives"/>.
    /// </summary>
    public static class CudaAnnKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "ann_compute_distances",
            "ann_ivf_assign",
            "ann_pq_distance_tables",
            "ann_pq_adc_scan",
        };

        public static string GetSource() => @"
#include <math.h>
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// metric: 0 = L2 (squared Euclidean, smaller nearer), 1 = InnerProduct (larger nearer).
__device__ __forceinline__ float ann_distance(
    const float* __restrict__ a, int aOff,
    const float* __restrict__ b, int bOff,
    int dim, int metric)
{
    if (metric == 1) {
        float ip = 0.0f;
        for (int k = 0; k < dim; k++) ip += a[aOff + k] * b[bOff + k];
        return ip;
    }
    float sum = 0.0f;
    for (int k = 0; k < dim; k++) {
        float d = a[aOff + k] - b[bOff + k];
        sum += d * d;
    }
    return sum;
}

// ----------------------------------------------------------------------------
// ComputeDistances — dense query x database distance matrix, row-major
// [numQueries, numDatabase]. One thread per (q, j) cell.
// distances[q * numDatabase + j] = metric(query_q, db_j).
// ----------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void ann_compute_distances(
    const float* __restrict__ queries, const float* __restrict__ database,
    float* __restrict__ distances,
    int numQueries, int numDatabase, int dim, int metric)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)numQueries * numDatabase;
    if (gid >= total) return;
    int q = gid / numDatabase;
    int j = gid % numDatabase;
    distances[gid] = ann_distance(queries, q * dim, database, j * dim, dim, metric);
}

// ----------------------------------------------------------------------------
// IvfAssign — nearest-centroid assignment. One thread per vector.
// argmin (L2) / argmax (IP) over centroids; ties resolve to the lowest index
// (ascending scan, strict better replaces). assignments is an int32 buffer.
// ----------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void ann_ivf_assign(
    const float* __restrict__ vectors, const float* __restrict__ centroids,
    int* __restrict__ assignments,
    int numVectors, int numCentroids, int dim, int metric)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVectors) return;
    int vOff = i * dim;
    int best = 0;
    float bestScore = (metric == 1) ? -INFINITY : INFINITY;
    for (int c = 0; c < numCentroids; c++) {
        float score = ann_distance(vectors, vOff, centroids, c * dim, dim, metric);
        bool better = (metric == 1) ? (score > bestScore) : (score < bestScore);
        if (better) {
            bestScore = score;
            best = c;
        }
    }
    assignments[i] = best;
}

// ----------------------------------------------------------------------------
// PqComputeDistanceTables — PQ asymmetric distance tables.
// One thread per (q, s, c). Query subvector at [q*m*dsub + s*dsub, len dsub];
// codebook subcentroid at [s*ksub*dsub + c*dsub, len dsub].
// tables[q*(m*ksub) + s*ksub + c] = metric(query subvec, subcentroid).
// ----------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void ann_pq_distance_tables(
    const float* __restrict__ queries, const float* __restrict__ codebooks,
    float* __restrict__ tables,
    int numQueries, int m, int ksub, int dsub, int metric)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)numQueries * m * ksub;
    if (gid >= total) return;
    int c = gid % ksub;
    int tmp = gid / ksub;
    int s = tmp % m;
    int q = tmp / m;
    int qSubOff = q * (m * dsub) + s * dsub;
    int cbOff = s * (ksub * dsub) + c * dsub;
    tables[gid] = ann_distance(queries, qSubOff, codebooks, cbOff, dsub, metric);
}

// ----------------------------------------------------------------------------
// PqAdcScan — sum precomputed table lookups per (query, coded vector).
// One thread per (q, i). codes is a uint8 buffer, layout [code][m].
// distances[q*numCodes + i] = sum over s of tables[q*m*ksub + s*ksub + codes[i*m + s]].
// ----------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void ann_pq_adc_scan(
    const unsigned char* __restrict__ codes, const float* __restrict__ tables,
    float* __restrict__ distances,
    int numQueries, int numCodes, int m, int ksub)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)numQueries * numCodes;
    if (gid >= total) return;
    int q = gid / numCodes;
    int i = gid % numCodes;
    int tblQOff = q * (m * ksub);
    int codeOff = i * m;
    float sum = 0.0f;
    for (int s = 0; s < m; s++)
        sum += tables[tblQOff + s * ksub + (int)codes[codeOff + s]];
    distances[gid] = sum;
}
";
    }
}
