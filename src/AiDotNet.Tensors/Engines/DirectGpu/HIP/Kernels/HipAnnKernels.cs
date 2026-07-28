// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for the fused ANN primitives (IAnnBackend). Supply-chain-clean
// replacement for FaissNet/MKL — mirrors AiDotNet.Tensors.Ann.AnnPrimitives
// function-for-function so the GPU path is numerically identical to the managed
// CPU oracle/fallback. HIP's hiprtc accepts CUDA-style source, so this file is
// structurally identical to the (would-be) CUDA variant.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels
{
    /// <summary>
    /// HIP/ROCm kernels backing <see cref="IAnnBackend"/>: dense distance matrix,
    /// nearest-centroid assignment, and PQ asymmetric distance table + ADC scan.
    /// Metric selection matches <c>AnnMetric</c> / <c>AnnPrimitives</c>: metric 0 =
    /// squared-L2 (smaller nearer), metric 1 = inner product (larger nearer).
    /// fp32 throughout except PQ codes (uint8) and IVF assignments (int32).
    /// </summary>
    public static class HipAnnKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "ann_compute_distances",
            "ann_ivf_assign",
            "ann_pq_distance_tables",
            "ann_pq_adc_scan",
        };

        public static string GetSource() => @"
// HIP-RTC device source — no #include needed.
// metric: 0 = squared Euclidean (L2), 1 = inner product. Mirrors AnnPrimitives.Distance.
__device__ __forceinline__ float ann_distance(
    const float* __restrict__ a, int aOffset,
    const float* __restrict__ b, int bOffset, int dim, int metric)
{
    if (metric == 1)
    {
        float ip = 0.0f;
        for (int k = 0; k < dim; k++) ip += a[aOffset + k] * b[bOffset + k];
        return ip;
    }
    float sum = 0.0f;
    for (int k = 0; k < dim; k++)
    {
        float d = a[aOffset + k] - b[bOffset + k];
        sum += d * d;
    }
    return sum;
}

// ----------------------------------------------------------------------------
// ComputeDistances — one thread per (q, j) cell of the [numQueries, numDatabase]
// matrix. distances[q*numDatabase + j] = metric(query_q, db_j).
// ----------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void ann_compute_distances(
    const float* __restrict__ queries, const float* __restrict__ database,
    float* __restrict__ distances, int numQueries, int numDatabase, int dim, int metric)
{
    long total = (long)numQueries * (long)numDatabase;
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int q = (int)(gid / numDatabase);
    int j = (int)(gid % numDatabase);
    distances[gid] = ann_distance(queries, q * dim, database, j * dim, dim, metric);
}

// ----------------------------------------------------------------------------
// IvfAssign — one thread per vector. argmin (L2) / argmax (IP), ties -> lowest
// index. Seed best = centroid 0 then scan 1..numCentroids using a strict
// comparison, matching AnnPrimitives (WorstScore + IsBetter with strict > / <).
// ----------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void ann_ivf_assign(
    const float* __restrict__ vectors, const float* __restrict__ centroids,
    int* __restrict__ assignments, int numVectors, int numCentroids, int dim, int metric)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVectors) return;
    if (numCentroids <= 0) return;
    int vOff = i * dim;
    int best = 0;
    float bestScore = ann_distance(vectors, vOff, centroids, 0, dim, metric);
    for (int c = 1; c < numCentroids; c++)
    {
        float score = ann_distance(vectors, vOff, centroids, c * dim, dim, metric);
        bool better = (metric == 1) ? (score > bestScore) : (score < bestScore);
        if (better)
        {
            bestScore = score;
            best = c;
        }
    }
    assignments[i] = best;
}

// ----------------------------------------------------------------------------
// PqComputeDistanceTables — one thread per (q, s, c). Codebooks are subspace-major
// [m][ksub][dsub]; tables are [q][m][ksub] row-major.
// tables[q*(m*ksub) + s*ksub + c] = metric(query subvector s, sub-centroid c of s).
// ----------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void ann_pq_distance_tables(
    const float* __restrict__ queries, const float* __restrict__ codebooks,
    float* __restrict__ tables, int numQueries, int m, int ksub, int dsub, int metric)
{
    long total = (long)numQueries * (long)m * (long)ksub;
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    long mk = (long)m * (long)ksub;
    int q = (int)(gid / mk);
    int rem = (int)(gid % mk);
    int s = rem / ksub;
    int c = rem % ksub;
    int qSubOff = q * (m * dsub) + s * dsub;
    int cbOff = s * (ksub * dsub) + c * dsub;
    tables[gid] = ann_distance(queries, qSubOff, codebooks, cbOff, dsub, metric);
}

// ----------------------------------------------------------------------------
// PqAdcScan — one thread per (q, i). Sums the per-subspace table lookups selected
// by the code bytes. codes are [numCodes][m] uint8; output [numQueries, numCodes].
// distances[q*numCodes + i] = sum_s tables[q*m*ksub + s*ksub + codes[i*m + s]].
// ----------------------------------------------------------------------------
extern ""C"" __global__ __launch_bounds__(256) void ann_pq_adc_scan(
    const unsigned char* __restrict__ codes, const float* __restrict__ tables,
    float* __restrict__ distances, int numQueries, int numCodes, int m, int ksub)
{
    long total = (long)numQueries * (long)numCodes;
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int q = (int)(gid / numCodes);
    int i = (int)(gid % numCodes);
    long tblQOff = (long)q * (long)m * (long)ksub;
    int codeOff = i * m;
    float sum = 0.0f;
    for (int s = 0; s < m; s++)
        sum += tables[tblQOff + (long)s * ksub + (int)codes[codeOff + s]];
    distances[gid] = sum;
}
";
    }
}
