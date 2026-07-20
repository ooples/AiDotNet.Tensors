// Copyright (c) AiDotNet. All rights reserved.
// Metal Shading Language (MSL) kernels backing the fused ANN primitives declared by
// IAnnBackend. Supply-chain-clean: custom kernels (no FAISS / cuVS / MKL), numerically
// mirroring the managed AnnPrimitives oracle function-for-function.
//
// Metric codes match AnnPrimitives / AnnMetric:
//   0 = L2  (squared Euclidean, sum((a-b)^2); smaller is nearer)
//   1 = InnerProduct (sum(a*b); larger is nearer)
//
// Thread mapping (one thread per output cell, matching the oracle's loop nests):
//   ann_compute_distances   — one thread per (query, database) pair
//   ann_ivf_assign          — one thread per vector
//   ann_pq_distance_tables  — one thread per (query, subspace, sub-centroid)
//   ann_pq_adc_scan         — one thread per (query, coded vector)
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    /// <summary>
    /// Metal Shading Language implementations for the ANN kernels. Bit-for-bit metric semantics
    /// with <see cref="AiDotNet.Tensors.Ann.AnnPrimitives"/> (the correctness oracle) so the GPU
    /// path and the managed CPU fallback agree.
    /// </summary>
    internal static class MetalAnnKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "ann_compute_distances",
            "ann_ivf_assign",
            "ann_pq_distance_tables",
            "ann_pq_adc_scan",
        };

        public const string Source = @"
#include <metal_stdlib>
using namespace metal;

// Metric codes (mirror AnnPrimitives.MetricL2 / MetricInnerProduct).
constant int ANN_METRIC_L2 = 0;
constant int ANN_METRIC_IP = 1;

// Distance between two length-`dim` subvectors at the given offsets.
// L2 returns the squared distance; InnerProduct returns the dot product.
// Accumulation order matches the oracle's k = 0..dim loop.
inline float ann_distance(
    device const float* a, int aOffset,
    device const float* b, int bOffset,
    int dim, int metric)
{
    if (metric == ANN_METRIC_IP)
    {
        float ip = 0.0f;
        for (int k = 0; k < dim; k++)
            ip += a[aOffset + k] * b[bOffset + k];
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
// ComputeDistances — dense query x database distance matrix, row-major
// [numQueries, numDatabase]. One thread per (q, j) cell.
//   distances[q * numDatabase + j] = metric(query_q, db_j)
// ----------------------------------------------------------------------------
kernel void ann_compute_distances(
    device const float* queries   [[buffer(0)]],
    device const float* database  [[buffer(1)]],
    device float* distances       [[buffer(2)]],
    constant int& numQueries      [[buffer(3)]],
    constant int& numDatabase     [[buffer(4)]],
    constant int& dim             [[buffer(5)]],
    constant int& metric          [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int total = numQueries * numDatabase;
    if ((int)gid >= total) return;
    int q = (int)gid / numDatabase;
    int j = (int)gid % numDatabase;
    distances[gid] = ann_distance(queries, q * dim, database, j * dim, dim, metric);
}

// ----------------------------------------------------------------------------
// IvfAssign — nearest-centroid assignment (argmin for L2, argmax for IP).
// Ties resolve to the lowest centroid index (strict IsBetter, matching the
// oracle). One thread per vector. assignments is an int32 buffer.
// ----------------------------------------------------------------------------
kernel void ann_ivf_assign(
    device const float* vectors   [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device int* assignments       [[buffer(2)]],
    constant int& numVectors      [[buffer(3)]],
    constant int& numCentroids    [[buffer(4)]],
    constant int& dim             [[buffer(5)]],
    constant int& metric          [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= numVectors) return;
    int vOff = i * dim;

    int best = 0;
    // Seed with the worst possible score for the metric (WorstScore in the oracle).
    float bestScore = (metric == ANN_METRIC_IP) ? -INFINITY : INFINITY;
    for (int c = 0; c < numCentroids; c++)
    {
        float score = ann_distance(vectors, vOff, centroids, c * dim, dim, metric);
        // IsBetter: strict comparison keeps the first (lowest-index) winner on ties.
        bool better = (metric == ANN_METRIC_IP) ? (score > bestScore) : (score < bestScore);
        if (better)
        {
            bestScore = score;
            best = c;
        }
    }
    assignments[i] = best;
}

// ----------------------------------------------------------------------------
// PqComputeDistanceTables — PQ asymmetric distance tables.
//   tables[q*(m*ksub) + s*ksub + c] =
//       metric(query subvector [q*m*dsub + s*dsub, len dsub],
//              codebook subcentroid [s*ksub*dsub + c*dsub, len dsub])
// Codebooks layout [subspace][ksub][dsub]. One thread per (q, s, c).
// ----------------------------------------------------------------------------
kernel void ann_pq_distance_tables(
    device const float* queries   [[buffer(0)]],
    device const float* codebooks [[buffer(1)]],
    device float* tables          [[buffer(2)]],
    constant int& numQueries      [[buffer(3)]],
    constant int& m               [[buffer(4)]],
    constant int& ksub            [[buffer(5)]],
    constant int& dsub            [[buffer(6)]],
    constant int& metric          [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int total = numQueries * m * ksub;
    if ((int)gid >= total) return;

    // Decode the flat table index (which is exactly gid: q*(m*ksub)+s*ksub+c).
    int c = (int)gid % ksub;
    int tmp = (int)gid / ksub;
    int s = tmp % m;
    int q = tmp / m;

    int qStride = m * dsub;                 // per-query stride in queries
    int qSubOff = q * qStride + s * dsub;   // query subvector offset
    int cbOff = s * (ksub * dsub) + c * dsub; // codebook sub-centroid offset

    tables[gid] = ann_distance(queries, qSubOff, codebooks, cbOff, dsub, metric);
}

// ----------------------------------------------------------------------------
// PqAdcScan — sum the per-subspace table lookups into an approximate distance.
//   distances[q*numCodes + i] = sum over s of tables[q*m*ksub + s*ksub + codes[i*m + s]]
// codes layout [code][m], uint8. One thread per (q, i).
// ----------------------------------------------------------------------------
kernel void ann_pq_adc_scan(
    device const uchar* codes  [[buffer(0)]],
    device const float* tables [[buffer(1)]],
    device float* distances    [[buffer(2)]],
    constant int& numQueries   [[buffer(3)]],
    constant int& numCodes     [[buffer(4)]],
    constant int& m            [[buffer(5)]],
    constant int& ksub         [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int total = numQueries * numCodes;
    if ((int)gid >= total) return;
    int q = (int)gid / numCodes;
    int i = (int)gid % numCodes;

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
