#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// OpenCL C kernels for the fused approximate-nearest-neighbour (ANN) primitives
/// declared by <see cref="IAnnBackend"/>. These replace the external FAISS/MKL
/// dependency with supply-chain-clean native kernels and are the numerical mirror
/// of the managed <c>AiDotNet.Tensors.Ann.AnnPrimitives</c> oracle.
/// <para>
/// Metric codes match <see cref="AnnMetric"/> / <c>AnnPrimitives</c>:
/// 0 = squared Euclidean (L2², smaller is nearer), 1 = inner product (larger is nearer).
/// </para>
/// All buffers are float32 except <c>codes</c> (uchar) and <c>assignments</c> (int).
/// </summary>
public static class OpenClAnnKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "ann_compute_distances",
        "ann_ivf_assign",
        "ann_pq_distance_tables",
        "ann_pq_adc_scan",
    };

    public static string GetSource() => @"
// ============================================================================
// Fused ANN primitives. Numerical mirror of AnnPrimitives (managed oracle).
// metric: 0 = squared Euclidean (L2^2, smaller nearer), 1 = inner product
// (larger nearer). Ties resolve to the lowest index (strict compare only).
// ============================================================================

// ----------------------------------------------------------------------------
// Dense query x database distance matrix, row-major [numQueries, numDatabase].
// One thread per (q, j) output cell.
// ----------------------------------------------------------------------------
__kernel void ann_compute_distances(
    __global const float* queries, __global const float* database,
    __global float* distances,
    const int numQueries, const int numDatabase, const int dim, const int metric)
{
    int gid = get_global_id(0);
    if (gid >= numQueries * numDatabase) return;
    int q = gid / numDatabase;
    int j = gid % numDatabase;
    int qOff = q * dim;
    int dOff = j * dim;

    float result;
    if (metric == 1) {
        float ip = 0.0f;
        for (int k = 0; k < dim; k++) ip += queries[qOff + k] * database[dOff + k];
        result = ip;
    } else {
        float sum = 0.0f;
        for (int k = 0; k < dim; k++) {
            float d = queries[qOff + k] - database[dOff + k];
            sum += d * d;
        }
        result = sum;
    }
    distances[gid] = result;
}

// ----------------------------------------------------------------------------
// Nearest-centroid assignment (IVF coarse quantizer / Lloyd k-means step).
// argmin (L2) / argmax (IP); ties -> lowest index. One thread per vector.
// ----------------------------------------------------------------------------
__kernel void ann_ivf_assign(
    __global const float* vectors, __global const float* centroids,
    __global int* assignments,
    const int numVectors, const int numCentroids, const int dim, const int metric)
{
    int i = get_global_id(0);
    if (i >= numVectors) return;
    int vOff = i * dim;

    int best = 0;
    float bestScore = (metric == 1) ? -INFINITY : INFINITY;
    for (int c = 0; c < numCentroids; c++) {
        int cOff = c * dim;
        float score;
        if (metric == 1) {
            float ip = 0.0f;
            for (int k = 0; k < dim; k++) ip += vectors[vOff + k] * centroids[cOff + k];
            score = ip;
        } else {
            float sum = 0.0f;
            for (int k = 0; k < dim; k++) {
                float d = vectors[vOff + k] - centroids[cOff + k];
                sum += d * d;
            }
            score = sum;
        }
        // Strict compare -> first (lowest-index) centroid wins on ties.
        bool better = (metric == 1) ? (score > bestScore) : (score < bestScore);
        if (better) {
            bestScore = score;
            best = c;
        }
    }
    assignments[i] = best;
}

// ----------------------------------------------------------------------------
// PQ asymmetric distance tables. tables[q][s][c] = metric(query subvector,
// sub-centroid). Queries [numQueries, m*dsub]; codebooks [m][ksub][dsub];
// tables [numQueries, m, ksub]. One thread per (q, s, c).
// ----------------------------------------------------------------------------
__kernel void ann_pq_distance_tables(
    __global const float* queries, __global const float* codebooks,
    __global float* tables,
    const int numQueries, const int m, const int ksub, const int dsub, const int metric)
{
    int gid = get_global_id(0);
    if (gid >= numQueries * m * ksub) return;
    // gid == q*(m*ksub) + s*ksub + c, so tables[gid] is the target cell.
    int c = gid % ksub;
    int tmp = gid / ksub;
    int s = tmp % m;
    int q = tmp / m;

    int qSubOff = q * (m * dsub) + s * dsub;
    int cbOff = s * (ksub * dsub) + c * dsub;

    float result;
    if (metric == 1) {
        float ip = 0.0f;
        for (int k = 0; k < dsub; k++) ip += queries[qSubOff + k] * codebooks[cbOff + k];
        result = ip;
    } else {
        float sum = 0.0f;
        for (int k = 0; k < dsub; k++) {
            float d = queries[qSubOff + k] - codebooks[cbOff + k];
            sum += d * d;
        }
        result = sum;
    }
    tables[gid] = result;
}

// ----------------------------------------------------------------------------
// PQ ADC scan. distances[q][i] = sum over s of tables[q][s][codes[i][s]].
// codes [numCodes, m] (uchar); output [numQueries, numCodes]. One thread per (q, i).
// ----------------------------------------------------------------------------
__kernel void ann_pq_adc_scan(
    __global const uchar* codes, __global const float* tables,
    __global float* distances,
    const int numQueries, const int numCodes, const int m, const int ksub)
{
    int gid = get_global_id(0);
    if (gid >= numQueries * numCodes) return;
    int q = gid / numCodes;
    int i = gid % numCodes;

    int tblQOff = q * (m * ksub);
    int codeOff = i * m;
    float sum = 0.0f;
    for (int s = 0; s < m; s++)
        sum += tables[tblQOff + s * ksub + codes[codeOff + s]];
    distances[gid] = sum;
}
";
}
#endif
