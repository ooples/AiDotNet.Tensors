// Copyright (c) AiDotNet. All rights reserved.
// WebGPU WGSL compute shaders for the native ANN (IVF / PQ / IVFPQ / HNSW)
// kernels backing IAnnBackend. Supply-chain-clean: no FAISS / cuVS / MKL.
// Each source mirrors AiDotNet.Tensors.Ann.AnnPrimitives function-for-function
// so the GPU path is numerically identical to the managed CPU oracle.
//
// Metric convention (matches AnnMetric / AnnPrimitives):
//   L2 (0)           = squared Euclidean sum((a-b)^2), smaller is nearer.
//   InnerProduct (1) = sum(a*b),                        larger  is nearer.
//
// @workgroup_size(256) + 1-D dispatch: one shader invocation per output
// element — per (query, db) cell for distances, per vector for IVF assign,
// per (query, subspace, sub-centroid) for PQ tables, per (query, code) for ADC.

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// WGSL compute shader sources for the ANN kernels. Bit-for-bit semantics with
/// <see cref="AiDotNet.Tensors.Ann.AnnPrimitives"/>.
/// </summary>
/// <remarks>
/// <para><b>Buffer element types.</b> All storage buffers are <c>array&lt;f32&gt;</c>
/// except:</para>
/// <list type="bullet">
///   <item><c>assignments</c> (IvfAssign output) is <c>array&lt;i32&gt;</c> — one
///     centroid index per vector.</item>
///   <item><c>codes</c> (PqAdcScan input) is <c>array&lt;u32&gt;</c>. WGSL has no
///     u8 type, so the <c>numCodes*m</c> PQ code bytes are uploaded packed 4-per-word
///     (little-endian: byte <c>b</c> occupies bits <c>[8*b, 8*b+8)</c> of the word,
///     matching a raw <c>Buffer.BlockCopy</c> of the host <c>byte[]</c>). The shader
///     recovers byte at linear index <c>idx</c> via
///     <c>(codes[idx/4] &gt;&gt; (8*(idx%4))) &amp; 0xFFu</c>.</item>
/// </list>
/// </remarks>
public static class WebGpuAnnKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "ann_compute_distances",
        "ann_ivf_assign",
        "ann_pq_distance_tables",
        "ann_pq_adc_scan",
    };

    // -----------------------------------------------------------------------
    // ComputeDistances — dense query×database distance matrix.
    // distances[q*numDatabase + j] = metric(query_q, db_j). One invocation
    // per (q, j) cell, 1-D over numQueries*numDatabase.
    // -----------------------------------------------------------------------

    public static string ComputeDistances => @"
@group(0) @binding(0) var<storage, read> queries : array<f32>;
@group(0) @binding(1) var<storage, read> database : array<f32>;
@group(0) @binding(2) var<storage, read_write> distances : array<f32>;
struct P { numQueries: i32, numDatabase: i32, dim: i32, metric: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.numQueries * p.numDatabase;
    if (gid >= total) { return; }
    let q = gid / p.numDatabase;
    let j = gid % p.numDatabase;
    let qOff = q * p.dim;
    let dOff = j * p.dim;
    var acc : f32 = 0.0;
    if (p.metric == 1) {
        for (var k : i32 = 0; k < p.dim; k = k + 1) {
            acc = acc + queries[qOff + k] * database[dOff + k];
        }
    } else {
        for (var k : i32 = 0; k < p.dim; k = k + 1) {
            let d = queries[qOff + k] - database[dOff + k];
            acc = acc + d * d;
        }
    }
    distances[gid] = acc;
}
";

    // -----------------------------------------------------------------------
    // IvfAssign — nearest-centroid assignment. argmin (L2) / argmax (IP),
    // ties resolved to the lowest centroid index (strict comparison, mirroring
    // AnnPrimitives.IsBetter). One invocation per vector, 1-D over numVectors.
    // Seeds bestScore with +inf (L2) / -inf (IP) exactly like WorstScore.
    // -----------------------------------------------------------------------

    public static string IvfAssign => @"
@group(0) @binding(0) var<storage, read> vectors : array<f32>;
@group(0) @binding(1) var<storage, read> centroids : array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments : array<i32>;
struct P { numVectors: i32, numCentroids: i32, dim: i32, metric: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let i = i32(id.x);
    if (i >= p.numVectors) { return; }
    let vOff = i * p.dim;
    var best : i32 = 0;
    var bestScore : f32 = bitcast<f32>(0x7f800000u);   // +inf (WorstScore for L2)
    if (p.metric == 1) { bestScore = bitcast<f32>(0xff800000u); }  // -inf (WorstScore for IP)
    for (var c : i32 = 0; c < p.numCentroids; c = c + 1) {
        let cOff = c * p.dim;
        var score : f32 = 0.0;
        if (p.metric == 1) {
            for (var k : i32 = 0; k < p.dim; k = k + 1) {
                score = score + vectors[vOff + k] * centroids[cOff + k];
            }
        } else {
            for (var k : i32 = 0; k < p.dim; k = k + 1) {
                let d = vectors[vOff + k] - centroids[cOff + k];
                score = score + d * d;
            }
        }
        var better : bool = score < bestScore;         // L2: smaller is nearer
        if (p.metric == 1) { better = score > bestScore; }  // IP: larger is nearer
        if (better) { bestScore = score; best = c; }
    }
    assignments[i] = best;
}
";

    // -----------------------------------------------------------------------
    // PqComputeDistanceTables — PQ asymmetric distance tables.
    // tables[q*(m*ksub) + s*ksub + c] = metric(query subvector s, sub-centroid c
    // of subspace s). Codebooks laid out [subspace][ksub][dsub]. One invocation
    // per (q, s, c), 1-D over numQueries*m*ksub.
    // -----------------------------------------------------------------------

    public static string PqComputeDistanceTables => @"
@group(0) @binding(0) var<storage, read> queries : array<f32>;
@group(0) @binding(1) var<storage, read> codebooks : array<f32>;
@group(0) @binding(2) var<storage, read_write> tables : array<f32>;
struct P { numQueries: i32, m: i32, ksub: i32, dsub: i32, metric: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let mksub = p.m * p.ksub;
    let total = p.numQueries * mksub;
    if (gid >= total) { return; }
    let q = gid / mksub;
    let rem = gid % mksub;
    let s = rem / p.ksub;
    let c = rem % p.ksub;
    let qSubOff = q * (p.m * p.dsub) + s * p.dsub;
    let cbOff = s * (p.ksub * p.dsub) + c * p.dsub;
    var acc : f32 = 0.0;
    if (p.metric == 1) {
        for (var k : i32 = 0; k < p.dsub; k = k + 1) {
            acc = acc + queries[qSubOff + k] * codebooks[cbOff + k];
        }
    } else {
        for (var k : i32 = 0; k < p.dsub; k = k + 1) {
            let d = queries[qSubOff + k] - codebooks[cbOff + k];
            acc = acc + d * d;
        }
    }
    tables[gid] = acc;
}
";

    // -----------------------------------------------------------------------
    // PqAdcScan — sum per-subspace table lookups into an approximate distance.
    // distances[q*numCodes + i] = sum_s tables[q*m*ksub + s*ksub + codes[i*m + s]].
    // codes is array<u32>: PQ code bytes packed 4-per-word, little-endian.
    // One invocation per (q, i), 1-D over numQueries*numCodes.
    // -----------------------------------------------------------------------

    public static string PqAdcScan => @"
@group(0) @binding(0) var<storage, read> codes : array<u32>;
@group(0) @binding(1) var<storage, read> tables : array<f32>;
@group(0) @binding(2) var<storage, read_write> distances : array<f32>;
struct P { numQueries: i32, numCodes: i32, m: i32, ksub: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.numQueries * p.numCodes;
    if (gid >= total) { return; }
    let q = gid / p.numCodes;
    let i = gid % p.numCodes;
    let tblQOff = q * p.m * p.ksub;
    let codeBase = i * p.m;
    var sum : f32 = 0.0;
    for (var s : i32 = 0; s < p.m; s = s + 1) {
        let idx = codeBase + s;                 // linear byte index into codes
        let word = codes[idx / 4];
        let shift = u32((idx % 4) * 8);
        let code = i32((word >> shift) & 0xFFu);
        sum = sum + tables[tblQOff + s * p.ksub + code];
    }
    distances[gid] = sum;
}
";
}
#endif
