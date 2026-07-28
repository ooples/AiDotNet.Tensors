// Copyright (c) AiDotNet. All rights reserved.
// Vulkan GLSL compute shaders for the fused ANN primitives declared by IAnnBackend.
// Numerically mirror AiDotNet.Tensors.Ann.AnnPrimitives function-for-function:
//   * Metric 0 (L2)  = squared Euclidean sum((a-b)^2), smaller is nearer.
//   * Metric 1 (IP)  = inner product sum(a*b),         larger is nearer.
// Each shader is compiled to SPIR-V at runtime via VulkanGlslCompiler and dispatched
// with 1-D workgroups of local_size_x = 256, exactly like VulkanDetectionKernels /
// VulkanFftKernels. Scalars (dims + metric) travel as a push-constant block; all
// data SSBOs are float32 except:
//   * IvfAssign 'assignments' — an int32 SSBO (matches AnnPrimitives' int[] output).
//   * PqAdcScan 'codes'       — logically a tightly-packed uint8 byte buffer, but
//     declared here as a uint[] SSBO and unpacked with a little-endian bit-shift
//     (byte b at linear index i lives in word i>>2, at bit (i&3)*8). Vulkan buffer
//     storage is little-endian, so this reproduces the CPU byte[] indexing without
//     requiring the optional GL_EXT_shader_8bit_storage extension.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// GLSL compute shaders (Vulkan) backing <see cref="IAnnBackend"/>. Mirrors
/// <see cref="AiDotNet.Tensors.Ann.AnnPrimitives"/> so the GPU path is numerically
/// identical to the managed correctness oracle.
/// </summary>
public static class VulkanAnnKernels
{
    // 256 is the repo-wide idiomatic work-group size (see VulkanDetectionKernels /
    // VulkanFftKernels); safe on every conformant Vulkan implementation.
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    /// <summary>Logical kernel names, mirroring how the peer kernel files enumerate their shaders.</summary>
    public static readonly string[] KernelNames =
    {
        "ann_compute_distances",
        "ann_ivf_assign",
        "ann_pq_distance_tables",
        "ann_pq_adc_scan",
    };

    // -----------------------------------------------------------------------
    // ann_compute_distances — dense query x database distance matrix.
    // distances[q*numDatabase + j] = metric(query_q, db_j).
    // One invocation per (q, j) cell; dispatched over numQueries*numDatabase.
    // -----------------------------------------------------------------------
    public static string ComputeDistances => Header + @"
layout(set = 0, binding = 0) readonly buffer Q { float queries[]; };
layout(set = 0, binding = 1) readonly buffer D { float database[]; };
layout(set = 0, binding = 2) writeonly buffer R { float distances[]; };
layout(push_constant) uniform P { uint numQueries; uint numDatabase; uint dim; uint metric; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total = numQueries * numDatabase;
    if (gid >= total) return;
    uint q = gid / numDatabase;
    uint j = gid % numDatabase;
    uint qOff = q * dim;
    uint dOff = j * dim;
    float acc = 0.0;
    if (metric == 1u) {
        for (uint k = 0u; k < dim; ++k) acc += queries[qOff + k] * database[dOff + k];
    } else {
        for (uint k = 0u; k < dim; ++k) { float d = queries[qOff + k] - database[dOff + k]; acc += d * d; }
    }
    distances[gid] = acc;
}";

    // -----------------------------------------------------------------------
    // ann_ivf_assign — nearest-centroid assignment (argmin for L2, argmax for IP).
    // Ties resolve to the lowest centroid index (strict comparison + ascending
    // scan, matching AnnPrimitives.IsBetter). assignments is an int32 SSBO.
    // One invocation per vector; dispatched over numVectors.
    // -----------------------------------------------------------------------
    public static string IvfAssign => Header + @"
layout(set = 0, binding = 0) readonly buffer V { float vectors[]; };
layout(set = 0, binding = 1) readonly buffer C { float centroids[]; };
layout(set = 0, binding = 2) writeonly buffer A { int assignments[]; };
layout(push_constant) uniform P { uint numVectors; uint numCentroids; uint dim; uint metric; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= numVectors) return;
    uint vOff = gid * dim;
    int best = 0;
    // WorstScore(metric): +inf for L2, -inf for InnerProduct.
    float bestScore = (metric == 1u) ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u);
    for (uint c = 0u; c < numCentroids; ++c) {
        uint cOff = c * dim;
        float score = 0.0;
        if (metric == 1u) {
            for (uint k = 0u; k < dim; ++k) score += vectors[vOff + k] * centroids[cOff + k];
        } else {
            for (uint k = 0u; k < dim; ++k) { float d = vectors[vOff + k] - centroids[cOff + k]; score += d * d; }
        }
        bool better = (metric == 1u) ? (score > bestScore) : (score < bestScore);
        if (better) { bestScore = score; best = int(c); }
    }
    assignments[gid] = best;
}";

    // -----------------------------------------------------------------------
    // ann_pq_distance_tables — PQ asymmetric distance tables.
    // tables[q*(m*ksub) + s*ksub + c] = metric(query subvector, codebook subcentroid).
    //   query subvector : queries[q*(m*dsub) + s*dsub, len dsub]
    //   subcentroid     : codebooks[s*ksub*dsub + c*dsub, len dsub]
    // One invocation per (q, s, c); dispatched over numQueries*m*ksub.
    // -----------------------------------------------------------------------
    public static string PqDistanceTables => Header + @"
layout(set = 0, binding = 0) readonly buffer Q { float queries[]; };
layout(set = 0, binding = 1) readonly buffer B { float codebooks[]; };
layout(set = 0, binding = 2) writeonly buffer T { float tables[]; };
layout(push_constant) uniform P { uint numQueries; uint m; uint ksub; uint dsub; uint metric; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint mk = m * ksub;
    uint total = numQueries * mk;
    if (gid >= total) return;
    uint q = gid / mk;
    uint rem = gid % mk;
    uint s = rem / ksub;
    uint c = rem % ksub;
    uint qSubOff = q * (m * dsub) + s * dsub;
    uint cbOff = s * ksub * dsub + c * dsub;
    float acc = 0.0;
    if (metric == 1u) {
        for (uint k = 0u; k < dsub; ++k) acc += queries[qSubOff + k] * codebooks[cbOff + k];
    } else {
        for (uint k = 0u; k < dsub; ++k) { float d = queries[qSubOff + k] - codebooks[cbOff + k]; acc += d * d; }
    }
    tables[gid] = acc;
}";

    // -----------------------------------------------------------------------
    // ann_pq_adc_scan — PQ ADC scan.
    // distances[q*numCodes + i] = sum over s of tables[q*m*ksub + s*ksub + codes[i*m + s]].
    // 'codes' is a tightly-packed uint8 byte buffer read here from a uint[] SSBO via
    // a little-endian bit-shift (see file header). One invocation per (q, i);
    // dispatched over numQueries*numCodes.
    // -----------------------------------------------------------------------
    public static string PqAdcScan => Header + @"
layout(set = 0, binding = 0) readonly buffer C { uint codesPacked[]; };
layout(set = 0, binding = 1) readonly buffer T { float tables[]; };
layout(set = 0, binding = 2) writeonly buffer R { float distances[]; };
layout(push_constant) uniform P { uint numQueries; uint numCodes; uint m; uint ksub; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total = numQueries * numCodes;
    if (gid >= total) return;
    uint q = gid / numCodes;
    uint i = gid % numCodes;
    uint tblQOff = q * m * ksub;
    uint codeBase = i * m;
    float sum = 0.0;
    for (uint s = 0u; s < m; ++s) {
        uint byteIdx = codeBase + s;
        uint word = codesPacked[byteIdx >> 2u];
        uint code = (word >> ((byteIdx & 3u) * 8u)) & 0xFFu;
        sum += tables[tblQOff + s * ksub + code];
    }
    distances[gid] = sum;
}";
}
