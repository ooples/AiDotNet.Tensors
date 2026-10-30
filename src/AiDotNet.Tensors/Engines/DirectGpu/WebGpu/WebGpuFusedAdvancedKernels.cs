// Copyright (c) CrowdTrainer LLC. All rights reserved.

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static class WebGpuFusedAdvancedKernels
{
    private const string ActivationHelper = @"
fn fused_apply_activation(xIn : f32, activation : i32) -> f32 {
    var x = xIn;
    if (activation == 1) { return max(x, 0.0); }
    if (activation == 2) {
        let k = sqrt(2.0 / 3.14159265358979323846);
        return 0.5 * x * (1.0 + tanh(k * (x + 0.044715 * x * x * x)));
    }
    if (activation == 3) { return 1.0 / (1.0 + exp(-x)); }
    if (activation == 4) { return tanh(x); }
    if (activation == 5) { return select(0.01 * x, x, x > 0.0); }
    if (activation == 6) {
        let sigmoid = 1.0 / (1.0 + exp(-x));
        return x * sigmoid;
    }
    return x;
}
";

    // Two-stage fused LoRA forward. See CudaFusedAdvancedKernels.cs for the
    // design rationale. Total work: O(batch · rank · (in + out)) instead of
    // O(batch · in · rank · out).
    //
    // Launch contract (set by the WebGPU dispatch wrapper):
    //   workgroups.x   = batchSize
    //   workgroup_size = (256, 1, 1)
    //   workgroup memory = 256 floats, one rank chunk; the kernel walks any
    //                      rank in chunks of 256 (#1165).
    public const string LoRAForward = @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read> baseOutput : array<f32>;
@group(0) @binding(2) var<storage, read> loraA : array<f32>;
@group(0) @binding(3) var<storage, read> loraB : array<f32>;
@group(0) @binding(4) var<storage, read_write> output : array<f32>;
struct P { batchSize : i32, inputFeatures : i32, rank : i32, outputFeatures : i32, scaling : f32, pad0 : f32, pad1 : f32, pad2 : f32 };
@group(0) @binding(5) var<uniform> p : P;

// The rank is processed in chunks of RANK_CHUNK (#1165): each chunk projects input . loraA[:, chunk] into
// workgroup memory, then the thread owning each output column adds that chunk's proj . loraB[chunk, :]. The first
// chunk writes base + scaling * delta and later chunks accumulate, so any rank works; a rank <= RANK_CHUNK is one
// chunk and computes exactly the single-pass result.
const RANK_CHUNK : u32 = 256u;
var<workgroup> proj : array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) tid : vec3<u32>
) {
    let b = i32(wg.x);
    if (b >= p.batchSize) { return; }
    // Fail closed if host validation regressed: the signed stride `i * p.rank` needs a positive rank.
    if (p.rank <= 0) { return; }

    let blockSize : u32 = 256u;
    let inBase = b * p.inputFeatures;
    let outBase = b * p.outputFeatures;
    let effRank = u32(p.rank);

    var c0 : u32 = 0u;
    loop {
        if (c0 >= effRank) { break; }
        let chunk = min(RANK_CHUNK, effRank - c0);

        // Stage 1: proj[r] = input . loraA[:, c0 + r] for this chunk (one thread per r).
        if (tid.x < chunk) {
            let r = c0 + tid.x;
            var acc : f32 = 0.0;
            for (var i : i32 = 0; i < p.inputFeatures; i = i + 1) {
                acc = acc + input_[inBase + i] * loraA[i * p.rank + i32(r)];
            }
            proj[tid.x] = acc;
        }
        workgroupBarrier();

        // Stage 2: each thread owns output columns o, o + 256, ...; it adds this chunk's contribution.
        var o : u32 = tid.x;
        loop {
            if (i32(o) >= p.outputFeatures) { break; }
            var delta : f32 = 0.0;
            for (var rr : u32 = 0u; rr < chunk; rr = rr + 1u) {
                delta = delta + proj[rr] * loraB[i32(c0 + rr) * p.outputFeatures + i32(o)];
            }
            if (c0 == 0u) {
                output[outBase + i32(o)] = baseOutput[outBase + i32(o)] + p.scaling * delta;
            } else {
                output[outBase + i32(o)] = output[outBase + i32(o)] + p.scaling * delta;
            }
            o = o + blockSize;
        }
        // Every thread is done reading proj before the next chunk overwrites it.
        workgroupBarrier();
        c0 = c0 + RANK_CHUNK;
    }
}
";

    public const string DDIMStep = @"
@group(0) @binding(0) var<storage, read> xT : array<f32>;
@group(0) @binding(1) var<storage, read> epsilonTheta : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
struct P { size : i32, alphaBarT : f32, alphaBarTMinus1 : f32, pad : f32 };
@group(0) @binding(3) var<uniform> p : P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }

    let eps = epsilonTheta[gid];
    // Defence in depth: host-side already validates alphaBarT > 0, but if a
    // regression let it through, sqrt(0) makes the divisor zero and the
    // output Inf/NaN. Clamp away from zero so the worst case is a near-final-
    // step result instead of poisoned output. Mirrors the Vulkan kernel.
    let clampedAt = max(p.alphaBarT, 1e-12);
    let clampedAtm1 = max(p.alphaBarTMinus1, 0.0);
    let x0Pred = (xT[gid] - sqrt(max(0.0, 1.0 - clampedAt)) * eps) / sqrt(clampedAt);
    output[gid] = sqrt(clampedAtm1) * x0Pred + sqrt(max(0.0, 1.0 - clampedAtm1)) * eps;
}
";

    public static string SparseLinear => ActivationHelper + @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read> packedCsr : array<f32>;
@group(0) @binding(2) var<storage, read> sparseValues : array<f32>;
@group(0) @binding(3) var<storage, read> bias : array<f32>;
@group(0) @binding(4) var<storage, read_write> output : array<f32>;
struct P { batchSize : i32, inputFeatures : i32, outputFeatures : i32, nnz : i32, hasBias : i32, activation : i32, pad0 : i32, pad1 : i32 };
@group(0) @binding(5) var<uniform> p : P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.batchSize * p.outputFeatures;
    if (gid >= total) { return; }

    let b = gid / p.outputFeatures;
    let o = gid - b * p.outputFeatures;
    let rowStart = bitcast<i32>(packedCsr[o]);
    let rowEnd = bitcast<i32>(packedCsr[o + 1]);
    let colBase = p.outputFeatures + 1;
    var sum : f32 = select(0.0, bias[o], p.hasBias != 0);

    // Defensive bounds matching CUDA / HIP / Metal / Vulkan: skip the row if
    // the CSR offsets are inverted, negative, or out of nnz range. A
    // malformed CSR otherwise walks past sparseValues / input_ with UB.
    if (rowStart < 0 || rowEnd < rowStart || rowEnd > p.nnz) {
        output[gid] = fused_apply_activation(sum, p.activation);
        return;
    }

    for (var idx : i32 = rowStart; idx < rowEnd; idx = idx + 1) {
        // Per-element bounds: idx must land inside sparseValues [0, nnz),
        // and col must land inside the input feature axis. Skip malformed
        // entries instead of reading past the buffer.
        if (idx >= p.nnz) { break; }
        let col = bitcast<i32>(packedCsr[colBase + idx]);
        if (col < 0 || col >= p.inputFeatures) { continue; }
        sum = sum + input_[b * p.inputFeatures + col] * sparseValues[idx];
    }

    output[gid] = fused_apply_activation(sum, p.activation);
}
";
}
#endif
