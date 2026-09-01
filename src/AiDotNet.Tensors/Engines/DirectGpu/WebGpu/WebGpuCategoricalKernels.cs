// Copyright (c) AiDotNet. All rights reserved.
// WGSL kernel for categorical sampling — the device twin of CpuEngine.TensorCategoricalSample and
// of the OpenCL/CUDA/HIP/Vulkan/Metal kernels.
//
// WHY THIS IS NOT A LINE-FOR-LINE PORT OF THE OTHERS.
// The CPU reference, and the OpenCL/CUDA/HIP/Vulkan kernels, accumulate the probability sum and the
// cumulative walk in `double`. WGSL HAS NO f64 TYPE — it is absent from the language, not a device
// capability. Accumulating in f32 instead is not an option: the running total drifts, a target
// sitting near a bucket edge selects the neighbouring category, and the exact-parity test treats a
// single differing category as a hard mismatch.
//
// So the accumulation is carried in a TWO-FLOAT (double-single) expansion: a hi/lo pair whose
// combined significand is roughly 48 bits against double's 53. Knuth's two_sum is exact under
// round-to-nearest, so summing N f32 inputs this way is exact whenever the true sum needs no more
// than ~48 bits — which covers any realistic class count, since each addend carries only 24.
//
// WGSL requires IEEE-754 binary32 with round-to-nearest and does NOT permit implementations to
// reassociate floating-point expressions, so unlike the Metal port this needs no pragma to stay
// exact.
//
// NOT EXECUTED IN VERIFICATION: this requires a WebGPU adapter (Dawn) and was written on a host
// without one, so the kernel is unverified against the CPU oracle. The algorithm mirrors the OpenCL
// kernel, which IS verified, with the accumulation type as the single deliberate difference.
#if NET8_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// WGSL implementation of categorical sampling, matching the managed reference's category selection
/// via compensated two-float accumulation (WGSL has no f64).
/// </summary>
internal static class WebGpuCategoricalKernels
{
    public const string CategoricalSample = @"
@group(0) @binding(0) var<storage, read> cs_probabilities: array<f32>;
@group(0) @binding(1) var<storage, read_write> cs_one_hot: array<f32>;
struct CsParams { rows: u32, classes: u32, seed_lo: u32, seed_hi: u32 }
@group(0) @binding(2) var<uniform> cs_params: CsParams;

// Knuth two_sum: exact under round-to-nearest. sum + err == a + b, with no rounding lost.
fn cs_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    return vec2<f32>(s, err);
}

// Add an f32 to a two-float accumulator, renormalising so the pair stays non-overlapping.
fn cs_df_add(acc: vec2<f32>, v: f32) -> vec2<f32> {
    let t = cs_two_sum(acc.x, v);
    return cs_two_sum(t.x, t.y + acc.y);
}

// Multiply a two-float by an f32. fma recovers the exact rounding error of the head product.
fn cs_df_mul(a: vec2<f32>, b: f32) -> vec2<f32> {
    let p = a.x * b;
    var e = fma(a.x, b, -p);
    e = fma(a.y, b, e);
    return cs_two_sum(p, e);
}

// Lexicographic compare of two non-overlapping pairs is a compare of the exact values.
fn cs_df_less(a: vec2<f32>, b: vec2<f32>) -> bool {
    return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

// StatelessRandom.Uniform01(seed, index) — identical constants to the managed helper and to every
// other backend's kernel. Keying on the ROW index lets an invocation compute its own uniform
// without replaying the rows before it.
fn cs_uniform01(seed32: u32, index: u32) -> f32 {
    let state = index * 747796405u + seed32 + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let draw = (word >> 22u) ^ word;
    return f32(draw >> 8u) * (1.0 / 16777216.0);
}

@compute @workgroup_size(256)
fn categorical_sample(@builtin(global_invocation_id) gid: vec3u) {
    let row = gid.x;
    if (row >= cs_params.rows) { return; }

    let seed32 = cs_params.seed_lo ^ cs_params.seed_hi;
    let u = cs_uniform01(seed32, row);
    let offset = row * cs_params.classes;

    var sum = vec2<f32>(0.0, 0.0);
    for (var c: u32 = 0u; c < cs_params.classes; c = c + 1u) {
        sum = cs_df_add(sum, cs_probabilities[offset + c]);
    }

    let target = cs_df_mul(sum, u);
    var cumulative = vec2<f32>(0.0, 0.0);
    var selected: u32 = cs_params.classes - 1u;
    for (var c: u32 = 0u; c < cs_params.classes; c = c + 1u) {
        cumulative = cs_df_add(cumulative, cs_probabilities[offset + c]);
        if (cs_df_less(target, cumulative)) { selected = c; break; }
    }

    for (var c: u32 = 0u; c < cs_params.classes; c = c + 1u) {
        cs_one_hot[offset + c] = 0.0;
    }
    cs_one_hot[offset + selected] = 1.0;
}
";
}
#endif
