// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// The forward max pools in WGSL, decided on IEEE-754 bit patterns (<see cref="ExactFloatKernels"/>) over u32-bound
/// buffers. They replace the max_pool kernels of <see cref="WebGpuKernels"/> (whose public source strings stay as
/// shipped): WGSL lets an implementation flush subnormals and assume no NaN in a float compare, and their -f32::MAX
/// seeds answered -f32::MAX for a -inf window. The rule is the CPU engine's: a window's maximum is its first in-bounds
/// cell, replaced only by a strictly greater one, so the first of equal maxima (and of +0 / -0) wins, a NaN first cell
/// stays and a later NaN never wins. Each module has the binding layout and uniforms of the kernel it replaces.
/// </summary>
internal static class WebGpuExactPoolKernels
{
    /// <summary>max_pool2d: binding 0 input, 1 output, 2 PoolParams (as PoolingSource).</summary>
    public const string MaxPool2D = ExactFloatKernels.Wgsl + @"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct PoolParams {
    batch_size: u32, channels: u32, in_height: u32, in_width: u32, out_height: u32, out_width: u32,
    kernel_height: u32, kernel_width: u32, stride_h: u32, stride_w: u32, pad_h: u32, pad_w: u32,
}
@group(0) @binding(2) var<uniform> params: PoolParams;

@compute @workgroup_size(8, 8, 1)
fn max_pool2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_x = gid.x;
    let out_y = gid.y;
    let c = gid.z % params.channels;
    let batch = gid.z / params.channels;
    if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch_size) { return; }
    var max_bits: u32 = 0xFF800000u;
    var have = false;
    for (var ky: u32 = 0u; ky < params.kernel_height; ky = ky + 1u) {
        for (var kx: u32 = 0u; kx < params.kernel_width; kx = kx + 1u) {
            let in_y = i32(out_y * params.stride_h + ky) - i32(params.pad_h);
            let in_x = i32(out_x * params.stride_w + kx) - i32(params.pad_w);
            if (in_y >= 0 && in_y < i32(params.in_height) && in_x >= 0 && in_x < i32(params.in_width)) {
                let v = input[(batch * params.channels + c) * params.in_height * params.in_width
                              + u32(in_y) * params.in_width + u32(in_x)];
                if (!have || xf_gt(v, max_bits)) { max_bits = v; have = true; }
            }
        }
    }
    output[(batch * params.channels + c) * params.out_height * params.out_width + out_y * params.out_width + out_x] = max_bits;
}
";

    /// <summary>max_pool2d_with_indices: binding 0 input, 1 output, 2 indices (i32 bits), 3 PoolParams (as PoolExtendedSource).</summary>
    public const string MaxPool2DWithIndices = ExactFloatKernels.Wgsl + @"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> indices_out: array<u32>;

struct PoolParams {
    batch_size: u32, channels: u32, in_height: u32, in_width: u32, out_height: u32, out_width: u32,
    kernel_height: u32, kernel_width: u32, stride_h: u32, stride_w: u32, pad_h: u32, pad_w: u32,
}
@group(0) @binding(3) var<uniform> params: PoolParams;

@compute @workgroup_size(8, 8, 1)
fn max_pool2d_with_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_x = gid.x;
    let out_y = gid.y;
    let c = gid.z % params.channels;
    let batch = gid.z / params.channels;
    if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch_size) { return; }
    var max_bits: u32 = 0xFF800000u;
    var max_idx: u32 = 0u;
    var have = false;
    for (var ky: u32 = 0u; ky < params.kernel_height; ky = ky + 1u) {
        for (var kx: u32 = 0u; kx < params.kernel_width; kx = kx + 1u) {
            let in_y = i32(out_y * params.stride_h + ky) - i32(params.pad_h);
            let in_x = i32(out_x * params.stride_w + kx) - i32(params.pad_w);
            if (in_y >= 0 && in_y < i32(params.in_height) && in_x >= 0 && in_x < i32(params.in_width)) {
                let v = input[(batch * params.channels + c) * params.in_height * params.in_width
                              + u32(in_y) * params.in_width + u32(in_x)];
                if (!have || xf_gt(v, max_bits)) {
                    max_bits = v;
                    max_idx = u32(in_y) * params.in_width + u32(in_x);
                    have = true;
                }
            }
        }
    }
    let out_idx = (batch * params.channels + c) * params.out_height * params.out_width + out_y * params.out_width + out_x;
    output[out_idx] = max_bits;
    indices_out[out_idx] = max_idx;
}
";

    /// <summary>
    /// Global max pool over each (batch, channel) plane: binding 0 input, 1 output, 2 {outer_size, spatial_size}
    /// (as GlobalPoolSource). global_max_pool writes the maximum; global_max_pool_with_indices writes
    /// output[2o] = maximum and output[2o + 1] = its spatial index (i32 bits), the layout the backend de-interleaves.
    /// </summary>
    public const string GlobalMaxPool = ExactFloatKernels.Wgsl + @"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params { outer_size: u32, spatial_size: u32, }
@group(0) @binding(2) var<uniform> params: Params;

fn plane_max(base: u32) -> vec2<u32> {
    var max_bits: u32 = 0xFF800000u;
    var max_idx: u32 = 0u;
    for (var i: u32 = 0u; i < params.spatial_size; i = i + 1u) {
        let v = input[base + i];
        if (i == 0u || xf_gt(v, max_bits)) { max_bits = v; max_idx = i; }
    }
    return vec2<u32>(max_bits, max_idx);
}

@compute @workgroup_size(256)
fn global_max_pool(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outer = gid.x;
    if (outer >= params.outer_size) { return; }
    output[outer] = plane_max(outer * params.spatial_size).x;
}

@compute @workgroup_size(256)
fn global_max_pool_with_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outer = gid.x;
    if (outer >= params.outer_size) { return; }
    let m = plane_max(outer * params.spatial_size);
    output[outer * 2u] = m.x;
    output[outer * 2u + 1u] = m.y;
}
";

    private const string Pool3DParams = @"
struct Pool3DParams {
    batch: u32, channels: u32,
    in_d: u32, in_h: u32, in_w: u32,
    out_d: u32, out_h: u32, out_w: u32,
    k_d: u32, k_h: u32, k_w: u32,
    s_d: u32, s_h: u32, s_w: u32,
    _pad1: u32, _pad2: u32,
}
";

    private const string Pool3DBody = @"
    let idx = gid.x;
    let out_spatial = params.out_d * params.out_h * params.out_w;
    let total = params.batch * params.channels * out_spatial;
    if (idx >= total) { return; }
    let ow = idx % params.out_w;
    let oh = (idx / params.out_w) % params.out_h;
    let od = (idx / (params.out_w * params.out_h)) % params.out_d;
    let nc = idx / out_spatial;
    let in_spatial = params.in_d * params.in_h * params.in_w;
    var max_bits: u32 = 0xFF800000u;
    var max_pos: u32 = 0u;
    var have = false;
    for (var kd: u32 = 0u; kd < params.k_d; kd = kd + 1u) {
        let id = od * params.s_d + kd;
        if (id >= params.in_d) { continue; }
        for (var kh: u32 = 0u; kh < params.k_h; kh = kh + 1u) {
            let ih = oh * params.s_h + kh;
            if (ih >= params.in_h) { continue; }
            for (var kw: u32 = 0u; kw < params.k_w; kw = kw + 1u) {
                let iw = ow * params.s_w + kw;
                if (iw >= params.in_w) { continue; }
                let flat_pos = id * params.in_h * params.in_w + ih * params.in_w + iw;
                let v = input[nc * in_spatial + flat_pos];
                if (!have || xf_gt(v, max_bits)) { max_bits = v; max_pos = flat_pos; have = true; }
            }
        }
    }
    output[idx] = max_bits;
";

    /// <summary>max_pool3d: binding 0 input, 1 output, 2 Pool3DParams (as Pool3DSource).</summary>
    public const string MaxPool3D = ExactFloatKernels.Wgsl + @"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
" + Pool3DParams + @"
@group(0) @binding(2) var<uniform> params: Pool3DParams;

@compute @workgroup_size(256)
fn max_pool3d(@builtin(global_invocation_id) gid: vec3<u32>) {" + Pool3DBody + @"}
";

    /// <summary>max_pool3d_with_indices: binding 0 input, 1 output, 2 indices (i32 bits), 3 Pool3DParams (as Pool3DWithIndicesSource).</summary>
    public const string MaxPool3DWithIndices = ExactFloatKernels.Wgsl + @"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> indices_out: array<u32>;
" + Pool3DParams + @"
@group(0) @binding(3) var<uniform> params: Pool3DParams;

@compute @workgroup_size(256)
fn max_pool3d_with_indices(@builtin(global_invocation_id) gid: vec3<u32>) {" + Pool3DBody + @"    indices_out[idx] = max_pos;
}
";

    /// <summary>max_pool1d: binding 0 input, 1 output, 2 PoolParams (as Pool1DSource).</summary>
    public const string MaxPool1D = ExactFloatKernels.Wgsl + @"
@group(0) @binding(0) var<storage, read> pool_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> pool_out: array<u32>;

struct PoolParams { batch: u32, channels: u32, inLength: u32, outLength: u32, kernelSize: u32, stride: u32, pad0: u32, pad1: u32 };
@group(0) @binding(2) var<uniform> pool_params: PoolParams;

@compute @workgroup_size(256)
fn max_pool1d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = pool_params.batch * pool_params.channels * pool_params.outLength;
    if (idx >= total) { return; }
    let o = idx % pool_params.outLength;
    let c = (idx / pool_params.outLength) % pool_params.channels;
    let b = idx / (pool_params.outLength * pool_params.channels);
    let inOff = (b * pool_params.channels + c) * pool_params.inLength;
    var max_bits: u32 = 0xFF800000u;
    var have = false;
    for (var k: u32 = 0u; k < pool_params.kernelSize; k = k + 1u) {
        let pos = o * pool_params.stride + k;
        if (pos < pool_params.inLength) {
            let v = pool_in[inOff + pos];
            if (!have || xf_gt(v, max_bits)) { max_bits = v; have = true; }
        }
    }
    pool_out[idx] = max_bits;
}
";
}
#endif
