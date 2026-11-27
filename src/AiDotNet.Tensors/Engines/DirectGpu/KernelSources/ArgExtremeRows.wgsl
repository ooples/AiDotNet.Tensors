@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    outer_size: u32,
    reduce_size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> ax_key: array<i32, 256>;
var<workgroup> ax_idx: array<u32, 256>;

fn arg_extreme_rows(outer: u32, tid: u32, find_max: bool) {
    let live = outer < params.outer_size;
    let offset = outer * params.reduce_size;
    // The max of key' = find_max ? key : -key. |xf_key| <= 0x7F800000, so the negation cannot overflow.
    var have = false;
    var best_key: i32 = 0;
    var best_idx: u32 = 0u;
    if (live) {
        for (var i: u32 = tid; i < params.reduce_size; i = i + 256u) {
            let v = bitcast<u32>(input[offset + i]);
            if (xf_isnan(v)) { continue; }
            let key = select(-xf_key(v), xf_key(v), find_max);
            if (!have || key > best_key) {
                have = true;
                best_key = key;
                best_idx = i;
            }
        }
    }
    ax_key[tid] = select(-2147483647 - 1, best_key, have);   // INT_MIN: below every finite or infinite key
    ax_idx[tid] = select(0xFFFFFFFFu, best_idx, have);
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let k2 = ax_key[tid + stride];
            let i2 = ax_idx[tid + stride];
            if (k2 > ax_key[tid] || (k2 == ax_key[tid] && i2 < ax_idx[tid])) {
                ax_key[tid] = k2;
                ax_idx[tid] = i2;
            }
        }
        workgroupBarrier();
    }
    if (tid == 0u && live) {
        output[outer] = select(f32(ax_idx[0]), 0.0, xf_isnan(bitcast<u32>(input[offset])));
    }
}

@compute @workgroup_size(256)
fn argmax_axis_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    arg_extreme_rows(wid.x, lid.x, true);
}

@compute @workgroup_size(256)
fn argmin_axis_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    arg_extreme_rows(wid.x, lid.x, false);
}
