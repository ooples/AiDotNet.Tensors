
struct Params { K: i32, D: i32, first: i32, pad0: i32, pad1: i32, pad2: i32, pad3: i32, pad4: i32, }
@group(0) @binding(0) var<storage, read> tb: array<f32>;
@group(0) @binding(1) var<storage, read_write> ot: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<f32>;
@group(0) @binding(3) var<uniform> pm: Params;

@compute @workgroup_size(256)
fn dt_gather_rows(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let i = i32(gidv.x) + pm.first * 256;
    if (i >= pm.K * pm.D) { return; }
    let k = i / pm.D;
    let d = i - k * pm.D;
    let r = i32(ids[k]);
    var v = 0.0;
    if (r >= 0) { v = tb[r * pm.D + d]; }
    ot[i] = v;
}
