
struct Params { N: i32, branch: i32, meanBits: i32, stdBits: i32, clampOn: i32, lowBits: i32, highBits: i32, first: i32, }
@group(0) @binding(0) var<storage, read> uv: array<f32>;
@group(0) @binding(1) var<storage, read_write> ot: array<f32>;
@group(0) @binding(2) var<uniform> pm: Params;

@compute @workgroup_size(256)
fn dt_box_muller(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let i = i32(gidv.x) + pm.first * 256;
    if (i >= pm.N) { return; }
    let u1 = uv[2 * i];
    let u2 = uv[2 * i + 1];
    let r = sqrt(-2.0 * log(u1));
    let ang = 6.2831855 * u2;
    var z = r * cos(ang);
    if (pm.branch == 1) { z = r * sin(ang); }
    var v = bitcast<f32>(pm.meanBits) + bitcast<f32>(pm.stdBits) * z;
    if (pm.clampOn != 0) {
        let lo = bitcast<f32>(pm.lowBits);
        let hi = bitcast<f32>(pm.highBits);
        if (v < lo) { v = lo; } else if (v > hi) { v = hi; }
    }
    ot[i] = v;
}
