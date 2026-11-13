
struct Params { N: i32, above: i32, perElement: i32, thrBits: i32, onBits: i32, offBits: i32, first: i32, pad0: i32, }
@group(0) @binding(0) var<storage, read> uv: array<f32>;
@group(0) @binding(1) var<storage, read_write> ot: array<f32>;
@group(0) @binding(2) var<storage, read> th: array<f32>;
@group(0) @binding(3) var<uniform> pm: Params;

@compute @workgroup_size(256)
fn dt_uniform_threshold(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let i = i32(gidv.x) + pm.first * 256;
    if (i >= pm.N) { return; }
    let a = uv[2 * i];
    let r = uv[2 * i + 1];
    var t = bitcast<f32>(pm.thrBits);
    if (pm.perElement != 0) { t = th[i]; }
    var hit = false;
    if (pm.above != 0) {
        hit = a > t || (a == t && r > 0.0);
    } else {
        hit = a < t || (a == t && r < 0.0);
    }
    var v = bitcast<f32>(pm.offBits);
    if (hit) { v = bitcast<f32>(pm.onBits); }
    ot[i] = v;
}
