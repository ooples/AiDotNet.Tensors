
struct Params { N: i32, mode: i32, lowBits: i32, highBits: i32, first: i32, pad0: i32, pad1: i32, pad2: i32, }
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> ys: array<f32>;
@group(0) @binding(2) var<uniform> pm: Params;

@compute @workgroup_size(256)
fn dt_round_integral(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let i = i32(gidv.x) + pm.first * 256;
    if (i >= pm.N) { return; }
    let v = xs[i];
    var t = 0.0;
    if ((bitcast<u32>(v) & 2147483647u) <= 2139095040u) {
        t = trunc(v);
        let f = v - t;
        var af = f;
        if (f < 0.0) { af = -f; }
        var sg = 1.0;
        if (v < 0.0) { sg = -1.0; }
        if (pm.mode == 1) {
            let h = t * 0.5;
            if (af > 0.5 || (af == 0.5 && h != trunc(h))) { t = t + sg; }
        } else if (pm.mode == 2) {
            if (af >= 0.5) { t = t + sg; }
        } else if (pm.mode == 3) {
            if (f < 0.0) { t = t - 1.0; }
        } else if (pm.mode == 4) {
            if (f > 0.0) { t = t + 1.0; }
        }
        let lo = bitcast<f32>(pm.lowBits);
        let hi = bitcast<f32>(pm.highBits);
        if (t < lo) { t = lo; } else if (t > hi) { t = hi; }
    }
    ys[i] = t;
}
