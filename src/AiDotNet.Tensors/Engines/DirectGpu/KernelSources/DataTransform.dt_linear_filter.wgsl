
struct Params { S: i32, L: i32, NB: i32, NA: i32, hasZi: i32, guardOn: i32, guardBits: i32, first: i32, }
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> ys: array<f32>;
@group(0) @binding(2) var<storage, read> bc: array<f32>;
@group(0) @binding(3) var<storage, read> ac: array<f32>;
@group(0) @binding(4) var<storage, read> zi: array<f32>;
@group(0) @binding(5) var<storage, read_write> zs: array<f32>;
@group(0) @binding(6) var<uniform> pm: Params;

@compute @workgroup_size(256)
fn dt_linear_filter(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let s = i32(gidv.x) + pm.first * 256;
    if (s >= pm.S) { return; }
    var K = pm.NB;
    if (pm.NA > K) { K = pm.NA; }
    let o = s * pm.L;
    let zo = s * (K - 1);
    let a0 = ac[0];
    for (var k = 0; k < K - 1; k = k + 1) {
        var z0 = 0.0;
        if (pm.hasZi != 0) { z0 = zi[zo + k]; }
        zs[zo + k] = z0;
    }
    let g = bitcast<f32>(pm.guardBits);
    let b0 = bc[0] / a0;
    for (var n = 0; n < pm.L; n = n + 1) {
        let xn = xs[o + n];
        var yn = b0 * xn;
        if (K > 1) { yn = yn + zs[zo]; }
        if (pm.guardOn != 0) {
            if ((bitcast<u32>(yn) & 2147483647u) > 2139095040u) { yn = 0.0; }
            else if (yn > g) { yn = g; }
            else if (yn < -g) { yn = -g; }
        }
        for (var k = 1; k < K; k = k + 1) {
            var t = 0.0;
            if (k < pm.NB) { t = (bc[k] / a0) * xn; }
            if (k < pm.NA) { t = t - (ac[k] / a0) * yn; }
            var zk = 0.0;
            if (k < K - 1) { zk = zs[zo + k]; }
            zs[zo + k - 1] = t + zk;
        }
        ys[o + n] = yn;
    }
}
