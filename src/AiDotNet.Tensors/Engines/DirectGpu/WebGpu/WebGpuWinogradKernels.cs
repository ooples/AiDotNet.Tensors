namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #1372: the WGSL instance of WinogradF23KernelTemplate's tile transforms. WGSL is not C-like enough to share the
// template's bodies (let/var, explicit casts, no ?: on buffer reads, uniform structs instead of push constants), so the
// arithmetic is restated here operation for operation; WinogradF23PortTests checks both against the same CPU reference.
// Each kernel is its own module: the read buffer at binding 0, the written one at 1, the int parameters in a uniform
// struct at 2, padded to a 16-byte multiple. Boundary reads use if, never select(), which would evaluate the
// out-of-bounds read too. Function-parameter arrays are copied into a var before they are indexed dynamically.
internal static class WebGpuWinogradKernels
{
    private const string FilterTail = @"
fn filter_tail(gIn: array<f32, 9>, idx: i32) {
    var g = gIn;
    var t: array<f32, 12>;
    for (var j = 0; j < 3; j = j + 1) {
        let g0 = g[j]; let g1 = g[3 + j]; let g2 = g[6 + j];
        t[j] = g0;
        t[3 + j] = 0.5 * (g0 + g1 + g2);
        t[6 + j] = 0.5 * (g0 - g1 + g2);
        t[9 + j] = g2;
    }
    let plane = pm.K * pm.C;
    for (var i = 0; i < 4; i = i + 1) {
        let a = t[i * 3]; let b = t[i * 3 + 1]; let e = t[i * 3 + 2];
        u[(i * 4 + 0) * plane + idx] = a;
        u[(i * 4 + 1) * plane + idx] = 0.5 * (a + b + e);
        u[(i * 4 + 2) * plane + idx] = 0.5 * (a - b + e);
        u[(i * 4 + 3) * plane + idx] = e;
    }
}
";

    private const string FilterHeader = @"
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
struct Params { K: i32, C: i32, pad0: i32, pad1: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
";

    public const string Filter = FilterHeader + FilterTail + @"
@compute @workgroup_size(256)
fn wino_f23_filter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    if (idx >= pm.K * pm.C) { return; }
    var g: array<f32, 9>;
    for (var q = 0; q < 9; q = q + 1) { g[q] = w[idx * 9 + q]; }
    filter_tail(g, idx);
}
";

    public const string FilterFlip = FilterHeader + FilterTail + @"
@compute @workgroup_size(256)
fn wino_f23_filter_flip(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    if (idx >= pm.K * pm.C) { return; }
    let c = idx / pm.K;
    let k = idx - c * pm.K;
    let src = (k * pm.C + c) * 9;
    var g: array<f32, 9>;
    for (var i = 0; i < 3; i = i + 1) {
        for (var j = 0; j < 3; j = j + 1) { g[i * 3 + j] = w[src + (2 - i) * 3 + (2 - j)]; }
    }
    filter_tail(g, idx);
}
";

    private const string InputHeader = @"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> v: array<f32>;
struct Params { N: i32, C: i32, H: i32, W: i32, pad: i32, tilesH: i32, tilesW: i32, pad0: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
fn input_tail(c: i32, p: i32, idx: i32) {
    let P = pm.N * pm.tilesH * pm.tilesW;
    let n = p / (pm.tilesH * pm.tilesW);
    let tt = p - n * (pm.tilesH * pm.tilesW);
    let th = tt / pm.tilesW;
    let tw = tt - th * pm.tilesW;
    let h0 = th * 2 - pm.pad;
    let w0 = tw * 2 - pm.pad;
    let src = (n * pm.C + c) * pm.H * pm.W;
    var d: array<f32, 16>;
    for (var i = 0; i < 4; i = i + 1) {
        let hh = h0 + i;
        for (var j = 0; j < 4; j = j + 1) {
            let ww = w0 + j;
            var value = 0.0;
            if (hh >= 0 && hh < pm.H && ww >= 0 && ww < pm.W) { value = x[src + hh * pm.W + ww]; }
            d[i * 4 + j] = value;
        }
    }
    var s: array<f32, 16>;
    for (var j = 0; j < 4; j = j + 1) {
        s[j] = d[j] - d[8 + j];
        s[4 + j] = d[4 + j] + d[8 + j];
        s[8 + j] = d[8 + j] - d[4 + j];
        s[12 + j] = d[4 + j] - d[12 + j];
    }
    let plane = pm.C * P;
    for (var i = 0; i < 4; i = i + 1) {
        v[(i * 4 + 0) * plane + idx] = s[i * 4] - s[i * 4 + 2];
        v[(i * 4 + 1) * plane + idx] = s[i * 4 + 1] + s[i * 4 + 2];
        v[(i * 4 + 2) * plane + idx] = s[i * 4 + 2] - s[i * 4 + 1];
        v[(i * 4 + 3) * plane + idx] = s[i * 4 + 1] - s[i * 4 + 3];
    }
}
";

    // V[x][c][p]; idx = c*P + p.
    public const string Input = InputHeader + @"
@compute @workgroup_size(256)
fn wino_f23_input(@builtin(global_invocation_id) gid: vec3<u32>) {
    let P = pm.N * pm.tilesH * pm.tilesW;
    let idx = i32(gid.x);
    if (idx >= pm.C * P) { return; }
    let c = idx / P;
    input_tail(c, idx - c * P, idx);
}
";

    // Weight gradient: V[x][p][c], tile-major; idx = p*C + c.
    public const string InputTilesTileMajor = InputHeader + @"
@compute @workgroup_size(256)
fn wino_f32w_input(@builtin(global_invocation_id) gid: vec3<u32>) {
    let P = pm.N * pm.tilesH * pm.tilesW;
    let idx = i32(gid.x);
    if (idx >= pm.C * P) { return; }
    let p = idx / pm.C;
    input_tail(idx - p * pm.C, p, idx);
}
";

    private const string TileHeader = @"
struct Params { N: i32, K: i32, outH: i32, outW: i32, tilesH: i32, tilesW: i32, pad0: i32, pad1: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
";

    // The convolution's epilogue: epi bit 0 adds bias[k], bit 1 applies ReLU (v < 0 is false for NaN, which passes). The
    // bias is a third buffer, so the uniform struct moves to binding 3.
    public const string Output = @"
@group(0) @binding(0) var<storage, read> mm: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
struct Params { N: i32, K: i32, outH: i32, outW: i32, tilesH: i32, tilesW: i32, epi: i32, pad1: i32, }
@group(0) @binding(3) var<uniform> pm: Params;
@compute @workgroup_size(256)
fn wino_f23_output(@builtin(global_invocation_id) gid: vec3<u32>) {
    let P = pm.N * pm.tilesH * pm.tilesW;
    let idx = i32(gid.x);
    if (idx >= pm.K * P) { return; }
    let k = idx / P;
    let p = idx - k * P;
    let n = p / (pm.tilesH * pm.tilesW);
    let tt = p - n * (pm.tilesH * pm.tilesW);
    let th = tt / pm.tilesW;
    let tw = tt - th * pm.tilesW;
    let plane = pm.K * P;
    var m: array<f32, 16>;
    for (var q = 0; q < 16; q = q + 1) { m[q] = mm[q * plane + idx]; }
    var s: array<f32, 8>;
    for (var j = 0; j < 4; j = j + 1) {
        s[j] = m[j] + m[4 + j] + m[8 + j];
        s[4 + j] = m[4 + j] - m[8 + j] - m[12 + j];
    }
    let dst = (n * pm.K + k) * pm.outH * pm.outW;
    let h = th * 2;
    let w2 = tw * 2;
    var bk: f32 = 0.0;
    if ((pm.epi & 1) != 0) { bk = bias[k]; }
    let relu = (pm.epi & 2) != 0;
    for (var i = 0; i < 2; i = i + 1) {
        if (h + i >= pm.outH) { break; }
        var o0 = s[i * 4] + s[i * 4 + 1] + s[i * 4 + 2] + bk;
        if (relu && o0 < 0.0) { o0 = 0.0; }
        y[dst + (h + i) * pm.outW + w2] = o0;
        if (w2 + 1 < pm.outW) {
            var o1 = s[i * 4 + 1] - s[i * 4 + 2] - s[i * 4 + 3] + bk;
            if (relu && o1 < 0.0) { o1 = 0.0; }
            y[dst + (h + i) * pm.outW + w2 + 1] = o1;
        }
    }
}
";

    public const string GradOutputTiles = @"
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
" + TileHeader + @"
@compute @workgroup_size(256)
fn wino_f32w_dy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let P = pm.N * pm.tilesH * pm.tilesW;
    let idx = i32(gid.x);
    if (idx >= pm.K * P) { return; }
    let k = idx / P;
    let p = idx - k * P;
    let n = p / (pm.tilesH * pm.tilesW);
    let tt = p - n * (pm.tilesH * pm.tilesW);
    let th = tt / pm.tilesW;
    let tw = tt - th * pm.tilesW;
    let src = (n * pm.K + k) * pm.outH * pm.outW;
    let h = th * 2;
    let w2 = tw * 2;
    let g00 = dy[src + h * pm.outW + w2];
    var g01 = 0.0;
    var g10 = 0.0;
    var g11 = 0.0;
    if (w2 + 1 < pm.outW) { g01 = dy[src + h * pm.outW + w2 + 1]; }
    if (h + 1 < pm.outH) { g10 = dy[src + (h + 1) * pm.outW + w2]; }
    if (h + 1 < pm.outH && w2 + 1 < pm.outW) { g11 = dy[src + (h + 1) * pm.outW + w2 + 1]; }
    // t = A g (4x2): rows g0, g0+g1, g0-g1, -g1 over the two columns.
    var r: array<f32, 8>;
    r[0] = g00; r[1] = g01;
    r[2] = g00 + g10; r[3] = g01 + g11;
    r[4] = g00 - g10; r[5] = g01 - g11;
    r[6] = -g10; r[7] = -g11;
    let plane = pm.K * P;
    for (var i = 0; i < 4; i = i + 1) {
        let a = r[i * 2];
        let b = r[i * 2 + 1];
        u[(i * 4 + 0) * plane + idx] = a;
        u[(i * 4 + 1) * plane + idx] = a + b;
        u[(i * 4 + 2) * plane + idx] = a - b;
        u[(i * 4 + 3) * plane + idx] = -b;
    }
}
";

    public const string FilterGradient = @"
@group(0) @binding(0) var<storage, read> mm: array<f32>;
@group(0) @binding(1) var<storage, read_write> dw: array<f32>;
struct Params { K: i32, C: i32, splits: i32, pad0: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
@compute @workgroup_size(256)
fn wino_f32w_output(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    if (idx >= pm.K * pm.C) { return; }
    let plane = pm.K * pm.C;
    var m: array<f32, 16>;
    for (var q = 0; q < 16; q = q + 1) {
        var acc = 0.0;
        for (var sp = 0; sp < pm.splits; sp = sp + 1) { acc = acc + mm[(q * pm.splits + sp) * plane + idx]; }
        m[q] = acc;
    }
    var t: array<f32, 12>;
    for (var j = 0; j < 4; j = j + 1) {
        let m0 = m[j]; let m1 = m[4 + j]; let m2 = m[8 + j]; let m3 = m[12 + j];
        t[j] = m0 + 0.5 * (m1 + m2);
        t[4 + j] = 0.5 * (m1 - m2);
        t[8 + j] = 0.5 * (m1 + m2) + m3;
    }
    let dst = idx * 9;
    for (var i = 0; i < 3; i = i + 1) {
        let a = t[i * 4]; let b = t[i * 4 + 1]; let e = t[i * 4 + 2]; let f = t[i * 4 + 3];
        dw[dst + i * 3 + 0] = a + 0.5 * (b + e);
        dw[dst + i * 3 + 1] = 0.5 * (b - e);
        dw[dst + i * 3 + 2] = 0.5 * (b + e) + f;
    }
}
";

    // The fused weight gradient (WinogradF23KernelTemplate's FusedMain and epilogues), restated: x at binding 0, the
    // written dW or partials at 1, dY at 2, the ints at 3. Group wid.x = (bz * gy + by) * gx + bx; 16 KB of workgroup
    // storage, WebGPU's guaranteed minimum. Every barrier sits in uniform control flow (the loop bounds come from the
    // workgroup id and the uniforms), which WGSL's uniformity analysis requires.
    private const string FusedMain = @"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> res: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
struct Params { N: i32, C: i32, H: i32, W: i32, K: i32, outH: i32, outW: i32, pd: i32, tilesH: i32, tilesW: i32, tilesPerSplit: i32, pad0: i32, }
@group(0) @binding(3) var<uniform> pm: Params;
var<workgroup> sm: array<f32, 4096>;
var<private> acc: array<f32, 64>;
var<private> r9: array<f32, 9>;

fn fused_accumulate(tid: i32, grp: i32) {
    let gx = (pm.C + 31) / 32;
    let gy = (pm.K + 31) / 32;
    let bz = grp / (gx * gy);
    let rem = grp - bz * (gx * gy);
    let by = rem / gx;
    let bx = rem - by * gx;
    let c0 = bx * 32;
    let k0 = by * 32;
    let TT = pm.tilesH * pm.tilesW;
    let P = pm.N * TT;
    let pBegin = bz * pm.tilesPerSplit;
    let pEnd = min(P, pBegin + pm.tilesPerSplit);
    let tc = tid & 7;
    let tk = (tid >> 3u) & 7;
    let tx = tid >> 6u;
    let tj = tid & 3;
    let tr = (tid >> 2u) & 31;
    let role = tid >> 7u;
    let col = tj * 32 + (tr ^ (tj * 8));
    for (var i = 0; i < 64; i = i + 1) { acc[i] = 0.0; }
    for (var pc = pBegin; pc < pEnd; pc = pc + 4) {
        let p = pc + tj;
        let pin = p < pEnd;
        var n = 0;
        var th = 0;
        var tw = 0;
        if (pin) { n = p / TT; let tt = p - n * TT; th = tt / pm.tilesW; tw = tt - th * pm.tilesW; }
        if (role == 0) {
            let k = k0 + tr;
            var g00 = 0.0;
            var g01 = 0.0;
            var g10 = 0.0;
            var g11 = 0.0;
            if (pin && k < pm.K) {
                let h = th * 2;
                let w2 = tw * 2;
                let src = (n * pm.K + k) * pm.outH * pm.outW + h * pm.outW + w2;
                g00 = dy[src];
                if (w2 + 1 < pm.outW) { g01 = dy[src + 1]; }
                if (h + 1 < pm.outH) {
                    g10 = dy[src + pm.outW];
                    if (w2 + 1 < pm.outW) { g11 = dy[src + pm.outW + 1]; }
                }
            }
            var ra: array<f32, 4>;
            var rb: array<f32, 4>;
            ra[0] = g00; ra[1] = g00 + g10; ra[2] = g00 - g10; ra[3] = -g10;
            rb[0] = g01; rb[1] = g01 + g11; rb[2] = g01 - g11; rb[3] = -g11;
            for (var i = 0; i < 4; i = i + 1) {
                sm[(i * 4 + 0) * 128 + col] = ra[i];
                sm[(i * 4 + 1) * 128 + col] = ra[i] + rb[i];
                sm[(i * 4 + 2) * 128 + col] = ra[i] - rb[i];
                sm[(i * 4 + 3) * 128 + col] = -rb[i];
            }
        } else {
            let c = c0 + tr;
            let cin = pin && c < pm.C;
            let h0 = th * 2 - pm.pd;
            let w0 = tw * 2 - pm.pd;
            var cs = 0;
            if (cin) { cs = c; }
            let src = (n * pm.C + cs) * pm.H * pm.W;
            var d: array<f32, 16>;
            for (var i = 0; i < 4; i = i + 1) {
                let hh = h0 + i;
                for (var j = 0; j < 4; j = j + 1) {
                    let ww = w0 + j;
                    d[i * 4 + j] = 0.0;
                    if (cin && hh >= 0 && hh < pm.H && ww >= 0 && ww < pm.W) { d[i * 4 + j] = x[src + hh * pm.W + ww]; }
                }
            }
            var s: array<f32, 16>;
            for (var j = 0; j < 4; j = j + 1) {
                s[j] = d[j] - d[8 + j];
                s[4 + j] = d[4 + j] + d[8 + j];
                s[8 + j] = d[8 + j] - d[4 + j];
                s[12 + j] = d[4 + j] - d[12 + j];
            }
            for (var i = 0; i < 4; i = i + 1) {
                sm[2048 + (i * 4 + 0) * 128 + col] = s[i * 4] - s[i * 4 + 2];
                sm[2048 + (i * 4 + 1) * 128 + col] = s[i * 4 + 1] + s[i * 4 + 2];
                sm[2048 + (i * 4 + 2) * 128 + col] = s[i * 4 + 2] - s[i * 4 + 1];
                sm[2048 + (i * 4 + 3) * 128 + col] = s[i * 4 + 1] - s[i * 4 + 3];
            }
        }
        workgroupBarrier();
        for (var j = 0; j < 4; j = j + 1) {
            for (var xi = 0; xi < 4; xi = xi + 1) {
                let row = (tx * 4 + xi) * 128 + j * 32;
                let uo = row + ((tk * 4) ^ (j * 8));
                let vo = 2048 + row + ((tc * 4) ^ (j * 8));
                var uu: array<f32, 4>;
                var vv: array<f32, 4>;
                for (var q = 0; q < 4; q = q + 1) { uu[q] = sm[uo + q]; vv[q] = sm[vo + q]; }
                for (var a = 0; a < 4; a = a + 1) {
                    for (var b = 0; b < 4; b = b + 1) {
                        let ai = xi * 16 + a * 4 + b;
                        acc[ai] = fma(uu[a], vv[b], acc[ai]);
                    }
                }
            }
        }
        workgroupBarrier();
    }
}

// Eighth e of the epilogue: the tk == e threads gather their (k, c) components into sm[x][4 k][33].
fn fused_gather(tid: i32, e: i32) {
    let tc = tid & 7;
    let tk = (tid >> 3u) & 7;
    let tx = tid >> 6u;
    if (tk == e) {
        for (var xi = 0; xi < 4; xi = xi + 1) {
            for (var a = 0; a < 4; a = a + 1) {
                for (var b = 0; b < 4; b = b + 1) {
                    sm[(tx * 4 + xi) * 132 + a * 33 + tc * 4 + b] = acc[xi * 16 + a * 4 + b];
                }
            }
        }
    }
}

// One (k, c) of the gathered eighth (threads 0..127): G^T M G into r9.
fn fused_transform(tid: i32) {
    let kl = tid >> 5u;
    let cl = tid & 31;
    var m: array<f32, 16>;
    for (var q = 0; q < 16; q = q + 1) { m[q] = sm[q * 132 + kl * 33 + cl]; }
    var t: array<f32, 12>;
    for (var j = 0; j < 4; j = j + 1) {
        let m0 = m[j]; let m1 = m[4 + j]; let m2 = m[8 + j]; let m3 = m[12 + j];
        t[j] = m0 + 0.5 * (m1 + m2);
        t[4 + j] = 0.5 * (m1 - m2);
        t[8 + j] = 0.5 * (m1 + m2) + m3;
    }
    for (var i = 0; i < 3; i = i + 1) {
        let a = t[i * 4]; let b = t[i * 4 + 1]; let cc = t[i * 4 + 2]; let dd = t[i * 4 + 3];
        r9[i * 3 + 0] = a + 0.5 * (b + cc);
        r9[i * 3 + 1] = 0.5 * (b - cc);
        r9[i * 3 + 2] = 0.5 * (b + cc) + dd;
    }
}
";

    public const string FusedFilterGradientDirect = FusedMain + @"
@compute @workgroup_size(256)
fn wino_f32w_fused_direct(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = i32(lidx);
    let grp = i32(wid.x);
    fused_accumulate(tid, grp);
    let gx = (pm.C + 31) / 32;
    let gy = (pm.K + 31) / 32;
    let rem = grp - (grp / (gx * gy)) * (gx * gy);
    let by = rem / gx;
    let c0 = (rem - by * gx) * 32;
    let k0 = by * 32;
    let span = min(32, pm.C - c0) * 9;
    for (var e = 0; e < 8; e = e + 1) {
        fused_gather(tid, e);
        workgroupBarrier();
        if (tid < 128) {
            fused_transform(tid);
            let kl = tid >> 5u;
            let cl = tid & 31;
            for (var q = 0; q < 9; q = q + 1) { sm[2112 + kl * 288 + cl * 9 + q] = r9[q]; }
        }
        workgroupBarrier();
        for (var i = tid; i < 4 * 288; i = i + 256) {
            let rowk = i / 288;
            let r = i - rowk * 288;
            let k = k0 + e * 4 + rowk;
            if (k < pm.K && r < span) { res[(k * pm.C + c0) * 9 + r] = sm[2112 + i]; }
        }
        workgroupBarrier();
    }
}
";

    public const string FusedFilterGradient = FusedMain + @"
@compute @workgroup_size(256)
fn wino_f32w_fused(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = i32(lidx);
    let grp = i32(wid.x);
    fused_accumulate(tid, grp);
    let gx = (pm.C + 31) / 32;
    let gy = (pm.K + 31) / 32;
    let bz = grp / (gx * gy);
    let rem = grp - bz * (gx * gy);
    let by = rem / gx;
    let c0 = (rem - by * gx) * 32;
    let k0 = by * 32;
    let plane = pm.K * pm.C;
    for (var e = 0; e < 8; e = e + 1) {
        fused_gather(tid, e);
        workgroupBarrier();
        let k = k0 + e * 4 + (tid >> 5u);
        let c = c0 + (tid & 31);
        if (tid < 128 && k < pm.K && c < pm.C) {
            fused_transform(tid);
            let dst = bz * 9 * plane + k * pm.C + c;
            for (var q = 0; q < 9; q = q + 1) { res[dst + q * plane] = r9[q]; }
        }
        workgroupBarrier();
    }
}
";

    public const string FusedFilterGradientFinish = @"
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> dw: array<f32>;
struct Params { KC: i32, splits: i32, pad0: i32, pad1: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
@compute @workgroup_size(256)
fn wino_f32w_fused_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    if (idx >= pm.KC * 9) { return; }
    let kc = idx / 9;
    let q = idx - kc * 9;
    var sum = 0.0;
    for (var z = 0; z < pm.splits; z = z + 1) { sum = sum + partials[(z * 9 + q) * pm.KC + kc]; }
    dw[idx] = sum;
}
";

    /// <summary>The module source for a <see cref="WinogradF23KernelTemplate"/> kernel name.</summary>
    internal static string SourceFor(string name) => name switch
    {
        WinogradF23KernelTemplate.Filter => Filter,
        WinogradF23KernelTemplate.FilterFlip => FilterFlip,
        WinogradF23KernelTemplate.Input => Input,
        WinogradF23KernelTemplate.InputTilesTileMajor => InputTilesTileMajor,
        WinogradF23KernelTemplate.Output => Output,
        WinogradF23KernelTemplate.GradOutputTiles => GradOutputTiles,
        WinogradF23KernelTemplate.FilterGradient => FilterGradient,
        WinogradF23KernelTemplate.FusedFilterGradient => FusedFilterGradient,
        WinogradF23KernelTemplate.FusedFilterGradientDirect => FusedFilterGradientDirect,
        WinogradF23KernelTemplate.FusedFilterGradientFinish => FusedFilterGradientFinish,
        _ => throw new System.ArgumentException($"No WGSL Winograd kernel named '{name}'.", nameof(name)),
    };
}