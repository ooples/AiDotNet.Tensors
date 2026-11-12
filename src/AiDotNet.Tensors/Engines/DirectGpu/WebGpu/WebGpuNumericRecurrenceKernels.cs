// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The WGSL instance of NumericRecurrenceKernelTemplate, restated operation for operation. Each kernel is its own module:
// the storage buffers in the template's binding order, the ints in a uniform struct after them (padded to whole 16-byte
// rows). The negative infinity and NaN written out are bit patterns passed through a function, so they are never
// constant expressions (which Dawn rejects); one read back is recognised by its bits. The CTC and assignment kernels keep
// their state in storage buffers behind storageBarrier, as the FFT kernels do. WGSL treats a value read from workgroup
// memory as non-uniform, so the assignment kernel's augmenting-path loop reads its exit flag through
// workgroupUniformLoad.
internal static class WebGpuNumericRecurrenceKernels
{
    /// <summary>Words of each uniform struct (every kernel's ints, padded to 8).</summary>
    internal const int UniformWords = 8;

    private const string TridiagonalSource = @"
struct Params { B: i32, N: i32, first: i32, pad0: i32, pad1: i32, pad2: i32, pad3: i32, pad4: i32, }
@group(0) @binding(0) var<storage, read> lo: array<f32>;
@group(0) @binding(1) var<storage, read_write> xs: array<f32>;
@group(0) @binding(2) var<storage, read> dg: array<f32>;
@group(0) @binding(3) var<storage, read> up: array<f32>;
@group(0) @binding(4) var<storage, read> rh: array<f32>;
@group(0) @binding(5) var<storage, read_write> cw: array<f32>;
@group(0) @binding(6) var<uniform> pm: Params;

@compute @workgroup_size(256)
fn nrec_tridiagonal(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let b = i32(gidv.x) + pm.first * 256;
    if (b >= pm.B) { return; }
    let n = pm.N;
    let o = b * n;
    let ob = b * (n - 1);
    var beta = dg[o];
    xs[o] = rh[o] / beta;
    for (var i = 1; i < n; i = i + 1) {
        let c = up[ob + i - 1] / beta;
        cw[ob + i - 1] = c;
        let a = lo[ob + i - 1];
        beta = dg[o + i] - a * c;
        xs[o + i] = (rh[o + i] - a * xs[o + i - 1]) / beta;
    }
    for (var i = n - 2; i >= 0; i = i - 1) {
        xs[o + i] = xs[o + i] - cw[ob + i] * xs[o + i + 1];
    }
}
";

    private const string FritschCarlsonSource = @"
struct Params { B: i32, N: i32, first: i32, pad0: i32, pad1: i32, pad2: i32, pad3: i32, pad4: i32, }
@group(0) @binding(0) var<storage, read> ms: array<f32>;
@group(0) @binding(1) var<storage, read_write> mo: array<f32>;
@group(0) @binding(2) var<storage, read> ds: array<f32>;
@group(0) @binding(3) var<uniform> pm: Params;

@compute @workgroup_size(256)
fn nrec_fritsch_carlson(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let b = i32(gidv.x) + pm.first * 256;
    if (b >= pm.B) { return; }
    let n = pm.N;
    let o = b * n;
    let od = b * (n - 1);
    for (var i = 0; i < n; i = i + 1) { mo[o + i] = ms[o + i]; }
    for (var i = 0; i < n - 1; i = i + 1) {
        let d = ds[od + i];
        if (d == 0.0) {
            mo[o + i] = 0.0;
            mo[o + i + 1] = 0.0;
        } else {
            // Slope space, as CpuEngine.FritschCarlsonHost: no a = m / d or a^2 + b^2 that overflows for finite inputs.
            let m0 = mo[o + i];
            let m1 = mo[o + i + 1];
            var ad = d;
            if (d < 0.0) { ad = -d; }
            var a0 = m0;
            if (m0 < 0.0) { a0 = -m0; }
            var a1 = m1;
            if (m1 < 0.0) { a1 = -m1; }
            var s = a1;
            if (a0 > a1) { s = a0; }
            if (s > 0.0) {
                let u0 = m0 / s;
                let u1 = m1 / s;
                let k = 3.0 / sqrt(u0 * u0 + u1 * u1);
                if (s / ad > k) {
                    mo[o + i] = u0 * k * ad;
                    mo[o + i + 1] = u1 * k * ad;
                }
            }
        }
    }
}
";

    private const string CtcForwardBackwardSource = @"
struct Params { T: i32, N: i32, C: i32, SM: i32, blank: i32, first: i32, pad0: i32, pad1: i32, }
@group(0) @binding(0) var<storage, read> lp: array<f32>;
@group(0) @binding(1) var<storage, read_write> la: array<f32>;
@group(0) @binding(2) var<storage, read> tg: array<f32>;
@group(0) @binding(3) var<storage, read> lens: array<f32>;
@group(0) @binding(4) var<storage, read_write> lb: array<f32>;
@group(0) @binding(5) var<storage, read_write> lz: array<f32>;
@group(0) @binding(6) var<uniform> pm: Params;

fn nrec_bits(b: u32) -> f32 { return bitcast<f32>(b); }

fn nrec_out(v: f32) -> f32 {
    if (v <= -1.0e29) { return nrec_bits(0xff800000u); }
    return v;
}

// A stored negative infinity read back as the finite sentinel, tested by its bits (WGSL may assume no infinities).
fn nrec_in(x: f32) -> f32 {
    if (bitcast<u32>(x) == 0xff800000u) { return -1.0e30; }
    return max(x, -1.0e30);
}

@compute @workgroup_size(256)
fn nrec_ctc_forward_backward(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let lid = i32(lidx);
    let n = i32(wid.x) + pm.first;
    if (n >= pm.N) { return; }
    let tn = i32(lens[n]);
    let un = i32(lens[pm.N + n]);
    let off = i32(lens[2 * pm.N + n]);
    let S = 2 * un + 1;
    let obase = n * pm.T * pm.SM;
    for (var t = 0; t < pm.T; t = t + 1) {
        let prv = obase + (t - 1) * pm.SM;
        for (var s = lid; s < pm.SM; s = s + 256) {
            var v = -1.0e30;
            if (t < tn && s < S) {
                var lab = pm.blank;
                if ((s & 1) == 1) { lab = i32(tg[off + (s >> 1u)]); }
                let e = nrec_in(lp[(t * pm.N + n) * pm.C + lab]);
                if (t == 0) {
                    if (s < 2) { v = e; }
                } else {
                    let a0 = nrec_in(la[prv + s]);
                    var a1 = -1.0e30;
                    if (s >= 1) { a1 = nrec_in(la[prv + s - 1]); }
                    var a2 = -1.0e30;
                    if (s >= 2 && lab != pm.blank) {
                        let lab2 = i32(tg[off + (s >> 1u) - 1]);
                        if (lab2 != lab) { a2 = nrec_in(la[prv + s - 2]); }
                    }
                    let m = max(max(a0, a1), a2);
                    if (m > -1.0e29) { v = max(m + log(exp(a0 - m) + exp(a1 - m) + exp(a2 - m)) + e, -1.0e30); }
                }
            }
            la[obase + t * pm.SM + s] = nrec_out(v);
        }
        storageBarrier();
    }
    if (lid == 0) {
        var z = -1.0e30;
        if (S == 1) { z = 0.0; }
        if (tn > 0) {
            let last = obase + (tn - 1) * pm.SM;
            let a0 = nrec_in(la[last + S - 1]);
            var a1 = -1.0e30;
            if (S >= 2) { a1 = nrec_in(la[last + S - 2]); }
            let m = max(a0, a1);
            z = -1.0e30;
            if (m > -1.0e29) { z = m + log(exp(a0 - m) + exp(a1 - m)); }
        }
        lz[n] = z;
    }
    for (var t = pm.T - 1; t >= 0; t = t - 1) {
        let nxt = obase + (t + 1) * pm.SM;
        for (var s = lid; s < pm.SM; s = s + 256) {
            var v = -1.0e30;
            if (t < tn && s < S) {
                if (t == tn - 1) {
                    if (s >= S - 2) { v = 0.0; }
                } else {
                    var lab = pm.blank;
                    if ((s & 1) == 1) { lab = i32(tg[off + (s >> 1u)]); }
                    let row = ((t + 1) * pm.N + n) * pm.C;
                    let b0 = max(nrec_in(lb[nxt + s]) + nrec_in(lp[row + lab]), -1.0e30);
                    var b1 = -1.0e30;
                    if (s + 1 < S) {
                        var l1 = pm.blank;
                        if ((s & 1) == 0) { l1 = i32(tg[off + (s >> 1u)]); }
                        b1 = max(nrec_in(lb[nxt + s + 1]) + nrec_in(lp[row + l1]), -1.0e30);
                    }
                    var b2 = -1.0e30;
                    if (s + 2 < S && (s & 1) == 1) {
                        let l2 = i32(tg[off + (s >> 1u) + 1]);
                        if (l2 != pm.blank && l2 != lab) { b2 = max(nrec_in(lb[nxt + s + 2]) + nrec_in(lp[row + l2]), -1.0e30); }
                    }
                    let m = max(max(b0, b1), b2);
                    if (m > -1.0e29) { v = max(m + log(exp(b0 - m) + exp(b1 - m) + exp(b2 - m)), -1.0e30); }
                }
            }
            lb[obase + t * pm.SM + s] = nrec_out(v);
        }
        storageBarrier();
    }
}
";

    private const string CtcPosteriorsSource = @"
struct Params { T: i32, N: i32, C: i32, SM: i32, blank: i32, first: i32, pad0: i32, pad1: i32, }
@group(0) @binding(0) var<storage, read> la: array<f32>;
@group(0) @binding(1) var<storage, read_write> gm: array<f32>;
@group(0) @binding(2) var<storage, read> lb: array<f32>;
@group(0) @binding(3) var<storage, read> tg: array<f32>;
@group(0) @binding(4) var<storage, read> lens: array<f32>;
@group(0) @binding(5) var<storage, read> lz: array<f32>;
@group(0) @binding(6) var<uniform> pm: Params;

fn nrec_bits(b: u32) -> f32 { return bitcast<f32>(b); }

// The forward-backward kernel's reader: a stored negative infinity becomes the finite sentinel by its bits, since
// WGSL may assume no infinities and a direct comparison on -inf is undefined.
fn nrec_in(x: f32) -> f32 {
    if (bitcast<u32>(x) == 0xff800000u) { return -1.0e30; }
    return max(x, -1.0e30);
}

@compute @workgroup_size(256)
fn nrec_ctc_posteriors(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let idx = i32(gidv.x) + pm.first * 256;
    if (idx >= pm.T * pm.N) { return; }
    let t = idx / pm.N;
    let n = idx - t * pm.N;
    let o = idx * pm.C;
    for (var c = 0; c < pm.C; c = c + 1) { gm[o + c] = 0.0; }
    let tn = i32(lens[n]);
    if (t >= tn) { return; }
    let z = nrec_in(lz[n]);
    if (z <= -1.0e29) {
        let qnan = nrec_bits(0x7fc00000u);
        for (var c = 0; c < pm.C; c = c + 1) { gm[o + c] = qnan; }
        return;
    }
    let un = i32(lens[pm.N + n]);
    let off = i32(lens[2 * pm.N + n]);
    let S = 2 * un + 1;
    let row = (n * pm.T + t) * pm.SM;
    for (var s = 0; s < S; s = s + 1) {
        let a = nrec_in(la[row + s]);
        let bt = nrec_in(lb[row + s]);
        if (a > -1.0e29 && bt > -1.0e29) {
            var lab = pm.blank;
            if ((s & 1) == 1) { lab = i32(tg[off + (s >> 1u)]); }
            gm[o + lab] = gm[o + lab] + exp(a + bt - z);
        }
    }
}
";

    private const string LinearSumAssignmentSource = @"
struct Params { B: i32, S: i32, first: i32, pad0: i32, pad1: i32, pad2: i32, pad3: i32, pad4: i32, }
@group(0) @binding(0) var<storage, read> cost: array<f32>;
@group(0) @binding(1) var<storage, read_write> asg: array<f32>;
@group(0) @binding(2) var<storage, read_write> ws: array<f32>;
@group(0) @binding(3) var<uniform> pm: Params;
var<workgroup> sm: array<f32, 515>;
var<workgroup> wgSink: i32;

@compute @workgroup_size(256)
fn nrec_linear_sum_assignment(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let lid = i32(lidx);
    let b = i32(wid.x) + pm.first;
    if (b >= pm.B) { return; }
    let S = pm.S;
    let obase = b * S * S;
    let wu = b * 8 * S;
    let wv = wu + S;
    let wd = wu + 2 * S;
    let wp = wu + 3 * S;
    let wrc = wu + 4 * S;
    let wcr = wu + 5 * S;
    let wsr = wu + 6 * S;
    let wsc = wu + 7 * S;
    var mx = 0.0;
    for (var k = lid; k < S * S; k = k + 256) {
        let c = cost[obase + k];
        if ((bitcast<i32>(c) & 2139095040) != 2139095040) { mx = max(mx, abs(c)); }
    }
    sm[lid] = mx;
    workgroupBarrier();
    for (var st = 128; st > 0; st = st / 2) {
        if (lid < st) { sm[lid] = max(sm[lid], sm[lid + st]); }
        workgroupBarrier();
    }
    let sz = f32(S);
    let exponent = max(0, i32(bitcast<u32>(sm[0]) >> 23u) - 127);
    let halfExponent = exponent / 2;
    let scaleA = bitcast<f32>(u32(halfExponent + 127) << 23u);
    let scaleB = bitcast<f32>(u32(exponent - halfExponent + 127) << 23u);
    let big = (sz + 1.0) * (2.0 * ((sm[0] / scaleA) / scaleB) + 1.0);
    for (var j = lid; j < S; j = j + 256) {
        ws[wu + j] = 0.0;
        ws[wv + j] = 0.0;
        ws[wp + j] = -1.0;
        ws[wrc + j] = -1.0;
        ws[wcr + j] = -1.0;
    }
    storageBarrier();
    workgroupBarrier();
    for (var cur = 0; cur < S; cur = cur + 1) {
        for (var j = lid; j < S; j = j + 256) {
            ws[wd + j] = 3.0e38;
            ws[wsr + j] = 0.0;
            ws[wsc + j] = 0.0;
        }
        if (lid == 0) {
            sm[512] = 0.0;
            sm[514] = f32(cur);
            wgSink = -1;
        }
        storageBarrier();
        workgroupBarrier();
        loop {
            if (workgroupUniformLoad(&wgSink) >= 0) { break; }
            let minv = sm[512];
            let i = i32(sm[514]);
            let ui = ws[wu + i];
            let crow = obase + i * S;
            var best = 3.0e38;
            var bkey = 1.0e9;
            for (var j = lid; j < S; j = j + 256) {
                if (ws[wsc + j] == 0.0) {
                    var c = cost[crow + j];
                    let bits = bitcast<i32>(c);
                    if ((bits & 2139095040) == 2139095040) {
                        c = big;
                        if (bits == -8388608) { c = -big; }
                    } else { c = (c / scaleA) / scaleB; }
                    let r = minv + c - ui - ws[wv + j];
                    if (r < ws[wd + j]) {
                        ws[wp + j] = f32(i);
                        ws[wd + j] = r;
                    }
                    let sj = ws[wd + j];
                    var key = f32(j);
                    if (ws[wrc + j] >= 0.0) { key = key + sz; }
                    if (sj < best || (sj == best && key < bkey)) {
                        best = sj;
                        bkey = key;
                    }
                }
            }
            sm[lid] = best;
            sm[256 + lid] = bkey;
            workgroupBarrier();
            for (var st = 128; st > 0; st = st / 2) {
                if (lid < st) {
                    let v1 = sm[lid];
                    let v2 = sm[lid + st];
                    let k2 = sm[256 + lid + st];
                    if (v2 < v1 || (v2 == v1 && k2 < sm[256 + lid])) {
                        sm[lid] = v2;
                        sm[256 + lid] = k2;
                    }
                }
                workgroupBarrier();
            }
            if (lid == 0) {
                let kk = i32(sm[256]);
                var j = kk;
                if (kk >= S) { j = kk - S; }
                ws[wsr + i] = 1.0;
                sm[512] = sm[0];
                ws[wsc + j] = 1.0;
                let r4 = ws[wrc + j];
                if (r4 < 0.0) { sm[513] = f32(j); wgSink = j; } else { sm[514] = r4; }
            }
            storageBarrier();
            workgroupBarrier();
        }
        let minv = sm[512];
        for (var r = lid; r < S; r = r + 256) {
            if (ws[wsr + r] != 0.0 && r != cur) { ws[wu + r] = ws[wu + r] + (minv - ws[wd + i32(ws[wcr + r])]); }
        }
        for (var j = lid; j < S; j = j + 256) {
            if (ws[wsc + j] != 0.0) { ws[wv + j] = ws[wv + j] - (minv - ws[wd + j]); }
        }
        storageBarrier();
        workgroupBarrier();
        if (lid == 0) {
            ws[wu + cur] = ws[wu + cur] + minv;
            var j = i32(sm[513]);
            loop {
                let row = i32(ws[wp + j]);
                ws[wrc + j] = f32(row);
                let prev = i32(ws[wcr + row]);
                ws[wcr + row] = f32(j);
                j = prev;
                if (row == cur) { break; }
            }
        }
        storageBarrier();
        workgroupBarrier();
    }
    for (var r = lid; r < S; r = r + 256) { asg[b * S + r] = ws[wcr + r]; }
}
";

    private static readonly System.Collections.Generic.Dictionary<string, string> Sources = new()
    {
        [NumericRecurrenceKernelTemplate.Tridiagonal] = TridiagonalSource,
        [NumericRecurrenceKernelTemplate.FritschCarlson] = FritschCarlsonSource,
        [NumericRecurrenceKernelTemplate.CtcForwardBackward] = CtcForwardBackwardSource,
        [NumericRecurrenceKernelTemplate.CtcPosteriors] = CtcPosteriorsSource,
        [NumericRecurrenceKernelTemplate.LinearSumAssignment] = LinearSumAssignmentSource,
    };

    /// <summary>The WGSL module for one kernel name, its entry point named after the kernel.</summary>
    internal static string SourceFor(string name) => Sources[name];
}
