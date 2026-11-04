// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// The deterministic global-norm kernels in WGSL (<see cref="Gpu.DeterministicGradientNorm"/> is the reference they
/// match bit for bit). WGSL has neither 64-bit integers nor a widening multiply, so 64- and 128-bit values are
/// vec2 / vec4 of u32 (low word first) and 32x32-bit products are formed from 16-bit halves.
/// </summary>
internal static class WebGpuDeterministicNormKernels
{
    internal const string ModuleName = "DeterministicNorm";

    // Bindings: every entry takes two u32 storage arrays and a 16-byte uniform of four u32.
    internal const string Source = @"
@group(0) @binding(0) var<storage, read_write> a: array<u32>;
@group(0) @binding(1) var<storage, read_write> b: array<u32>;
struct Params { p0: u32, p1: u32, p2: u32, p3: u32, }
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> su: array<u32, 256>;
var<workgroup> su2: array<u32, 256>;
var<workgroup> sn: array<u32, 256>;
var<workgroup> si: array<u32, 256>;
var<workgroup> sacc: array<vec4<u32>, 256>;

fn dgn_add64(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> {
    let lo = x.x + y.x;
    return vec2<u32>(lo, x.y + y.y + select(0u, 1u, lo < x.x));
}
fn dgn_shr64(v: vec2<u32>, sh: i32) -> vec2<u32> {
    if (sh <= 0) { return v; }
    if (sh >= 64) { return vec2<u32>(0u, 0u); }
    if (sh >= 32) { return vec2<u32>(v.y >> u32(sh - 32), 0u); }
    return vec2<u32>((v.x >> u32(sh)) | (v.y << u32(32 - sh)), v.y >> u32(sh));
}
fn dgn_shl64(v: vec2<u32>, sh: i32) -> vec2<u32> {
    if (sh <= 0) { return v; }
    if (sh >= 64) { return vec2<u32>(0u, 0u); }
    if (sh >= 32) { return vec2<u32>(0u, v.x << u32(sh - 32)); }
    return vec2<u32>(v.x << u32(sh), (v.y << u32(sh)) | (v.x >> u32(32 - sh)));
}
fn dgn_bitlen32(x: u32) -> i32 { return 32 - i32(countLeadingZeros(x)); }
fn dgn_bitlen64(v: vec2<u32>) -> i32 { if (v.y != 0u) { return 32 + dgn_bitlen32(v.y); } return dgn_bitlen32(v.x); }
fn dgn_low64(v: vec2<u32>, bits: i32) -> bool {
    if (bits <= 0) { return false; }
    if (bits >= 64) { return (v.x | v.y) != 0u; }
    let t = dgn_shl64(v, 64 - bits);
    return (t.x | t.y) != 0u;
}
fn dgn_bit64(v: vec2<u32>, bit: i32) -> bool {
    if (bit < 32) { return ((v.x >> u32(bit)) & 1u) != 0u; }
    return ((v.y >> u32(bit - 32)) & 1u) != 0u;
}
fn dgn_mul32(x: u32, y: u32) -> vec2<u32> {
    let x0 = x & 0xFFFFu; let x1 = x >> 16u; let y0 = y & 0xFFFFu; let y1 = y >> 16u;
    let p00 = x0 * y0; let p01 = x0 * y1; let p10 = x1 * y0; let p11 = x1 * y1;
    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    return vec2<u32>((p00 & 0xFFFFu) | (mid << 16u), p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}

fn dgn_add128(x: vec4<u32>, y: vec4<u32>) -> vec4<u32> {
    var r = vec4<u32>(0u, 0u, 0u, 0u);
    var c = 0u;
    for (var i = 0; i < 4; i++) {
        let s = x[i] + y[i];
        let c1 = select(0u, 1u, s < x[i]);
        let s2 = s + c;
        let c2 = select(0u, 1u, s2 < s);
        r[i] = s2;
        c = c1 + c2;
    }
    return r;
}
fn dgn_word(v: vec4<u32>, i: i32) -> u32 { if (i < 0 || i > 3) { return 0u; } return v[i]; }
fn dgn_shr128(v: vec4<u32>, sh: i32) -> vec4<u32> {
    if (sh <= 0) { return v; }
    if (sh >= 128) { return vec4<u32>(0u, 0u, 0u, 0u); }
    let ws = sh >> 5u; let bs = sh & 31;
    var r = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i = 0; i < 4; i++) {
        let lo = dgn_word(v, i + ws);
        let hi = dgn_word(v, i + ws + 1);
        if (bs == 0) { r[i] = lo; } else { r[i] = (lo >> u32(bs)) | (hi << u32(32 - bs)); }
    }
    return r;
}
fn dgn_shl128(v: vec4<u32>, sh: i32) -> vec4<u32> {
    if (sh <= 0) { return v; }
    if (sh >= 128) { return vec4<u32>(0u, 0u, 0u, 0u); }
    let ws = sh >> 5u; let bs = sh & 31;
    var r = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i = 0; i < 4; i++) {
        let x = dgn_word(v, i - ws);
        let y = dgn_word(v, i - ws - 1);
        if (bs == 0) { r[i] = x; } else { r[i] = (x << u32(bs)) | (y >> u32(32 - bs)); }
    }
    return r;
}
fn dgn_bitlen128(v: vec4<u32>) -> i32 {
    if (v.w != 0u) { return 96 + dgn_bitlen32(v.w); }
    if (v.z != 0u) { return 64 + dgn_bitlen32(v.z); }
    if (v.y != 0u) { return 32 + dgn_bitlen32(v.y); }
    return dgn_bitlen32(v.x);
}
fn dgn_low128(v: vec4<u32>, bits: i32) -> bool {
    if (bits <= 0) { return false; }
    if (bits >= 128) { return (v.x | v.y | v.z | v.w) != 0u; }
    let t = dgn_shl128(v, 128 - bits);
    return (t.x | t.y | t.z | t.w) != 0u;
}
fn dgn_bit128(v: vec4<u32>, bit: i32) -> bool { return ((dgn_word(v, bit >> 5u) >> u32(bit & 31)) & 1u) != 0u; }

fn dgn_square(bits: u32, emax: u32) -> vec4<u32> {
    var e = (bits >> 23u) & 0xFFu;
    if (e == 0xFFu) { return vec4<u32>(0u, 0u, 0u, 0u); }
    let m = (bits & 0x7FFFFFu) | select(0u, 0x800000u, e != 0u);
    if (e == 0u) { e = 1u; }
    let sq = dgn_mul32(m, m);
    let sh = 68 - 2 * i32(emax - e);
    let v = vec4<u32>(sq.x, sq.y, 0u, 0u);
    if (sh >= 0) { return dgn_shl128(v, sh); }
    return dgn_shr128(v, -sh);
}

struct Root { r: u32, rem: u32, }
fn dgn_isqrt(x: vec2<u32>) -> Root {
    var r = 0u; var q = 0u;
    for (var i = 26; i >= 0; i--) {
        var two: u32;
        if (i >= 16) { two = (x.y >> u32(2 * i - 32)) & 3u; } else { two = (x.x >> u32(2 * i)) & 3u; }
        q = (q << 2u) | two;
        let trial = (r << 2u) | 1u;
        if (q >= trial) { q = q - trial; r = (r << 1u) | 1u; } else { r = r << 1u; }
    }
    return Root(r, q);
}

fn dgn_roundshift(m: vec2<u32>, sh: i32, sticky: bool) -> vec2<u32> {
    if (sh > 64) { return vec2<u32>(0u, 0u); }
    var q = dgn_shr64(m, sh);
    let roundBit = dgn_bit64(m, sh - 1);
    let rest = sticky || dgn_low64(m, sh - 1);
    if (roundBit && (rest || (q.x & 1u) != 0u)) { q = dgn_add64(q, vec2<u32>(1u, 0u)); }
    return q;
}

fn dgn_round_to_float(m: vec2<u32>, e: i32, sticky: bool) -> u32 {
    if ((m.x | m.y) == 0u) { return 0u; }
    let lead = e + dgn_bitlen64(m) - 1;
    var ulp = max(lead - 23, -149);
    let sh = ulp - e;
    var q: vec2<u32>;
    if (sh > 0) { q = dgn_roundshift(m, sh, sticky); } else { q = dgn_shl64(m, -sh); }
    if (q.y != 0u || q.x >= 0x1000000u) { q = dgn_shr64(q, 1); ulp = ulp + 1; }
    if (q.x < 0x800000u) { return q.x; }
    let biased = ulp + 23 + 127;
    if (biased >= 255) { return 0x7F800000u; }
    return (u32(biased) << 23u) | (q.x & 0x7FFFFFu);
}

struct Ext { m: u32, e: i32, }
fn dgn_round24(m: vec2<u32>, e: i32, sticky: bool) -> Ext {
    if ((m.x | m.y) == 0u) { return Ext(0u, 0); }
    let len = dgn_bitlen64(m);
    if (len <= 24) { return Ext(m.x << u32(24 - len), e - (24 - len)); }
    let sh = len - 24;
    var q = dgn_roundshift(m, sh, sticky);
    var ee = e + sh;
    if (q.y != 0u || q.x >= 0x1000000u) { q = dgn_shr64(q, 1); ee = ee + 1; }
    return Ext(q.x, ee);
}

fn dgn_addround24(a0: u32, ae0: i32, b0: u32, be0: i32) -> Ext {
    var am = a0; var ae = ae0; var bm = b0; var be = be0;
    if (am == 0u) { return Ext(bm, be); }
    if (bm == 0u) { return Ext(am, ae); }
    if (ae < be) { let tm = am; am = bm; bm = tm; let te = ae; ae = be; be = te; }
    let d = ae - be;
    if (d > 30) { return dgn_round24(vec2<u32>((am << 3u) | 1u, am >> 29u), ae - 3, false); }
    return dgn_round24(dgn_add64(dgn_shl64(vec2<u32>(am, 0u), d), vec2<u32>(bm, 0u)), be, false);
}

struct Sq { m: u32, e: i32, sticky: bool, }
fn dgn_sqrt(x: vec4<u32>, v: i32) -> Sq {
    let len = dgn_bitlen128(x);
    if (len == 0) { return Sq(0u, 0, false); }
    var k = len - 54;
    if (((v + k) & 1) != 0) { k = k + 1; }
    var t: vec4<u32>;
    if (k >= 0) { t = dgn_shr128(x, k); } else { t = dgn_shl128(x, -k); }
    let dropped = k > 0 && dgn_low128(x, k);
    let root = dgn_isqrt(t.xy);
    return Sq(root.r, (v + k) / 2, dropped || root.rem != 0u);
}

fn dgn_clip_coef(nm: u32, ne: i32, ns: bool, mb: u32) -> u32 {
    let n = dgn_round24(vec2<u32>(nm, 0u), ne, ns);
    let d = dgn_addround24(n.m, n.e, 0x8637BDu, 107 - 150);
    let me = (mb >> 23u) & 0xFFu;
    var mM = (mb & 0x7FFFFFu) | select(0u, 0x800000u, me != 0u);
    var mE = select(1, i32(me), me != 0u) - 150;
    if (mM == 0u) { return 0u; }
    while (mM < 0x800000u) { mM = mM << 1u; mE = mE - 1; }
    if (mE > d.e || (mE == d.e && mM >= d.m)) { return 0x3F800000u; }
    let num = dgn_shl64(vec2<u32>(mM, 0u), 40);
    var q = vec2<u32>(0u, 0u);
    var rem = 0u;
    for (var i = 63; i >= 0; i--) {
        rem = (rem << 1u) | select(0u, 1u, dgn_bit64(num, i));
        q = dgn_shl64(q, 1);
        if (rem >= d.m) { rem = rem - d.m; q.x = q.x | 1u; }
    }
    return dgn_round_to_float(q, mE - 40 - d.e, rem != 0u);
}

fn dgn_to_double(x: vec4<u32>, v: i32) -> vec2<u32> {
    let len = dgn_bitlen128(x);
    if (len == 0) { return vec2<u32>(0u, 0u); }
    var m: vec2<u32>;
    var e: i32;
    if (len > 53) {
        let sh = len - 53;
        m = dgn_shr128(x, sh).xy;
        let roundBit = dgn_bit128(x, sh - 1);
        let rest = sh > 1 && dgn_low128(x, sh - 1);
        if (roundBit && (rest || (m.x & 1u) != 0u)) { m = dgn_add64(m, vec2<u32>(1u, 0u)); }
        e = v + sh;
        if (m.y >= 0x200000u) { m = dgn_shr64(m, 1); e = e + 1; }
    } else {
        m = dgn_shl128(x, 53 - len).xy;
        e = v - (53 - len);
    }
    let lead = e + 52;
    if (lead >= -1022) { return vec2<u32>(m.x, (u32(lead + 1023) << 20u) | (m.y & 0xFFFFFu)); }
    return dgn_shr64(m, -1022 - lead);
}

fn dgn_mul(g: u32, s: u32) -> u32 {
    let eg = (g >> 23u) & 0xFFu; let es = (s >> 23u) & 0xFFu;
    if (eg - 1u < 254u && es - 1u < 254u && eg + es >= 128u) { return bitcast<u32>(bitcast<f32>(g) * bitcast<f32>(s)); }
    let sign = (g ^ s) & 0x80000000u;
    let fg = g & 0x7FFFFFu; let fs = s & 0x7FFFFFu;
    if ((eg == 0xFFu && fg != 0u) || (es == 0xFFu && fs != 0u)) { return 0x7FC00000u; }
    let gzero = eg == 0u && fg == 0u; let szero = es == 0u && fs == 0u;
    if (eg == 0xFFu || es == 0xFFu) { return select(sign | 0x7F800000u, 0x7FC00000u, gzero || szero); }
    if (gzero || szero) { return sign; }
    let prod = dgn_mul32(fg | select(0u, 0x800000u, eg != 0u), fs | select(0u, 0x800000u, es != 0u));
    let e = select(1, i32(eg), eg != 0u) + select(1, i32(es), es != 0u) - 300;
    return dgn_round_to_float(prod, e, false) | sign;
}

fn dgn_satadd(x: u32, y: u32) -> u32 { let s = x + y; return select(s, 0xFFFFFFFFu, s < x); }

// a = tensor, b = partials; params = (n, tensor, first partial). One work group per 4096-element chunk.
@compute @workgroup_size(256)
fn dgn_chunk_partials(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let c = wid.x;
    let start = c * 4096u;
    let end = min(start + 4096u, params.p0);
    var vals: array<u32, 16>;
    var emax = 1u; var nan = 0u; var inf = 0u;
    for (var j = 0u; j < 16u; j++) {
        let i = start + j * 256u + tid;
        var v = 0u;
        if (i < end) { v = a[i]; }
        vals[j] = v;
        let e = (v >> 23u) & 0xFFu;
        if (e == 0xFFu) { if ((v & 0x7FFFFFu) != 0u) { nan = nan + 1u; } else { inf = inf + 1u; } }
        else if (e > emax) { emax = e; }
    }
    su[tid] = emax;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) { if (tid < s) { su[tid] = max(su[tid], su[tid + s]); } workgroupBarrier(); }
    emax = su[0];
    workgroupBarrier();
    su[tid] = (nan << 16u) | inf;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) { if (tid < s) { su[tid] = su[tid] + su[tid + s]; } workgroupBarrier(); }
    let counts = su[0];
    var acc = vec4<u32>(0u, 0u, 0u, 0u);
    for (var j = 0u; j < 16u; j++) { acc = dgn_add128(acc, dgn_square(vals[j], emax)); }
    sacc[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) { if (tid < s) { sacc[tid] = dgn_add128(sacc[tid], sacc[tid + s]); } workgroupBarrier(); }
    if (tid == 0u) {
        let o = (params.p2 + c) * 8u;
        let r = sacc[0];
        b[o] = emax; b[o + 1u] = counts >> 16u; b[o + 2u] = counts & 0xFFFFu; b[o + 3u] = params.p1;
        b[o + 4u] = r.x; b[o + 5u] = r.y; b[o + 6u] = r.z; b[o + 7u] = r.w;
    }
}

// a = partials, b = record; params = (count, headroom). One work group.
@compute @workgroup_size(256)
fn dgn_finish(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let count = params.p0;
    let headroom = i32(params.p1);
    var g = 0u; var first = 0xFFFFFFFFu; var nan = 0u; var inf = 0u;
    for (var i = tid; i < count; i = i + 256u) {
        let o = i * 8u;
        if ((a[o + 4u] | a[o + 5u] | a[o + 6u] | a[o + 7u]) != 0u && a[o] > g) { g = a[o]; }
        nan = dgn_satadd(nan, a[o + 1u]); inf = dgn_satadd(inf, a[o + 2u]);
        if ((a[o + 1u] | a[o + 2u]) != 0u && a[o + 3u] < first) { first = a[o + 3u]; }
    }
    su[tid] = g; su2[tid] = first; sn[tid] = nan; si[tid] = inf;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            su[tid] = max(su[tid], su[tid + s]); su2[tid] = min(su2[tid], su2[tid + s]);
            sn[tid] = dgn_satadd(sn[tid], sn[tid + s]); si[tid] = dgn_satadd(si[tid], si[tid + s]);
        }
        workgroupBarrier();
    }
    g = su[0]; first = su2[0]; nan = sn[0]; inf = si[0];
    var acc = vec4<u32>(0u, 0u, 0u, 0u);
    if (g != 0u) {
        for (var i = tid; i < count; i = i + 256u) {
            let o = i * 8u;
            let x = vec4<u32>(a[o + 4u], a[o + 5u], a[o + 6u], a[o + 7u]);
            if ((x.x | x.y | x.z | x.w) != 0u) { acc = dgn_add128(acc, dgn_shr128(x, 2 * i32(g - a[o]) + headroom)); }
        }
    }
    sacc[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) { if (tid < s) { sacc[tid] = dgn_add128(sacc[tid], sacc[tid + s]); } workgroupBarrier(); }
    if (tid == 0u) {
        let x = sacc[0];
        var v = 0;
        if (g != 0u) { v = 2 * i32(g) - 368 + headroom; }
        b[2] = x.x; b[3] = x.y; b[4] = x.z; b[5] = x.w;
        b[6] = bitcast<u32>(v); b[7] = nan; b[8] = inf; b[9] = first;
        var dbl = vec2<u32>(0u, 0u);
        var norm: u32;
        if (nan != 0u) { dbl = vec2<u32>(0u, 0x7FF80000u); norm = 0x7FC00000u; }
        else if (inf != 0u) { dbl = vec2<u32>(0u, 0x7FF00000u); norm = 0x7F800000u; }
        else {
            dbl = dgn_to_double(x, v);
            let r = dgn_sqrt(x, v);
            norm = dgn_round_to_float(vec2<u32>(r.m, 0u), r.e, r.sticky);
        }
        b[0] = dbl.x; b[1] = dbl.y; b[10] = norm; b[11] = 0x3F800000u;
        b[12] = select(0u, 1u, (nan | inf) != 0u); b[13] = 0u; b[14] = 0u; b[15] = 0u;
    }
}

// a = record, b = scale; params = (maxNorm bits, policy).
@compute @workgroup_size(1)
fn dgn_clip_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    var s: u32;
    if (a[7] != 0u) { s = select(0x3F800000u, 0x7FC00000u, params.p1 != 0u); }
    else if (a[8] != 0u) { s = select(0x3F800000u, 0u, params.p1 != 0u); }
    else {
        let r = dgn_sqrt(vec4<u32>(a[2], a[3], a[4], a[5]), bitcast<i32>(a[6]));
        s = dgn_clip_coef(r.m, r.e, r.sticky, params.p0);
    }
    a[11] = s; a[13] = params.p0;
    b[0] = s;
}

// a = scale, b = tensor; params = (n).
@compute @workgroup_size(256)
fn dgn_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.p0) { b[i] = dgn_mul(b[i], a[0]); }
}
";
}
#endif
