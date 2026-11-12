// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The WGSL instance of DataDecodeKernelTemplate, restated operation for operation: every storage buffer an array of u32
// or i32 (raw bytes four per word, float results as their bits), the ints in a uniform struct after the buffers. Each
// kernel is its own module whose entry point is named after the template's kernel.
internal static class WebGpuDataDecodeKernels
{
    /// <summary>Words of every uniform struct (the largest kernel's five ints, padded to a 16-byte multiple).</summary>
    internal const int UniformWords = 8;

    private const string Helpers = @"
fn dd_msb(x: u32) -> i32 {
    var r = 0;
    var v = x;
    if (v >= 65536u) { v = v >> 16u; r = r + 16; }
    if (v >= 256u) { v = v >> 8u; r = r + 8; }
    if (v >= 16u) { v = v >> 4u; r = r + 4; }
    if (v >= 4u) { v = v >> 2u; r = r + 2; }
    if (v >= 2u) { r = r + 1; }
    return r;
}

fn dd_shr64_lo(hi: u32, lo: u32, s: i32) -> u32 {
    if (s <= 0) { return lo; }
    if (s >= 64) { return 0u; }
    if (s >= 32) { return hi >> u32(s - 32); }
    return (lo >> u32(s)) | (hi << u32(32 - s));
}

fn dd_bit64(hi: u32, lo: u32, s: i32) -> u32 {
    if (s >= 32) { return (hi >> u32(s - 32)) & 1u; }
    return (lo >> u32(s)) & 1u;
}

fn dd_low_bits_set(hi: u32, lo: u32, s: i32) -> bool {
    if (s <= 0) { return false; }
    if (s < 32) { return (lo & ((1u << u32(s)) - 1u)) != 0u; }
    if (lo != 0u) { return true; }
    if (s == 32) { return false; }
    if (s >= 64) { return hi != 0u; }
    return (hi & ((1u << u32(s - 32)) - 1u)) != 0u;
}

fn dd_round_shift(hi: u32, lo: u32, s: i32) -> u32 {
    var m = dd_shr64_lo(hi, lo, s);
    var rb = 0u;
    if (s <= 64) { rb = dd_bit64(hi, lo, s - 1); }
    let sticky = dd_low_bits_set(hi, lo, s - 1);
    if (rb != 0u && (sticky || (m & 1u) != 0u)) { m = m + 1u; }
    return m;
}

fn dd_u64_to_f32(hi: u32, lo: u32) -> u32 {
    if (hi == 0u && lo == 0u) { return 0u; }
    var p = 0;
    if (hi != 0u) { p = 32 + dd_msb(hi); } else { p = dd_msb(lo); }
    var m = 0u;
    if (p <= 23) {
        m = lo << u32(23 - p);
    } else {
        m = dd_round_shift(hi, lo, p - 23);
        if (m == 16777216u) { m = 8388608u; p = p + 1; }
    }
    return (u32(p + 127) << 23u) | (m & 8388607u);
}

fn dd_i32_to_f32(v: u32) -> u32 {
    if ((v & 2147483648u) != 0u) { return 2147483648u | dd_u64_to_f32(0u, (~v) + 1u); }
    return dd_u64_to_f32(0u, v);
}

fn dd_i64_to_f32(hi: u32, lo: u32) -> u32 {
    if ((hi & 2147483648u) != 0u) {
        let nlo = (~lo) + 1u;
        var nhi = ~hi;
        if (nlo == 0u) { nhi = nhi + 1u; }
        return 2147483648u | dd_u64_to_f32(nhi, nlo);
    }
    return dd_u64_to_f32(hi, lo);
}

fn dd_f16_to_f32(h: u32) -> u32 {
    let sgn = (h & 32768u) << 16u;
    let e = (h >> 10u) & 31u;
    let m = h & 1023u;
    if (e == 31u) { return sgn | 2139095040u | (m << 13u); }
    if (e == 0u) {
        if (m == 0u) { return sgn; }
        let p = dd_msb(m);
        return sgn | (u32(p + 103) << 23u) | ((m << u32(23 - p)) & 8388607u);
    }
    return sgn | ((e + 112u) << 23u) | (m << 13u);
}

fn dd_f64_to_f32(hi: u32, lo: u32) -> u32 {
    let sgn = hi & 2147483648u;
    let e = i32((hi >> 20u) & 2047u);
    let mhi = hi & 1048575u;
    if (e == 2047) {
        if (mhi != 0u || lo != 0u) { return sgn | 2143289344u | (((mhi << 3u) | (lo >> 29u)) & 8388607u); }
        return sgn | 2139095040u;
    }
    if (e == 0) { return sgn; }
    var ue = e - 1023;
    if (ue > 127) { return sgn | 2139095040u; }
    let shi = mhi | 1048576u;
    if (ue >= -126) {
        var m = dd_round_shift(shi, lo, 29);
        if (m == 16777216u) { m = 8388608u; ue = ue + 1; }
        if (ue > 127) { return sgn | 2139095040u; }
        return sgn | (u32(ue + 127) << 23u) | (m & 8388607u);
    }
    return sgn | dd_round_shift(shi, lo, -97 - ue);
}

fn dd_mul10_lo(lo: u32, d: u32) -> u32 {
    let l0 = (lo & 65535u) * 10u + d;
    let l1 = (lo >> 16u) * 10u + (l0 >> 16u);
    return (l1 << 16u) | (l0 & 65535u);
}

fn dd_mul10_carry(lo: u32, d: u32) -> u32 {
    let l0 = (lo & 65535u) * 10u + d;
    let l1 = (lo >> 16u) * 10u + (l0 >> 16u);
    return l1 >> 16u;
}

fn dd_big_mul10(x: ptr<function, array<u32, 8>>) {
    var carry = 0u;
    for (var i = 0; i < 8; i = i + 1) {
        let v = (*x)[i];
        let l0 = (v & 65535u) * 10u + carry;
        let l1 = (v >> 16u) * 10u + (l0 >> 16u);
        (*x)[i] = (l1 << 16u) | (l0 & 65535u);
        carry = l1 >> 16u;
    }
}

fn dd_big_bitlen(x: ptr<function, array<u32, 8>>) -> i32 {
    var n = 0;
    for (var i = 7; i >= 0; i = i - 1) {
        if (n == 0 && (*x)[i] != 0u) { n = i * 32 + dd_msb((*x)[i]) + 1; }
    }
    return n;
}

fn dd_big_shl(x: ptr<function, array<u32, 8>>, s: i32) {
    let ls = s >> 5u;
    let bs = s & 31;
    for (var i = 7; i >= 0; i = i - 1) {
        let j = i - ls;
        var v = 0u;
        if (j >= 0) {
            v = (*x)[j] << u32(bs);
            if (bs != 0 && j >= 1) { v = v | ((*x)[j - 1] >> u32(32 - bs)); }
        }
        (*x)[i] = v;
    }
}

fn dd_big_ge(x: ptr<function, array<u32, 8>>, y: ptr<function, array<u32, 8>>) -> bool {
    var r = true;
    var decided = false;
    for (var i = 7; i >= 0; i = i - 1) {
        if (!decided && (*x)[i] != (*y)[i]) {
            decided = true;
            if ((*x)[i] < (*y)[i]) { r = false; }
        }
    }
    return r;
}

fn dd_big_sub(x: ptr<function, array<u32, 8>>, y: ptr<function, array<u32, 8>>) {
    var borrow = 0u;
    for (var i = 0; i < 8; i = i + 1) {
        let u = (*x)[i];
        let w = (*y)[i];
        var nb = 0u;
        if (u < w || (u == w && borrow != 0u)) { nb = 1u; }
        (*x)[i] = u - w - borrow;
        borrow = nb;
    }
}

fn dd_exact_to_f32(sHi: u32, sLo: u32, e10: i32, stickyIn: bool) -> u32 {
    var a: array<u32, 8>;
    var b: array<u32, 8>;
    a[0] = sLo;
    a[1] = sHi;
    b[0] = 1u;
    for (var t = 0; t < e10; t = t + 1) { dd_big_mul10(&a); }
    for (var t = 0; t < -e10; t = t + 1) { dd_big_mul10(&b); }
    let la = dd_big_bitlen(&a);
    let lb = dd_big_bitlen(&b);
    var ex2 = la - lb;
    if (la > lb) { dd_big_shl(&b, la - lb); }
    else if (lb > la) { dd_big_shl(&a, lb - la); }
    if (!dd_big_ge(&a, &b)) { dd_big_shl(&a, 1); ex2 = ex2 - 1; }
    var m = 0u;
    for (var k = 0; k < 25; k = k + 1) {
        var bit = 0u;
        if (dd_big_ge(&a, &b)) { dd_big_sub(&a, &b); bit = 1u; }
        m = (m << 1u) | bit;
        dd_big_shl(&a, 1);
    }
    var sticky = stickyIn;
    if (dd_big_bitlen(&a) != 0) { sticky = true; }
    let rb = m & 1u;
    m = m >> 1u;
    if (rb != 0u && (sticky || (m & 1u) != 0u)) { m = m + 1u; }
    if (m == 16777216u) { m = 8388608u; ex2 = ex2 + 1; }
    if (ex2 < -126 || ex2 > 127) { return 0u; }
    return (u32(ex2 + 127) << 23u) | (m & 8388607u);
}

fn dd_is_white(b: u32) -> bool { return b == 32u || (b >= 9u && b <= 13u); }
";

    // A text byte from the packed words (each kernel declares `words`).
    private const string ByteFn = @"
fn dd_byte(k: i32) -> u32 { return (words[u32(k) >> 2u] >> ((u32(k) & 3u) * 8u)) & 255u; }
";

    internal const string Decode = @"
struct Params { nItems: i32, typeCode: i32, elemSize: i32, littleEndian: i32, firstItem: i32, pad0: i32, pad1: i32, pad2: i32, }
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read_write> outBits: array<u32>;
@group(0) @binding(2) var<uniform> pm: Params;
" + Helpers + ByteFn + @"
@compute @workgroup_size(256)
fn dd_decode_bytes(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let i = i32(gidv.x) + pm.firstItem;
    if (i >= pm.nItems) { return; }
    let at = i * pm.elemSize;
    var lo = 0u;
    var hi = 0u;
    for (var j = 0; j < pm.elemSize; j = j + 1) {
        var k = at + j;
        if (pm.littleEndian == 0) { k = at + pm.elemSize - 1 - j; }
        let b = dd_byte(k);
        if (j < 4) { lo = lo | (b << u32(j * 8)); } else { hi = hi | (b << u32((j - 4) * 8)); }
    }
    var bits = 0u;
    let tc = pm.typeCode;
    if (tc == 0) { bits = dd_f16_to_f32(lo); }
    else if (tc == 1) { bits = lo; }
    else if (tc == 2) { bits = dd_f64_to_f32(hi, lo); }
    else if (tc == 3) { if ((lo & 128u) != 0u) { lo = lo | 4294967040u; } bits = dd_i32_to_f32(lo); }
    else if (tc == 5) { if ((lo & 32768u) != 0u) { lo = lo | 4294901760u; } bits = dd_i32_to_f32(lo); }
    else if (tc == 7) { bits = dd_i32_to_f32(lo); }
    else if (tc == 9) { bits = dd_i64_to_f32(hi, lo); }
    else { bits = dd_u64_to_f32(0u, lo); }
    outBits[i] = bits;
}
";

    internal const string CountSeparators = @"
struct Params { textLen: i32, delim: i32, nSegs: i32, firstItem: i32, pad0: i32, pad1: i32, pad2: i32, pad3: i32, }
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts: array<i32>;
@group(0) @binding(2) var<uniform> pm: Params;
" + ByteFn + @"
@compute @workgroup_size(256)
fn dd_count_separators(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let seg = i32(gidv.x) + pm.firstItem;
    if (seg >= pm.nSegs) { return; }
    let start = seg * 64;
    var stop = pm.textLen;
    if (start + 64 < pm.textLen) { stop = start + 64; }
    let dl = u32(pm.delim);
    var c = 0;
    for (var k = start; k < stop; k = k + 1) {
        let b = dd_byte(k);
        if (b == 10u || b == dl) { c = c + 1; }
    }
    counts[seg] = c;
}
";

    internal const string Scan = @"
struct Params { nItems: i32, pad0: i32, pad1: i32, pad2: i32, pad3: i32, pad4: i32, pad5: i32, pad6: i32, }
@group(0) @binding(0) var<storage, read_write> vals: array<i32>;
@group(0) @binding(1) var<storage, read_write> total: array<i32>;
@group(0) @binding(2) var<uniform> pm: Params;
var<workgroup> sm: array<i32, 256>;

@compute @workgroup_size(256)
fn dd_exclusive_scan(@builtin(local_invocation_index) lidx: u32) {
    let lid = i32(lidx);
    let n = pm.nItems;
    let per = (n + 255) / 256;
    let lo0 = lid * per;
    var hi0 = n;
    if (lo0 + per < n) { hi0 = lo0 + per; }
    var s = 0;
    for (var k = lo0; k < hi0; k = k + 1) { s = s + vals[k]; }
    sm[lid] = s;
    workgroupBarrier();
    if (lid == 0) {
        var run = 0;
        for (var t = 0; t < 256; t = t + 1) { let v = sm[t]; sm[t] = run; run = run + v; }
        total[0] = run;
    }
    workgroupBarrier();
    var acc = sm[lid];
    for (var k = lo0; k < hi0; k = k + 1) { let v = vals[k]; vals[k] = acc; acc = acc + v; }
}
";

    internal const string ScatterBounds = @"
struct Params { textLen: i32, delim: i32, nSegs: i32, trailingNewline: i32, firstItem: i32, pad0: i32, pad1: i32, pad2: i32, }
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> offs: array<i32>;
@group(0) @binding(2) var<storage, read_write> bounds: array<i32>;
@group(0) @binding(3) var<uniform> pm: Params;
" + ByteFn + @"
@compute @workgroup_size(256)
fn dd_scatter_bounds(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let seg = i32(gidv.x) + pm.firstItem;
    if (seg >= pm.nSegs) { return; }
    let start = seg * 64;
    var stop = pm.textLen;
    if (start + 64 < pm.textLen) { stop = start + 64; }
    let dl = u32(pm.delim);
    var f = offs[seg];
    if (seg == 0) { bounds[0] = 0; }
    for (var k = start; k < stop; k = k + 1) {
        let b = dd_byte(k);
        if (b == 10u || b == dl) {
            var e = k;
            if (b == 10u && pm.delim != 13 && k > 0) {
                if (dd_byte(k - 1) == 13u) { e = k - 1; }
            }
            bounds[2 * f + 1] = e;
            bounds[2 * f + 2] = k + 1;
            f = f + 1;
        }
    }
    if (seg == pm.nSegs - 1) {
        var e = pm.textLen;
        if (pm.trailingNewline != 0 && pm.delim != 13 && pm.textLen > 0) {
            if (dd_byte(pm.textLen - 1) == 13u) { e = pm.textLen - 1; }
        }
        bounds[2 * f + 1] = e;
    }
}
";

    internal const string ParseFields = @"
struct Params { nItems: i32, styleMask: i32, missingBits: i32, firstItem: i32, pad0: i32, pad1: i32, pad2: i32, pad3: i32, }
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> bounds: array<i32>;
@group(0) @binding(2) var<storage, read_write> outBits: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: array<i32>;
@group(0) @binding(4) var<uniform> pm: Params;
" + Helpers + ByteFn + @"
@compute @workgroup_size(256)
fn dd_parse_fields(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let f = i32(gidv.x) + pm.firstItem;
    if (f >= pm.nItems) { return; }
    let p = bounds[2 * f];
    let e = bounds[2 * f + 1];
    let sm = pm.styleMask;
    var state = 0;
    var resultBits = u32(pm.missingBits);
    if (p < e) {
        var q = p;
        loop { if (q >= e) { break; } if (dd_is_white(dd_byte(q))) { q = q + 1; } else { break; } }
        if (q > p && (sm & 1) == 0) { state = 1; }
        var r = e;
        loop { if (r <= q) { break; } if (dd_is_white(dd_byte(r - 1))) { r = r - 1; } else { break; } }
        if (r < e && (sm & 2) == 0) { state = 1; }
        if (q == r) { state = 1; }
        var neg = false;
        if (state == 0) {
            let b = dd_byte(q);
            if (b == 45u || b == 43u) {
                if ((sm & 4) == 0) { state = 1; }
                neg = b == 45u;
                q = q + 1;
            }
        }
        var sLo = 0u;
        var sHi = 0u;
        var nsig = 0;
        var dropped = false;
        var scale = 0;
        var digits = 0;
        loop {
            if (state != 0 || q >= r) { break; }
            let b = dd_byte(q);
            if (b < 48u || b > 57u) { break; }
            let d = b - 48u;
            digits = digits + 1;
            if (nsig == 0 && d == 0u) { }
            else if (nsig < 19) { sHi = sHi * 10u + dd_mul10_carry(sLo, d); sLo = dd_mul10_lo(sLo, d); nsig = nsig + 1; }
            else { scale = scale + 1; if (d != 0u) { dropped = true; } }
            q = q + 1;
        }
        if (state == 0 && q < r) {
            if (dd_byte(q) == 46u) {
                if ((sm & 8) == 0) { state = 1; }
                q = q + 1;
                loop {
                    if (state != 0 || q >= r) { break; }
                    let c = dd_byte(q);
                    if (c < 48u || c > 57u) { break; }
                    let d = c - 48u;
                    digits = digits + 1;
                    if (nsig == 0 && d == 0u) { scale = scale - 1; }
                    else if (nsig < 19) {
                        sHi = sHi * 10u + dd_mul10_carry(sLo, d);
                        sLo = dd_mul10_lo(sLo, d);
                        nsig = nsig + 1;
                        scale = scale - 1;
                    }
                    else if (d != 0u) { dropped = true; }
                    q = q + 1;
                }
            }
        }
        if (digits == 0) { state = 1; }
        var ex = 0;
        if (state == 0 && q < r) {
            let b = dd_byte(q);
            if (b == 101u || b == 69u) {
                if ((sm & 16) == 0) { state = 1; }
                q = q + 1;
                var eneg = false;
                if (q < r) {
                    let c = dd_byte(q);
                    if (c == 45u || c == 43u) { eneg = c == 45u; q = q + 1; }
                }
                var ed = 0;
                loop {
                    if (q >= r) { break; }
                    let c = dd_byte(q);
                    if (c < 48u || c > 57u) { break; }
                    if (ex < 100000) { ex = ex * 10 + i32(c - 48u); }
                    ed = ed + 1;
                    q = q + 1;
                }
                if (ed == 0) { state = 1; }
                if (eneg) { ex = -ex; }
            }
        }
        if (q != r) { state = 1; }
        if (state == 0) {
            if (nsig == 0) {
                if (neg) { state = 1; } else { resultBits = 0u; }
            } else {
                let e10 = scale + ex;
                if (nsig - 1 + e10 < -37 || nsig + e10 > 38) { state = 1; }
                else {
                    let lob = dd_exact_to_f32(sHi, sLo, e10, dropped);
                    if (lob == 0u) { state = 1; }
                    else if (dropped) {
                        let hLo = sLo + 1u;
                        var hHi = sHi;
                        if (hLo == 0u) { hHi = hHi + 1u; }
                        if (dd_exact_to_f32(hHi, hLo, e10, true) != lob) { state = 1; }
                    }
                    if (state == 0) {
                        resultBits = lob;
                        if (neg) { resultBits = lob | 2147483648u; }
                    }
                }
            }
        }
    }
    if (state != 0) { resultBits = u32(pm.missingBits); }
    outBits[f] = resultBits;
    status[f] = state;
}
";

    private static readonly System.Collections.Generic.Dictionary<string, string> Sources = new()
    {
        [DataDecodeKernelTemplate.Decode] = Decode,
        [DataDecodeKernelTemplate.CountSeparators] = CountSeparators,
        [DataDecodeKernelTemplate.Scan] = Scan,
        [DataDecodeKernelTemplate.ScatterBounds] = ScatterBounds,
        [DataDecodeKernelTemplate.ParseFields] = ParseFields,
    };

    /// <summary>The WGSL module for one kernel name, its entry point named after the kernel.</summary>
    internal static string SourceFor(string name) => Sources[name];
}
