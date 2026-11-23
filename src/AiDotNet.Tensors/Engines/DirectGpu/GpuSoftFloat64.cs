// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// IEEE-754 binary64 arithmetic in 32-bit integer operations, as shader source: what a device whose shading language
/// has no 64-bit float (WGSL, Metal Shading Language) computes doubles with. A double is two 32-bit words, low word
/// first, the layout a <see cref="GpuElementType.Float64"/> buffer has on every backend.
/// </summary>
/// <remarks>
/// <para>
/// Add, subtract, multiply, divide and square root are correctly rounded (round to nearest even), including
/// subnormals and infinities: they equal the host's results bit for bit, except that every NaN result is the one
/// quiet NaN <c>0xFFF8000000000000</c>, whatever NaN operand produced it (an operand's payload and sign are not
/// carried). The arithmetic follows Berkeley SoftFloat 3; division and square root are digit recurrences with a sticky
/// bit. Exp and log follow fdlibm, and tanh and the logistic function are built on exp; measured against the host's
/// <c>Math</c> functions over 262,144 arguments each, exp, log and tanh are within 1, 0 and 4 ulp.
/// </para>
/// <para>
/// The source is written once, in WGSL, and translated to Metal Shading Language by <see cref="ToMetal"/>, so both
/// backends run the algorithm that is verified on a WebGPU device. The WGSL is restricted to the constructs the
/// translation handles: untyped <c>let</c>/<c>var</c>, <c>vec2&lt;u32&gt;</c>/<c>vec4&lt;u32&gt;</c>, <c>u32</c>/<c>i32</c>
/// conversions, <c>select</c> and <c>countLeadingZeros</c>.
/// </para>
/// <para>
/// Emulated arithmetic costs hundreds of integer operations an element, so it is slower than the host's doubles. A
/// backend that computes with it reports <see cref="Gpu.GpuPrecisionImplementation.Emulated"/> and the engine routes
/// doubles to it only when a policy asks for device residency.
/// </para>
/// </remarks>
internal static class GpuSoftFloat64
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(5);

    /// <summary>The arithmetic: 64-bit integer helpers, pack and round, add, subtract, multiply, divide, square root,
    /// comparison and int conversion.</summary>
    public const string WgslArithmetic = @"// IEEE-754 binary64 in 32-bit integer arithmetic. A double is vec2<u32>(low word, high word), as is a 64-bit integer.
// Round to nearest even throughout. The arithmetic follows Berkeley SoftFloat 3 (addMags/subMags/mul/roundPack);
// division and square root are digit recurrences, exact with a sticky bit.

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}

fn u64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

fn u64_shl(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) { return a; }
    if (n >= 64u) { return vec2<u32>(0u, 0u); }
    if (n >= 32u) { return vec2<u32>(0u, a.x << (n - 32u)); }
    return vec2<u32>(a.x << n, (a.y << n) | (a.x >> (32u - n)));
}

fn u64_shr(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) { return a; }
    if (n >= 64u) { return vec2<u32>(0u, 0u); }
    if (n >= 32u) { return vec2<u32>(a.y >> (n - 32u), 0u); }
    return vec2<u32>((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}

// Shift right, ORing every bit shifted out into bit 0.
fn u64_shr_jam(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) { return a; }
    if (n >= 64u) { return vec2<u32>(select(0u, 1u, (a.x | a.y) != 0u), 0u); }
    let lost = u64_shl(a, 64u - n);
    let r = u64_shr(a, n);
    return vec2<u32>(r.x | select(0u, 1u, (lost.x | lost.y) != 0u), r.y);
}

fn u64_clz(a: vec2<u32>) -> u32 {
    if (a.y != 0u) { return countLeadingZeros(a.y); }
    return 32u + countLeadingZeros(a.x);
}

fn u32_mul(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xffffu;
    let ah = a >> 16u;
    let bl = b & 0xffffu;
    let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let mid = lh + hl;
    let lo = ll + (mid << 16u);
    let hi = ah * bh + (mid >> 16u) + select(0u, 0x10000u, mid < lh) + select(0u, 1u, lo < ll);
    return vec2<u32>(lo, hi);
}

// The 128-bit product: xy is the low 64 bits, zw the high 64.
fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let p00 = u32_mul(a.x, b.x);
    let p01 = u32_mul(a.x, b.y);
    let p10 = u32_mul(a.y, b.x);
    let p11 = u32_mul(a.y, b.y);
    let cross = u64_add(p01, p10);
    let crossCarry = select(0u, 1u, u64_lt(cross, p01));
    let lo = u64_add(p00, vec2<u32>(0u, cross.x));
    let loCarry = select(0u, 1u, u64_lt(lo, p00));
    let hi = u64_add(u64_add(p11, vec2<u32>(cross.y, crossCarry)), vec2<u32>(loCarry, 0u));
    return vec4<u32>(lo.x, lo.y, hi.x, hi.y);
}

fn f64_sign(a: vec2<u32>) -> u32 { return a.y >> 31u; }
fn f64_exp(a: vec2<u32>) -> i32 { return i32((a.y >> 20u) & 0x7ffu); }
fn f64_frac(a: vec2<u32>) -> vec2<u32> { return vec2<u32>(a.x, a.y & 0xfffffu); }
fn f64_is_nan(a: vec2<u32>) -> bool { return (a.y & 0x7fffffffu) > 0x7ff00000u || ((a.y & 0x7fffffffu) == 0x7ff00000u && a.x != 0u); }
fn f64_nan() -> vec2<u32> { return vec2<u32>(0u, 0xfff80000u); }
fn f64_neg(a: vec2<u32>) -> vec2<u32> { return vec2<u32>(a.x, a.y ^ 0x80000000u); }
fn f64_abs(a: vec2<u32>) -> vec2<u32> { return vec2<u32>(a.x, a.y & 0x7fffffffu); }

// sn, biased exponent and significand added together: a significand with bit 52 set carries into the exponent.
fn f64_pack(sn: u32, ex: i32, sig: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(sig.x, (sn << 31u) + (u32(ex) << 20u) + sig.y);
}

// sig holds the significand with its leading bit at 62 and ten rounding bits; ex is the biased exponent less one.
fn f64_round_pack(sn: u32, expIn: i32, sigIn: vec2<u32>) -> vec2<u32> {
    var ex = expIn;
    var sig = sigIn;
    if (ex < 0) {
        sig = u64_shr_jam(sig, u32(-ex));
        ex = 0;
    } else if (ex > 0x7fd || (ex == 0x7fd && u64_add(sig, vec2<u32>(0x200u, 0u)).y >= 0x80000000u)) {
        return f64_pack(sn, 0x7ff, vec2<u32>(0u, 0u));
    }
    let roundBits = sig.x & 0x3ffu;
    sig = u64_shr(u64_add(sig, vec2<u32>(0x200u, 0u)), 10u);
    if (roundBits == 0x200u) { sig.x = sig.x & 0xfffffffeu; }
    if ((sig.x | sig.y) == 0u) { ex = 0; }
    return f64_pack(sn, ex, sig);
}

fn f64_norm_round_pack(sn: u32, ex: i32, sig: vec2<u32>) -> vec2<u32> {
    if ((sig.x | sig.y) == 0u) { return f64_pack(sn, 0, sig); }
    let shift = u64_clz(sig) - 1u;
    return f64_round_pack(sn, ex - i32(shift), u64_shl(sig, shift));
}

fn f64_add_mags(a: vec2<u32>, b: vec2<u32>, sn: u32) -> vec2<u32> {
    let expA = f64_exp(a);
    let expB = f64_exp(b);
    var sigA = f64_frac(a);
    var sigB = f64_frac(b);
    let expDiff = expA - expB;
    var expZ = expA;
    var sigZ = vec2<u32>(0u, 0u);
    if (expDiff == 0) {
        if (expA == 0) { return u64_add(a, sigB); }
        if (expA == 0x7ff) {
            if ((sigA.x | sigA.y | sigB.x | sigB.y) != 0u) { return f64_nan(); }
            return a;
        }
        sigZ = u64_shl(u64_add(u64_add(vec2<u32>(0u, 0x00200000u), sigA), sigB), 9u);
    } else {
        sigA = u64_shl(sigA, 9u);
        sigB = u64_shl(sigB, 9u);
        if (expDiff < 0) {
            if (expB == 0x7ff) {
                if ((sigB.x | sigB.y) != 0u) { return f64_nan(); }
                return f64_pack(sn, 0x7ff, vec2<u32>(0u, 0u));
            }
            expZ = expB;
            if (expA != 0) { sigA = u64_add(sigA, vec2<u32>(0u, 0x20000000u)); } else { sigA = u64_shl(sigA, 1u); }
            sigA = u64_shr_jam(sigA, u32(-expDiff));
        } else {
            if (expA == 0x7ff) {
                if ((sigA.x | sigA.y) != 0u) { return f64_nan(); }
                return a;
            }
            if (expB != 0) { sigB = u64_add(sigB, vec2<u32>(0u, 0x20000000u)); } else { sigB = u64_shl(sigB, 1u); }
            sigB = u64_shr_jam(sigB, u32(expDiff));
        }
        sigZ = u64_add(u64_add(vec2<u32>(0u, 0x20000000u), sigA), sigB);
        if (sigZ.y < 0x40000000u) {
            expZ = expZ - 1;
            sigZ = u64_shl(sigZ, 1u);
        }
    }
    return f64_round_pack(sn, expZ, sigZ);
}

fn f64_sub_mags(a: vec2<u32>, b: vec2<u32>, snIn: u32) -> vec2<u32> {
    var sn = snIn;
    var expA = f64_exp(a);
    let expB = f64_exp(b);
    var sigA = f64_frac(a);
    var sigB = f64_frac(b);
    let expDiff = expA - expB;
    if (expDiff == 0) {
        if (expA == 0x7ff) { return f64_nan(); }
        if (sigA.x == sigB.x && sigA.y == sigB.y) { return vec2<u32>(0u, 0u); }
        if (expA != 0) { expA = expA - 1; }
        var diff = u64_sub(sigA, sigB);
        if (u64_lt(sigA, sigB)) {
            sn = sn ^ 1u;
            diff = u64_sub(sigB, sigA);
        }
        var shift = i32(u64_clz(diff)) - 11;
        var expZ = expA - shift;
        if (expZ < 0) {
            shift = expA;
            expZ = 0;
        }
        return f64_pack(sn, expZ, u64_shl(diff, u32(shift)));
    }
    sigA = u64_shl(sigA, 10u);
    sigB = u64_shl(sigB, 10u);
    var expZ = expA;
    var sigZ = vec2<u32>(0u, 0u);
    if (expDiff < 0) {
        sn = sn ^ 1u;
        if (expB == 0x7ff) {
            if ((sigB.x | sigB.y) != 0u) { return f64_nan(); }
            return f64_pack(sn, 0x7ff, vec2<u32>(0u, 0u));
        }
        if (expA != 0) { sigA = u64_add(sigA, vec2<u32>(0u, 0x40000000u)); } else { sigA = u64_shl(sigA, 1u); }
        sigA = u64_shr_jam(sigA, u32(-expDiff));
        sigB.y = sigB.y | 0x40000000u;
        expZ = expB;
        sigZ = u64_sub(sigB, sigA);
    } else {
        if (expA == 0x7ff) {
            if ((sigA.x | sigA.y) != 0u) { return f64_nan(); }
            return a;
        }
        if (expB != 0) { sigB = u64_add(sigB, vec2<u32>(0u, 0x40000000u)); } else { sigB = u64_shl(sigB, 1u); }
        sigB = u64_shr_jam(sigB, u32(expDiff));
        sigA.y = sigA.y | 0x40000000u;
        sigZ = u64_sub(sigA, sigB);
    }
    return f64_norm_round_pack(sn, expZ - 1, sigZ);
}

fn f64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    if (f64_is_nan(a) || f64_is_nan(b)) { return f64_nan(); }
    let sn = f64_sign(a);
    if (sn == f64_sign(b)) { return f64_add_mags(a, b, sn); }
    return f64_sub_mags(a, b, sn);
}

fn f64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    if (f64_is_nan(a) || f64_is_nan(b)) { return f64_nan(); }
    let sn = f64_sign(a);
    if (sn == f64_sign(b)) { return f64_sub_mags(a, b, sn); }
    return f64_add_mags(a, b, sn);
}

fn f64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    if (f64_is_nan(a) || f64_is_nan(b)) { return f64_nan(); }
    let sn = f64_sign(a) ^ f64_sign(b);
    var expA = f64_exp(a);
    var expB = f64_exp(b);
    var sigA = f64_frac(a);
    var sigB = f64_frac(b);
    let zeroA = expA == 0 && (sigA.x | sigA.y) == 0u;
    let zeroB = expB == 0 && (sigB.x | sigB.y) == 0u;
    if (expA == 0x7ff || expB == 0x7ff) {
        if (zeroA || zeroB) { return f64_nan(); }
        return f64_pack(sn, 0x7ff, vec2<u32>(0u, 0u));
    }
    if (zeroA || zeroB) { return f64_pack(sn, 0, vec2<u32>(0u, 0u)); }
    if (expA == 0) {
        let shift = u64_clz(sigA) - 11u;
        expA = 1 - i32(shift);
        sigA = u64_shl(sigA, shift);
    }
    if (expB == 0) {
        let shift = u64_clz(sigB) - 11u;
        expB = 1 - i32(shift);
        sigB = u64_shl(sigB, shift);
    }
    var expZ = expA + expB - 0x3ff;
    sigA = u64_shl(vec2<u32>(sigA.x, sigA.y | 0x00100000u), 10u);
    sigB = u64_shl(vec2<u32>(sigB.x, sigB.y | 0x00100000u), 11u);
    let product = u64_mul(sigA, sigB);
    var sigZ = vec2<u32>(product.z | select(0u, 1u, (product.x | product.y) != 0u), product.w);
    if (sigZ.y < 0x40000000u) {
        expZ = expZ - 1;
        sigZ = u64_shl(sigZ, 1u);
    }
    return f64_round_pack(sn, expZ, sigZ);
}

fn f64_div(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    if (f64_is_nan(a) || f64_is_nan(b)) { return f64_nan(); }
    let sn = f64_sign(a) ^ f64_sign(b);
    var expA = f64_exp(a);
    var expB = f64_exp(b);
    var sigA = f64_frac(a);
    var sigB = f64_frac(b);
    let zeroA = expA == 0 && (sigA.x | sigA.y) == 0u;
    let zeroB = expB == 0 && (sigB.x | sigB.y) == 0u;
    if (expA == 0x7ff) {
        if (expB == 0x7ff) { return f64_nan(); }
        return f64_pack(sn, 0x7ff, vec2<u32>(0u, 0u));
    }
    if (expB == 0x7ff) { return f64_pack(sn, 0, vec2<u32>(0u, 0u)); }
    if (zeroB) {
        if (zeroA) { return f64_nan(); }
        return f64_pack(sn, 0x7ff, vec2<u32>(0u, 0u));
    }
    if (zeroA) { return f64_pack(sn, 0, vec2<u32>(0u, 0u)); }
    if (expA == 0) {
        let shift = u64_clz(sigA) - 11u;
        expA = 1 - i32(shift);
        sigA = u64_shl(sigA, shift);
    }
    if (expB == 0) {
        let shift = u64_clz(sigB) - 11u;
        expB = 1 - i32(shift);
        sigB = u64_shl(sigB, shift);
    }
    var expZ = expA - expB + 0x3fe;
    sigA.y = sigA.y | 0x00100000u;
    sigB.y = sigB.y | 0x00100000u;
    if (u64_lt(sigA, sigB)) {
        expZ = expZ - 1;
        sigA = u64_shl(sigA, 1u);
    }
    // sigA / sigB is in [1, 2): 63 quotient bits by restoring division, the remainder as the sticky bit.
    var rem = u64_sub(sigA, sigB);
    var q = vec2<u32>(1u, 0u);
    for (var i = 0u; i < 62u; i++) {
        rem = u64_shl(rem, 1u);
        q = u64_shl(q, 1u);
        if (!u64_lt(rem, sigB)) {
            rem = u64_sub(rem, sigB);
            q.x = q.x | 1u;
        }
    }
    q.x = q.x | select(0u, 1u, (rem.x | rem.y) != 0u);
    return f64_round_pack(sn, expZ, q);
}

fn f64_sqrt(a: vec2<u32>) -> vec2<u32> {
    if (f64_is_nan(a)) { return f64_nan(); }
    var expA = f64_exp(a);
    var sigA = f64_frac(a);
    let zero = expA == 0 && (sigA.x | sigA.y) == 0u;
    if (zero) { return a; }
    if (f64_sign(a) != 0u) { return f64_nan(); }
    if (expA == 0x7ff) { return a; }
    if (expA == 0) {
        let shift = u64_clz(sigA) - 11u;
        expA = 1 - i32(shift);
        sigA = u64_shl(sigA, shift);
    }
    let e = expA - 0x3ff;
    let expZ = (e >> 1u) + 0x3fe;
    sigA.y = sigA.y | 0x00100000u;
    if ((e & 1) != 0) { sigA = u64_shl(sigA, 1u); }
    // The root of sigA * 2^56, two radicand bits a step: 55 root bits and an exact remainder.
    var rem = vec2<u32>(0u, 0u);
    var root = vec2<u32>(0u, 0u);
    for (var i = 0u; i < 55u; i++) {
        let position = 108u - 2u * i;
        var pair = 0u;
        if (position >= 56u) { pair = u64_shr(sigA, position - 56u).x & 3u; }
        rem = u64_shl(rem, 2u);
        rem.x = rem.x | pair;
        root = u64_shl(root, 1u);
        let trial = vec2<u32>((root.x << 1u) | 1u, (root.y << 1u) | (root.x >> 31u));
        if (!u64_lt(rem, trial)) {
            rem = u64_sub(rem, trial);
            root.x = root.x | 1u;
        }
    }
    var sigZ = u64_shl(root, 8u);
    sigZ.x = sigZ.x | select(0u, 1u, (rem.x | rem.y) != 0u);
    return f64_round_pack(0u, expZ, sigZ);
}

// a < b, false when either is NaN.
fn f64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
    if (f64_is_nan(a) || f64_is_nan(b)) { return false; }
    let signA = f64_sign(a);
    let signB = f64_sign(b);
    if (signA != signB) {
        return signA != 0u && (((a.y | b.y) & 0x7fffffffu) | a.x | b.x) != 0u;
    }
    if (a.x == b.x && a.y == b.y) { return false; }
    return (signA != 0u) != u64_lt(a, b);
}

fn f64_from_i32(v: i32) -> vec2<u32> {
    if (v == 0) { return vec2<u32>(0u, 0u); }
    let sn = select(0u, 1u, v < 0);
    let magnitude = select(u32(v), u32(-v), v < 0);
    let shift = countLeadingZeros(magnitude) + 21u;
    return f64_pack(sn, 0x432 - i32(shift), u64_shl(vec2<u32>(magnitude, 0u), shift));
}

// Truncation toward zero of a value known to lie within the int range.
fn f64_to_i32(a: vec2<u32>) -> i32 {
    let ex = f64_exp(a);
    if (ex < 0x3ff) { return 0; }
    let sig = vec2<u32>(a.x, (a.y & 0xfffffu) | 0x00100000u);
    let magnitude = i32(u64_shr(sig, u32(0x433 - ex)).x);
    return select(magnitude, -magnitude, f64_sign(a) != 0u);
}

// 2^k for k in [-1022, 1023].
fn f64_pow2(k: i32) -> vec2<u32> { return vec2<u32>(0u, u32(k + 0x3ff) << 20u); }
";

    /// <summary>Exp, log, tanh and the logistic function over <see cref="WgslArithmetic"/>.</summary>
    public const string WgslElementary = @"
// Elementary functions over the arithmetic above, after fdlibm (e_exp.c, e_log.c); within a few ulp.

fn f64_scale_pow2(y: vec2<u32>, k: i32) -> vec2<u32> {
    if (k > 1023) { return f64_mul(f64_mul(y, f64_pow2(1023)), f64_pow2(k - 1023)); }
    if (k < -1022) { return f64_mul(f64_mul(y, f64_pow2(k + 1000)), f64_pow2(-1000)); }
    return f64_mul(y, f64_pow2(k));
}

fn f64_exp_fn(xIn: vec2<u32>) -> vec2<u32> {
    if (f64_is_nan(xIn)) { return f64_nan(); }
    var x = xIn;
    let hx = x.y & 0x7fffffffu;
    let negative = f64_sign(x) != 0u;
    if (hx >= 0x40862e42u) {
        if (hx >= 0x7ff00000u) { return select(x, vec2<u32>(0u, 0u), negative); }
        if (f64_lt(vec2<u32>(0xfefa39efu, 0x40862e42u), x)) { return vec2<u32>(0u, 0x7ff00000u); }
        if (f64_lt(x, vec2<u32>(0xd52d3051u, 0xc0874910u))) { return vec2<u32>(0u, 0u); }
    }
    var k = 0;
    var hi = x;
    var lo = vec2<u32>(0u, 0u);
    if (hx > 0x3fd62e42u) {
        k = f64_to_i32(f64_add(f64_mul(vec2<u32>(0x652b82feu, 0x3ff71547u), x), select(vec2<u32>(0x00000000u, 0x3fe00000u), vec2<u32>(0x00000000u, 0xbfe00000u), negative)));
        let t = f64_from_i32(k);
        hi = f64_sub(x, f64_mul(t, vec2<u32>(0xfee00000u, 0x3fe62e42u)));
        lo = f64_mul(t, vec2<u32>(0x35793c76u, 0x3dea39efu));
        x = f64_sub(hi, lo);
    } else if (hx < 0x3e300000u) {
        return f64_add(vec2<u32>(0x00000000u, 0x3ff00000u), x);
    }
    let t = f64_mul(x, x);
    var p = f64_add(vec2<u32>(0xc5d26bf1u, 0xbebbbd41u), f64_mul(t, vec2<u32>(0x72bea4d0u, 0x3e663769u)));
    p = f64_add(vec2<u32>(0xaf25de2cu, 0x3f11566au), f64_mul(t, p));
    p = f64_add(vec2<u32>(0x16bebd93u, 0xbf66c16cu), f64_mul(t, p));
    p = f64_add(vec2<u32>(0x5555553eu, 0x3fc55555u), f64_mul(t, p));
    let cc = f64_sub(x, f64_mul(t, p));
    if (k == 0) {
        return f64_sub(vec2<u32>(0x00000000u, 0x3ff00000u), f64_sub(f64_div(f64_mul(x, cc), f64_sub(cc, vec2<u32>(0x00000000u, 0x40000000u))), x));
    }
    let y = f64_sub(vec2<u32>(0x00000000u, 0x3ff00000u), f64_sub(f64_sub(lo, f64_div(f64_mul(x, cc), f64_sub(vec2<u32>(0x00000000u, 0x40000000u), cc))), hi));
    return f64_scale_pow2(y, k);
}

fn f64_log_fn(xIn: vec2<u32>) -> vec2<u32> {
    if (f64_is_nan(xIn)) { return f64_nan(); }
    var x = xIn;
    var k = 0;
    if ((x.y & 0x7fffffffu) == 0u && x.x == 0u) { return vec2<u32>(0u, 0xfff00000u); }
    if (f64_sign(x) != 0u) { return f64_nan(); }
    if (x.y < 0x00100000u) {
        k = -54;
        x = f64_mul(x, vec2<u32>(0x00000000u, 0x43500000u));
    }
    if (x.y >= 0x7ff00000u) { return x; }
    var hx = x.y;
    k = k + i32(hx >> 20u) - 1023;
    hx = hx & 0x000fffffu;
    let i = (hx + 0x95f64u) & 0x100000u;
    x = vec2<u32>(x.x, hx | (i ^ 0x3ff00000u));
    k = k + i32(i >> 20u);
    let f = f64_sub(x, vec2<u32>(0x00000000u, 0x3ff00000u));
    let dk = f64_from_i32(k);
    if ((0x000fffffu & (2u + hx)) < 3u) {
        if ((f.x | (f.y & 0x7fffffffu)) == 0u) {
            if (k == 0) { return vec2<u32>(0u, 0u); }
            return f64_add(f64_mul(dk, vec2<u32>(0xfee00000u, 0x3fe62e42u)), f64_mul(dk, vec2<u32>(0x35793c76u, 0x3dea39efu)));
        }
        let r = f64_mul(f64_mul(f, f), f64_sub(vec2<u32>(0x00000000u, 0x3fe00000u), f64_mul(vec2<u32>(0x55555555u, 0x3fd55555u), f)));
        if (k == 0) { return f64_sub(f, r); }
        return f64_sub(f64_mul(dk, vec2<u32>(0xfee00000u, 0x3fe62e42u)),
            f64_sub(f64_sub(r, f64_mul(dk, vec2<u32>(0x35793c76u, 0x3dea39efu))), f));
    }
    let s = f64_div(f, f64_add(vec2<u32>(0x00000000u, 0x40000000u), f));
    let z = f64_mul(s, s);
    let w = f64_mul(z, z);
    var t1 = f64_add(vec2<u32>(0x9997fa04u, 0x3fd99999u), f64_mul(w, f64_add(vec2<u32>(0x1d8e78afu, 0x3fcc71c5u), f64_mul(w, vec2<u32>(0xd078c69fu, 0x3fc39a09u)))));
    t1 = f64_mul(w, t1);
    var t2 = f64_add(vec2<u32>(0x96cb03deu, 0x3fc74664u), f64_mul(w, vec2<u32>(0xdf3e5244u, 0x3fc2f112u)));
    t2 = f64_add(vec2<u32>(0x94229359u, 0x3fd24924u), f64_mul(w, t2));
    t2 = f64_add(vec2<u32>(0x55555593u, 0x3fe55555u), f64_mul(w, t2));
    t2 = f64_mul(z, t2);
    let r = f64_add(t2, t1);
    let big = (i32(hx) - 0x6147a) | (0x6b851 - i32(hx));
    if (big > 0) {
        let hfsq = f64_mul(vec2<u32>(0x00000000u, 0x3fe00000u), f64_mul(f, f));
        if (k == 0) { return f64_sub(f, f64_sub(hfsq, f64_mul(s, f64_add(hfsq, r)))); }
        return f64_sub(f64_mul(dk, vec2<u32>(0xfee00000u, 0x3fe62e42u)),
            f64_sub(f64_sub(hfsq, f64_add(f64_mul(s, f64_add(hfsq, r)), f64_mul(dk, vec2<u32>(0x35793c76u, 0x3dea39efu)))), f));
    }
    if (k == 0) { return f64_sub(f, f64_mul(s, f64_sub(f, r))); }
    return f64_sub(f64_mul(dk, vec2<u32>(0xfee00000u, 0x3fe62e42u)),
        f64_sub(f64_sub(f64_mul(s, f64_sub(f, r)), f64_mul(dk, vec2<u32>(0x35793c76u, 0x3dea39efu))), f));
}

fn f64_tanh_fn(x: vec2<u32>) -> vec2<u32> {
    if (f64_is_nan(x)) { return f64_nan(); }
    let ax = f64_abs(x);
    let sn = x.y & 0x80000000u;
    if (f64_lt(vec2<u32>(0x00000000u, 0x40340000u), ax)) { return vec2<u32>(0u, 0x3ff00000u | sn); }
    // Below 0.5, sinh by its series (e^x - e^-x cancels there) over cosh: 2 sinh e / (e^2 + 1) with e = e^|x|.
    // From 0.5 up, (e - 1) / (e + 1) with e = e^(2|x|). One exponential and one division either way.
    let small = f64_lt(ax, vec2<u32>(0x00000000u, 0x3fe00000u));
    let e = f64_exp_fn(select(f64_add(ax, ax), ax, small));
    var numerator = f64_sub(e, vec2<u32>(0x00000000u, 0x3ff00000u));
    var denominator = f64_add(e, vec2<u32>(0x00000000u, 0x3ff00000u));
    if (small) {
        let x2 = f64_mul(ax, ax);
        var p = vec2<u32>(0x13163819u, 0x3b4761b4u);
        p = f64_add(vec2<u32>(0xf6dcf572u, 0x3bd71b8eu), f64_mul(x2, p));
        p = f64_add(vec2<u32>(0x46814157u, 0x3c62f49bu), f64_mul(x2, p));
        p = f64_add(vec2<u32>(0x7030ad4au, 0x3ce952c7u), f64_mul(x2, p));
        p = f64_add(vec2<u32>(0xe733b81fu, 0x3d6ae7f3u), f64_mul(x2, p));
        p = f64_add(vec2<u32>(0x13a86d09u, 0x3de61246u), f64_mul(x2, p));
        p = f64_add(vec2<u32>(0x67f544e4u, 0x3e5ae645u), f64_mul(x2, p));
        p = f64_add(vec2<u32>(0xa556c734u, 0x3ec71de3u), f64_mul(x2, p));
        p = f64_add(vec2<u32>(0x1a01a01au, 0x3f2a01a0u), f64_mul(x2, p));
        p = f64_add(vec2<u32>(0x11111111u, 0x3f811111u), f64_mul(x2, p));
        p = f64_add(vec2<u32>(0x55555555u, 0x3fc55555u), f64_mul(x2, p));
        p = f64_add(vec2<u32>(0x00000000u, 0x3ff00000u), f64_mul(x2, p));
        numerator = f64_mul(f64_mul(f64_add(ax, ax), p), e);
        denominator = f64_add(f64_mul(e, e), vec2<u32>(0x00000000u, 0x3ff00000u));
    }
    let r = f64_div(numerator, denominator);
    return vec2<u32>(r.x, r.y | sn);
}

// exp(x) / (1 + exp(x)) below zero: exp(-x) overflows there from x = -710, while the logistic is still representable.
fn f64_sigmoid_fn(x: vec2<u32>) -> vec2<u32> {
    if (f64_sign(x) != 0u) {
        let e = f64_exp_fn(x);
        return f64_div(e, f64_add(vec2<u32>(0x00000000u, 0x3ff00000u), e));
    }
    return f64_div(vec2<u32>(0x00000000u, 0x3ff00000u), f64_add(vec2<u32>(0x00000000u, 0x3ff00000u), f64_exp_fn(f64_neg(x))));
}
";

    private static readonly Lazy<string> MetalArithmeticSource = new Lazy<string>(() => MetalPrelude + ToMetal(WgslArithmetic));
    private static readonly Lazy<string> MetalElementarySource = new Lazy<string>(() => ToMetal(WgslElementary));

    /// <summary><see cref="WgslArithmetic"/> in Metal Shading Language.</summary>
    public static string MetalArithmetic => MetalArithmeticSource.Value;

    /// <summary><see cref="WgslElementary"/> in Metal Shading Language.</summary>
    public static string MetalElementary => MetalElementarySource.Value;

    // WGSL's select takes a scalar condition for vector operands; Metal's wants a matching bool vector.
    private const string MetalPrelude = @"
static inline uint sf_select(uint a, uint b, bool c) { return c ? b : a; }
static inline int sf_select(int a, int b, bool c) { return c ? b : a; }
static inline uint2 sf_select(uint2 a, uint2 b, bool c) { return c ? b : a; }
";

    private static readonly Regex Function = new Regex(
        @"^fn\s+(\w+)\(([^)]*)\)\s*->\s*([\w<>]+)\s*\{", RegexOptions.CultureInvariant, RegexTimeout);
    private static readonly Regex Local = new Regex(@"\b(?:let|var)\s+(\w+)\s*=", RegexOptions.CultureInvariant, RegexTimeout);
    private static readonly Regex Conversion = new Regex(@"\b(u32|i32)\(", RegexOptions.CultureInvariant, RegexTimeout);

    private static string MetalType(string wgsl) => wgsl.Trim() switch
    {
        "u32" => "uint",
        "i32" => "int",
        "bool" => "bool",
        "vec2<u32>" => "uint2",
        "vec4<u32>" => "uint4",
        var other => throw new InvalidOperationException($"The soft-float source uses a type the Metal translation does not know: '{other}'."),
    };

    /// <summary>
    /// Translates the restricted WGSL of this class to Metal Shading Language: function signatures, untyped locals
    /// (<c>auto</c>), the integer vector types and conversions, <c>select</c> and <c>countLeadingZeros</c>. Statements
    /// and expressions are otherwise the same text in both languages. WGSL resolves a function wherever it is declared
    /// and Metal only after its declaration, so every function is declared ahead of the definitions.
    /// </summary>
    internal static string ToMetal(string wgsl)
    {
        if (wgsl is null) throw new ArgumentNullException(nameof(wgsl));
        var metal = new StringBuilder(wgsl.Length + wgsl.Length / 8);
        var prototypes = new StringBuilder();
        foreach (string line in wgsl.Replace("\r\n", "\n").Split('\n'))
            metal.Append(TranslateLine(line, prototypes)).Append('\n');

        return prototypes.Append(metal).ToString();
    }

    // One line of the soft-float WGSL in Metal; a function signature also adds its prototype to `prototypes`.
    private static string TranslateLine(string line, StringBuilder prototypes)
    {
        var function = Function.Match(line);
        if (function.Success)
        {
            var parameters = new StringBuilder();
            foreach (string parameter in function.Groups[2].Value.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries))
            {
                int colon = parameter.IndexOf(':');
                if (colon < 0)
                    throw new InvalidOperationException($"The soft-float source has an untyped parameter: '{parameter}'.");
                if (parameters.Length > 0) parameters.Append(", ");
                parameters.Append(MetalType(parameter.Substring(colon + 1))).Append(' ').Append(parameter.Substring(0, colon).Trim());
            }

            string signature = "static inline " + MetalType(function.Groups[3].Value) + " " + function.Groups[1].Value + "(" + parameters + ")";
            prototypes.Append(signature).Append(';').Append('\n');
            line = signature + " {" + line.Substring(function.Length);
        }

        line = Local.Replace(line, "auto $1 =");
        line = line.Replace("vec2<u32>", "uint2").Replace("vec4<u32>", "uint4");
        line = Conversion.Replace(line, match => match.Groups[1].Value == "u32" ? "uint(" : "int(");
        return line.Replace("countLeadingZeros(", "clz(").Replace("select(", "sf_select(");
    }
}
