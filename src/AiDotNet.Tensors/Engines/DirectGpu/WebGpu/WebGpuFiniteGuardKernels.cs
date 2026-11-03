#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #1522: the device-side finiteness guard on WebGPU, the WGSL counterpart of OpenCL's (#1502). The multi-tensor sum of
// squares accumulates a float sum and a non-finite mark per tensor and finishes as the double IDeviceScalarOps promises
// (NaN when any element was not finite); the clip reads that double. The accumulator is declared as u32 words so the
// mark and the double's words move as bits. The guarded optimizer module is OptimizerSource with the guard bound after
// its buffers and read before any write.
internal static class WebGpuFiniteGuardKernels
{
    public const string Source = @"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> acc: array<u32>;
struct Params { n: u32, maxNorm: f32, _p0: u32, _p1: u32, }
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> partialSum: array<f32, 256>;
var<workgroup> partialScaled: array<f32, 256>;
var<workgroup> partialBad: array<u32, 256>;

// a = tensor. Only work-group 0 runs; thread 0 adds the partials in order (dispatches run in order, so no atomics) to
// acc[0] (the sum of squares) and acc[1] (the sum of (x * 2^-80)^2), or sets acc[1] to NaN bits when any element was
// not finite (read from the exponent bits; the mark is kept by its bits, as WGSL need not propagate NaN).
// The scaled sum carries a norm past the float range: when acc[0] overflows (the sum is at least 2^128) the scaled sum
// is at least 2^-32, a normal float, and it cannot itself overflow below ~4e9 elements at FLT_MAX.
@compute @workgroup_size(256)
fn guard_sum_squares_accumulate(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    if (wid.x != 0u) { return; }
    let down = bitcast<f32>(0x17800000u); // 2^-80, exact
    var v: f32 = 0.0;
    var w: f32 = 0.0;
    var bad: u32 = 0u;
    for (var i: u32 = lid; i < params.n; i = i + 256u) {
        let e = a[i];
        let d = e * down;
        v = v + e * e;
        w = w + d * d;
        if ((bitcast<u32>(e) & 0x7f800000u) == 0x7f800000u) { bad = 1u; }
    }
    partialSum[lid] = v;
    partialScaled[lid] = w;
    partialBad[lid] = bad;
    workgroupBarrier();
    if (lid == 0u) {
        var s: f32 = 0.0;
        var t: f32 = 0.0;
        var f: u32 = 0u;
        for (var j: u32 = 0u; j < 256u; j = j + 1u) { s = s + partialSum[j]; t = t + partialScaled[j]; f = f | partialBad[j]; }
        acc[0] = bitcast<u32>(bitcast<f32>(acc[0]) + s);
        let prev = acc[1];
        if (f != 0u || (prev & 0x7f800000u) == 0x7f800000u) { acc[1] = 0x7fc00000u; }
        else { acc[1] = bitcast<u32>(bitcast<f32>(prev) + t); }
    }
}

// The accumulated (float sum, scaled sum) as a double built from bits: NaN when the scaled sum is NaN (an element was
// not finite), the scaled sum times 2^160 when only the float sum overflowed (the finite value CUDA's double
// accumulation gives), else the float sum exactly (a subnormal normalised into the double's wider exponent). Writing
// DBL_MAX for an overflow made the clip coefficient ~0 and erased the gradients.
@compute @workgroup_size(256)
fn guard_sum_squares_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    // The module's layout is derived per entry point: binding 0 and the uniform stay in it only if this one uses them.
    _ = a[0];
    _ = params.n;
    if (gid.x != 0u) { return; }
    let u = acc[0];
    let e = (u >> 23u) & 0xffu;
    let mant = u & 0x7fffffu;
    var hi: u32;
    var lo: u32;
    let su = acc[1];
    let se = (su >> 23u) & 0xffu;
    let smant = su & 0x7fffffu;
    if (se == 0xffu) { hi = 0x7ff80000u; lo = 0u; }
    else if (e == 0xffu) {
        // The scaled sum is >= 2^-32 here, so normal; re-bias its exponent by the 2^160 the scaling removed.
        if (se == 0u) { hi = 0x7fefffffu; lo = 0xffffffffu; }
        else { hi = ((se + 896u + 160u) << 20u) | (smant >> 3u); lo = smant << 29u; }
    }
    else if (e == 0u && mant == 0u) { hi = 0u; lo = 0u; }
    else if (e == 0u) {
        let shift = countLeadingZeros(mant) - 8u;
        let m = (mant << shift) & 0x7fffffu;
        hi = (u32(1 - i32(shift) - 127 + 1023) << 20u) | (m >> 3u);
        lo = m << 29u;
    }
    else { hi = ((e + 896u) << 20u) | (mant >> 3u); lo = mant << 29u; }
    acc[0] = lo;
    acc[1] = hi;
}

// a = scale: min(1, maxNorm / (norm + 1e-6)) from the finished double, the norm taken from the double's exponent and
// mantissa (sum = m * 2^k, norm = sqrt(m) * 2^(k/2)) so a sum past the float range, carried by the finish from the
// scaled sum, gets the coefficient CUDA's double norm gives. A non-finite norm leaves the gradients unscaled.
@compute @workgroup_size(256)
fn guard_clip_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let lo = acc[0];
    let hi = acc[1];
    let ef = (hi >> 20u) & 0x7ffu;
    var c: f32 = 1.0;
    if (ef == 0u) {
        c = params.maxNorm / 1e-6;
    } else if (ef != 0x7ffu) {
        var k = i32(ef) - 1023;
        var m = bitcast<f32>(0x3f800000u | ((hi & 0xfffffu) << 3u) | (lo >> 29u));
        if ((k & 1) != 0) { m = m * 2.0; k = k - 1; }
        let h = k / 2;
        let root = sqrt(m);
        if (h < 64) { c = params.maxNorm / (ldexp(root, h) + 1e-6); } else { c = ldexp(params.maxNorm / root, -h); }
    }
    a[0] = min(c, 1.0);
}
";

    /// <summary>OptimizerSource with the guard at binding 4 (the uniform moves to 5) and tested in every entry point's
    /// bounds check.</summary>
    public static readonly string GuardedOptimizerSource = Guarded(WebGpuKernels.OptimizerSource);

    private static string Guarded(string source)
    {
        const string Uniform = "@group(0) @binding(4) var<uniform> opt_params: OptimizerParams;";
        const string Bounds = "if (idx < opt_params.size) {";
        if (source.IndexOf(Uniform, StringComparison.Ordinal) < 0 || source.IndexOf(Bounds, StringComparison.Ordinal) < 0)
            throw new InvalidOperationException("OptimizerSource's layout moved; update the guarded form.");
        return source
            .Replace(Uniform, "@group(0) @binding(4) var<storage, read> guard_bits: array<u32>;\n"
                + "@group(0) @binding(5) var<uniform> opt_params: OptimizerParams;")
            .Replace(Bounds, "if (idx < opt_params.size && (guard_bits[1] & 0x7ff00000u) != 0x7ff00000u) {");
    }
}
#endif
