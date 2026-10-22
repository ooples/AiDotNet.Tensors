#if NET7_0_OR_GREATER
// Copyright (c) AiDotNet. All rights reserved.
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
var<workgroup> partialBad: array<u32, 256>;

// a = tensor. Only work-group 0 runs; thread 0 adds the partials in order (dispatches run in order, so no atomics) and
// marks acc[1] when any element was not finite (read from the exponent bits).
@compute @workgroup_size(256)
fn guard_sum_squares_accumulate(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    if (wid.x != 0u) { return; }
    var v: f32 = 0.0;
    var bad: u32 = 0u;
    for (var i: u32 = lid; i < params.n; i = i + 256u) {
        let e = a[i];
        v = v + e * e;
        if ((bitcast<u32>(e) & 0x7f800000u) == 0x7f800000u) { bad = 1u; }
    }
    partialSum[lid] = v;
    partialBad[lid] = bad;
    workgroupBarrier();
    if (lid == 0u) {
        var s: f32 = 0.0;
        var f: u32 = 0u;
        for (var j: u32 = 0u; j < 256u; j = j + 1u) { s = s + partialSum[j]; f = f | partialBad[j]; }
        acc[0] = bitcast<u32>(bitcast<f32>(acc[0]) + s);
        if (f != 0u) { acc[1] = 1u; }
    }
}

// The accumulated (float sum, mark) as a double built from bits: NaN when marked, DBL_MAX when only the float sum
// overflowed, else the float sum exactly (a subnormal normalised into the double's wider exponent).
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
    if (acc[1] != 0u) { hi = 0x7ff80000u; lo = 0u; }
    else if (e == 0xffu) { hi = 0x7fefffffu; lo = 0xffffffffu; }
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

// a = scale: min(1, maxNorm / (norm + 1e-6)) from the finished double (narrowed back to a float: it came from one);
// a non-finite norm, or a sum past the float range, leaves the gradients unscaled.
@compute @workgroup_size(256)
fn guard_clip_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let lo = acc[0];
    let hi = acc[1];
    let e = i32((hi >> 20u) & 0x7ffu) - 1023;
    var c: f32 = 1.0;
    // Past the float range (NaN and infinity included) the norm is not finite: c stays 1.
    if (e <= 127) {
        var sum: f32 = 0.0;
        if (e >= -126) { sum = bitcast<f32>((u32(e + 127) << 23u) | ((hi & 0xfffffu) << 3u) | (lo >> 29u)); }
        let norm = sqrt(sum);
        if ((bitcast<u32>(norm) & 0x7f800000u) != 0x7f800000u) { c = params.maxNorm / (norm + 1e-6); }
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
