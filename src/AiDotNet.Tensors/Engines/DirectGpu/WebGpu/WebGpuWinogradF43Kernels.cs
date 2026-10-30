#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #1432: the WGSL instance of WinogradF43KernelTemplate's tile transforms, restated operation for operation (WGSL is not
// C-like enough to share the template's bodies; see WebGpuWinogradKernels). Each kernel is its own module: the read
// buffer at binding 0, the written one at 1, the int parameters in a uniform struct padded to a 16-byte multiple at 2
// (at 3 for the output transform, whose bias is a third buffer). Boundary reads use if, never select().
internal static class WebGpuWinogradF43Kernels
{
    public const string Filter = @"
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
struct Params { K: i32, C: i32, pad0: i32, pad1: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
@compute @workgroup_size(256)
fn wino_f43_filter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    if (idx >= pm.K * pm.C) { return; }
    var g: array<f32, 9>;
    for (var q = 0; q < 9; q = q + 1) { g[q] = w[idx * 9 + q]; }
    var t: array<f32, 18>;
    for (var j = 0; j < 3; j = j + 1) {
        let g0 = g[j]; let g1 = g[3 + j]; let g2 = g[6 + j];
        t[j] = 0.25 * g0;
        t[3 + j] = -(g0 + g1 + g2) / 6.0;
        t[6 + j] = -(g0 - g1 + g2) / 6.0;
        t[9 + j] = g0 / 24.0 + g1 / 12.0 + g2 / 6.0;
        t[12 + j] = g0 / 24.0 - g1 / 12.0 + g2 / 6.0;
        t[15 + j] = g2;
    }
    let plane = pm.K * pm.C;
    for (var i = 0; i < 6; i = i + 1) {
        let a = t[i * 3]; let b = t[i * 3 + 1]; let e = t[i * 3 + 2];
        u[(i * 6 + 0) * plane + idx] = 0.25 * a;
        u[(i * 6 + 1) * plane + idx] = -(a + b + e) / 6.0;
        u[(i * 6 + 2) * plane + idx] = -(a - b + e) / 6.0;
        u[(i * 6 + 3) * plane + idx] = a / 24.0 + b / 12.0 + e / 6.0;
        u[(i * 6 + 4) * plane + idx] = a / 24.0 - b / 12.0 + e / 6.0;
        u[(i * 6 + 5) * plane + idx] = e;
    }
}
";

    public const string Input = @"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> v: array<f32>;
struct Params { N: i32, C: i32, H: i32, W: i32, pad: i32, tilesH: i32, tilesW: i32, pad1: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
@compute @workgroup_size(256)
fn wino_f43_input(@builtin(global_invocation_id) gid: vec3<u32>) {
    let P = pm.N * pm.tilesH * pm.tilesW;
    let idx = i32(gid.x);
    if (idx >= pm.C * P) { return; }
    let c = idx / P;
    let p = idx - c * P;
    let n = p / (pm.tilesH * pm.tilesW);
    let tt = p - n * (pm.tilesH * pm.tilesW);
    let th = tt / pm.tilesW;
    let tw = tt - th * pm.tilesW;
    let h0 = th * 4 - pm.pad;
    let w0 = tw * 4 - pm.pad;
    let src = (n * pm.C + c) * pm.H * pm.W;
    var d: array<f32, 36>;
    for (var i = 0; i < 6; i = i + 1) {
        let hh = h0 + i;
        for (var j = 0; j < 6; j = j + 1) {
            let ww = w0 + j;
            d[i * 6 + j] = 0.0;
            if (hh >= 0 && hh < pm.H && ww >= 0 && ww < pm.W) { d[i * 6 + j] = x[src + hh * pm.W + ww]; }
        }
    }
    var s: array<f32, 36>;
    for (var j = 0; j < 6; j = j + 1) {
        let z0 = d[j]; let z1 = d[6 + j]; let z2 = d[12 + j]; let z3 = d[18 + j]; let z4 = d[24 + j]; let z5 = d[30 + j];
        s[j] = 4.0 * z0 - 5.0 * z2 + z4;
        s[6 + j] = -4.0 * z1 - 4.0 * z2 + z3 + z4;
        s[12 + j] = 4.0 * z1 - 4.0 * z2 - z3 + z4;
        s[18 + j] = -2.0 * z1 - z2 + 2.0 * z3 + z4;
        s[24 + j] = 2.0 * z1 - z2 - 2.0 * z3 + z4;
        s[30 + j] = 4.0 * z1 - 5.0 * z3 + z5;
    }
    let plane = pm.C * P;
    for (var i = 0; i < 6; i = i + 1) {
        let z0 = s[i * 6]; let z1 = s[i * 6 + 1]; let z2 = s[i * 6 + 2]; let z3 = s[i * 6 + 3]; let z4 = s[i * 6 + 4];
        let z5 = s[i * 6 + 5];
        v[(i * 6 + 0) * plane + idx] = 4.0 * z0 - 5.0 * z2 + z4;
        v[(i * 6 + 1) * plane + idx] = -4.0 * z1 - 4.0 * z2 + z3 + z4;
        v[(i * 6 + 2) * plane + idx] = 4.0 * z1 - 4.0 * z2 - z3 + z4;
        v[(i * 6 + 3) * plane + idx] = -2.0 * z1 - z2 + 2.0 * z3 + z4;
        v[(i * 6 + 4) * plane + idx] = 2.0 * z1 - z2 - 2.0 * z3 + z4;
        v[(i * 6 + 5) * plane + idx] = 4.0 * z1 - 5.0 * z3 + z5;
    }
}
";

    public const string Output = @"
@group(0) @binding(0) var<storage, read> mm: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
struct Params { N: i32, K: i32, outH: i32, outW: i32, tilesH: i32, tilesW: i32, epi: i32, pad1: i32, }
@group(0) @binding(3) var<uniform> pm: Params;
@compute @workgroup_size(256)
fn wino_f43_output(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    var m: array<f32, 36>;
    for (var q = 0; q < 36; q = q + 1) { m[q] = mm[q * plane + idx]; }
    var s: array<f32, 24>;
    for (var j = 0; j < 6; j = j + 1) {
        let z0 = m[j]; let z1 = m[6 + j]; let z2 = m[12 + j]; let z3 = m[18 + j]; let z4 = m[24 + j]; let z5 = m[30 + j];
        s[j] = z0 + z1 + z2 + z3 + z4;
        s[6 + j] = z1 - z2 + 2.0 * z3 - 2.0 * z4;
        s[12 + j] = z1 + z2 + 4.0 * z3 + 4.0 * z4;
        s[18 + j] = z1 - z2 + 8.0 * z3 - 8.0 * z4 + z5;
    }
    let dst = (n * pm.K + k) * pm.outH * pm.outW;
    let h = th * 4;
    let w4 = tw * 4;
    var bk: f32 = 0.0;
    if ((pm.epi & 1) != 0) { bk = bias[k]; }
    let relu = (pm.epi & 2) != 0;
    for (var i = 0; i < 4; i = i + 1) {
        if (h + i >= pm.outH) { break; }
        let z0 = s[i * 6]; let z1 = s[i * 6 + 1]; let z2 = s[i * 6 + 2]; let z3 = s[i * 6 + 3]; let z4 = s[i * 6 + 4];
        let z5 = s[i * 6 + 5];
        var o: array<f32, 4>;
        o[0] = z0 + z1 + z2 + z3 + z4;
        o[1] = z1 - z2 + 2.0 * z3 - 2.0 * z4;
        o[2] = z1 + z2 + 4.0 * z3 + 4.0 * z4;
        o[3] = z1 - z2 + 8.0 * z3 - 8.0 * z4 + z5;
        let row = dst + (h + i) * pm.outW;
        for (var j = 0; j < 4; j = j + 1) {
            if (w4 + j >= pm.outW) { break; }
            var ov = o[j] + bk;
            if (relu && ov < 0.0) { ov = 0.0; }
            y[row + w4 + j] = ov;
        }
    }
}
";

    /// <summary>The module source for a <see cref="WinogradF43KernelTemplate"/> kernel name.</summary>
    internal static string SourceFor(string name) => name switch
    {
        WinogradF43KernelTemplate.Filter => Filter,
        WinogradF43KernelTemplate.Input => Input,
        WinogradF43KernelTemplate.Output => Output,
        _ => throw new System.ArgumentException($"No WGSL Winograd F(4x4) kernel named '{name}'.", nameof(name)),
    };
}
#endif
