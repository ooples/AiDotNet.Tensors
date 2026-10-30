#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #1432: the WGSL instance of WinogradFusedForwardTemplate at a 4-channel reduction step and 8-channel epilogue parts, so
// its workgroup storage is 4096 floats (16 KB, WebGPU's guaranteed maxComputeWorkgroupStorageSize). Restated operation
// for operation (WGSL is not C-like enough to share the template's body). Bindings: x, wt (read), y (written), bias
// (read), then the uniform struct.
internal static class WebGpuWinogradFusedKernels
{
    internal const int ChannelStep = 4;
    internal const int PartChannels = 8;

    internal static string Name => WinogradFusedForwardTemplate.Name(ChannelStep);

    public static readonly string Source = @"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
struct Params { N: i32, R: i32, IH: i32, IW: i32, O: i32, OH: i32, OW: i32, pad: i32, tilesH: i32, tilesW: i32,
    epi: i32, pBlocks: i32, }
@group(0) @binding(4) var<uniform> pm: Params;
var<workgroup> sm: array<f32, 4096>;

@compute @workgroup_size(256)
fn " + Name + @"(@builtin(local_invocation_index) lidU: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = i32(lidU);
    let grp = i32(wid.x);
    let tilesPerImage = pm.tilesH * pm.tilesW;
    let P = pm.N * tilesPerImage;
    let p0 = (grp % pm.pBlocks) * 32;
    let o0 = (grp / pm.pBlocks) * 32;
    let vBase = 16 * 4 * 32;

    let lp = tid & 31;
    let lc = tid >> 5u;
    let gp = p0 + lp;
    var pmask: u32 = 0u;
    var inOff: i32 = 0;
    if (gp < P && lc < 4) {
        let n = gp / tilesPerImage;
        let t = gp - n * tilesPerImage;
        let th = t / pm.tilesW;
        let tw = t - th * pm.tilesW;
        let h0 = th * 2 - pm.pad;
        let w0 = tw * 2 - pm.pad;
        for (var i = 0; i < 4; i = i + 1) {
            for (var j = 0; j < 4; j = j + 1) {
                if (h0 + i >= 0 && h0 + i < pm.IH && w0 + j >= 0 && w0 + j < pm.IW) { pmask = pmask | (1u << u32(i * 4 + j)); }
            }
        }
        inOff = n * pm.R * pm.IH * pm.IW + h0 * pm.IW + w0;
    }
    let chanStride = pm.IH * pm.IW;
    let fr = tid % 4;
    let fo = tid / 4;
    let xc = tid >> 4u;
    let lane = tid & 15;
    let oo = (lane >> 2u) * 8;
    let pp = (lane & 3) * 8;
    var acc: array<f32, 64>;
    for (var q = 0; q < 64; q = q + 1) { acc[q] = 0.0; }

    let steps = (pm.R + 3) / 4;
    for (var step = 0; step < steps; step = step + 1) {
        let r0 = step * 4;
        if (lc < 4) {
            let r = r0 + lc;
            var m: u32 = 0u;
            if (r < pm.R) { m = pmask; }
            var d: array<f32, 16>;
            for (var i = 0; i < 4; i = i + 1) {
                for (var j = 0; j < 4; j = j + 1) {
                    d[i * 4 + j] = 0.0;
                    if (((m >> u32(i * 4 + j)) & 1u) != 0u) { d[i * 4 + j] = x[inOff + r * chanStride + i * pm.IW + j]; }
                }
            }
            var t: array<f32, 16>;
            for (var j = 0; j < 4; j = j + 1) {
                t[j] = d[j] - d[8 + j];
                t[4 + j] = d[4 + j] + d[8 + j];
                t[8 + j] = d[8 + j] - d[4 + j];
                t[12 + j] = d[4 + j] - d[12 + j];
            }
            let vs = vBase + lc * 32 + lp;
            for (var i = 0; i < 4; i = i + 1) {
                sm[vs + (i * 4 + 0) * 4 * 32] = t[i * 4] - t[i * 4 + 2];
                sm[vs + (i * 4 + 1) * 4 * 32] = t[i * 4 + 1] + t[i * 4 + 2];
                sm[vs + (i * 4 + 2) * 4 * 32] = t[i * 4 + 2] - t[i * 4 + 1];
                sm[vs + (i * 4 + 3) * 4 * 32] = t[i * 4 + 1] - t[i * 4 + 3];
            }
        }
        if (tid < 4 * 32) {
            let wr = r0 + fr;
            let wo = o0 + fo;
            var g: array<f32, 9>;
            for (var q = 0; q < 9; q = q + 1) { g[q] = 0.0; }
            if (wr < pm.R && wo < pm.O) {
                for (var q = 0; q < 9; q = q + 1) { g[q] = wt[(wo * pm.R + wr) * 9 + q]; }
            }
            var t: array<f32, 12>;
            for (var j = 0; j < 3; j = j + 1) {
                let g0 = g[j]; let g1 = g[3 + j]; let g2 = g[6 + j];
                t[j] = g0;
                t[3 + j] = 0.5 * (g0 + g1 + g2);
                t[6 + j] = 0.5 * (g0 - g1 + g2);
                t[9 + j] = g2;
            }
            let us = fr * 32 + fo;
            for (var i = 0; i < 4; i = i + 1) {
                let a = t[i * 3]; let b = t[i * 3 + 1]; let c = t[i * 3 + 2];
                sm[us + (i * 4 + 0) * 4 * 32] = a;
                sm[us + (i * 4 + 1) * 4 * 32] = 0.5 * (a + b + c);
                sm[us + (i * 4 + 2) * 4 * 32] = 0.5 * (a - b + c);
                sm[us + (i * 4 + 3) * 4 * 32] = c;
            }
        }
        workgroupBarrier();
        let ub = xc * 4 * 32 + oo;
        let vb = vBase + xc * 4 * 32 + pp;
        for (var r = 0; r < 4; r = r + 1) {
            var a: array<f32, 8>;
            var b: array<f32, 8>;
            for (var i = 0; i < 8; i = i + 1) { a[i] = sm[ub + r * 32 + i]; b[i] = sm[vb + r * 32 + i]; }
            for (var i = 0; i < 8; i = i + 1) {
                for (var j = 0; j < 8; j = j + 1) { acc[i * 8 + j] = fma(a[i], b[j], acc[i * 8 + j]); }
            }
        }
        workgroupBarrier();
    }

    for (var part = 0; part < 4; part = part + 1) {
        if (oo / 8 == part) {
            for (var i = 0; i < 8; i = i + 1) {
                for (var j = 0; j < 8; j = j + 1) { sm[(xc * 8 + oo - 8 * part + i) * 32 + pp + j] = acc[i * 8 + j]; }
            }
        }
        workgroupBarrier();
        for (var pair = tid; pair < 8 * 32; pair = pair + 256) {
            let ol = pair >> 5u;
            let pl = pair & 31;
            let o = o0 + 8 * part + ol;
            let p = p0 + pl;
            if (o < pm.O && p < P) {
                var mm: array<f32, 16>;
                for (var c = 0; c < 16; c = c + 1) { mm[c] = sm[(c * 8 + ol) * 32 + pl]; }
                var t0: array<f32, 4>;
                var t1: array<f32, 4>;
                for (var j = 0; j < 4; j = j + 1) {
                    t0[j] = mm[j] + mm[4 + j] + mm[8 + j];
                    t1[j] = mm[4 + j] - mm[8 + j] - mm[12 + j];
                }
                let n = p / tilesPerImage;
                let t = p - n * tilesPerImage;
                let th = t / pm.tilesW;
                let tw = t - th * pm.tilesW;
                let h = th * 2;
                let w = tw * 2;
                let dst = ((n * pm.O + o) * pm.OH + h) * pm.OW + w;
                var bk: f32 = 0.0;
                if ((pm.epi & 1) != 0) { bk = bias[o]; }
                let relu = (pm.epi & 2) != 0;
                var v00 = t0[0] + t0[1] + t0[2] + bk;
                if (relu && v00 < 0.0) { v00 = 0.0; }
                y[dst] = v00;
                if (w + 1 < pm.OW) {
                    var v01 = t0[1] - t0[2] - t0[3] + bk;
                    if (relu && v01 < 0.0) { v01 = 0.0; }
                    y[dst + 1] = v01;
                }
                if (h + 1 < pm.OH) {
                    var v10 = t1[0] + t1[1] + t1[2] + bk;
                    if (relu && v10 < 0.0) { v10 = 0.0; }
                    y[dst + pm.OW] = v10;
                    if (w + 1 < pm.OW) {
                        var v11 = t1[1] - t1[2] - t1[3] + bk;
                        if (relu && v11 < 0.0) { v11 = 0.0; }
                        y[dst + pm.OW + 1] = v11;
                    }
                }
            }
        }
        workgroupBarrier();
    }
}
";
}
#endif
