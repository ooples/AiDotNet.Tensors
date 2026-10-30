// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The WGSL instance of ConvDgradS2KernelTemplate (#1483): the stride-2 3 x 3 four-phase input gradient, its projection-
// shortcut pair, the 1 x 1 stride-2 input gradient and the split reduce, restated operation for operation. Each kernel is
// its own module: the storage buffers in the template's binding order (dX at 1), then the ints in a uniform struct (padded
// to 16 bytes). Every workgroupBarrier sits in uniform control flow: the step counts derive from the uniforms and the
// workgroup id alone.
internal static class WebGpuConvDgradS2Kernels
{
    private const string Header3 = @"
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read_write> dx: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
";

    private const string Prologue = @"
    let tid = i32(lidx);
    let grp = i32(wid.x);
    let N = pm.N; let C = pm.C; let K = pm.K; let OH = pm.OH; let OW = pm.OW;
    let RB = 32 / OW; let YS = OW + 4;
    let rows = N * OH; let H = 2 * OH; let W = 2 * OW;
    let gx = (rows + RB - 1) / RB; let gy = (C + 31) / 32;
    let bz = grp / (gx * gy);
    let rem = grp - bz * (gx * gy);
    let by = rem / gx; let bx = rem - by * gx;
    let R0 = bx * RB; let c0 = by * 32;
    let coBegin = bz * pm.coPerSplit; let coEnd = min(K, coBegin + pm.coPerSplit);
    let dxBase = bz * N * C * H * W;
    let cg = tid & 7; let ch = tid >> 3u;
    let l = (cg * 4) / OW; let b0 = cg * 4 - l * OW;
    let R = R0 + l;
    let aRow = R % OH;
    let lastRow = aRow == OH - 1;
    let c = c0 + ch;
    var acc: array<f32, 16>;
    var steps = 0;
    if (coBegin < coEnd) { steps = (coEnd - coBegin + 7) / 8; }
";

    private static string StageY(string buffer, string channels, string first, string end, int smBase) => $@"
        for (var s = tid; s < 576; s = s + 256) {{
            let kl = s / 72; let o = s - kl * 72; let r = o / YS; let col = o - r * YS;
            let rr = R0 + r; let co = {first} + kl;
            var v = 0.0;
            if (r <= RB && col < OW && rr < rows && co < {end}) {{
                let nn = rr / OH; let a = rr - nn * OH;
                v = {buffer}[((nn * {channels} + co) * OH + a) * OW + col];
            }}
            sm[{smBase} + s] = v;
        }}
";

    private const string StageW = @"
        for (var s = tid; s < 2304; s = s + 256) {
            let kl = s / 288; let o = s - kl * 288; let cl = o / 9; let tap = o - cl * 9;
            let co = co0 + kl;
            var v = 0.0;
            if (co < coEnd && c0 + cl < C) { v = wt[(co * C + c0 + cl) * 9 + tap]; }
            sm[576 + kl * 288 + tap * 32 + cl] = v;
        }
";

    private const string Compute = @"
        for (var k = 0; k < 8; k = k + 1) {
            let yb = k * 72 + l * YS + b0;
            var ya: array<f32, 5>;
            var yb1: array<f32, 5>;
            for (var j = 0; j < 5; j = j + 1) {
                ya[j] = sm[yb + j];
                yb1[j] = select(sm[yb + YS + j], 0.0, lastRow);
            }
            var wq: array<f32, 9>;
            for (var t = 0; t < 9; t = t + 1) { wq[t] = sm[576 + k * 288 + t * 32 + ch]; }
            for (var j = 0; j < 4; j = j + 1) {
                acc[j * 4] = fma(ya[j], wq[4], acc[j * 4]);
                acc[j * 4 + 1] = fma(ya[j + 1], wq[3], fma(ya[j], wq[5], acc[j * 4 + 1]));
                acc[j * 4 + 2] = fma(yb1[j], wq[1], fma(ya[j], wq[7], acc[j * 4 + 2]));
                acc[j * 4 + 3] = fma(yb1[j + 1], wq[0], fma(yb1[j], wq[2], fma(ya[j + 1], wq[6], fma(ya[j], wq[8], acc[j * 4 + 3]))));
            }
        }
";

    private const string Store = @"
    if (R < rows && c < C) {
        let nn = R / OH;
        let o0 = dxBase + ((nn * C + c) * H + 2 * aRow) * W + 2 * b0;
        for (var j = 0; j < 4; j = j + 1) {
            dx[o0 + 2 * j] = acc[j * 4];
            dx[o0 + 2 * j + 1] = acc[j * 4 + 1];
            dx[o0 + W + 2 * j] = acc[j * 4 + 2];
            dx[o0 + W + 2 * j + 1] = acc[j * 4 + 3];
        }
    }
}
";

    private const string Entry = @"
@compute @workgroup_size(256)
fn NAME(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {";

    private static readonly string K3Source = Header3 + @"
struct Params { N: i32, C: i32, K: i32, OH: i32, OW: i32, coPerSplit: i32, pad0: i32, pad1: i32, }
@group(0) @binding(3) var<uniform> pm: Params;
var<workgroup> sm: array<f32, 2880>;
" + Entry + Prologue + @"
    for (var st = 0; st < steps; st = st + 1) {
        let co0 = coBegin + st * 8;
" + StageY("dy", "K", "co0", "coEnd", 0) + StageW + @"
        workgroupBarrier();
" + Compute + @"
        workgroupBarrier();
    }
" + Store;

    private static readonly string K3PairSource = Header3 + @"
@group(0) @binding(3) var<storage, read> dy2: array<f32>;
@group(0) @binding(4) var<storage, read> ws: array<f32>;
struct Params { N: i32, C: i32, K: i32, OH: i32, OW: i32, coPerSplit: i32, K2: i32, co2PerSplit: i32, }
@group(0) @binding(5) var<uniform> pm: Params;
var<workgroup> sm: array<f32, 3712>;
" + Entry + Prologue + @"
    let K2 = pm.K2;
    let co2Begin = bz * pm.co2PerSplit; let co2End = min(K2, co2Begin + pm.co2PerSplit);
    var steps2 = 0;
    if (co2Begin < co2End) { steps2 = (co2End - co2Begin + 7) / 8; }
    let stepsAll = max(steps, steps2);
    for (var st = 0; st < stepsAll; st = st + 1) {
        let co0 = coBegin + st * 8; let co20 = co2Begin + st * 8;
" + StageY("dy", "K", "co0", "coEnd", 0) + StageW + StageY("dy2", "K2", "co20", "co2End", 2880) + @"
        for (var s = tid; s < 256; s = s + 256) {
            let kl = s >> 5u; let cl = s & 31;
            let co = co20 + kl;
            var v = 0.0;
            if (co < co2End && c0 + cl < C) { v = ws[co * C + c0 + cl]; }
            sm[3456 + s] = v;
        }
        workgroupBarrier();
" + Compute + @"
        for (var k = 0; k < 8; k = k + 1) {
            let yb = 2880 + k * 72 + l * YS + b0;
            let w2 = sm[3456 + k * 32 + ch];
            for (var j = 0; j < 4; j = j + 1) { acc[j * 4] = fma(sm[yb + j], w2, acc[j * 4]); }
        }
        workgroupBarrier();
    }
" + Store;

    private static readonly string K1Source = Header3 + @"
struct Params { N: i32, C: i32, K: i32, OH: i32, OW: i32, coPerSplit: i32, pad0: i32, pad1: i32, }
@group(0) @binding(3) var<uniform> pm: Params;
var<workgroup> sm: array<f32, 512>;
" + Entry + @"
    let tid = i32(lidx);
    let grp = i32(wid.x);
    let N = pm.N; let C = pm.C; let K = pm.K; let OH = pm.OH; let OW = pm.OW;
    let HW = OH * OW; let M = N * HW; let H = 2 * OH; let W = 2 * OW;
    let gx = (M + 31) / 32; let gy = (C + 31) / 32;
    let bz = grp / (gx * gy);
    let rem = grp - bz * (gx * gy);
    let by = rem / gx; let bx = rem - by * gx;
    let m0 = bx * 32; let c0 = by * 32;
    let coBegin = bz * pm.coPerSplit; let coEnd = min(K, coBegin + pm.coPerSplit);
    let dxBase = bz * N * C * H * W;
    let cg = tid & 7; let ch = tid >> 3u;
    let c = c0 + ch;
    var acc: array<f32, 4>;
    var steps = 0;
    if (coBegin < coEnd) { steps = (coEnd - coBegin + 7) / 8; }
    for (var st = 0; st < steps; st = st + 1) {
        let co0 = coBegin + st * 8;
        let kl = tid >> 5u; let ci = tid & 31;
        let co = co0 + kl; let m = m0 + ci;
        var v = 0.0;
        if (m < M && co < coEnd) {
            let nn = m / HW;
            v = dy[(nn * K + co) * HW + (m - nn * HW)];
        }
        sm[tid] = v;
        var w = 0.0;
        if (co < coEnd && c0 + ci < C) { w = wt[co * C + c0 + ci]; }
        sm[256 + tid] = w;
        workgroupBarrier();
        for (var k = 0; k < 8; k = k + 1) {
            let w1 = sm[256 + k * 32 + ch];
            for (var j = 0; j < 4; j = j + 1) { acc[j] = fma(sm[k * 32 + cg * 4 + j], w1, acc[j]); }
        }
        workgroupBarrier();
    }
    let m4 = m0 + cg * 4;
    if (m4 < M && c < C) {
        let nn = m4 / HW; let rm = m4 - nn * HW; let a = rm / OW; let b0 = rm - a * OW;
        let o0 = dxBase + ((nn * C + c) * H + 2 * a) * W + 2 * b0;
        for (var j = 0; j < 4; j = j + 1) {
            dx[o0 + 2 * j] = acc[j];
            dx[o0 + 2 * j + 1] = 0.0;
            dx[o0 + W + 2 * j] = 0.0;
            dx[o0 + W + 2 * j + 1] = 0.0;
        }
    }
}
";

    private const string ReduceSource = @"
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> dx: array<f32>;
struct Params { total: i32, splits: i32, pad0: i32, pad1: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
@compute @workgroup_size(256)
fn NAME(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let gid = i32(gidv.x);
    if (gid < pm.total) {
        var s = part[gid];
        for (var i = 1; i < pm.splits; i = i + 1) { s = s + part[i * pm.total + gid]; }
        dx[gid] = s;
    }
}
";

    private static readonly Dictionary<string, string> Sources = new()
    {
        [ConvDgradS2KernelTemplate.K3] = K3Source.Replace("NAME", ConvDgradS2KernelTemplate.K3),
        [ConvDgradS2KernelTemplate.K3Pair] = K3PairSource.Replace("NAME", ConvDgradS2KernelTemplate.K3Pair),
        [ConvDgradS2KernelTemplate.K1] = K1Source.Replace("NAME", ConvDgradS2KernelTemplate.K1),
        [ConvDgradS2KernelTemplate.Reduce] = ReduceSource.Replace("NAME", ConvDgradS2KernelTemplate.Reduce),
    };

    /// <summary>The WGSL module of one <see cref="ConvDgradS2KernelTemplate"/> kernel.</summary>
    internal static string SourceFor(string name) => Sources[name];

    /// <summary>The uniform words a kernel's Params struct holds (its ints, padded to 16 bytes).</summary>
    internal static int UniformWords(string name) => name == ConvDgradS2KernelTemplate.Reduce ? 4 : 8;
}
