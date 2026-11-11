// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The WGSL instance of GroupedGemmKernelTemplate (64 x 64 output tiles, 256 threads with 4 x 4 values each, operands
// staged 16 deep in workgroup memory), restated operation for operation. Each kernel is its own module: the storage
// buffers in the template's binding order, the ints in a uniform struct after them. WGSL treats a value read from a
// storage buffer as non-uniform, so the group offsets that bound the barrier loops pass through workgroupUniformLoad.
internal static class WebGpuGroupedGemmKernels
{
    /// <summary>Words of the uniform struct (eight ints; the weight gradient's seven leave one as padding).</summary>
    internal const int UniformWords = 8;

    private const string ForwardSource = @"
struct Params { M: i32, N: i32, K: i32, G: i32, tilesN: i32, transB: i32, biasMode: i32, tileBase: i32, }
@group(0) @binding(0) var<storage, read> xa: array<f32>;
@group(0) @binding(1) var<storage, read_write> yc: array<f32>;
@group(0) @binding(2) var<storage, read> wb: array<f32>;
@group(0) @binding(3) var<storage, read> ends: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<uniform> pm: Params;
var<workgroup> sm: array<f32, 2048>;
var<workgroup> wgStart: i32;
var<workgroup> wgEnd: i32;

// The first group whose rows reach past `row` (G when none does).
fn first_group_past(row: i32) -> i32 {
    var lo = 0;
    var hi = pm.G;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2;
        if (i32(ends[mid]) > row) { hi = mid; } else { lo = mid + 1; }
    }
    return lo;
}

@compute @workgroup_size(256)
fn NAME(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = i32(lidx);
    let t = i32(wid.x) + pm.tileBase;
    let rt = t / pm.tilesN;
    let ct = t - rt * pm.tilesN;
    let row0 = rt * 64;
    let col0 = ct * 64;
    let rowHi = min(row0 + 64, pm.M);
    let tr = tid / 16;
    let tc = tid - tr * 16;
    var acc: array<f32, 16>;
    if (tid == 0) { wgStart = first_group_past(row0); }
    var g = workgroupUniformLoad(&wgStart);
    workgroupBarrier();
    loop {
        if (g >= pm.G) { break; }
        if (tid == 0) {
            var s = 0;
            if (g > 0) { s = i32(ends[g - 1]); }
            wgStart = s;
            wgEnd = i32(ends[g]);
        }
        let gs = workgroupUniformLoad(&wgStart);
        let ge = workgroupUniformLoad(&wgEnd);
        if (gs >= rowHi) { break; }
        let r0 = max(gs, row0);
        let r1 = min(ge, rowHi);
        if (r1 > r0) {
            for (var k0 = 0; k0 < pm.K; k0 = k0 + 16) {
                for (var q = 0; q < 4; q = q + 1) {
                    let idx = tid + q * 256;
                    let ar = idx / 16;
                    let ak = idx - ar * 16;
                    let arow = row0 + ar;
                    let acol = k0 + ak;
                    var av = 0.0;
                    if (arow >= r0 && arow < r1 && acol < pm.K) { av = xa[arow * pm.K + acol]; }
                    sm[idx] = av;
                    var bc = 0;
                    var bk = 0;
                    if (pm.transB != 0) { bc = idx / 16; bk = idx - bc * 16; } else { bk = idx / 64; bc = idx - bk * 64; }
                    let bcol = col0 + bc;
                    let brow = k0 + bk;
                    var bv = 0.0;
                    if (bcol < pm.N && brow < pm.K) {
                        if (pm.transB != 0) { bv = wb[(g * pm.N + bcol) * pm.K + brow]; }
                        else { bv = wb[(g * pm.K + brow) * pm.N + bcol]; }
                    }
                    sm[1024 + bk * 64 + bc] = bv;
                }
                workgroupBarrier();
                for (var kk = 0; kk < 16; kk = kk + 1) {
                    for (var i = 0; i < 4; i = i + 1) {
                        // Only this group's rows accumulate: another group's weights (a NaN, say) never reach them.
                        let orow = row0 + tr * 4 + i;
                        if (orow >= r0 && orow < r1) {
                            let a = sm[(tr * 4 + i) * 16 + kk];
                            for (var j = 0; j < 4; j = j + 1) {
                                acc[i * 4 + j] = fma(a, sm[1024 + kk * 64 + tc * 4 + j], acc[i * 4 + j]);
                            }
                        }
                    }
                }
                workgroupBarrier();
            }
        }
        g = g + 1;
        workgroupBarrier();
    }
    var last = 0;
    if (pm.G > 0) { last = i32(ends[pm.G - 1]); }
    for (var i = 0; i < 4; i = i + 1) {
        let row = row0 + tr * 4 + i;
        if (row < pm.M) {
            var rg = 0;
            if (pm.biasMode == 2 && row < last) { rg = first_group_past(row); }
            for (var j = 0; j < 4; j = j + 1) {
                let col = col0 + tc * 4 + j;
                if (col < pm.N) {
                    var v = acc[i * 4 + j];
                    if (pm.biasMode == 1 && row < last) { v = v + bias[col]; }
                    if (pm.biasMode == 2 && row < last) { v = v + bias[rg * pm.N + col]; }
                    yc[row * pm.N + col] = v;
                }
            }
        }
    }
}
";

    private const string WeightGradSource = @"
struct Params { M: i32, K: i32, N: i32, G: i32, tilesK: i32, tilesN: i32, tileBase: i32, pad0: i32, }
@group(0) @binding(0) var<storage, read> xa: array<f32>;
@group(0) @binding(1) var<storage, read_write> wout: array<f32>;
@group(0) @binding(2) var<storage, read> cg: array<f32>;
@group(0) @binding(3) var<storage, read> ends: array<f32>;
@group(0) @binding(4) var<uniform> pm: Params;
var<workgroup> sm: array<f32, 2048>;
var<workgroup> wgStart: i32;
var<workgroup> wgEnd: i32;

@compute @workgroup_size(256)
fn NAME(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = i32(lidx);
    let t = i32(wid.x) + pm.tileBase;
    let per = pm.tilesK * pm.tilesN;
    let g = t / per;
    let rem = t - g * per;
    let kt = rem / pm.tilesN;
    let nt = rem - kt * pm.tilesN;
    let k0 = kt * 64;
    let n0 = nt * 64;
    let tr = tid / 16;
    let tc = tid - tr * 16;
    var acc: array<f32, 16>;
    if (tid == 0) {
        var s = 0;
        if (g > 0) { s = i32(ends[g - 1]); }
        wgStart = s;
        wgEnd = i32(ends[g]);
    }
    let gs = workgroupUniformLoad(&wgStart);
    let ge = workgroupUniformLoad(&wgEnd);
    for (var r0 = gs; r0 < ge; r0 = r0 + 16) {
        for (var q = 0; q < 4; q = q + 1) {
            let idx = tid + q * 256;
            let rr = idx / 64;
            let cc = idx - rr * 64;
            let row = r0 + rr;
            var av = 0.0;
            if (row < ge && k0 + cc < pm.K) { av = xa[row * pm.K + k0 + cc]; }
            sm[idx] = av;
            var cv = 0.0;
            if (row < ge && n0 + cc < pm.N) { cv = cg[row * pm.N + n0 + cc]; }
            sm[1024 + idx] = cv;
        }
        workgroupBarrier();
        for (var rr = 0; rr < 16; rr = rr + 1) {
            for (var i = 0; i < 4; i = i + 1) {
                let a = sm[rr * 64 + tr * 4 + i];
                for (var j = 0; j < 4; j = j + 1) {
                    acc[i * 4 + j] = fma(a, sm[1024 + rr * 64 + tc * 4 + j], acc[i * 4 + j]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i = 0; i < 4; i = i + 1) {
        let k = k0 + tr * 4 + i;
        for (var j = 0; j < 4; j = j + 1) {
            let n = n0 + tc * 4 + j;
            if (k < pm.K && n < pm.N) { wout[(g * pm.K + k) * pm.N + n] = acc[i * 4 + j]; }
        }
    }
}
";

    private static readonly System.Collections.Generic.Dictionary<string, string> Sources = new()
    {
        [GroupedGemmKernelTemplate.Forward] = ForwardSource.Replace("NAME", GroupedGemmKernelTemplate.Forward),
        [GroupedGemmKernelTemplate.WeightGrad] = WeightGradSource.Replace("NAME", GroupedGemmKernelTemplate.WeightGrad),
    };

    /// <summary>The WGSL module for one kernel name, its entry point named after the kernel.</summary>
    internal static string SourceFor(string name) => Sources[name];
}
