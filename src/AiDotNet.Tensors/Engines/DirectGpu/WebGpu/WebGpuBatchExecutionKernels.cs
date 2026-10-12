#if NET7_0_OR_GREATER
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// WGSL ports of <see cref="BatchExecutionCudaKernels"/> for <see cref="IGpuBatchExecution"/> on WebGPU (#1142):
/// same kernels, same mode numbering, same counter hash. Storage buffers bind at 0..n-1 in declaration order and
/// the parameter struct at binding n, with every field 4 bytes in declaration order. WGSL has no 64-bit integers,
/// so the hash builds its 32x32-bit product from 16-bit halves and the seed arrives as two words.
/// </summary>
internal static class WebGpuBatchExecutionKernels
{
    private const string Common = @"
fn bx_mul_wide(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xFFFFu; let ah = a >> 16u; let bl = b & 0xFFFFu; let bh = b >> 16u;
    let p0 = al * bl; let p1 = al * bh; let p2 = ah * bl; let p3 = ah * bh;
    let mid = (p0 >> 16u) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
    return vec2<u32>((p0 & 0xFFFFu) | (mid << 16u), p3 + (p1 >> 16u) + (p2 >> 16u) + (mid >> 16u));
}
fn bx_hash(counter: u32, key: u32) -> u32 {
    var w = bx_mul_wide(counter, 0xCD9E8D57u);
    let c = w.y ^ key ^ w.x;
    w = bx_mul_wide(c, 0xCD9E8D57u);
    return w.y ^ (key + 1u) ^ w.x;
}
// Built at run time: an infinite constant expression is a shader-creation error.
fn bx_inf() -> f32 { let bits = 0x7f800000u; return bitcast<f32>(bits); }
fn bx_unit(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }
// (a * b) mod m in 32 bits; a, b < m <= 2^30.
fn bx_mulmod(a0: u32, b0: u32, m: u32) -> u32 {
    var a = a0; var b = b0; var r = 0u;
    loop {
        if (b == 0u) { break; }
        if ((b & 1u) != 0u) { r = r + a; if (r >= m) { r = r - m; } }
        a = a + a; if (a >= m) { a = a - m; }
        b = b >> 1u;
    }
    return r;
}
// mode 1: IEEE binary16, mode 2: bfloat16; both round to nearest even.
fn bx_to_16(x: f32, mode: i32) -> u32 {
    let f = bitcast<u32>(x);
    if (mode == 2) {
        if ((f & 0x7fffffffu) > 0x7f800000u) { return (f >> 16u) | 0x40u; }
        return (f + 0x7fffu + ((f >> 16u) & 1u)) >> 16u;
    }
    let sgn = (f >> 16u) & 0x8000u; let e = i32((f >> 23u) & 0xffu); var mant = f & 0x7fffffu;
    if (e == 255) { return sgn | 0x7c00u | select(0u, 0x200u, mant != 0u); }
    let ex = e - 112;
    if (ex >= 31) { return sgn | 0x7c00u; }
    if (ex <= 0) {
        if (ex < -10) { return sgn; }
        mant = mant | 0x800000u; let shift = u32(14 - ex);
        var h = mant >> shift; let rem = mant & ((1u << shift) - 1u); let hw = 1u << (shift - 1u);
        if (rem > hw || (rem == hw && (h & 1u) != 0u)) { h = h + 1u; }
        return sgn | h;
    }
    var h = sgn | (u32(ex) << 10u) | (mant >> 13u); let rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) { h = h + 1u; }
    return h;
}
fn bx_from_16(h: u32, mode: i32) -> f32 {
    if (mode == 2) { return bitcast<f32>(h << 16u); }
    let sgn = (h & 0x8000u) << 16u; let e = (h >> 10u) & 0x1fu; var mant = h & 0x3ffu;
    if (e == 0u) {
        if (mant == 0u) { return bitcast<f32>(sgn); }
        var ex = 1;
        loop { if ((mant & 0x400u) != 0u) { break; } mant = mant << 1u; ex = ex - 1; }
        return bitcast<f32>(sgn | (u32(ex + 112) << 23u) | ((mant & 0x3ffu) << 13u));
    }
    if (e == 31u) { return bitcast<f32>(sgn | 0x7f800000u | (mant << 13u)); }
    return bitcast<f32>(sgn | ((e + 112u) << 23u) | (mant << 13u));
}
";

    private const string Main = "@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n    let i = i32(id.x);\n";

    /// <summary>Buffer specs are "r:name" or "rw:name"; <paramref name="fields"/> is the parameter struct body.</summary>
    private static string Shader(string[] buffers, string fields, string body)
    {
        var sb = new StringBuilder();
        for (int b = 0; b < buffers.Length; b++)
        {
            string[] parts = buffers[b].Split(':');
            sb.Append("@group(0) @binding(").Append(b).Append(") var<storage, ")
              .Append(parts[0] == "r" ? "read" : "read_write").Append("> ").Append(parts[1])
              .Append(" : array<").Append(parts.Length > 2 ? parts[2] : "f32").Append(">;\n");
        }
        sb.Append("struct P { ").Append(fields).Append(" };\n");
        sb.Append("@group(0) @binding(").Append(buffers.Length).Append(") var<uniform> p : P;\n");
        return sb.Append(Common).Append(body).ToString();
    }

    public static readonly string FullReduce = Shader(new[] { "r:inp", "rw:outp" }, "size: i32, mode: i32, maxVal: f32", @"
var<workgroup> scratch: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = i32(lid3.x);
    var acc = 0.0;
    if (p.mode == 2) { acc = 1.0; } else if (p.mode == 4) { acc = bx_inf(); }
    for (var k = lid; k < p.size; k += 256) {
        let v = inp[k];
        if (p.mode <= 1) { acc += v * v; }
        else if (p.mode == 2) { acc *= v; }
        else if (p.mode == 3) { acc = max(acc, abs(v)); }
        else if (p.mode == 4) { acc = min(acc, abs(v)); }
        else { acc += exp(v - p.maxVal); }
    }
    scratch[lid] = acc;
    workgroupBarrier();
    for (var s = 128; s > 0; s = s / 2) {
        if (lid < s) {
            let x = scratch[lid]; let y = scratch[lid + s];
            var r = x + y;
            if (p.mode == 2) { r = x * y; } else if (p.mode == 3) { r = max(x, y); } else if (p.mode == 4) { r = min(x, y); }
            scratch[lid] = r;
        }
        workgroupBarrier();
    }
    if (lid == 0) {
        let r = scratch[0];
        var o = r;
        if (p.mode == 1) { o = sqrt(r); } else if (p.mode == 5) { o = p.maxVal + log(r); }
        outp[0] = o;
    }
}");

    public static readonly string RowReduce = Shader(new[] { "r:inp", "rw:outp" }, "outerSize: i32, innerSize: i32, mode: i32", Main + @"
    if (i >= p.outerSize) { return; }
    let rb = i * p.innerSize; let n = p.innerSize;
    var r = 0.0;
    if (p.mode == 2) { r = 1.0; for (var j = 0; j < n; j++) { r *= inp[rb + j]; } }
    else if (p.mode == 3) { for (var j = 0; j < n; j++) { r += inp[rb + j] * inp[rb + j]; } r = sqrt(r); }
    else if (p.mode == 4) {
        var m = -bx_inf(); for (var j = 0; j < n; j++) { m = max(m, inp[rb + j]); }
        var s = 0.0; for (var j = 0; j < n; j++) { s += exp(inp[rb + j] - m); }
        r = m + log(s);
    } else {
        var sum = 0.0; for (var j = 0; j < n; j++) { sum += inp[rb + j]; }
        let mu = sum / f32(n);
        var sq = 0.0; for (var j = 0; j < n; j++) { let d = inp[rb + j] - mu; sq += d * d; }
        let v = sq / f32(n);
        if (p.mode == 0) { r = v; } else if (p.mode == 1) { r = sqrt(v); } else { r = log(v + 1e-8); }
    }
    outp[i] = r;
}");

    public static readonly string RowMap = Shader(new[] { "r:inp", "rw:outp" }, "outerSize: i32, innerSize: i32, mode: i32", Main + @"
    if (i >= p.outerSize) { return; }
    let rb = i * p.innerSize; let n = p.innerSize;
    if (p.mode == 0) {
        var acc = 0.0; for (var j = 0; j < n; j++) { acc += inp[rb + j]; outp[rb + j] = acc; }
    } else if (p.mode == 1) {
        var sq = 0.0; for (var j = 0; j < n; j++) { sq += inp[rb + j] * inp[rb + j]; }
        let inv = 1.0 / (sqrt(sq) + 1e-12);
        for (var j = 0; j < n; j++) { outp[rb + j] = inp[rb + j] * inv; }
    } else if (p.mode == 2) {
        var m = -bx_inf(); for (var j = 0; j < n; j++) { m = max(m, inp[rb + j]); }
        var s = 0.0; for (var j = 0; j < n; j++) { s += exp(inp[rb + j] - m); }
        let l = log(s);
        for (var j = 0; j < n; j++) { outp[rb + j] = inp[rb + j] - m - l; }
    } else if (p.mode == 3) {
        var s = 0.0;
        for (var j = 0; j < n; j++) { let x = inp[rb + j]; let t = max(1.0 + x + 0.5 * x * x, 1e-10); outp[rb + j] = t; s += t; }
        let inv = 1.0 / s;
        for (var j = 0; j < n; j++) { outp[rb + j] = outp[rb + j] * inv; }
    } else if (p.mode == 4) {
        var sq = 0.0; for (var j = 0; j < n; j++) { sq += inp[rb + j] * inp[rb + j]; }
        let nrm = sqrt(sq + 1e-12);
        var m = -bx_inf(); for (var j = 0; j < n; j++) { m = max(m, inp[rb + j] / nrm); }
        var s = 0.0; for (var j = 0; j < n; j++) { let e = exp(inp[rb + j] / nrm - m); outp[rb + j] = e; s += e; }
        let inv = 1.0 / (s + 1e-10);
        for (var j = 0; j < n; j++) { outp[rb + j] = outp[rb + j] * inv; }
    } else {
        var supportSum = 0.0; var supportCount = 0;
        for (var j = 0; j < n; j++) {
            let z = inp[rb + j]; var k = 0; var s = 0.0;
            for (var q = 0; q < n; q++) { let v = inp[rb + q]; if (v >= z) { k++; s += v; } }
            if (1.0 + f32(k) * z > s) { supportSum += z; supportCount++; }
        }
        let tau = (supportSum - 1.0) / f32(supportCount);
        for (var j = 0; j < n; j++) { outp[rb + j] = max(inp[rb + j] - tau, 0.0); }
    }
}");

    public static readonly string Scalar = Shader(new[] { "r:inp", "rw:outp" }, "scalar: f32, size: i32, mode: i32", Main + @"
    if (i >= p.size) { return; }
    outp[i] = select(inp[i] / p.scalar, p.scalar - inp[i], p.mode == 0);
}");

    public static readonly string Unary = Shader(new[] { "r:inp", "rw:outp" }, "size: i32, mode: i32", Main + @"
    if (i >= p.size) { return; }
    let v = inp[i];
    var r = 1.0 - v * v;
    if (p.mode == 0) { r = inverseSqrt(v + 1e-12); } else if (p.mode == 1) { r = select(0.0, 1.0, v > 0.0); } else if (p.mode == 2) { r = v * (1.0 - v); }
    outp[i] = r;
}");

    public static readonly string SinCos = Shader(new[] { "r:inp", "rw:sinOut", "rw:cosOut" }, "size: i32", Main + @"
    if (i >= p.size) { return; }
    sinOut[i] = sin(inp[i]);
    cosOut[i] = cos(inp[i]);
}");

    public static readonly string ReduceBackward = Shader(new[] { "r:gradOut", "r:inp", "r:stat", "r:stat2", "rw:gradIn" },
        "outerSize: i32, reduceSize: i32, mode: i32", Main + @"
    if (i >= p.outerSize * p.reduceSize) { return; }
    let o = i / p.reduceSize;
    let g = gradOut[o];
    var r = g;
    if (p.mode == 1) { r = select(0.0, g, inp[i] == stat[o]); }
    else if (p.mode == 2) { r = g * 2.0 * (inp[i] - stat[o]) / f32(p.reduceSize); }
    else if (p.mode == 3) { r = g / (stat2[o] + 1e-8) * 2.0 * (inp[i] - stat[o]) / f32(p.reduceSize); }
    gradIn[i] = r;
}");

    public static readonly string BroadcastFirst = Shader(new[] { "r:a", "r:b", "rw:outp" }, "outerSize: i32, innerSize: i32, mode: i32", Main + @"
    if (i >= p.outerSize * p.innerSize) { return; }
    let x = a[i / p.innerSize];
    outp[i] = select(x * b[i], x + b[i], p.mode == 0);
}");

    public static readonly string GatedBackward = Shader(new[] { "r:gradOut", "r:inp", "rw:gradIn" }, "outerSize: i32, halfDim: i32, mode: i32", Main + @"
    if (i >= p.outerSize * p.halfDim) { return; }
    let o = i / p.halfDim; let d = i % p.halfDim; let full = p.halfDim * 2;
    let v = inp[o * full + d]; let g = inp[o * full + p.halfDim + d]; let go = gradOut[i];
    var act = 0.0; var dact = 0.0;
    if (p.mode == 0) { let s = 1.0 / (1.0 + exp(-g)); act = s; dact = s * (1.0 - s); }
    else if (p.mode == 1) {
        let t = tanh(0.7978845608 * (g + 0.044715 * g * g * g));
        act = 0.5 * g * (1.0 + t);
        dact = 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * 0.7978845608 * (1.0 + 0.134145 * g * g);
    }
    else if (p.mode == 2) { act = max(g, 0.0); dact = select(0.0, 1.0, g > 0.0); }
    else { let s = 1.0 / (1.0 + exp(-g)); act = g * s; dact = s + g * s * (1.0 - s); }
    gradIn[o * full + d] = go * act;
    gradIn[o * full + p.halfDim + d] = go * v * dact;
}");

    public static readonly string ConcatAxis = Shader(new[] { "r:a", "r:b", "rw:outp" }, "outerSize: i32, aInner: i32, bInner: i32", Main + @"
    let total = p.aInner + p.bInner;
    if (i >= p.outerSize * total) { return; }
    let o = i / total; let j = i % total;
    if (j < p.aInner) { outp[i] = a[o * p.aInner + j]; } else { outp[i] = b[o * p.bInner + j - p.aInner]; }
}");

    public static readonly string SliceLastAxis = Shader(new[] { "r:inp", "rw:outp" }, "outerSize: i32, inputInner: i32, start: i32, sliceSize: i32", Main + @"
    if (i >= p.outerSize * p.sliceSize) { return; }
    outp[i] = inp[(i / p.sliceSize) * p.inputInner + p.start + i % p.sliceSize];
}");

    public static readonly string SetSliceLastAxis = Shader(new[] { "rw:outp", "r:vals" }, "outerSize: i32, outputInner: i32, start: i32, sliceSize: i32", Main + @"
    if (i >= p.outerSize * p.sliceSize) { return; }
    outp[(i / p.sliceSize) * p.outputInner + p.start + i % p.sliceSize] = vals[i];
}");

    public static readonly string SliceAxis = Shader(new[] { "r:inp", "rw:outp" }, "outerSize: i32, axisSize: i32, stride: i32, index: i32", Main + @"
    if (i >= p.outerSize * p.stride) { return; }
    outp[i] = inp[((i / p.stride) * p.axisSize + p.index) * p.stride + i % p.stride];
}");

    public static readonly string SetSliceAxis = Shader(new[] { "rw:outp", "r:vals" }, "outerSize: i32, axisSize: i32, stride: i32, index: i32", Main + @"
    if (i >= p.outerSize * p.stride) { return; }
    outp[((i / p.stride) * p.axisSize + p.index) * p.stride + i % p.stride] = vals[i];
}");

    public static readonly string Stack2 = Shader(new[] { "r:a", "r:b", "rw:outp" }, "size: i32", Main + @"
    if (i >= p.size) { return; }
    outp[2 * i] = a[i];
    outp[2 * i + 1] = b[i];
}");

    public static readonly string Window2D = Shader(new[] { "r:inp", "rw:outp" },
        "batch: i32, channels: i32, inH: i32, inW: i32, outH: i32, outW: i32, h0: i32, w0: i32, padValue: f32, mode: i32", Main + @"
    if (p.mode == 0) {
        if (i >= p.batch * p.channels * p.outH * p.outW) { return; }
        let w = i % p.outW - p.w0; let h = i / p.outW % p.outH - p.h0; let bc = i / (p.outW * p.outH);
        if (h >= 0 && h < p.inH && w >= 0 && w < p.inW) { outp[i] = inp[(bc * p.inH + h) * p.inW + w]; } else { outp[i] = p.padValue; }
    } else if (p.mode == 1) {
        if (i >= p.batch * p.channels * p.inH * p.inW) { return; }
        let w = i % p.inW; let h = i / p.inW % p.inH; let bc = i / (p.inW * p.inH);
        outp[i] = inp[(bc * p.outH + h + p.h0) * p.outW + w + p.w0];
    } else if (p.mode == 2) {
        if (i >= p.batch * p.channels * p.outH * p.outW) { return; }
        let w = i % p.outW; let h = i / p.outW % p.outH; let bc = i / (p.outW * p.outH);
        outp[i] = inp[(bc * p.inH + h + p.h0) * p.inW + w + p.w0];
    } else {
        if (i >= p.batch * p.channels * p.inH * p.inW) { return; }
        let w = i % p.inW - p.w0; let h = i / p.inW % p.inH - p.h0; let bc = i / (p.inW * p.inH);
        if (h >= 0 && h < p.outH && w >= 0 && w < p.outW) { outp[i] = inp[(bc * p.outH + h) * p.outW + w]; } else { outp[i] = 0.0; }
    }
}");

    public static readonly string TileLastAxis = Shader(new[] { "r:inp", "rw:outp" }, "outerSize: i32, innerSize: i32, repeats: i32", Main + @"
    let tiled = p.innerSize * p.repeats;
    if (i >= p.outerSize * tiled) { return; }
    outp[i] = inp[(i / tiled) * p.innerSize + (i % tiled) % p.innerSize];
}");

    public static readonly string RepeatElements = Shader(new[] { "r:inp", "rw:outp" }, "outerSize: i32, innerSize: i32, repeats: i32", Main + @"
    let tiled = p.innerSize * p.repeats;
    if (i >= p.outerSize * tiled) { return; }
    outp[i] = inp[(i / tiled) * p.innerSize + (i % tiled) / p.repeats];
}");

    public static readonly string Linspace = Shader(new[] { "rw:outp" }, "start: f32, stepSize: f32, size: i32", Main + @"
    if (i >= p.size) { return; }
    outp[i] = p.start + p.stepSize * f32(i);
}");

    public static readonly string Diag = Shader(new[] { "r:inp", "rw:outp" }, "n: i32", Main + @"
    if (i >= p.n * p.n) { return; }
    outp[i] = select(0.0, inp[i / p.n], i / p.n == i % p.n);
}");

    public static readonly string ExtractDiag = Shader(new[] { "r:inp", "rw:outp" }, "n: i32, cols: i32", Main + @"
    if (i >= p.n || i >= p.cols) { return; }
    outp[i] = inp[i * p.cols + i];
}");

    public static readonly string TriangularMask = Shader(new[] { "rw:outp" }, "rows: i32, cols: i32, diagonal: i32, maskValue: f32", Main + @"
    if (i >= p.rows * p.cols) { return; }
    outp[i] = select(0.0, p.maskValue, i % p.cols > i / p.cols + p.diagonal);
}");

    public static readonly string IndexSelect = Shader(new[] { "r:inp", "r:indices", "rw:outp" }, "numIndices: i32, innerSize: i32", Main + @"
    if (i >= p.numIndices * p.innerSize) { return; }
    outp[i] = inp[i32(indices[i / p.innerSize]) * p.innerSize + i % p.innerSize];
}");

    public static readonly string RowLoss = Shader(new[] { "r:pred", "r:targ", "rw:lossOut" }, "batchSize: i32, numClasses: i32, mode: i32", Main + @"
    if (i >= p.batchSize) { return; }
    let rb = i * p.numClasses;
    var r = 0.0;
    if (p.mode == 0) {
        var m = -bx_inf(); for (var c = 0; c < p.numClasses; c++) { m = max(m, pred[rb + c]); }
        var s = 0.0; for (var c = 0; c < p.numClasses; c++) { s += exp(pred[rb + c] - m); }
        let lse = m + log(s);
        for (var c = 0; c < p.numClasses; c++) { if (targ[rb + c] > 0.0) { r -= targ[rb + c] * (pred[rb + c] - lse); } }
    } else {
        for (var c = 0; c < p.numClasses; c++) { let d = pred[rb + c] - targ[rb + c]; r += d * d; }
        r = r / f32(p.numClasses);
    }
    lossOut[i] = r;
}");

    public static readonly string DropoutMask = Shader(new[] { "rw:maskOut" }, "size: i32, keepProb: f32, seedLo: u32, seedHi: u32", Main + @"
    if (i >= p.size) { return; }
    let u = bx_unit(bx_hash(u32(i), p.seedLo ^ p.seedHi));
    let invKeep = select(0.0, 1.0 / p.keepProb, p.keepProb > 1e-7);
    maskOut[i] = select(0.0, invKeep, u < p.keepProb);
}");

    public static readonly string GaussianNoise = Shader(new[] { "rw:outp" }, "size: i32, mean: f32, stdDev: f32, seedLo: u32, seedHi: u32", Main + @"
    if (i >= p.size) { return; }
    let key = p.seedLo ^ p.seedHi;
    let u1 = bx_unit(bx_hash(u32(2 * i), key)) + (0.5 / 16777216.0);
    let u2 = bx_unit(bx_hash(u32(2 * i + 1), key + 2u));
    outp[i] = p.mean + p.stdDev * sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}");

    public static readonly string GumbelSoftmax = Shader(new[] { "r:logits", "rw:outp" },
        "outerSize: i32, innerSize: i32, temperature: f32, seedLo: u32, seedHi: u32", Main + @"
    if (i >= p.outerSize) { return; }
    let key = p.seedLo ^ p.seedHi;
    let t = max(p.temperature, 1e-7);
    let rb = i * p.innerSize;
    var m = -bx_inf();
    for (var j = 0; j < p.innerSize; j++) {
        let u = bx_unit(bx_hash(u32(rb + j), key)) + (0.5 / 16777216.0);
        let v = (logits[rb + j] - log(-log(u))) / t;
        outp[rb + j] = v; m = max(m, v);
    }
    var s = 0.0;
    for (var j = 0; j < p.innerSize; j++) { let e = exp(outp[rb + j] - m); outp[rb + j] = e; s += e; }
    let inv = 1.0 / (s + 1e-10);
    for (var j = 0; j < p.innerSize; j++) { outp[rb + j] = outp[rb + j] * inv; }
}");

    public static readonly string BatchOuterProduct = Shader(new[] { "r:a", "r:b", "rw:outp" }, "batchSize: i32, M: i32, N: i32", Main + @"
    if (i >= p.batchSize * p.M * p.N) { return; }
    let mn = p.M * p.N; let bt = i / mn; let r = i % mn;
    outp[i] = a[bt * p.M + r / p.N] * b[bt * p.N + r % p.N];
}");

    public static readonly string Meshgrid2D = Shader(new[] { "r:source0", "r:source1", "rw:output0", "rw:output1" }, "n0: i32, n1: i32, xy: i32", Main + @"
    if (i >= p.n0 * p.n1) { return; }
    let cols = select(p.n1, p.n0, p.xy != 0); let r = i / cols; let c = i % cols;
    output0[i] = source0[select(r, c, p.xy != 0)];
    output1[i] = source1[select(c, r, p.xy != 0)];
}");

    public static readonly string FftChirp = Shader(new[] { "rw:chirpRe", "rw:chirpIm", "rw:filterRe", "rw:filterIm" }, "n: i32, m: i32, inverse: i32", Main + @"
    if (i >= p.m) { return; }
    let twoN = 2u * u32(p.n);
    let sgn = select(-1.0, 1.0, p.inverse != 0);
    if (i < p.n) {
        let a = 3.14159265358979323846 * f32(bx_mulmod(u32(i), u32(i), twoN)) / f32(p.n);
        chirpRe[i] = cos(a); chirpIm[i] = sgn * sin(a);
    }
    var t = -1;
    if (i < p.n) { t = i; } else if (i > p.m - p.n) { t = p.m - i; }
    if (t < 0) { filterRe[i] = 0.0; filterIm[i] = 0.0; return; }
    let b = 3.14159265358979323846 * f32(bx_mulmod(u32(t), u32(t), twoN)) / f32(p.n);
    filterRe[i] = cos(b); filterIm[i] = -sgn * sin(b);
}");

    public static readonly string FftBluesteinPre = Shader(new[] { "r:re", "r:im", "r:chirpRe", "r:chirpIm", "rw:padRe", "rw:padIm" }, "batch: i32, n: i32, m: i32", Main + @"
    if (i >= p.batch * p.m) { return; }
    let b = i / p.m; let k = i % p.m;
    if (k >= p.n) { padRe[i] = 0.0; padIm[i] = 0.0; return; }
    let xr = re[b * p.n + k]; let xi = im[b * p.n + k]; let cr = chirpRe[k]; let ci = chirpIm[k];
    padRe[i] = xr * cr - xi * ci;
    padIm[i] = xr * ci + xi * cr;
}");

    public static readonly string FftCmul = Shader(new[] { "rw:aRe", "rw:aIm", "r:bRe", "r:bIm" }, "batch: i32, m: i32", Main + @"
    if (i >= p.batch * p.m) { return; }
    let k = i % p.m;
    let xr = aRe[i]; let xi = aIm[i]; let yr = bRe[k]; let yi = bIm[k];
    aRe[i] = xr * yr - xi * yi;
    aIm[i] = xr * yi + xi * yr;
}");

    public static readonly string FftBluesteinPost = Shader(new[] { "r:padRe", "r:padIm", "r:chirpRe", "r:chirpIm", "rw:re", "rw:im" },
        "batch: i32, n: i32, m: i32, scale: f32", Main + @"
    if (i >= p.batch * p.n) { return; }
    let b = i / p.n; let k = i % p.n;
    let xr = padRe[b * p.m + k]; let xi = padIm[b * p.m + k]; let cr = chirpRe[k]; let ci = chirpIm[k];
    re[i] = (xr * cr - xi * ci) * p.scale;
    im[i] = (xr * ci + xi * cr) * p.scale;
}");

    public static readonly string FftToStorage = Shader(new[] { "r:source", "rw:words:u32" }, "count: i32, mode: i32", Main + @"
    if (i >= (p.count + 1) / 2) { return; }
    let lo = bx_to_16(source[2 * i], p.mode);
    var hi = 0u;
    if (2 * i + 1 < p.count) { hi = bx_to_16(source[2 * i + 1], p.mode); }
    words[i] = lo | (hi << 16u);
}");

    public static readonly string FftFromStorage = Shader(new[] { "r:words:u32", "rw:destination" }, "count: i32, mode: i32", Main + @"
    if (i >= p.count) { return; }
    let word = words[i / 2];
    destination[i] = bx_from_16(select(word & 0xffffu, word >> 16u, (i & 1) != 0), p.mode);
}");

    // IExtendedLinalgBackend solvers (#1142): line-for-line ports of BatchExecutionCudaKernels.SolverSource. Scalars
    // are read into locals first because a local pivot would otherwise shadow the parameter struct p.
    private const string LaCommon = @"
fn la_nan() -> f32 { let bits = 0x7fc00000u; return bitcast<f32>(bits); }
";

    public static readonly string LaSymmetrize = Shader(new[] { "r:a", "rw:outp" }, "batch: i32, n: i32, upper: i32", LaCommon + Main + @"
    let n = p.n;
    if (i >= p.batch * n * n) { return; }
    let o = (i / (n * n)) * n * n; let r = (i / n) % n; let c = i % n;
    let keep = select(r >= c, r <= c, p.upper != 0);
    outp[i] = select(a[o + c * n + r], a[i], keep);
}");

    public static readonly string LaLuSolve = Shader(new[] { "r:lu", "r:piv:i32", "r:rhs", "rw:x" }, "batch: i32, n: i32, nrhs: i32", LaCommon + Main + @"
    let b = i; let n = p.n; let nrhs = p.nrhs;
    if (b >= p.batch) { return; }
    let ao = b * n * n; let po = b * n; let xo = b * n * nrhs;
    for (var k = 0; k < n * nrhs; k++) { x[xo + k] = rhs[xo + k]; }
    for (var r = 0; r < n; r++) {
        let pr = piv[po + r];
        if (pr != r && pr >= 0 && pr < n) {
            for (var c = 0; c < nrhs; c++) {
                let t = x[xo + r * nrhs + c]; x[xo + r * nrhs + c] = x[xo + pr * nrhs + c]; x[xo + pr * nrhs + c] = t;
            }
        }
    }
    for (var c = 0; c < nrhs; c++) {
        for (var r = 1; r < n; r++) {
            var s = x[xo + r * nrhs + c];
            for (var j = 0; j < r; j++) { s -= lu[ao + r * n + j] * x[xo + j * nrhs + c]; }
            x[xo + r * nrhs + c] = s;
        }
        for (var r = n - 1; r >= 0; r--) {
            var s = x[xo + r * nrhs + c];
            for (var j = r + 1; j < n; j++) { s -= lu[ao + r * n + j] * x[xo + j * nrhs + c]; }
            x[xo + r * nrhs + c] = s / lu[ao + r * n + r];
        }
    }
}");

    public static readonly string LaLdlFactor = Shader(new[] { "r:a", "rw:ld", "rw:piv:i32" }, "batch: i32, n: i32, upper: i32", LaCommon + Main + @"
    let b = i; let n = p.n; let upper = p.upper;
    if (b >= p.batch) { return; }
    let o = b * n * n; let po = b * n;
    for (var k = 0; k < n * n; k++) { ld[o + k] = a[o + k]; }
    for (var j = 0; j < n; j++) {
        var pr = j; var best = abs(ld[o + j * n + j]);
        for (var r = j + 1; r < n; r++) {
            let v = abs(ld[o + r * n + r]);
            if (v > best) { best = v; pr = r; }
        }
        piv[po + j] = pr;
        if (pr != j) {
            for (var k = 0; k < n; k++) { let t = ld[o + j * n + k]; ld[o + j * n + k] = ld[o + pr * n + k]; ld[o + pr * n + k] = t; }
            for (var k = 0; k < n; k++) { let t = ld[o + k * n + j]; ld[o + k * n + j] = ld[o + k * n + pr]; ld[o + k * n + pr] = t; }
        }
        let d = ld[o + j * n + j];
        if (d == 0.0) { continue; }
        for (var r = j + 1; r < n; r++) {
            let lij = ld[o + r * n + j] / d;
            ld[o + r * n + j] = lij;
            for (var k = j + 1; k < n; k++) { ld[o + r * n + k] -= lij * ld[o + j * n + k]; }
        }
    }
    for (var r = 0; r < n; r++) { for (var j = r + 1; j < n; j++) { ld[o + r * n + j] = 0.0; } }
    if (upper != 0) {
        for (var r = 0; r < n; r++) {
            for (var j = 0; j < r; j++) { ld[o + j * n + r] = ld[o + r * n + j]; ld[o + r * n + j] = 0.0; }
        }
    }
}");

    public static readonly string LaLdlSolve = Shader(new[] { "r:ld", "r:piv:i32", "r:rhs", "rw:x" }, "batch: i32, n: i32, nrhs: i32, upper: i32", LaCommon + Main + @"
    let b = i; let n = p.n; let nrhs = p.nrhs; let upper = p.upper;
    if (b >= p.batch) { return; }
    let o = b * n * n; let po = b * n; let xo = b * n * nrhs;
    for (var k = 0; k < n * nrhs; k++) { x[xo + k] = rhs[xo + k]; }
    for (var c = 0; c < nrhs; c++) {
        for (var j = 0; j < n; j++) {
            let pr = piv[po + j];
            if (pr != j && pr >= 0 && pr < n) { let t = x[xo + j * nrhs + c]; x[xo + j * nrhs + c] = x[xo + pr * nrhs + c]; x[xo + pr * nrhs + c] = t; }
        }
        for (var r = 0; r < n; r++) {
            var s = x[xo + r * nrhs + c];
            for (var k = 0; k < r; k++) { s -= select(ld[o + r * n + k], ld[o + k * n + r], upper != 0) * x[xo + k * nrhs + c]; }
            x[xo + r * nrhs + c] = s;
        }
        for (var r = 0; r < n; r++) {
            let d = ld[o + r * n + r];
            x[xo + r * nrhs + c] = select(x[xo + r * nrhs + c] / d, la_nan(), d == 0.0);
        }
        for (var r = n - 1; r >= 0; r--) {
            var s = x[xo + r * nrhs + c];
            for (var k = r + 1; k < n; k++) { s -= select(ld[o + k * n + r], ld[o + r * n + k], upper != 0) * x[xo + k * nrhs + c]; }
            x[xo + r * nrhs + c] = s;
        }
        for (var j = n - 1; j >= 0; j--) {
            let pr = piv[po + j];
            if (pr != j && pr >= 0 && pr < n) { let t = x[xo + j * nrhs + c]; x[xo + j * nrhs + c] = x[xo + pr * nrhs + c]; x[xo + pr * nrhs + c] = t; }
        }
    }
}");

    // Partial-pivot LU of the n x n matrix in w (row-major), pivots into pv; shared by solve and solve-backward.
    private const string LaLuInRegisters = @"
fn la_lu(w: ptr<function, array<f32, 64>>, pv: ptr<function, array<i32, 8>>, n: i32) {
    for (var j = 0; j < n; j++) {
        var pr = j; var best = abs((*w)[j * n + j]);
        for (var r = j + 1; r < n; r++) { let v = abs((*w)[r * n + j]); if (v > best) { best = v; pr = r; } }
        (*pv)[j] = pr;
        if (pr != j) { for (var c = 0; c < n; c++) { let t = (*w)[pr * n + c]; (*w)[pr * n + c] = (*w)[j * n + c]; (*w)[j * n + c] = t; } }
        let d = (*w)[j * n + j];
        if (d == 0.0) { continue; }
        for (var r = j + 1; r < n; r++) {
            let f = (*w)[r * n + j] / d;
            (*w)[r * n + j] = f;
            for (var c = j + 1; c < n; c++) { (*w)[r * n + c] -= f * (*w)[j * n + c]; }
        }
    }
}
";

    public static readonly string LaSolveVector = Shader(new[] { "r:a", "r:rhs", "rw:x", "rw:info:i32" }, "batch: i32, n: i32", LaCommon + LaLuInRegisters + Main + @"
    let b = i; let n = p.n;
    if (b >= p.batch) { return; }
    var w: array<f32, 64>; var pv: array<i32, 8>;
    let o = b * n * n; let xo = b * n;
    for (var k = 0; k < n * n; k++) { w[k] = a[o + k]; }
    la_lu(&w, &pv, n);
    var status = 0;
    for (var k = 0; k < n; k++) { if (w[k * n + k] == 0.0) { status = k + 1; break; } }
    info[b] = status;
    for (var r = 0; r < n; r++) { x[xo + r] = rhs[xo + r]; }
    for (var r = 0; r < n; r++) { let pr = pv[r]; if (pr != r) { let t = x[xo + r]; x[xo + r] = x[xo + pr]; x[xo + pr] = t; } }
    for (var r = 1; r < n; r++) { var s = x[xo + r]; for (var j = 0; j < r; j++) { s -= w[r * n + j] * x[xo + j]; } x[xo + r] = s; }
    for (var r = n - 1; r >= 0; r--) {
        var s = x[xo + r];
        for (var j = r + 1; j < n; j++) { s -= w[r * n + j] * x[xo + j]; }
        x[xo + r] = s / w[r * n + r];
    }
}");

    public static readonly string LaTriangularSolveVector = Shader(new[] { "r:a", "r:rhs", "rw:x" }, "batch: i32, n: i32, upper: i32, unitDiagonal: i32", LaCommon + Main + @"
    let b = i; let n = p.n; let upper = p.upper; let unitDiagonal = p.unitDiagonal;
    if (b >= p.batch) { return; }
    let o = b * n * n; let xo = b * n;
    for (var k = 0; k < n; k++) { x[xo + k] = rhs[xo + k]; }
    for (var st = 0; st < n; st++) {
        let r = select(st, n - 1 - st, upper != 0);
        var s = x[xo + r];
        if (upper != 0) { for (var j = r + 1; j < n; j++) { s -= a[o + r * n + j] * x[xo + j]; } }
        else { for (var j = 0; j < r; j++) { s -= a[o + r * n + j] * x[xo + j]; } }
        x[xo + r] = select(s / a[o + r * n + r], s, unitDiagonal != 0);
    }
}");

    public static readonly string LaCholeskyBackwardLower = Shader(new[] { "r:l", "r:g", "rw:ga" }, "batch: i32, n: i32", LaCommon + Main + @"
    let b = i; let n = p.n;
    if (b >= p.batch) { return; }
    var t: array<f32, 64>; var inv: array<f32, 64>; var row: array<f32, 8>;
    let o = b * n * n;
    for (var r = 0; r < n; r++) {
        for (var j = 0; j < n; j++) {
            var s = 0.0;
            for (var k = 0; k < n; k++) { s += l[o + k * n + r] * g[o + k * n + j]; }
            t[r * n + j] = s;
        }
    }
    for (var r = 0; r < n; r++) { for (var j = r + 1; j < n; j++) { t[r * n + j] = t[j * n + r]; } }
    for (var c = 0; c < n; c++) {
        for (var r = 0; r < n; r++) {
            var s = select(0.0, 1.0, r == c);
            for (var k = 0; k < r; k++) { s -= l[o + r * n + k] * inv[k * n + c]; }
            inv[r * n + c] = s / l[o + r * n + r];
        }
    }
    for (var r = 0; r < n; r++) {
        for (var k = 0; k < n; k++) {
            var s = 0.0;
            for (var q = 0; q < n; q++) { s += inv[q * n + r] * t[q * n + k]; }
            row[k] = s;
        }
        for (var j = 0; j < n; j++) {
            var s = 0.0;
            for (var k = 0; k < n; k++) { s += row[k] * inv[k * n + j]; }
            ga[o + r * n + j] = 0.5 * s;
        }
    }
}");

    public static readonly string LaSolveBackwardVector = Shader(new[] { "r:a", "r:xs", "r:g", "rw:ga", "rw:gb" }, "batch: i32, n: i32", LaCommon + LaLuInRegisters + Main + @"
    let b = i; let n = p.n;
    if (b >= p.batch) { return; }
    var w: array<f32, 64>; var pv: array<i32, 8>; var y: array<f32, 8>;
    let o = b * n * n; let vo = b * n;
    for (var r = 0; r < n; r++) { for (var j = 0; j < n; j++) { w[r * n + j] = a[o + j * n + r]; } }
    la_lu(&w, &pv, n);
    for (var r = 0; r < n; r++) { y[r] = g[vo + r]; }
    for (var r = 0; r < n; r++) { let pr = pv[r]; if (pr != r) { let t = y[r]; y[r] = y[pr]; y[pr] = t; } }
    for (var r = 1; r < n; r++) { var s = y[r]; for (var j = 0; j < r; j++) { s -= w[r * n + j] * y[j]; } y[r] = s; }
    for (var r = n - 1; r >= 0; r--) { var s = y[r]; for (var j = r + 1; j < n; j++) { s -= w[r * n + j] * y[j]; } y[r] = s / w[r * n + r]; }
    for (var r = 0; r < n; r++) {
        gb[vo + r] = y[r];
        for (var j = 0; j < n; j++) { ga[o + r * n + j] = -y[r] * xs[vo + j]; }
    }
}");

    public static readonly string LaSvdReduced = Shader(new[] { "r:a", "rw:u", "rw:s", "rw:vh" }, "batch: i32, m: i32, n: i32", LaCommon + Main + @"
    let b = i; let m = p.m; let n = p.n;
    if (b >= p.batch) { return; }
    var gm: array<f32, 64>; var v: array<f32, 64>; var w: array<f32, 8>; var ord: array<i32, 8>;
    let tall = m >= n; let k = min(m, n); let d = select(m, n, tall);
    let ao = b * m * n; let uo = b * m * k; let so = b * k; let vo = b * k * n;
    for (var r = 0; r < d; r++) {
        for (var j = 0; j < d; j++) {
            var acc = 0.0;
            if (tall) { for (var q = 0; q < m; q++) { acc += a[ao + q * n + r] * a[ao + q * n + j]; } }
            else { for (var q = 0; q < n; q++) { acc += a[ao + r * n + q] * a[ao + j * n + q]; } }
            gm[r * d + j] = acc;
            v[r * d + j] = select(0.0, 1.0, r == j);
        }
    }
    var scale = 0.0;
    for (var q = 0; q < d * d; q++) { scale += gm[q] * gm[q]; }
    scale = sqrt(scale);
    for (var sweep = 0; sweep < 100; sweep++) {
        var off = 0.0;
        for (var pp = 0; pp < d - 1; pp++) { for (var qq = pp + 1; qq < d; qq++) { off += gm[pp * d + qq] * gm[pp * d + qq]; } }
        off = sqrt(off);
        if (off < 1e-12 || off <= 1e-7 * scale) { break; }
        for (var pp = 0; pp < d - 1; pp++) {
            for (var qq = pp + 1; qq < d; qq++) {
                let apq = gm[pp * d + qq];
                if (abs(apq) < 1e-12) { continue; }
                let theta = (gm[qq * d + qq] - gm[pp * d + pp]) / (2.0 * apq);
                let sg = select(select(0.0, -1.0, theta < 0.0), 1.0, theta > 0.0);
                let t = select(sg / (abs(theta) + sqrt(1.0 + theta * theta)), 1.0, theta == 0.0);
                let c = 1.0 / sqrt(1.0 + t * t);
                let sn = t * c;
                for (var q = 0; q < d; q++) {
                    let aip = gm[q * d + pp]; let aiq = gm[q * d + qq];
                    gm[q * d + pp] = c * aip - sn * aiq; gm[q * d + qq] = sn * aip + c * aiq;
                }
                for (var q = 0; q < d; q++) {
                    let api = gm[pp * d + q]; let aqi = gm[qq * d + q];
                    gm[pp * d + q] = c * api - sn * aqi; gm[qq * d + q] = sn * api + c * aqi;
                }
                for (var q = 0; q < d; q++) {
                    let vip = v[q * d + pp]; let viq = v[q * d + qq];
                    v[q * d + pp] = c * vip - sn * viq; v[q * d + qq] = sn * vip + c * viq;
                }
            }
        }
    }
    for (var q = 0; q < d; q++) { w[q] = gm[q * d + q]; ord[q] = q; }
    for (var q = 1; q < d; q++) {
        let cur = ord[q]; var j = q - 1;
        while (j >= 0 && w[ord[j]] < w[cur]) { ord[j + 1] = ord[j]; j--; }
        ord[j + 1] = cur;
    }
    for (var j = 0; j < k; j++) {
        let sigma = sqrt(max(0.0, w[ord[j]]));
        s[so + j] = sigma;
        let e = ord[j];
        if (tall) {
            for (var c = 0; c < n; c++) { vh[vo + j * n + c] = v[c * d + e]; }
            for (var r = 0; r < m; r++) {
                var acc = 0.0;
                for (var c = 0; c < n; c++) { acc += a[ao + r * n + c] * v[c * d + e]; }
                u[uo + r * k + j] = select(0.0, acc / sigma, sigma > 1e-14);
            }
        } else {
            for (var r = 0; r < m; r++) { u[uo + r * k + j] = v[r * d + e]; }
            for (var c = 0; c < n; c++) {
                var acc = 0.0;
                for (var r = 0; r < m; r++) { acc += v[r * d + e] * a[ao + r * n + c]; }
                vh[vo + j * n + c] = select(0.0, acc / sigma, sigma > 1e-14);
            }
        }
    }
}");
    public static readonly string LaQrReduced = Shader(new[] { "r:a", "rw:q", "rw:r", "rw:work" }, "batch: i32, m: i32, n: i32", LaCommon + Main + @"
    let b = i; let m = p.m; let n = p.n;
    if (b >= p.batch) { return; }
    let k = min(m, n);
    let ao = b * m * n; let wo = b * (m * n + 2 * k); let bo = wo + m * n; let dg = bo + k;
    let qo = b * m * k; let ro = b * k * n;
    for (var t = 0; t < m * n; t++) { work[wo + t] = a[ao + t]; }
    for (var j = 0; j < k; j++) {
        var norm2 = 0.0;
        for (var q2 = j; q2 < m; q2++) { norm2 += work[wo + q2 * n + j] * work[wo + q2 * n + j]; }
        let norm = sqrt(norm2);
        let x0 = work[wo + j * n + j];
        var alpha = select(select(0.0, norm, x0 < 0.0), -norm, x0 > 0.0);
        if (x0 == 0.0 && alpha == 0.0) { alpha = norm; }
        work[wo + j * n + j] = x0 - alpha;
        var vnorm2 = 0.0;
        for (var q2 = j; q2 < m; q2++) { vnorm2 += work[wo + q2 * n + j] * work[wo + q2 * n + j]; }
        let beta = select(2.0 / vnorm2, 0.0, vnorm2 == 0.0);
        work[bo + j] = beta;
        work[dg + j] = alpha;
        for (var c = j + 1; c < n; c++) {
            var dot = 0.0;
            for (var q2 = j; q2 < m; q2++) { dot += work[wo + q2 * n + j] * work[wo + q2 * n + c]; }
            let sc = beta * dot;
            for (var q2 = j; q2 < m; q2++) { work[wo + q2 * n + c] -= sc * work[wo + q2 * n + j]; }
        }
    }
    for (var t = 0; t < k; t++) {
        for (var c = 0; c < n; c++) {
            r[ro + t * n + c] = select(select(work[wo + t * n + c], work[dg + t], c == t), 0.0, c < t);
        }
    }
    for (var t = 0; t < m; t++) { for (var c = 0; c < k; c++) { q[qo + t * k + c] = select(0.0, 1.0, t == c); } }
    for (var j = k - 1; j >= 0; j--) {
        let beta = work[bo + j];
        if (beta == 0.0) { continue; }
        for (var c = 0; c < k; c++) {
            var dot = 0.0;
            for (var q2 = j; q2 < m; q2++) { dot += work[wo + q2 * n + j] * q[qo + q2 * k + c]; }
            let sc = beta * dot;
            for (var q2 = j; q2 < m; q2++) { q[qo + q2 * k + c] -= sc * work[wo + q2 * n + j]; }
        }
    }
}");
    // #1179 fused attention: line-for-line ports of BatchExecutionCudaKernels.AttentionSource (WGSL reserves "meta",
    // hence "fam" for the metadata buffer on every port). -inf is tested on its bits, since WGSL lets an implementation
    // assume floats are finite. No kernel binds more than 8 storage buffers, WebGPU's guaranteed per-stage limit.
    private const string FaLoad = @"
struct Fa { B: i32, H: i32, Sq: i32, Sk: i32, D: i32, qb: i32, qh: i32, qr: i32, kb: i32, kh: i32, kr: i32, vb: i32,
    vh: i32, vr: i32, ob: i32, oh: i32, orw: i32, mb: i32, mh: i32, mi: i32, mj: i32, causal: i32, coff: i32,
    hasMask: i32, scale: f32, cap: f32 }
fn fa_load() -> Fa {
    return Fa(fam[0], fam[1], fam[2], fam[3], fam[4], fam[5], fam[6], fam[7], fam[8], fam[9], fam[10], fam[11], fam[12],
        fam[13], fam[14], fam[15], fam[16], fam[17], fam[18], fam[19], fam[20], fam[21], fam[22], fam[23],
        bitcast<f32>(fam[24]), bitcast<f32>(fam[25]));
}
fn fa_neg_inf() -> f32 { let bits = 0xff800000u; return bitcast<f32>(bits); }
fn fa_has_keys(lr: f32) -> bool { return bitcast<u32>(lr) != 0xff800000u; }
";

    private const string FaScore = @"
fn fa_allowed(f: Fa, b: i32, h: i32, i: i32, j: i32) -> bool {
    if (f.causal != 0 && j > i + f.coff) { return false; }
    if (f.hasMask != 0 && mask[b * f.mb + h * f.mh + i * f.mi + j * f.mj] == 0) { return false; }
    return true;
}
// The capped score and d(capped) / d(raw).
fn fa_score(f: Fa, b: i32, h: i32, i: i32, j: i32) -> vec2<f32> {
    var s = 0.0;
    for (var d = 0; d < f.D; d++) { s += (q[b * f.qb + h * f.qh + i * f.qr + d] * f.scale) * k[b * f.kb + h * f.kh + j * f.kr + d]; }
    if (f.cap > 0.0) { let t = tanh(s / f.cap); return vec2<f32>(f.cap * t, 1.0 - t * t); }
    return vec2<f32>(s, 1.0);
}
";

    private const string FaDp = @"
fn fa_dp(f: Fa, b: i32, h: i32, i: i32, j: i32) -> f32 {
    var dp = 0.0;
    for (var e = 0; e < f.D; e++) { dp += dout[b * f.ob + h * f.oh + i * f.orw + e] * v[b * f.vb + h * f.vh + j * f.vr + e]; }
    return dp;
}
";

    public static readonly string FaLse = Shader(new[] { "r:q", "r:k", "r:fam:i32", "r:mask:i32", "rw:lse" }, "total: i32",
        FaLoad + FaScore + Main + @"
    if (i >= p.total) { return; }
    let f = fa_load();
    let qi = i % f.Sq; let bh = i / f.Sq; let b = bh / f.H; let h = bh % f.H;
    var m = 0.0; var seen = false;
    for (var j = 0; j < f.Sk; j++) {
        if (!fa_allowed(f, b, h, qi, j)) { continue; }
        let s = fa_score(f, b, h, qi, j).x;
        if (!seen || s > m) { m = s; }
        seen = true;
    }
    if (!seen) { lse[i] = fa_neg_inf(); return; }
    var l = 0.0;
    for (var j = 0; j < f.Sk; j++) {
        if (!fa_allowed(f, b, h, qi, j)) { continue; }
        l += exp(fa_score(f, b, h, qi, j).x - m);
    }
    lse[i] = m + log(l);
}");

    public static readonly string FaOut = Shader(new[] { "r:q", "r:k", "r:v", "r:fam:i32", "r:mask:i32", "r:lse", "rw:o" },
        "total: i32", FaLoad + FaScore + Main + @"
    if (i >= p.total) { return; }
    let f = fa_load();
    let e = i % f.D; let r = i / f.D; let qi = r % f.Sq; let bh = r / f.Sq; let b = bh / f.H; let h = bh % f.H;
    let lr = lse[r]; var acc = 0.0;
    if (fa_has_keys(lr)) {
        for (var j = 0; j < f.Sk; j++) {
            if (!fa_allowed(f, b, h, qi, j)) { continue; }
            acc += exp(fa_score(f, b, h, qi, j).x - lr) * v[b * f.vb + h * f.vh + j * f.vr + e];
        }
    }
    o[b * f.ob + h * f.oh + qi * f.orw + e] = acc;
}");

    public static readonly string FaDelta = Shader(new[] { "r:o", "r:dout", "r:fam:i32", "rw:st" }, "total: i32", FaLoad + Main + @"
    if (i >= p.total) { return; }
    let f = fa_load();
    let qi = i % f.Sq; let bh = i / f.Sq; let b = bh / f.H; let h = bh % f.H;
    var acc = 0.0;
    for (var e = 0; e < f.D; e++) { acc += dout[b * f.ob + h * f.oh + qi * f.orw + e] * o[b * f.ob + h * f.oh + qi * f.orw + e]; }
    st[p.total + i] = acc;
}");

    public static readonly string FaDq = Shader(
        new[] { "r:q", "r:k", "r:v", "r:dout", "r:st", "r:fam:i32", "r:mask:i32", "rw:dq" }, "total: i32",
        FaLoad + FaScore + FaDp + Main + @"
    if (i >= p.total) { return; }
    let f = fa_load();
    let rows = f.B * f.H * f.Sq;
    let e = i % f.D; let r = i / f.D; let qi = r % f.Sq; let bh = r / f.Sq; let b = bh / f.H; let h = bh % f.H;
    let lr = st[r]; var acc = 0.0;
    if (fa_has_keys(lr)) {
        let dl = st[rows + r];
        for (var j = 0; j < f.Sk; j++) {
            if (!fa_allowed(f, b, h, qi, j)) { continue; }
            let s = fa_score(f, b, h, qi, j);
            acc += exp(s.x - lr) * (fa_dp(f, b, h, qi, j) - dl) * s.y * k[b * f.kb + h * f.kh + j * f.kr + e];
        }
    }
    dq[b * f.qb + h * f.qh + qi * f.qr + e] = acc * f.scale;
}");

    public static readonly string FaDk = Shader(
        new[] { "r:q", "r:k", "r:v", "r:dout", "r:st", "r:fam:i32", "r:mask:i32", "rw:dk" }, "total: i32",
        FaLoad + FaScore + FaDp + Main + @"
    if (i >= p.total) { return; }
    let f = fa_load();
    let rows = f.B * f.H * f.Sq;
    let e = i % f.D; let r = i / f.D; let j = r % f.Sk; let bh = r / f.Sk; let b = bh / f.H; let h = bh % f.H;
    var ak = 0.0;
    for (var qi = 0; qi < f.Sq; qi++) {
        let lr = st[bh * f.Sq + qi];
        if (!fa_has_keys(lr) || !fa_allowed(f, b, h, qi, j)) { continue; }
        let s = fa_score(f, b, h, qi, j);
        ak += exp(s.x - lr) * (fa_dp(f, b, h, qi, j) - st[rows + bh * f.Sq + qi]) * s.y * q[b * f.qb + h * f.qh + qi * f.qr + e] * f.scale;
    }
    dk[b * f.kb + h * f.kh + j * f.kr + e] = ak;
}");

    public static readonly string FaDv = Shader(
        new[] { "r:q", "r:k", "r:dout", "r:st", "r:fam:i32", "r:mask:i32", "rw:dv" }, "total: i32",
        FaLoad + FaScore + Main + @"
    if (i >= p.total) { return; }
    let f = fa_load();
    let e = i % f.D; let r = i / f.D; let j = r % f.Sk; let bh = r / f.Sk; let b = bh / f.H; let h = bh % f.H;
    var av = 0.0;
    for (var qi = 0; qi < f.Sq; qi++) {
        let lr = st[bh * f.Sq + qi];
        if (!fa_has_keys(lr) || !fa_allowed(f, b, h, qi, j)) { continue; }
        av += exp(fa_score(f, b, h, qi, j).x - lr) * dout[b * f.ob + h * f.oh + qi * f.orw + e];
    }
    dv[b * f.vb + h * f.vh + j * f.vr + e] = av;
}");

    // #974 fused decode-linear: line-for-line ports of BatchExecutionCudaKernels.DecodeLinearSource. dl_half is the
    // DL_HALF macro: one FP16 value widened to f32 on its bit pattern (zero, subnormals, infinity and NaN included).
    private const string DlCommon = @"
fn dl_gelu(s: f32) -> f32 { return 0.5 * s * (1.0 + tanh(0.7978845608028654 * (s + 0.044715 * s * s * s))); }
";

    private const string DlHalf = @"
fn dl_half(word: i32, slot: i32) -> f32 {
    let b = (word >> (u32(slot) * 16u)) & 65535;
    let e = (b >> 10u) & 31;
    let m = b & 1023;
    var v = 0.0;
    if (e == 0) {
        v = f32(m) * 5.9604644775390625e-8;
    } else {
        let exponent = select(e + 112, 255, e == 31);
        v = bitcast<f32>((exponent << 23u) | (m << 13u));
    }
    if ((b & 32768) != 0) { v = -v; }
    return v;
}
";

    public static readonly string DlLinearGelu = Shader(new[] { "r:x", "r:w", "r:bias", "rw:y" }, "M: i32, K: i32, N: i32",
        DlCommon + Main + @"
    let K = p.K; let N = p.N;
    if (i >= p.M * N) { return; }
    let m = i / N; let n = i - m * N;
    var s = 0.0;
    for (var k = 0; k < K; k++) { s += x[m * K + k] * w[n * K + k]; }
    s += bias[n];
    y[i] = dl_gelu(s);
}");

    public static readonly string DlLinearGeluFp16 = Shader(new[] { "r:xh:i32", "r:wh:i32", "r:bias", "rw:y" }, "M: i32, K: i32, N: i32",
        DlCommon + DlHalf + Main + @"
    let K = p.K; let N = p.N;
    if (i >= p.M * N) { return; }
    let m = i / N; let n = i - m * N;
    var s = 0.0;
    for (var k = 0; k < K; k++) {
        let xi = m * K + k; let wi = n * K + k;
        s += dl_half(xh[xi >> 1u], xi & 1) * dl_half(wh[wi >> 1u], wi & 1);
    }
    s += bias[n];
    y[i] = dl_gelu(s);
}");

    public static readonly string DlW8A8LinearGelu = Shader(
        new[] { "r:x8:i32", "r:w8:i32", "r:actScale", "r:wScales", "r:bias", "rw:y" }, "K: i32, N: i32", DlCommon + Main + @"
    let K = p.K;
    if (i >= p.N) { return; }
    var total = 0;
    for (var k = 0; k < K; k++) {
        var xi = (x8[k >> 2u] >> (u32(k & 3) * 8u)) & 255;
        if (xi > 127) { xi -= 256; }
        let j = i * K + k;
        var wi = (w8[j >> 2u] >> (u32(j & 3) * 8u)) & 255;
        if (wi > 127) { wi -= 256; }
        total += xi * wi;
    }
    let s = f32(total) * actScale[0] * wScales[i] + bias[i];
    y[i] = dl_gelu(s);
}");

    public static readonly string DlW8A8Epilogue = Shader(
        new[] { "r:acc:i32", "r:actScale", "r:wScales", "r:bias", "rw:y" }, "N: i32", DlCommon + Main + @"
    if (i >= p.N) { return; }
    let s = f32(acc[i]) * actScale[0] * wScales[i] + bias[i];
    y[i] = dl_gelu(s);
}");

    // #1107 / #1108 fused LSTM: line-for-line ports of BatchExecutionCudaKernels.LstmSource.
    private const string LstmSig = @"
fn lstm_sig(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
";

    public static readonly string LstmGates = Shader(
        new[] { "r:x", "r:wih", "r:whh", "r:bih", "r:bhh", "r:allH", "rw:gates" }, "t: i32, B: i32, T: i32, In: i32, H: i32",
        LstmSig + Main + @"
    let H = p.H; let In = p.In; let B = p.B;
    if (i >= B * 4 * H) { return; }
    let b = i / (4 * H); let row = i - b * 4 * H;
    var s = bih[row] + bhh[row];
    for (var k = 0; k < In; k++) { s += wih[row * In + k] * x[(b * p.T + p.t) * In + k]; }
    for (var j = 0; j < H; j++) { s += whh[row * H + j] * allH[(p.t * B + b) * H + j]; }
    gates[(p.t * B + b) * 4 * H + row] = select(lstm_sig(s), tanh(s), row / H == 2);
}");

    public static readonly string LstmCell = Shader(
        new[] { "r:gates", "rw:allC", "rw:allH", "rw:y" }, "t: i32, B: i32, T: i32, H: i32", Main + @"
    let H = p.H; let B = p.B; let t = p.t;
    if (i >= B * H) { return; }
    let b = i / H; let h = i - b * H;
    let gb = (t * B + b) * 4 * H;
    let c = gates[gb + H + h] * allC[(t * B + b) * H + h] + gates[gb + h] * gates[gb + 2 * H + h];
    let hv = gates[gb + 3 * H + h] * tanh(c);
    allC[((t + 1) * B + b) * H + h] = c;
    allH[((t + 1) * B + b) * H + h] = hv;
    y[(b * p.T + t) * H + h] = hv;
}");

    public static readonly string LstmCellBackward = Shader(
        new[] { "r:gy", "r:gates", "r:allC", "r:dh", "rw:dc", "rw:dpre" }, "t: i32, B: i32, T: i32, H: i32", Main + @"
    let H = p.H; let B = p.B; let t = p.t;
    if (i >= B * H) { return; }
    let b = i / H; let h = i - b * H;
    let gb = (t * B + b) * 4 * H;
    let ig = gates[gb + h]; let fg = gates[gb + H + h]; let gg = gates[gb + 2 * H + h]; let og = gates[gb + 3 * H + h];
    let tc = tanh(allC[((t + 1) * B + b) * H + h]);
    let dhv = gy[(b * p.T + t) * H + h] + dh[i];
    let dcv = dhv * og * (1.0 - tc * tc) + dc[i];
    dpre[gb + h] = dcv * gg * ig * (1.0 - ig);
    dpre[gb + H + h] = dcv * allC[(t * B + b) * H + h] * fg * (1.0 - fg);
    dpre[gb + 2 * H + h] = dcv * ig * (1.0 - gg * gg);
    dpre[gb + 3 * H + h] = dhv * tc * og * (1.0 - og);
    dc[i] = dcv * fg;
}");

    public static readonly string LstmHiddenBackward = Shader(
        new[] { "r:dpre", "r:whh", "rw:dh" }, "t: i32, B: i32, H: i32", Main + @"
    let H = p.H; let B = p.B;
    if (i >= B * H) { return; }
    let b = i / H; let j = i - b * H;
    let gb = (p.t * B + b) * 4 * H;
    var s = 0.0;
    for (var row = 0; row < 4 * H; row++) { s += dpre[gb + row] * whh[row * H + j]; }
    dh[i] = s;
}");

    public static readonly string LstmInputGrad = Shader(
        new[] { "r:dpre", "r:wih", "rw:gin" }, "B: i32, T: i32, In: i32, H: i32", Main + @"
    let H = p.H; let In = p.In;
    if (i >= p.B * p.T * In) { return; }
    let k = i % In; let bt = i / In; let t = bt % p.T; let b = bt / p.T;
    let gb = (t * p.B + b) * 4 * H;
    var s = 0.0;
    for (var row = 0; row < 4 * H; row++) { s += dpre[gb + row] * wih[row * In + k]; }
    gin[i] = s;
}");

    public static readonly string LstmWeightIhGrad = Shader(
        new[] { "r:dpre", "r:x", "rw:gwih" }, "B: i32, T: i32, In: i32, H: i32", Main + @"
    let H = p.H; let In = p.In;
    if (i >= 4 * H * In) { return; }
    let row = i / In; let k = i - row * In;
    var s = 0.0;
    for (var t = 0; t < p.T; t++) {
        for (var b = 0; b < p.B; b++) { s += dpre[(t * p.B + b) * 4 * H + row] * x[(b * p.T + t) * In + k]; }
    }
    gwih[i] = s;
}");

    public static readonly string LstmWeightHhGrad = Shader(
        new[] { "r:dpre", "r:allH", "rw:gwhh" }, "B: i32, T: i32, H: i32", Main + @"
    let H = p.H;
    if (i >= 4 * H * H) { return; }
    let row = i / H; let j = i - row * H;
    var s = 0.0;
    for (var t = 0; t < p.T; t++) {
        for (var b = 0; b < p.B; b++) { s += dpre[(t * p.B + b) * 4 * H + row] * allH[(t * p.B + b) * H + j]; }
    }
    gwhh[i] = s;
}");

    public static readonly string LstmBiasGrad = Shader(
        new[] { "r:dpre", "rw:gbih", "rw:gbhh" }, "B: i32, T: i32, H: i32", Main + @"
    let H = p.H;
    if (i >= 4 * H) { return; }
    var s = 0.0;
    for (var t = 0; t < p.T; t++) {
        for (var b = 0; b < p.B; b++) { s += dpre[(t * p.B + b) * 4 * H + i]; }
    }
    gbih[i] = s;
    gbhh[i] = s;
}");
    /// <summary>The WGSL port of each shared kernel, keyed by its <see cref="BatchExecutionCudaKernels"/> name.</summary>
    public static readonly IReadOnlyDictionary<string, string> ByKernelName = new Dictionary<string, string>
    {
        ["bx_full_reduce"] = FullReduce, ["bx_row_reduce"] = RowReduce, ["bx_row_map"] = RowMap,
        ["bx_scalar"] = Scalar, ["bx_unary"] = Unary, ["bx_sincos"] = SinCos,
        ["bx_reduce_backward"] = ReduceBackward, ["bx_broadcast_first"] = BroadcastFirst,
        ["bx_gated_backward"] = GatedBackward, ["bx_concat_axis"] = ConcatAxis,
        ["bx_slice_last_axis"] = SliceLastAxis, ["bx_set_slice_last_axis"] = SetSliceLastAxis,
        ["bx_slice_axis"] = SliceAxis, ["bx_set_slice_axis"] = SetSliceAxis, ["bx_stack2"] = Stack2,
        ["bx_window_2d"] = Window2D, ["bx_tile_last_axis"] = TileLastAxis,
        ["bx_repeat_elements"] = RepeatElements, ["bx_linspace"] = Linspace, ["bx_diag"] = Diag,
        ["bx_extract_diag"] = ExtractDiag, ["bx_triangular_mask"] = TriangularMask,
        ["bx_index_select"] = IndexSelect, ["bx_row_loss"] = RowLoss, ["bx_dropout_mask"] = DropoutMask,
        ["bx_gaussian_noise"] = GaussianNoise, ["bx_gumbel_softmax"] = GumbelSoftmax,
        ["bx_batch_outer_product"] = BatchOuterProduct, ["bx_meshgrid2d"] = Meshgrid2D,
        ["bx_fft_chirp"] = FftChirp, ["bx_fft_bluestein_pre"] = FftBluesteinPre, ["bx_fft_cmul"] = FftCmul,
        ["bx_fft_bluestein_post"] = FftBluesteinPost, ["bx_fft_to_storage"] = FftToStorage, ["bx_fft_from_storage"] = FftFromStorage,
        [BatchExecutionCudaKernels.LaSymmetrize] = LaSymmetrize, [BatchExecutionCudaKernels.LaLuSolve] = LaLuSolve,
        [BatchExecutionCudaKernels.LaLdlFactor] = LaLdlFactor, [BatchExecutionCudaKernels.LaLdlSolve] = LaLdlSolve,
        [BatchExecutionCudaKernels.LaSolveVector] = LaSolveVector, [BatchExecutionCudaKernels.LaTriangularSolveVector] = LaTriangularSolveVector,
        [BatchExecutionCudaKernels.LaCholeskyBackwardLower] = LaCholeskyBackwardLower,
        [BatchExecutionCudaKernels.LaSolveBackwardVector] = LaSolveBackwardVector, [BatchExecutionCudaKernels.LaSvdReduced] = LaSvdReduced,
        [BatchExecutionCudaKernels.LaQrReduced] = LaQrReduced,
        [BatchExecutionCudaKernels.FaLse] = FaLse, [BatchExecutionCudaKernels.FaOut] = FaOut, [BatchExecutionCudaKernels.FaDelta] = FaDelta,
        [BatchExecutionCudaKernels.FaDq] = FaDq, [BatchExecutionCudaKernels.FaDk] = FaDk, [BatchExecutionCudaKernels.FaDv] = FaDv,
        [BatchExecutionCudaKernels.LstmGates] = LstmGates, [BatchExecutionCudaKernels.LstmCell] = LstmCell,
        [BatchExecutionCudaKernels.LstmCellBackward] = LstmCellBackward, [BatchExecutionCudaKernels.LstmHiddenBackward] = LstmHiddenBackward,
        [BatchExecutionCudaKernels.LstmInputGrad] = LstmInputGrad, [BatchExecutionCudaKernels.LstmWeightIhGrad] = LstmWeightIhGrad,
        [BatchExecutionCudaKernels.LstmWeightHhGrad] = LstmWeightHhGrad, [BatchExecutionCudaKernels.LstmBiasGrad] = LstmBiasGrad,
        [BatchExecutionCudaKernels.DlLinearGelu] = DlLinearGelu, [BatchExecutionCudaKernels.DlLinearGeluFp16] = DlLinearGeluFp16,
        [BatchExecutionCudaKernels.DlW8A8LinearGelu] = DlW8A8LinearGelu, [BatchExecutionCudaKernels.DlW8A8Epilogue] = DlW8A8Epilogue,
    };
}
#endif