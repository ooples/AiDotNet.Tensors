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
              .Append(parts[0] == "r" ? "read" : "read_write").Append("> ").Append(parts[1]).Append(" : array<f32>;\n");
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
        ["bx_batch_outer_product"] = BatchOuterProduct,
    };
}
#endif