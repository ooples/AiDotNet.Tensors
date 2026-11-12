// Copyright (c) CrowdTrainer LLC. All rights reserved.
// WebGPU WGSL compute shader — radix-2 Cooley-Tukey FFT. Matches the Metal/
// Vulkan peer kernels. One workgroup per batch slice.

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu
{
    internal static class WebGpuFftKernels
    {
        public const int SharedBatchedFftMaxN = 2048;

        /// <summary>Threads per workgroup of <see cref="SharedBatchedFft"/>; the host dispatches this many per row.</summary>
        public const int SharedBatchedFftWorkgroupSize = 256;

        public static string[] GetKernelNames() => new[]
        {
            "parity212_fft",
            "batched_fft_shared",
            "fft_1d",
        };

        /// <summary>
        /// Arbitrary-length split-complex batched DFT with modular angle reduction (prod = (k * j) mod n)
        /// so non-power-of-two transforms preserve full FP32 phase precision across large indices.
        /// </summary>
        public const string DirectDft1D = @"
@group(0) @binding(0) var<storage, read> input_real: array<f32>;
@group(0) @binding(1) var<storage, read> input_imag: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_real: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_imag: array<f32>;

struct Params {
    n: u32,
    batch_size: u32,
    inverse: u32,
    _pad: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn fft_1d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch_size * params.n;
    if (idx >= total) { return; }
    let b = idx / params.n;
    let k = idx % params.n;
    let off = b * params.n;
    let sign = select(-1.0, 1.0, params.inverse > 0u);
    let PI2 = 6.283185307179586;
    var sum_r: f32 = 0.0;
    var sum_i: f32 = 0.0;
    var prod: u32 = 0u;
    for (var j: u32 = 0u; j < params.n; j = j + 1u) {
        let angle = sign * PI2 * f32(prod) / f32(params.n);
        prod = prod + k;
        if (prod >= params.n) { prod = prod - params.n; }
        let c = cos(angle);
        let s = sin(angle);
        sum_r = sum_r + input_real[off + j] * c - input_imag[off + j] * s;
        sum_i = sum_i + input_real[off + j] * s + input_imag[off + j] * c;
    }
    if (params.inverse > 0u) {
        let inv_n = 1.0 / f32(params.n);
        output_real[idx] = sum_r * inv_n;
        output_imag[idx] = sum_i * inv_n;
    } else {
        output_real[idx] = sum_r;
        output_imag[idx] = sum_i;
    }
}
";

        /// <summary>
        /// Power-of-two split-complex batched FFT, one workgroup per row with the row held in workgroup memory
        /// (2 * 2048 floats = 16 KB, fitting the WebGPU guaranteed minimum maxComputeWorkgroupStorageSize).
        /// Loads in natural order into the bit-reversed slot, runs every radix-2 DIT stage on chip, and writes once
        /// (scaled by 1/n for an inverse): O(n log n) in a single dispatch.
        /// </summary>
        // The shader's workgroup size, its loop strides and its row storage come from the constants above, which the
        // host dispatch also reads, so one of them cannot change without the others.
        public static readonly string SharedBatchedFft =
            "const WG: u32 = " + SharedBatchedFftWorkgroupSize + "u;\n" +
            "const MAX_N: u32 = " + SharedBatchedFftMaxN + "u;\n" + SharedBatchedFftBody;

        private const string SharedBatchedFftBody = @"
@group(0) @binding(0) var<storage, read> input_real: array<f32>;
@group(0) @binding(1) var<storage, read> input_imag: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_real: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_imag: array<f32>;

struct Params {
    row_base: u32,
    n: u32,
    log2n: u32,
    inverse: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> sre: array<f32, MAX_N>;
var<workgroup> sim: array<f32, MAX_N>;

@compute @workgroup_size(WG)
fn batched_fft_shared(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) li: vec3<u32>)
{
    let lid = li.x;
    let base = (params.row_base + wg.x) * params.n;
    let shift = 32u - params.log2n;
    for (var i: u32 = lid; i < params.n; i = i + WG) {
        let r = reverseBits(i) >> shift;
        sre[r] = input_real[base + i];
        sim[r] = input_imag[base + i];
    }
    workgroupBarrier();

    let sgn = select(-1.0, 1.0, params.inverse != 0u);
    let half_n = params.n >> 1u;
    for (var lg: u32 = 0u; lg < params.log2n; lg = lg + 1u) {
        let hs = 1u << lg;
        for (var k: u32 = lid; k < half_n; k = k + WG) {
            let w = k & (hs - 1u);
            let top = ((k >> lg) << (lg + 1u)) + w;
            let bot = top + hs;
            let a = 3.14159265358979323846 * sgn * f32(w) / f32(hs);
            let c = cos(a);
            let s = sin(a);
            let br = sre[bot];
            let bi = sim[bot];
            let tr = br * c - bi * s;
            let ti = br * s + bi * c;
            let ar = sre[top];
            let ai = sim[top];
            sre[top] = ar + tr;
            sim[top] = ai + ti;
            sre[bot] = ar - tr;
            sim[bot] = ai - ti;
        }
        workgroupBarrier();
    }

    let scale = select(1.0, 1.0 / f32(params.n), params.inverse != 0u);
    for (var i: u32 = lid; i < params.n; i = i + WG) {
        output_real[base + i] = sre[i] * scale;
        output_imag[base + i] = sim[i] * scale;
    }
}
";

        public static string Fft => @"
@group(0) @binding(0) var<storage, read_write> buf : array<f32>;
struct P { batchCount : i32, n : i32, inverse : i32 };
@group(0) @binding(1) var<uniform> p : P;

@compute @workgroup_size(256) fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) li : vec3<u32>)
{
    let b = i32(wg.x);
    if (b >= p.batchCount) { return; }
    let tid = i32(li.x);
    let bs = 256;
    let n = p.n;
    let bOff = b * 2 * n;

    // Bit reversal permutation.
    var bits : i32 = 0;
    var t : i32 = n;
    loop { if (t <= 1) { break; } t = t >> 1; bits = bits + 1; }

    var i : i32 = tid;
    loop {
        if (i >= n) { break; }
        var j : i32 = 0;
        var v : i32 = i;
        for (var bb : i32 = 0; bb < bits; bb = bb + 1) { j = (j << 1) | (v & 1); v = v >> 1; }
        if (j > i) {
            let tr = buf[bOff + 2 * i];
            let ti = buf[bOff + 2 * i + 1];
            buf[bOff + 2 * i] = buf[bOff + 2 * j];
            buf[bOff + 2 * i + 1] = buf[bOff + 2 * j + 1];
            buf[bOff + 2 * j] = tr;
            buf[bOff + 2 * j + 1] = ti;
        }
        i = i + bs;
    }
    storageBarrier();
    workgroupBarrier();

    let sign = select(-1.0, 1.0, p.inverse != 0);
    var size : i32 = 2;
    loop {
        if (size > n) { break; }
        let half = size >> 1;
        let theta = sign * 2.0 * 3.14159265358979323846 / f32(size);
        let numButterflies = n >> 1;
        var bf : i32 = tid;
        loop {
            if (bf >= numButterflies) { break; }
            let group = bf / half;
            let k = bf - group * half;
            let e = group * size + k;
            let o = e + half;
            let angle = theta * f32(k);
            let wRe = cos(angle);
            let wIm = sin(angle);
            let eRe = buf[bOff + 2 * e];
            let eIm = buf[bOff + 2 * e + 1];
            let oRe = buf[bOff + 2 * o];
            let oIm = buf[bOff + 2 * o + 1];
            let tRe2 = wRe * oRe - wIm * oIm;
            let tIm2 = wRe * oIm + wIm * oRe;
            buf[bOff + 2 * e] = eRe + tRe2;
            buf[bOff + 2 * e + 1] = eIm + tIm2;
            buf[bOff + 2 * o] = eRe - tRe2;
            buf[bOff + 2 * o + 1] = eIm - tIm2;
            bf = bf + bs;
        }
        storageBarrier();
    workgroupBarrier();
        size = size << 1;
    }

    if (p.inverse != 0) {
        let invN = 1.0 / f32(n);
        var i2 : i32 = tid;
        loop {
            if (i2 >= n) { break; }
            buf[bOff + 2 * i2] = buf[bOff + 2 * i2] * invN;
            buf[bOff + 2 * i2 + 1] = buf[bOff + 2 * i2 + 1] * invN;
            i2 = i2 + bs;
        }
    }
}
";
    }
}
