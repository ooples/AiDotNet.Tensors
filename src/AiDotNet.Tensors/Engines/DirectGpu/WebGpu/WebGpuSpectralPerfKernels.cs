// Copyright (c) AiDotNet. All rights reserved.
// WGSL compute shaders for Issue #160 spectral perf kernels.

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu
{
    internal static class WebGpuSpectralPerfKernels
    {
        public const string Atan2Source = @"
@group(0) @binding(0) var<storage, read> imag: array<f32>;
@group(0) @binding(1) var<storage, read> real: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
struct Params { n: u32 }
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn atan2_elementwise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < p.n) { output[idx] = atan2(imag[idx], real[idx]); }
}";

        public const string NormalizeRowsSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
struct Params { rows: u32, cols: u32 }
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64)
fn normalize_rows_fused(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    let rowOff = row * p.cols;
    var sumSq: f32 = 0.0;
    for (var c: u32 = 0u; c < p.cols; c = c + 1u) {
        let v = input[rowOff + c];
        sumSq = sumSq + v * v;
    }
    var invNorm: f32 = 0.0;
    if (sumSq > 0.0) { invNorm = inverseSqrt(sumSq); }
    for (var c: u32 = 0u; c < p.cols; c = c + 1u) {
        output[rowOff + c] = input[rowOff + c] * invNorm;
    }
}";

        public const string AnalyticSignalMaskSource = @"
@group(0) @binding(0) var<storage, read> specReal: array<f32>;
@group(0) @binding(1) var<storage, read_write> outReal: array<f32>;
struct Params { batch: u32, fftSize: u32, binLow: u32, binHigh: u32 }
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn analytic_signal_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.batch * p.fftSize;
    if (idx >= total) { return; }
    let k = idx % p.fftSize;
    let halfN = p.fftSize >> 1u;
    var gain: f32;
    if (k == 0u || k == halfN) {
        if (k < p.binLow || k >= p.binHigh) { gain = 0.0; } else { gain = 1.0; }
    } else if (k < halfN) {
        if (k < p.binLow || k >= p.binHigh) { gain = 0.0; } else { gain = 2.0; }
    } else {
        gain = 0.0;
    }
    outReal[idx] = specReal[idx] * gain;
}";

        public const string BispectrumSource = @"
@group(0) @binding(0) var<storage, read> specReal: array<f32>;
@group(0) @binding(1) var<storage, read> specImag: array<f32>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<f32>;
struct Params { maxF1: u32, maxF2: u32, mode: u32 }
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn bispectrum_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.maxF1 * p.maxF2;
    if (idx >= total) { return; }
    let f1 = idx / p.maxF2;
    let f2 = idx % p.maxF2;
    let sumIdx = f1 + f2;
    let ar = specReal[f1]; let ai = specImag[f1];
    let br = specReal[f2]; let bi = specImag[f2];
    let cr = specReal[sumIdx]; let ci = -specImag[sumIdx];
    let abr = ar * br - ai * bi;
    let abi = ar * bi + ai * br;
    if (p.mode == 0u) { outBuf[idx] = abr * cr - abi * ci; }
    else { outBuf[idx] = abr * ci + abi * cr; }
}";

        public const string TrispectrumSource = @"
@group(0) @binding(0) var<storage, read> specReal: array<f32>;
@group(0) @binding(1) var<storage, read> specImag: array<f32>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<f32>;
struct Params { maxF1: u32, maxF2: u32, maxF3: u32, mode: u32 }
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn trispectrum_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.maxF1 * p.maxF2 * p.maxF3;
    if (idx >= total) { return; }
    let f1 = idx / (p.maxF2 * p.maxF3);
    let rem = idx - f1 * p.maxF2 * p.maxF3;
    let f2 = rem / p.maxF3;
    let f3 = rem - f2 * p.maxF3;
    let sumIdx = f1 + f2 + f3;
    let ar = specReal[f1]; let ai = specImag[f1];
    let br = specReal[f2]; let bi = specImag[f2];
    let cr = specReal[f3]; let ci = specImag[f3];
    let dr = specReal[sumIdx]; let di = -specImag[sumIdx];
    let t1r = ar * br - ai * bi;
    let t1i = ar * bi + ai * br;
    let t2r = t1r * cr - t1i * ci;
    let t2i = t1r * ci + t1i * cr;
    if (p.mode == 0u) { outBuf[idx] = t2r * dr - t2i * di; }
    else { outBuf[idx] = t2r * di + t2i * dr; }
}";

        public const string CavityBounceSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
struct Params { total: u32, invN: f32 }
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn cavity_bounce_real(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.total) { return; }
    var r = input[idx] * p.invN;
    r = clamp(r, -20.0, 20.0);
    output[idx] = tanh(r);
}";

        public const string ZeroBufferSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
struct Params { total: u32 }
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn zero_buffer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.total) { return; }
    output[idx] = 0.0;
}";

        public const string WidebandLogBinPoolSource = @"
@group(0) @binding(0) var<storage, read> magBuf: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
struct Params { totalSegBatch: u32, fftSize: u32, numBins: u32, usable: u32 }
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn wideband_log_bin_pool(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outIdx = gid.x;
    let total = p.totalSegBatch * p.numBins;
    if (outIdx >= total) { return; }
    let seg = outIdx / p.numBins;
    let k = outIdx % p.numBins;
    let r0 = f32(k) / f32(p.numBins);
    let r1 = f32(k + 1u) / f32(p.numBins);
    var binStart = 1 + i32(r0 * r0 * f32(p.usable - 1u));
    var binEnd = 1 + i32(r1 * r1 * f32(p.usable - 1u));
    if (binEnd <= binStart) { binEnd = binStart + 1; }
    if (binEnd > i32(p.usable)) { binEnd = i32(p.usable); }
    let magOff = seg * p.fftSize;
    var sum: f32 = 0.0; var cnt: i32 = 0;
    for (var i = binStart; i < binEnd; i = i + 1) {
        sum = sum + magBuf[magOff + u32(i)];
        cnt = cnt + 1;
    }
    var avg: f32 = 0.0;
    if (cnt > 0) { avg = sum / f32(cnt); }
    output[outIdx] = log(1.0 + avg);
}";

        public const string MelFilterbankSource = @"
@group(0) @binding(0) var<storage, read> powerSpec: array<f32>;
@group(0) @binding(1) var<storage, read> melFilters: array<f32>;
@group(0) @binding(2) var<storage, read_write> melEnergy: array<f32>;
struct Params { totalSegBatch: u32, specBins: u32, melBins: u32 }
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn mel_filterbank_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outIdx = gid.x;
    let total = p.totalSegBatch * p.melBins;
    if (outIdx >= total) { return; }
    let seg = outIdx / p.melBins;
    let m = outIdx % p.melBins;
    let powerOff = seg * p.specBins;
    let filtOff = m * p.specBins;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < p.specBins; i = i + 1u) {
        sum = sum + powerSpec[powerOff + i] * melFilters[filtOff + i];
    }
    melEnergy[outIdx] = sum;
}";

        public const string MfccLog1pSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
struct Params { n: u32 }
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn mfcc_log1p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.n) { return; }
    output[idx] = log(1.0 + input[idx]);
}";

        public const string PacPhaseBinMiSource = @"
@group(0) @binding(0) var<storage, read> thetaPhase: array<f32>;
@group(0) @binding(1) var<storage, read> gammaAmp: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
struct Params { batch: u32, numSamples: u32, numGammaBands: u32, gammaIdx: u32 }
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(64)
fn pac_phase_bin_mi(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= p.batch) { return; }
    var binSum: array<f32, 18>;
    var binCount: array<f32, 18>;
    for (var k: u32 = 0u; k < 18u; k = k + 1u) { binSum[k] = 0.0; binCount[k] = 0.0; }
    let sampleOff = b * p.numSamples;
    let gammaOff = (p.gammaIdx * p.batch + b) * p.numSamples;
    for (var i: u32 = 0u; i < p.numSamples; i = i + 1u) {
        let phase = thetaPhase[sampleOff + i];
        let amp = gammaAmp[gammaOff + i];
        var fbin = (phase + 3.14159265358979) / (2.0 * 3.14159265358979) * 18.0;
        var bin = i32(fbin);
        if (bin < 0) { bin = 0; }
        if (bin >= 18) { bin = 17; }
        binSum[bin] = binSum[bin] + amp;
        binCount[bin] = binCount[bin] + 1.0;
    }
    var totalAmp: f32 = 0.0;
    for (var k: u32 = 0u; k < 18u; k = k + 1u) {
        var avg: f32 = 0.0;
        if (binCount[k] > 0.0) { avg = binSum[k] / binCount[k]; }
        totalAmp = totalAmp + avg;
    }
    var mi: f32 = 0.0;
    if (totalAmp > 0.0) {
        var entropy: f32 = 0.0;
        for (var k: u32 = 0u; k < 18u; k = k + 1u) {
            var avg: f32 = 0.0;
            if (binCount[k] > 0.0) { avg = binSum[k] / binCount[k]; }
            let pp = avg / totalAmp;
            if (pp > 1e-12) { entropy = entropy - pp * log(pp); }
        }
        mi = (log(18.0) - entropy) / log(18.0);
    }
    output[b * p.numGammaBands + p.gammaIdx] = mi;
}";
    }
}
