// Copyright (c) CrowdTrainer LLC. All rights reserved.
// One kernel each for members that were compositions of other backend members (TENS0008, #1638): the split-complex
// element-wise family (multiply was six dispatches through four scratch buffers, each with its own submit-and-wait), the
// batched dot product (a multiply into a temporary, then a row sum) and the CSR SpMM with its bias added in the same pass.

#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    private const string FusedCompositesModuleName = "FusedComposites";

    /// <summary>Selector of the split_complex_binary entry point.</summary>
    private enum SplitComplexBinaryOp { Multiply = 0, Add = 1, CrossSpectral = 2 }

    /// <summary>Selector of the split_complex_unary entry point.</summary>
    private enum SplitComplexUnaryOp { Conjugate = 0, Scale = 1 }

    /// <summary>Threads per row of batched_dot (its workgroup size).</summary>
    private const int BatchedDotWorkgroupSize = 256;

    // Each entry point declares only the bindings it uses; a module may hold several layouts because each pipeline is
    // built for one entry point.
    private const string SplitComplexBinarySource = @"
@group(0) @binding(0) var<storage, read> aR: array<f32>;
@group(0) @binding(1) var<storage, read> aI: array<f32>;
@group(0) @binding(2) var<storage, read> bR: array<f32>;
@group(0) @binding(3) var<storage, read> bI: array<f32>;
@group(0) @binding(4) var<storage, read_write> oR: array<f32>;
@group(0) @binding(5) var<storage, read_write> oI: array<f32>;
struct Params { n: u32, op: u32, }
@group(0) @binding(6) var<uniform> params: Params;

// (aR, aI) op (bR, bI); CrossSpectral is a * conj(b).
@compute @workgroup_size(256)
fn split_complex_binary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let xr = aR[i]; let xi = aI[i]; let yr = bR[i]; let yi = bI[i];
    if (params.op == 0u) { oR[i] = xr * yr - xi * yi; oI[i] = xr * yi + xi * yr; }
    else if (params.op == 1u) { oR[i] = xr + yr; oI[i] = xi + yi; }
    else { oR[i] = xr * yr + xi * yi; oI[i] = xi * yr - xr * yi; }
}
";

    private const string SplitComplexUnarySource = @"
@group(0) @binding(0) var<storage, read> iR: array<f32>;
@group(0) @binding(1) var<storage, read> iI: array<f32>;
@group(0) @binding(2) var<storage, read_write> oR: array<f32>;
@group(0) @binding(3) var<storage, read_write> oI: array<f32>;
struct Params { n: u32, op: u32, s: f32, pad: f32, }
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn split_complex_unary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    if (params.op == 0u) { oR[i] = iR[i]; oI[i] = -iI[i]; }
    else { oR[i] = iR[i] * params.s; oI[i] = iI[i] * params.s; }
}
";

    private const string SplitComplexMagnitudeSquaredSource = @"
@group(0) @binding(0) var<storage, read> iR: array<f32>;
@group(0) @binding(1) var<storage, read> iI: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
struct Params { n: u32, }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn split_complex_magnitude_squared(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    o[i] = iR[i] * iR[i] + iI[i] * iI[i];
}
";

    // One workgroup per row: strided partial dot products, then a workgroup-memory tree.
    private const string BatchedDotSource = @"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> r: array<f32>;
struct Params { batchSize: u32, vecSize: u32, }
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn batched_dot(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let t = lid.x;
    var s: f32 = 0.0;
    if (row < params.batchSize) {
        let base = row * params.vecSize;
        for (var i: u32 = t; i < params.vecSize; i = i + 256u) { s = s + a[base + i] * b[base + i]; }
    }
    partial[t] = s;
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
    }
    if (t == 0u && row < params.batchSize) { r[row] = partial[0]; }
}
";

    // WebGpuKernels.CsrSpMMSource with the bias of the output column added before the store.
    private const string CsrSpMMBiasSource = @"
@group(0) @binding(0) var<storage, read> csr_values: array<f32>;
@group(0) @binding(1) var<storage, read> csr_col_indices: array<f32>;
@group(0) @binding(2) var<storage, read> csr_row_ptrs: array<f32>;
@group(0) @binding(3) var<storage, read> dense_B: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
struct CsrParams { M: u32, K: u32, N: u32, nnz: u32, }
@group(0) @binding(6) var<uniform> params: CsrParams;

@compute @workgroup_size(256)
fn csr_spmm_bias(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.M * params.N) { return; }
    let row = idx / params.N;
    let j = idx % params.N;
    let start = bitcast<u32>(bitcast<i32>(csr_row_ptrs[row]));
    let end = bitcast<u32>(bitcast<i32>(csr_row_ptrs[row + 1u]));
    var sum_val: f32 = 0.0;
    for (var i: u32 = start; i < end; i = i + 1u) {
        let col = bitcast<u32>(bitcast<i32>(csr_col_indices[i]));
        sum_val = sum_val + csr_values[i] * dense_B[col * params.N + j];
    }
    output[idx] = sum_val + bias[j];
}
";

    // expm1 and log1p accurate for a small x: AdditionalUnarySource's exp(x) - 1.0 and log(1.0 + x) keep only a few of its
    // digits (that source is a public constant, so these live here). WGSL's log is too coarse near 1 for Kahan's
    // correction, so a small x takes a series: Taylor for expm1, 2 atanh(x / (2 + x)) for log1p (both below 1e-9 relative).
    private const string AccurateExpLogSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
struct Params { size: u32, }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn expm1_accurate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.size) { return; }
    let x = A[i];
    var y = exp(x) - 1.0;
    if (abs(x) < 0.35) {
        y = x * (1.0 + x / 2.0 * (1.0 + x / 3.0 * (1.0 + x / 4.0 * (1.0 + x / 5.0 * (1.0 + x / 6.0
            * (1.0 + x / 7.0 * (1.0 + x / 8.0)))))));
    }
    B[i] = y;
}

@compute @workgroup_size(256)
fn log1p_accurate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.size) { return; }
    let x = A[i];
    var y = log(1.0 + x);
    if (abs(x) < 0.25) {
        let s = x / (2.0 + x);
        let s2 = s * s;
        y = 2.0 * s * (1.0 + s2 * (1.0 / 3.0 + s2 * (1.0 / 5.0 + s2 * (1.0 / 7.0 + s2 * (1.0 / 9.0 + s2 / 11.0)))));
    }
    if (x == -1.0) { y = wgsl_f32_bits(0xff800000u); }        // log1p(-1) = -inf
    if (x < -1.0) { y = wgsl_f32_bits(0x7fc00000u); }         // outside the domain (and -inf): NaN
    B[i] = y;
}
";

    // -sum(target * log_softmax(row)) per row, from logits and dense targets: the shifted logits minus log(sumExp).
    private const string DenseCrossEntropyRowsSource = @"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> rowLoss: array<f32>;
struct Params { batch_size: u32, num_classes: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn cross_entropy_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= params.batch_size) { return; }
    let base = b * params.num_classes;
    var max_val: f32 = -3.402823e+38;
    for (var c: u32 = 0u; c < params.num_classes; c = c + 1u) { max_val = max(max_val, logits[base + c]); }
    var sum_exp: f32 = 0.0;
    for (var c: u32 = 0u; c < params.num_classes; c = c + 1u) { sum_exp = sum_exp + exp(logits[base + c] - max_val); }
    var shifted: f32 = 0.0;
    var target_sum: f32 = 0.0;
    for (var c: u32 = 0u; c < params.num_classes; c = c + 1u) {
        shifted = shifted + targets[base + c] * (logits[base + c] - max_val);
        target_sum = target_sum + targets[base + c];
    }
    rowLoss[b] = log(sum_exp) * target_sum - shifted;
}
";

    // melSpec[b, m] = sum_f powerSpec[b, f] * filterbank[m, f]: one thread per output, reading the filterbank row in place
    // (this was a transposed copy of the filterbank and a GEMM).
    private const string MelFilterbankSource = @"
@group(0) @binding(0) var<storage, read> power_spec: array<f32>;
@group(0) @binding(1) var<storage, read> filterbank: array<f32>;
@group(0) @binding(2) var<storage, read_write> mel_spec: array<f32>;
struct Params { num_frames: u32, num_freqs: u32, n_mels: u32, _pad: u32, }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn mel_filterbank(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_frames * params.n_mels) { return; }
    let b = idx / params.n_mels;
    let m = idx % params.n_mels;
    var sum: f32 = 0.0;
    for (var f: u32 = 0u; f < params.num_freqs; f = f + 1u) {
        sum = sum + power_spec[b * params.num_freqs + f] * filterbank[m * params.num_freqs + f];
    }
    mel_spec[idx] = sum;
}
";

    public void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec,
        int numFrames, int numFreqs, int nMels)
    {
        if (numFrames <= 0 || numFreqs <= 0 || nMels <= 0) return;
        Dispatch3BufferAsync(FusedCompositesModuleName + ".Mel", MelFilterbankSource, "mel_filterbank",
            powerSpec, filterbank, melSpec, new[] { Bits(numFrames), Bits(numFreqs), Bits(nMels), 0f },
            checked(numFrames * nMels)).GetAwaiter().GetResult();
    }
    public void ExpM1(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (size <= 0) return;
        Dispatch2BufferAsync(FusedCompositesModuleName + ".ExpLog", AccurateExpLogSource, "expm1_accurate", A, B,
            MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void Log1P(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (size <= 0) return;
        Dispatch2BufferAsync(FusedCompositesModuleName + ".ExpLog", AccurateExpLogSource, "log1p_accurate", A, B,
            MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void SplitComplexMultiply(IGpuBuffer aR, IGpuBuffer aI, IGpuBuffer bR, IGpuBuffer bI, IGpuBuffer oR, IGpuBuffer oI, int n)
        => SplitComplexBinary(aR, aI, bR, bI, oR, oI, n, SplitComplexBinaryOp.Multiply);

    public void SplitComplexAdd(IGpuBuffer aR, IGpuBuffer aI, IGpuBuffer bR, IGpuBuffer bI, IGpuBuffer oR, IGpuBuffer oI, int n)
        => SplitComplexBinary(aR, aI, bR, bI, oR, oI, n, SplitComplexBinaryOp.Add);

    public void SplitComplexCrossSpectral(IGpuBuffer xR, IGpuBuffer xI, IGpuBuffer yR, IGpuBuffer yI, IGpuBuffer oR, IGpuBuffer oI, int n)
        => SplitComplexBinary(xR, xI, yR, yI, oR, oI, n, SplitComplexBinaryOp.CrossSpectral);

    public void SplitComplexConjugate(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer oR, IGpuBuffer oI, int n)
        => SplitComplexUnary(iR, iI, oR, oI, n, SplitComplexUnaryOp.Conjugate, 0f);

    public void SplitComplexScale(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer oR, IGpuBuffer oI, float s, int n)
        => SplitComplexUnary(iR, iI, oR, oI, n, SplitComplexUnaryOp.Scale, s);

    public void SplitComplexMagnitudeSquared(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer o, int n)
    {
        if (n <= 0) return;
        DispatchNBufferAsync(FusedCompositesModuleName + ".MagSq", SplitComplexMagnitudeSquaredSource,
            "split_complex_magnitude_squared", new[] { iR, iI, o }, MakeUniform1(n), n).GetAwaiter().GetResult();
    }

    private void SplitComplexBinary(IGpuBuffer aR, IGpuBuffer aI, IGpuBuffer bR, IGpuBuffer bI, IGpuBuffer oR, IGpuBuffer oI,
        int n, SplitComplexBinaryOp op)
    {
        if (n <= 0) return;
        DispatchNBufferAsync(FusedCompositesModuleName + ".Binary", SplitComplexBinarySource, "split_complex_binary",
            new[] { aR, aI, bR, bI, oR, oI }, MakeUniformInts2(n, (int)op), n).GetAwaiter().GetResult();
    }

    private void SplitComplexUnary(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer oR, IGpuBuffer oI, int n, SplitComplexUnaryOp op, float s)
    {
        if (n <= 0) return;
        DispatchNBufferAsync(FusedCompositesModuleName + ".Unary", SplitComplexUnarySource, "split_complex_unary",
            new[] { iR, iI, oR, oI }, new[] { Bits(n), Bits((int)op), s, 0f }, n).GetAwaiter().GetResult();
    }

    public void BatchedDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer result, int batchSize, int vecSize)
    {
        if (batchSize <= 0 || vecSize <= 0) return;
        long totalElements = (long)batchSize * vecSize;
        if (totalElements > a.Size) throw new ArgumentOutOfRangeException(nameof(batchSize), $"batchSize*vecSize ({totalElements}) exceeds buffer A length ({a.Size}).");
        if (totalElements > b.Size) throw new ArgumentOutOfRangeException(nameof(batchSize), $"batchSize*vecSize ({totalElements}) exceeds buffer B length ({b.Size}).");
        if (batchSize > result.Size) throw new ArgumentOutOfRangeException(nameof(batchSize), $"batchSize ({batchSize}) exceeds result length ({result.Size}).");
        // One workgroup per row: the work size is rows times the workgroup size.
        DispatchNBufferAsync(FusedCompositesModuleName + ".Dot", BatchedDotSource, "batched_dot",
            new[] { a, b, result }, MakeUniformInts2(batchSize, vecSize), checked(batchSize * BatchedDotWorkgroupSize))
            .GetAwaiter().GetResult();
    }

    public void CsrSpMMBias(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer denseB, IGpuBuffer bias, IGpuBuffer output, int M, int K, int N, int nnz)
    {
        if (M <= 0 || N <= 0) return;
        DispatchNBufferAsync(FusedCompositesModuleName + ".CsrBias", CsrSpMMBiasSource, "csr_spmm_bias",
            new[] { csrValues, csrColIndices, csrRowPointers, denseB, bias, output },
            new[] { Bits(M), Bits(K), Bits(N), Bits(nnz) }, checked(M * N)).GetAwaiter().GetResult();
    }
}
#endif
