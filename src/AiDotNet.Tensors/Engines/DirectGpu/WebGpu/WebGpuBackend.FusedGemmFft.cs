// Copyright (c) CrowdTrainer LLC. All rights reserved.
// One kernel each for GEMM and FFT members that were compositions of other backend members (TENS0008, #1638): the
// swish and leaky-ReLU GEMM epilogues (a GEMM into a temporary, then the activation), the 2:4 sparse GEMMs (a dense
// decompression of A, then a dense GEMM), the real FFT (a zero-filled imaginary buffer, then a complex transform that
// wrote n bins into buffers sized for n/2 + 1), the 2-D FFT (row transforms, four transposes and four temporaries).

#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    private const string FusedGemmFftModuleName = "FusedGemmFft";

    /// <summary>Selector of the gemm_bias_epilogue entry point's activation.</summary>
    private enum GemmEpilogue { Swish = 0, LeakyRelu = 1 }

    // WebGpuKernels.FusedGemmBiasSource with the two activations it lacks; swish is its silu, leaky ReLU its leaky_relu.
    private const string GemmBiasEpilogueSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
struct EpilogueParams { M: u32, N: u32, K: u32, activation: u32, alpha: f32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(4) var<uniform> params: EpilogueParams;

@compute @workgroup_size(256)
fn gemm_bias_epilogue(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.M * params.N) { return; }
    let row = idx / params.N;
    let col = idx % params.N;
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        acc = acc + A[row * params.K + k] * B[k * params.N + col];
    }
    let val = acc + bias[col];
    if (params.activation == 0u) {
        output[idx] = val / (1.0 + exp(-val));
    } else {
        output[idx] = select(params.alpha * val, val, val > 0.0);
    }
}
";

    // C = alpha * A * B + beta * C with A in the interface's 2:4 layout (Sparse24LayoutSource). Only the kept values are
    // multiplied; C is not read when beta is zero, as in a BLAS GEMM.
    private const string Sparse24GemmSource = @"
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> packed: array<u32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
struct SparseParams { M: u32, N: u32, K: u32, _p0: u32, alpha: f32, beta: f32, _p1: u32, _p2: u32, }
@group(0) @binding(4) var<uniform> params: SparseParams;

@compute @workgroup_size(256)
fn sparse24_gemm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.M * params.N) { return; }
    let row = idx / params.N;
    let col = idx % params.N;
    let blocks = params.K / 4u;
    var acc: f32 = 0.0;
    for (var blk: u32 = 0u; blk < blocks; blk = blk + 1u) {
        let g = row * blocks + blk;
        let offsets = (packed[g / 4u] >> ((g % 4u) * 8u)) & 255u;
        acc = acc + values[g * 2u] * B[(blk * 4u + (offsets & 3u)) * params.N + col];
        acc = acc + values[g * 2u + 1u] * B[(blk * 4u + ((offsets >> 2u) & 3u)) * params.N + col];
    }
    if (params.beta == 0.0) {
        C[idx] = params.alpha * acc;
    } else {
        C[idx] = params.alpha * acc + params.beta * C[idx];
    }
}
";

    // The interface's 2:4 layout: per block of four, the two largest magnitudes (values, K / 2 per row) and one byte whose
    // low two bit pairs are their offsets, lower offset first -- four blocks to a u32 word of the byte buffer. The
    // selection is OpenCL's enforce_2_4_sparsity. (WebGpuKernels.Sparse2x4Source wrote one u32 per kept value, eight
    // times the M * K / 4 bytes the contract allocates.) One invocation per word, so no two write the same word.
    private const string Sparse24LayoutSource = @"
@group(0) @binding(0) var<storage, read> dense: array<f32>;
@group(0) @binding(1) var<storage, read_write> values: array<f32>;
@group(0) @binding(2) var<storage, read_write> packed: array<u32>;
struct LayoutParams { groups: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(3) var<uniform> params: LayoutParams;

@compute @workgroup_size(256)
fn sparse24_enforce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let first = w * 4u;
    if (first >= params.groups) { return; }
    var word = 0u;
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
        let g = first + q;
        if (g >= params.groups) { break; }
        var v: array<f32, 4>;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) { v[i] = dense[g * 4u + i]; }
        var i0 = 0u;
        var i1 = 1u;
        var m0 = abs(v[0]);
        var m1 = abs(v[1]);
        if (m1 > m0) { i0 = 1u; i1 = 0u; let t = m0; m0 = m1; m1 = t; }
        for (var i: u32 = 2u; i < 4u; i = i + 1u) {
            let a = abs(v[i]);
            if (a > m0) { i1 = i0; m1 = m0; i0 = i; m0 = a; }
            else if (a > m1) { i1 = i; m1 = a; }
        }
        if (i0 > i1) { let t = i0; i0 = i1; i1 = t; }
        values[g * 2u] = v[i0];
        values[g * 2u + 1u] = v[i1];
        word = word | (((i1 << 2u) | i0) << (q * 8u));
    }
    packed[w] = word;
}
";

    private const string Sparse24DecompressSource = @"
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> dense: array<f32>;
struct LayoutParams { count: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(3) var<uniform> params: LayoutParams;

@compute @workgroup_size(256)
fn sparse24_decompress(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    let g = idx / 4u;
    let lane = idx % 4u;
    let offsets = (packed[g / 4u] >> ((g % 4u) * 8u)) & 255u;
    var x: f32 = 0.0;
    if (lane == (offsets & 3u)) { x = values[g * 2u]; }
    else if (lane == ((offsets >> 2u) & 3u)) { x = values[g * 2u + 1u]; }
    dense[idx] = x;
}
";

    // max(0, A * B + bias) with A in the same 2:4 layout.
    private const string Sparse24GemmBiasReluSource = @"
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> packed: array<u32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
struct SparseParams { M: u32, N: u32, K: u32, _p0: u32, }
@group(0) @binding(5) var<uniform> params: SparseParams;

@compute @workgroup_size(256)
fn sparse24_gemm_bias_relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.M * params.N) { return; }
    let row = idx / params.N;
    let col = idx % params.N;
    let blocks = params.K / 4u;
    var acc: f32 = 0.0;
    for (var blk: u32 = 0u; blk < blocks; blk = blk + 1u) {
        let g = row * blocks + blk;
        let offsets = (packed[g / 4u] >> ((g % 4u) * 8u)) & 255u;
        acc = acc + values[g * 2u] * B[(blk * 4u + (offsets & 3u)) * params.N + col];
        acc = acc + values[g * 2u + 1u] * B[(blk * 4u + ((offsets >> 2u) & 3u)) * params.N + col];
    }
    output[idx] = max(0.0, acc + bias[col]);
}
";

    // The n / 2 + 1 non-negative-frequency bins of a real input. The twiddle index k * j is kept modulo n as it advances,
    // so neither the product nor the angle loses precision for a large n.
    private const string RealDftSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> outRe: array<f32>;
@group(0) @binding(2) var<storage, read_write> outIm: array<f32>;
struct RealDftParams { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(3) var<uniform> params: RealDftParams;

@compute @workgroup_size(256)
fn real_dft(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let n = params.n;
    if (k > n / 2u) { return; }
    let step = -6.283185307179586 / f32(n);
    var re: f32 = 0.0;
    var im: f32 = 0.0;
    var phase: u32 = 0u;
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        let a = step * f32(phase);
        let x = input[j];
        re = re + x * cos(a);
        im = im + x * sin(a);
        phase = phase + k;
        if (phase >= n) { phase = phase - n; }
    }
    outRe[k] = re;
    outIm[k] = im;
}
";

    // Radix-2 transforms of strided lines (a row: element stride 1, line stride width; a column: element stride width,
    // line stride 1), so a 2-D FFT runs its column transforms where the rows left them, without transposes. The
    // arithmetic is that of WebGpuBatchExecutionKernels' bx_fft_bit_reverse and bx_fft_butterfly.
    private const string StridedFftParams = @"
struct StridedFftParams { lines: u32, n: u32, elem_stride: u32, line_stride: u32, log2n: u32, inverse: u32, scale: f32, span: u32, }
";

    private const string StridedFftBitReverseSource = @"
@group(0) @binding(0) var<storage, read> inRe: array<f32>;
@group(0) @binding(1) var<storage, read> inIm: array<f32>;
@group(0) @binding(2) var<storage, read_write> outRe: array<f32>;
@group(0) @binding(3) var<storage, read_write> outIm: array<f32>;
" + StridedFftParams + @"
@group(0) @binding(4) var<uniform> params: StridedFftParams;

fn reverse_bits(k0: u32, bits: u32) -> u32 {
    var k = k0;
    var r = 0u;
    for (var b: u32 = 0u; b < bits; b = b + 1u) { r = (r << 1u) | (k & 1u); k = k >> 1u; }
    return r;
}

@compute @workgroup_size(256)
fn strided_fft_bit_reverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.lines * params.n) { return; }
    let line = i / params.n;
    let k = i % params.n;
    let base = line * params.line_stride;
    let dst = base + reverse_bits(k, params.log2n) * params.elem_stride;
    let src = base + k * params.elem_stride;
    outRe[dst] = inRe[src] * params.scale;
    outIm[dst] = inIm[src] * params.scale;
}
";

    // In place: the lower index of each reversed pair swaps both and scales both; a self-paired index scales itself.
    private const string StridedFftInPlaceSource = @"
@group(0) @binding(0) var<storage, read_write> re: array<f32>;
@group(0) @binding(1) var<storage, read_write> im: array<f32>;
" + StridedFftParams + @"
@group(0) @binding(2) var<uniform> params: StridedFftParams;

fn reverse_bits(k0: u32, bits: u32) -> u32 {
    var k = k0;
    var r = 0u;
    for (var b: u32 = 0u; b < bits; b = b + 1u) { r = (r << 1u) | (k & 1u); k = k >> 1u; }
    return r;
}

@compute @workgroup_size(256)
fn strided_fft_swap_reverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.lines * params.n) { return; }
    let line = i / params.n;
    let k = i % params.n;
    let r = reverse_bits(k, params.log2n);
    if (r < k) { return; }
    let base = line * params.line_stride;
    let a = base + k * params.elem_stride;
    let b = base + r * params.elem_stride;
    let ar = re[a]; let ai = im[a];
    re[a] = re[b] * params.scale; im[a] = im[b] * params.scale;
    re[b] = ar * params.scale; im[b] = ai * params.scale;
}

@compute @workgroup_size(256)
fn strided_fft_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let pairs = params.n / 2u;
    if (i >= params.lines * pairs) { return; }
    let line = i / pairs;
    let j = i % pairs;
    let span = params.span;
    let pos = j % span;
    let first = (j / span) * 2u * span + pos;
    let i0 = line * params.line_stride + first * params.elem_stride;
    let i1 = i0 + span * params.elem_stride;
    let a = select(-3.14159265358979323846, 3.14159265358979323846, params.inverse != 0u) * f32(pos) / f32(span);
    let wr = cos(a); let wi = sin(a);
    let xr = re[i1]; let xi = im[i1];
    let tr = wr * xr - wi * xi; let ti = wr * xi + wi * xr;
    let ur = re[i0]; let ui = im[i0];
    re[i0] = ur + tr; im[i0] = ui + ti;
    re[i1] = ur - tr; im[i1] = ui - ti;
}
";

    // A direct DFT of strided lines, for a length that is not a power of two (out-of-place, same layout as the input).
    private const string StridedDftSource = @"
@group(0) @binding(0) var<storage, read> inRe: array<f32>;
@group(0) @binding(1) var<storage, read> inIm: array<f32>;
@group(0) @binding(2) var<storage, read_write> outRe: array<f32>;
@group(0) @binding(3) var<storage, read_write> outIm: array<f32>;
" + StridedFftParams + @"
@group(0) @binding(4) var<uniform> params: StridedFftParams;

@compute @workgroup_size(256)
fn strided_dft(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.lines * params.n) { return; }
    let n = params.n;
    let line = i / n;
    let k = i % n;
    let base = line * params.line_stride;
    let step = select(-6.283185307179586, 6.283185307179586, params.inverse != 0u) / f32(n);
    var re: f32 = 0.0;
    var im: f32 = 0.0;
    var phase: u32 = 0u;
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        let a = step * f32(phase);
        let c = cos(a); let s = sin(a);
        let src = base + j * params.elem_stride;
        let xr = inRe[src]; let xi = inIm[src];
        re = re + xr * c - xi * s;
        im = im + xr * s + xi * c;
        phase = phase + k;
        if (phase >= n) { phase = phase - n; }
    }
    let dst = base + k * params.elem_stride;
    outRe[dst] = re * params.scale;
    outIm[dst] = im * params.scale;
}
";

    public IGpuBuffer GemmBiasSwish(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasEpilogueKernel(A, B, bias, M, N, K, GemmEpilogue.Swish, 0f);

    public IGpuBuffer GemmBiasLeakyRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K, float alpha = 0.01f)
        => GemmBiasEpilogueKernel(A, B, bias, M, N, K, GemmEpilogue.LeakyRelu, alpha);

    private IGpuBuffer GemmBiasEpilogueKernel(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K,
        GemmEpilogue epilogue, float alpha)
    {
        var output = AllocateBuffer(checked(M * N));
        DispatchNBufferAsync(FusedGemmFftModuleName + ".Epilogue", GemmBiasEpilogueSource, "gemm_bias_epilogue",
            new[] { A, B, bias, output }, new[] { Bits(M), Bits(N), Bits(K), Bits((int)epilogue), alpha, 0f, 0f, 0f },
            M * N).GetAwaiter().GetResult();
        return output;
    }

    public void Enforce2x4Sparsity(IGpuBuffer denseInput, IGpuBuffer sparseValues, IGpuBuffer sparseIndices, int M, int K)
    {
        ValidateSparse24(M, 1, K);
        int groups = checked(M * (K / 4));
        DispatchNBufferAsync(FusedGemmFftModuleName + ".Sparse24Layout", Sparse24LayoutSource, "sparse24_enforce",
            new[] { denseInput, sparseValues, sparseIndices }, new[] { Bits(groups), 0f, 0f, 0f }, (groups + 3) / 4)
            .GetAwaiter().GetResult();
    }

    public void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
    {
        ValidateSparse24(M, 1, K);
        int count = checked(M * K);
        DispatchNBufferAsync(FusedGemmFftModuleName + ".Sparse24Decompress", Sparse24DecompressSource, "sparse24_decompress",
            new[] { sparseValues, sparseIndices, denseOutput }, new[] { Bits(count), 0f, 0f, 0f }, count)
            .GetAwaiter().GetResult();
    }

    public void SparseGemm(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, float alpha = 1f, float beta = 0f)
    {
        EnsureInitialized();
        ValidateSparse24(M, N, K);
        DispatchNBufferAsync(FusedGemmFftModuleName + ".Sparse", Sparse24GemmSource, "sparse24_gemm",
            new[] { sparseAValues, sparseAIndices, B, C }, new[] { Bits(M), Bits(N), Bits(K), 0f, alpha, beta, 0f, 0f },
            checked(M * N)).GetAwaiter().GetResult();
    }

    public IGpuBuffer SparseGemmBiasRelu(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer bias,
        int M, int N, int K)
    {
        EnsureInitialized();
        ValidateSparse24(M, N, K);
        var output = AllocateBuffer(checked(M * N));
        DispatchNBufferAsync(FusedGemmFftModuleName + ".SparseBiasRelu", Sparse24GemmBiasReluSource, "sparse24_gemm_bias_relu",
            new[] { sparseAValues, sparseAIndices, B, bias, output }, new[] { Bits(M), Bits(N), Bits(K), 0f },
            M * N).GetAwaiter().GetResult();
        return output;
    }

    private static void ValidateSparse24(int M, int N, int K)
    {
        if (M <= 0) throw new ArgumentOutOfRangeException(nameof(M), "M must be positive.");
        if (N <= 0) throw new ArgumentOutOfRangeException(nameof(N), "N must be positive.");
        if (K <= 0 || K % 4 != 0) throw new ArgumentException($"K ({K}) must be a positive multiple of 4 for 2:4 sparsity.", nameof(K));
    }

    public void RFFT(IGpuBuffer input, IGpuBuffer outputReal, IGpuBuffer outputImag, int n)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "n must be positive.");
        DispatchNBufferAsync(FusedGemmFftModuleName + ".RealDft", RealDftSource, "real_dft",
            new[] { input, outputReal, outputImag }, new[] { Bits(n), 0f, 0f, 0f }, n / 2 + 1).GetAwaiter().GetResult();
    }

    public void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag,
        int height, int width, bool inverse)
    {
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), "height must be positive.");
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "width must be positive.");
        int count = checked(height * width);
        bool powerOfTwo = (height & (height - 1)) == 0 && (width & (width - 1)) == 0;
        if (!powerOfTwo)
        {
            // Rows into a temporary, then columns into the output: any aliasing of input and output is safe.
            using var rowsRe = AllocateBuffer(count);
            using var rowsIm = AllocateBuffer(count);
            StridedDft(inputReal, inputImag, rowsRe, rowsIm, height, width, 1, width, inverse);
            StridedDft(rowsRe, rowsIm, outputReal, outputImag, width, height, width, 1, inverse);
            return;
        }

        bool inPlace = ReferenceEquals(inputReal, outputReal) && ReferenceEquals(inputImag, outputImag);
        bool aliased = ReferenceEquals(inputReal, outputReal) || ReferenceEquals(inputImag, outputImag)
            || ReferenceEquals(inputReal, outputImag) || ReferenceEquals(inputImag, outputReal);
        if (inPlace)
        {
            StridedRadix2(outputReal, outputImag, height, width, 1, width, inverse, reverseInPlace: true);
        }
        else if (aliased)
        {
            using var re = AllocateBuffer(count);
            using var im = AllocateBuffer(count);
            Copy(inputReal, re, count);
            Copy(inputImag, im, count);
            StridedBitReverse(re, im, outputReal, outputImag, height, width, 1, width, inverse);
            StridedButterflies(outputReal, outputImag, height, width, 1, width, inverse);
        }
        else
        {
            StridedBitReverse(inputReal, inputImag, outputReal, outputImag, height, width, 1, width, inverse);
            StridedButterflies(outputReal, outputImag, height, width, 1, width, inverse);
        }
        StridedRadix2(outputReal, outputImag, width, height, width, 1, inverse, reverseInPlace: true);
    }

    private static int Log2(int n)
    {
        int log2n = 0;
        while ((1 << log2n) < n) log2n++;
        return log2n;
    }

    private static float[] StridedFftUniforms(int lines, int n, int elemStride, int lineStride, bool inverse, float scale, int span)
        => new[] { Bits(lines), Bits(n), Bits(elemStride), Bits(lineStride), Bits(Log2(n)), Bits(inverse ? 1 : 0), scale, Bits(span) };

    private void StridedBitReverse(IGpuBuffer inRe, IGpuBuffer inIm, IGpuBuffer outRe, IGpuBuffer outIm,
        int lines, int n, int elemStride, int lineStride, bool inverse)
        => DispatchNBufferAsync(FusedGemmFftModuleName + ".BitReverse", StridedFftBitReverseSource, "strided_fft_bit_reverse",
            new[] { inRe, inIm, outRe, outIm }, StridedFftUniforms(lines, n, elemStride, lineStride, inverse, inverse ? 1f / n : 1f, 0),
            checked(lines * n)).GetAwaiter().GetResult();

    private void StridedRadix2(IGpuBuffer re, IGpuBuffer im, int lines, int n, int elemStride, int lineStride, bool inverse,
        bool reverseInPlace)
    {
        if (reverseInPlace)
        {
            DispatchNBufferAsync(FusedGemmFftModuleName + ".InPlace", StridedFftInPlaceSource, "strided_fft_swap_reverse",
                new[] { re, im }, StridedFftUniforms(lines, n, elemStride, lineStride, inverse, inverse ? 1f / n : 1f, 0),
                checked(lines * n)).GetAwaiter().GetResult();
        }
        StridedButterflies(re, im, lines, n, elemStride, lineStride, inverse);
    }

    private void StridedButterflies(IGpuBuffer re, IGpuBuffer im, int lines, int n, int elemStride, int lineStride, bool inverse)
    {
        for (int span = 1; span < n; span <<= 1)
        {
            DispatchNBufferAsync(FusedGemmFftModuleName + ".InPlace", StridedFftInPlaceSource, "strided_fft_butterfly",
                new[] { re, im }, StridedFftUniforms(lines, n, elemStride, lineStride, inverse, 1f, span),
                checked(lines * (n / 2))).GetAwaiter().GetResult();
        }
    }

    private void StridedDft(IGpuBuffer inRe, IGpuBuffer inIm, IGpuBuffer outRe, IGpuBuffer outIm,
        int lines, int n, int elemStride, int lineStride, bool inverse)
        => DispatchNBufferAsync(FusedGemmFftModuleName + ".Dft", StridedDftSource, "strided_dft",
            new[] { inRe, inIm, outRe, outIm }, StridedFftUniforms(lines, n, elemStride, lineStride, inverse, inverse ? 1f / n : 1f, 0),
            checked(lines * n)).GetAwaiter().GetResult();

    /// <inheritdoc/>
    public void Hgemm(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp16, int m, int n, int k)
    {
        ValidateHalfGemmArgs(aFp16, bFp16, cFp16, m, n, k);
        HalfGemm(aFp16, bFp16, cFp16, m, n, k, transA: false, transB: false, HalfGemmOutput.Half);
    }
}
#endif
