// Copyright (c) CrowdTrainer LLC. All rights reserved.
// One kernel each for GEMM and FFT members that were compositions of other backend members (TENS0008, #1638): the
// swish and leaky-ReLU GEMM epilogues (a GEMM into a temporary, then the activation), the 2:4 sparse GEMMs (a dense
// decompression of A, then a dense GEMM), the real FFT (a zero-filled imaginary buffer, then a complex transform that
// wrote n bins into buffers sized for n/2 + 1), the 2-D FFT (row transforms, four transposes and four temporaries) and
// the FP16 GEMM (an FP32 GEMM into a temporary, then a rounding pass).

#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    private const string FusedGemmFftModuleName = "FusedGemmFft";

    /// <summary>Selector of the gemm_bias_epilogue entry point's activation.</summary>
    private enum GemmEpilogue { Swish = 0, LeakyRelu = 1 }

    private const int HgemmTile = 16;

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

    // C = alpha * A * B + beta * C with A in the 2:4 layout of WebGpuKernels.Sparse2x4Source: per row, K / 2 kept values,
    // each with its offset inside its block of four (u32 bits). Only the kept values are multiplied; C is not read when
    // beta is zero, as in a BLAS GEMM.
    private const string Sparse24GemmSource = @"
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> offsets: array<f32>;
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
    let kept = params.K / 2u;
    var acc: f32 = 0.0;
    for (var s: u32 = 0u; s < kept; s = s + 1u) {
        let v = row * kept + s;
        let k = (s / 2u) * 4u + bitcast<u32>(offsets[v]);
        acc = acc + values[v] * B[k * params.N + col];
    }
    if (params.beta == 0.0) {
        C[idx] = params.alpha * acc;
    } else {
        C[idx] = params.alpha * acc + params.beta * C[idx];
    }
}
";

    // max(0, A * B + bias) with A in the same 2:4 layout.
    private const string Sparse24GemmBiasReluSource = @"
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> offsets: array<f32>;
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
    let kept = params.K / 2u;
    var acc: f32 = 0.0;
    for (var s: u32 = 0u; s < kept; s = s + 1u) {
        let v = row * kept + s;
        let k = (s / 2u) * 4u + bitcast<u32>(offsets[v]);
        acc = acc + values[v] * B[k * params.N + col];
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

    // FP16 GEMM: 16 x 16 output tiles staged through workgroup memory, accumulated in f32, rounded once to FP16 precision
    // with WebGpuKernels' f16_trunc (the rounding of ConvertToFp16). One workgroup per tile.
    private const string HgemmSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
struct HgemmParams { M: u32, N: u32, K: u32, tiles_n: u32, }
@group(0) @binding(3) var<uniform> params: HgemmParams;
var<workgroup> a_tile: array<f32, 256>;
var<workgroup> b_tile: array<f32, 256>;

fn f16_trunc(val: f32) -> f32 {
    let bits = bitcast<u32>(val);
    let sign = (bits >> 31u) & 1u;
    let exp_bits = (bits >> 23u) & 0xFFu;
    let mantissa = bits & 0x7FFFFFu;
    if (exp_bits == 0xFFu) { return val; }
    let exp_val: i32 = i32(exp_bits) - 127;
    if (exp_val < -14) { return select(-0.0, 0.0, sign == 0u); }
    if (exp_val > 15) { return bitcast<f32>((sign << 31u) | 0x7F800000u); }
    return bitcast<f32>((sign << 31u) | (exp_bits << 23u) | (mantissa & 0x7FE000u));
}

@compute @workgroup_size(256)
fn hgemm_tiled(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_row = wid.x / params.tiles_n;
    let tile_col = wid.x % params.tiles_n;
    let ly = lid.x / 16u;
    let lx = lid.x % 16u;
    let row = tile_row * 16u + ly;
    let col = tile_col * 16u + lx;
    var acc: f32 = 0.0;
    let steps = (params.K + 15u) / 16u;
    for (var t: u32 = 0u; t < steps; t = t + 1u) {
        let ak = t * 16u + lx;
        let bk = t * 16u + ly;
        a_tile[lid.x] = select(0.0, A[row * params.K + ak], row < params.M && ak < params.K);
        b_tile[lid.x] = select(0.0, B[bk * params.N + col], bk < params.K && col < params.N);
        workgroupBarrier();
        for (var i: u32 = 0u; i < 16u; i = i + 1u) { acc = acc + a_tile[ly * 16u + i] * b_tile[i * 16u + lx]; }
        workgroupBarrier();
    }
    if (row < params.M && col < params.N) { C[row * params.N + col] = f16_trunc(acc); }
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
        int tilesN = (n + HgemmTile - 1) / HgemmTile;
        int tiles = checked(((m + HgemmTile - 1) / HgemmTile) * tilesN);
        DispatchNBufferAsync(FusedGemmFftModuleName + ".Hgemm", HgemmSource, "hgemm_tiled",
            new[] { aFp16, bFp16, cFp16 }, new[] { Bits(m), Bits(n), Bits(k), Bits(tilesN) },
            checked(tiles * HgemmTile * HgemmTile)).GetAwaiter().GetResult();
    }
}
#endif
