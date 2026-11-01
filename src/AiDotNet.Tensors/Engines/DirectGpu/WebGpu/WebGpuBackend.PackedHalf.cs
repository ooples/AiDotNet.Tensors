// Copyright (c) CrowdTrainer LLC. All rights reserved.
// FP16 buffers in the interface's layout: IEEE binary16, two to a 32-bit word, low half first -- the AllocateByteBuffer
// (elementCount * 2) buffers the engine and every other backend use, and the layout the shared FP16 batch kernels
// (fused decode linear, FFT storage) already read. This backend used to keep one f32 per "half" (its value truncated to
// a 10-bit mantissa), so a contract-sized buffer held only half the elements and everything past them was dropped.
// Conversion to binary16 rounds to nearest even, as CUDA's __float2half and PyTorch do.

#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    private const string PackedHalfModuleName = "PackedHalf";

    /// <summary>Where half_gemm writes: FP32 row-major, binary16 row-major, or FP32 in NCHW image order.</summary>
    private enum HalfGemmOutput { Float32 = 0, Half = 1, Float32Nchw = 2 }

    /// <summary>The packed element-wise half operations.</summary>
    private enum HalfElementwiseOp { Relu = 0, Gelu = 1, Add = 2 }

    // binary16 <-> f32 by bits: f32_to_half rounds to nearest even (overflow to inf, gradual underflow, NaN kept quiet);
    // half_to_f32 is exact (every binary16 is an f32).
    private const string HalfConversionFunctions = @"
fn f32_to_half(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    let sign = (b >> 16u) & 0x8000u;
    let e = i32((b >> 23u) & 0xFFu);
    let m = b & 0x7FFFFFu;
    if (e == 255) { return sign | 0x7C00u | select(0u, 0x200u | (m >> 13u), m != 0u); }
    let he = e - 112;
    if (he >= 31) { return sign | 0x7C00u; }
    if (he <= 0) {
        if (he < -10) { return sign; }
        let full = m | 0x800000u;
        let shift = u32(14 - he);
        var hm = full >> shift;
        let rem = full & ((1u << shift) - 1u);
        let halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (hm & 1u) == 1u)) { hm = hm + 1u; }
        return sign | hm;
    }
    var r = sign | (u32(he) << 10u) | (m >> 13u);
    let rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u) == 1u)) { r = r + 1u; }
    return r;
}

fn half_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u) << 16u;
    let e = (h >> 10u) & 0x1Fu;
    let m = h & 0x3FFu;
    if (e == 0u) { return select(1.0, -1.0, sign != 0u) * f32(m) * 5.9604644775390625e-8; }
    if (e == 31u) { return bitcast<f32>(sign | 0x7F800000u | (m << 13u)); }
    return bitcast<f32>(sign | ((e + 112u) << 23u) | (m << 13u));
}
";

    private const string HalfConvertSource = @"
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination: array<u32>;
struct ConvertParams { count: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(2) var<uniform> params: ConvertParams;
" + HalfConversionFunctions + @"
// One invocation per word: two f32 in, one word of two halves out (a padding half past an odd count is zero).
@compute @workgroup_size(256)
fn half_pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let i = w * 2u;
    if (i >= params.count) { return; }
    let lo = f32_to_half(bitcast<f32>(source[i]));
    var hi = 0u;
    if (i + 1u < params.count) { hi = f32_to_half(bitcast<f32>(source[i + 1u])); }
    destination[w] = lo | (hi << 16u);
}

@compute @workgroup_size(256)
fn half_unpack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    destination[i] = bitcast<u32>(half_to_f32((source[i / 2u] >> ((i % 2u) * 16u)) & 0xFFFFu));
}
";

    // C = op(A) * op(B) over binary16 operands, accumulated in f32. One invocation per output word: the two outputs at flat
    // indices 2w and 2w + 1, so a binary16 output word is written whole by one invocation. op(A)[i, k] is A[k * M + i]
    // when transA (A stored [K, M]) and A[i * K + k] otherwise; op(B)[k, j] is B[j * K + k] when transB (B stored
    // [N, K]) and B[k * N + j] otherwise. Float32Nchw writes C[(j / P) * M + i) * P + j % P] (columns are image-major
    // pixels), the layout of GemmFp16ConvBatched.
    private const string HalfGemmSource = @"
@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<u32>;
struct HalfGemmParams { M: u32, N: u32, K: u32, transA: u32, transB: u32, output_mode: u32, P: u32, _p0: u32, }
@group(0) @binding(3) var<uniform> params: HalfGemmParams;
" + HalfConversionFunctions + @"
fn a_at(i: u32) -> f32 { return half_to_f32((A[i / 2u] >> ((i % 2u) * 16u)) & 0xFFFFu); }
fn b_at(i: u32) -> f32 { return half_to_f32((B[i / 2u] >> ((i % 2u) * 16u)) & 0xFFFFu); }

fn dot_at(flat: u32) -> f32 {
    let i = flat / params.N;
    let j = flat % params.N;
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        let a = select(a_at(i * params.K + k), a_at(k * params.M + i), params.transA != 0u);
        let b = select(b_at(k * params.N + j), b_at(j * params.K + k), params.transB != 0u);
        acc = acc + a * b;
    }
    return acc;
}

fn store_f32(flat: u32, v: f32) {
    if (params.output_mode == 2u) {
        let i = flat / params.N;
        let j = flat % params.N;
        C[((j / params.P) * params.M + i) * params.P + j % params.P] = bitcast<u32>(v);
    } else {
        C[flat] = bitcast<u32>(v);
    }
}

@compute @workgroup_size(256)
fn half_gemm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let first = w * 2u;
    let total = params.M * params.N;
    if (first >= total) { return; }
    let v0 = dot_at(first);
    let has_second = first + 1u < total;
    var v1: f32 = 0.0;
    if (has_second) { v1 = dot_at(first + 1u); }
    if (params.output_mode == 1u) {
        C[w] = f32_to_half(v0) | (select(0u, f32_to_half(v1), has_second) << 16u);
    } else {
        store_f32(first, v0);
        if (has_second) { store_f32(first + 1u, v1); }
    }
}
";

    // Relu / Gelu / Add over packed halves, computed in f32 (Gelu is WebGpuKernels' tanh form), one word per invocation.
    private const string HalfElementwiseSource = @"
@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var<storage, read> Y: array<u32>;
@group(0) @binding(2) var<storage, read_write> Z: array<u32>;
struct HalfElementwiseParams { count: u32, op: u32, _p0: u32, _p1: u32, }
@group(0) @binding(3) var<uniform> params: HalfElementwiseParams;
" + HalfConversionFunctions + @"
fn apply(x: f32, y: f32) -> f32 {
    if (params.op == 0u) { return max(x, 0.0); }
    if (params.op == 1u) {
        let t = tanh(clamp(0.7978845608 * (x + 0.044715 * x * x * x), -10.0, 10.0));
        return 0.5 * x * (1.0 + t);
    }
    return x + y;
}

@compute @workgroup_size(256)
fn half_elementwise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w * 2u >= params.count) { return; }
    let x = X[w];
    let y = select(0u, Y[w], params.op == 2u);
    let lo = f32_to_half(apply(half_to_f32(x & 0xFFFFu), half_to_f32(y & 0xFFFFu)));
    var hi = 0u;
    if (w * 2u + 1u < params.count) { hi = f32_to_half(apply(half_to_f32(x >> 16u), half_to_f32(y >> 16u))); }
    Z[w] = lo | (hi << 16u);
}
";

    // WebGpuKernels.Im2colKnFp16Source in the packed layout: the transposed [K, N] column matrix, one word (two
    // consecutive col elements) per invocation.
    private const string HalfIm2colSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
struct Im2colParams {
    batch: u32, channels: u32, height: u32, width: u32, kernel_h: u32, kernel_w: u32, stride_h: u32, stride_w: u32,
    pad_h: u32, pad_w: u32, dilation_h: u32, dilation_w: u32, out_h: u32, out_w: u32, _p0: u32, _p1: u32,
}
@group(0) @binding(2) var<uniform> params: Im2colParams;
" + HalfConversionFunctions + @"
fn col_at(t: u32) -> f32 {
    let n_total = params.batch * params.out_h * params.out_w;
    let n = t % n_total;
    let k = t / n_total;
    let hw = params.out_h * params.out_w;
    let b = n / hw;
    let rem = n % hw;
    let oh = rem / params.out_w;
    let ow = rem % params.out_w;
    let kw = k % params.kernel_w;
    let kh = (k / params.kernel_w) % params.kernel_h;
    let c = k / (params.kernel_w * params.kernel_h);
    let ih = i32(oh * params.stride_h) - i32(params.pad_h) + i32(kh * params.dilation_h);
    let iw = i32(ow * params.stride_w) - i32(params.pad_w) + i32(kw * params.dilation_w);
    if (ih < 0 || ih >= i32(params.height) || iw < 0 || iw >= i32(params.width)) { return 0.0; }
    return input[((b * params.channels + c) * params.height + u32(ih)) * params.width + u32(iw)];
}

@compute @workgroup_size(256)
fn half_im2col_kn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let total = params.batch * params.out_h * params.out_w * params.channels * params.kernel_h * params.kernel_w;
    let first = w * 2u;
    if (first >= total) { return; }
    var hi = 0u;
    if (first + 1u < total) { hi = f32_to_half(col_at(first + 1u)); }
    output[w] = f32_to_half(col_at(first)) | (hi << 16u);
}
";

    private static int HalfWords(int count) => (count + 1) / 2;

    public void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size)
    {
        if (size <= 0) return;
        DispatchNBufferAsync(PackedHalfModuleName + ".Convert", HalfConvertSource, "half_pack",
            new[] { input, output }, new[] { Bits(size), 0f, 0f, 0f }, HalfWords(size)).GetAwaiter().GetResult();
    }

    public void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
    {
        if (size <= 0) return;
        DispatchNBufferAsync(PackedHalfModuleName + ".Convert", HalfConvertSource, "half_unpack",
            new[] { input, output }, new[] { Bits(size), 0f, 0f, 0f }, size).GetAwaiter().GetResult();
    }

    private void HalfGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k, bool transA, bool transB,
        HalfGemmOutput output, int pixels = 1)
    {
        int total = checked(m * n);
        DispatchNBufferAsync(PackedHalfModuleName + ".Gemm", HalfGemmSource, "half_gemm",
            new[] { a, b, c },
            new[] { Bits(m), Bits(n), Bits(k), Bits(transA ? 1 : 0), Bits(transB ? 1 : 0), Bits((int)output), Bits(pixels), 0f },
            HalfWords(total)).GetAwaiter().GetResult();
    }

    private void HalfElementwise(HalfElementwiseOp op, IGpuBuffer x, IGpuBuffer y, IGpuBuffer z, int count)
        => DispatchNBufferAsync(PackedHalfModuleName + ".Elementwise", HalfElementwiseSource, "half_elementwise",
            new[] { x, y, z }, new[] { Bits(count), Bits((int)op), 0f, 0f }, HalfWords(count)).GetAwaiter().GetResult();
}
#endif
