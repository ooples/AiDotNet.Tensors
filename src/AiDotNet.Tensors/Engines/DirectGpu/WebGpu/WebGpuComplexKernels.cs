// Copyright (c) AiDotNet. All rights reserved.
// WebGPU WGSL compute shaders for native Complex<T> tensor operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static class WebGpuComplexKernels
{
    public static string GetSource() => @"
// Complex multiply: (a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re)
@group(0) @binding(0) var<storage, read> aReal: array<f32>;
@group(0) @binding(1) var<storage, read> aImag: array<f32>;
@group(0) @binding(2) var<storage, read> bReal: array<f32>;
@group(0) @binding(3) var<storage, read> bImag: array<f32>;
@group(0) @binding(4) var<storage, read_write> outReal: array<f32>;
@group(0) @binding(5) var<storage, read_write> outImag: array<f32>;
struct ComplexParams { size: u32, }
@group(0) @binding(6) var<uniform> cparams: ComplexParams;

@compute @workgroup_size(256)
fn split_complex_multiply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cparams.size) { return; }
    let ar = aReal[idx]; let ai = aImag[idx];
    let br = bReal[idx]; let bi = bImag[idx];
    outReal[idx] = ar * br - ai * bi;
    outImag[idx] = ar * bi + ai * br;
}

// Complex conjugate: (re, -im)
@compute @workgroup_size(256)
fn split_complex_conjugate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cparams.size) { return; }
    outReal[idx] = aReal[idx];
    outImag[idx] = -aImag[idx];
}

// Complex magnitude: sqrt(re^2 + im^2)
@group(1) @binding(0) var<storage, read> mInReal: array<f32>;
@group(1) @binding(1) var<storage, read> mInImag: array<f32>;
@group(1) @binding(2) var<storage, read_write> mOut: array<f32>;
struct MagParams { size: u32, }
@group(1) @binding(3) var<uniform> mparams: MagParams;

@compute @workgroup_size(256)
fn split_complex_magnitude(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= mparams.size) { return; }
    let re = mInReal[idx]; let im = mInImag[idx];
    mOut[idx] = sqrt(re * re + im * im);
}

@compute @workgroup_size(256)
fn split_complex_magnitude_squared(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= mparams.size) { return; }
    let re = mInReal[idx]; let im = mInImag[idx];
    mOut[idx] = re * re + im * im;
}

@compute @workgroup_size(256)
fn split_complex_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= mparams.size) { return; }
    mOut[idx] = atan2(mInImag[idx], mInReal[idx]);
}

// Complex from polar: (mag*cos(phase), mag*sin(phase))
@compute @workgroup_size(256)
fn split_complex_from_polar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cparams.size) { return; }
    let m = aReal[idx]; let p = aImag[idx];  // reusing bindings: mag in aReal, phase in aImag
    outReal[idx] = m * cos(p);
    outImag[idx] = m * sin(p);
}

// Complex add: (a.re+b.re, a.im+b.im)
@compute @workgroup_size(256)
fn split_complex_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cparams.size) { return; }
    outReal[idx] = aReal[idx] + bReal[idx];
    outImag[idx] = aImag[idx] + bImag[idx];
}

// Cross-spectral density: X * conj(Y)
@compute @workgroup_size(256)
fn split_complex_cross_spectral(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cparams.size) { return; }
    let xr = aReal[idx]; let xi = aImag[idx];
    let yr = bReal[idx]; let yi = bImag[idx];
    outReal[idx] = xr * yr + xi * yi;
    outImag[idx] = xi * yr - xr * yi;
}
";
}
