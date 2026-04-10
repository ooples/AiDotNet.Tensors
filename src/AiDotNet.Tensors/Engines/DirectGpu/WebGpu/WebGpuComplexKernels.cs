// Copyright (c) AiDotNet. All rights reserved.
// WebGPU WGSL compute shaders for native Complex<T> tensor operations.
// Each kernel is a self-contained WGSL module with its own @group(0) bindings.

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static class WebGpuComplexKernels
{
    // Note: WebGPU dispatches each kernel as a separate pipeline, so each needs
    // its own complete binding layout. All use @group(0) consistently.

    public static string SplitComplexMultiply => @"
@group(0) @binding(0) var<storage, read> aReal: array<f32>;
@group(0) @binding(1) var<storage, read> aImag: array<f32>;
@group(0) @binding(2) var<storage, read> bReal: array<f32>;
@group(0) @binding(3) var<storage, read> bImag: array<f32>;
@group(0) @binding(4) var<storage, read_write> outReal: array<f32>;
@group(0) @binding(5) var<storage, read_write> outImag: array<f32>;
struct Params { size: u32, }
@group(0) @binding(6) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let ar = aReal[idx]; let ai = aImag[idx];
    let br = bReal[idx]; let bi = bImag[idx];
    outReal[idx] = ar * br - ai * bi;
    outImag[idx] = ar * bi + ai * br;
}";

    public static string SplitComplexConjugate => @"
@group(0) @binding(0) var<storage, read> inReal: array<f32>;
@group(0) @binding(1) var<storage, read> inImag: array<f32>;
@group(0) @binding(2) var<storage, read_write> outReal: array<f32>;
@group(0) @binding(3) var<storage, read_write> outImag: array<f32>;
struct Params { size: u32, }
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    outReal[idx] = inReal[idx];
    outImag[idx] = -inImag[idx];
}";

    public static string SplitComplexMagnitude => @"
@group(0) @binding(0) var<storage, read> inReal: array<f32>;
@group(0) @binding(1) var<storage, read> inImag: array<f32>;
@group(0) @binding(2) var<storage, read_write> outMag: array<f32>;
struct Params { size: u32, }
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let re = inReal[idx]; let im = inImag[idx];
    outMag[idx] = sqrt(re * re + im * im);
}";

    public static string SplitComplexMagnitudeSquared => @"
@group(0) @binding(0) var<storage, read> inReal: array<f32>;
@group(0) @binding(1) var<storage, read> inImag: array<f32>;
@group(0) @binding(2) var<storage, read_write> outMagSq: array<f32>;
struct Params { size: u32, }
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let re = inReal[idx]; let im = inImag[idx];
    outMagSq[idx] = re * re + im * im;
}";

    public static string SplitComplexPhase => @"
@group(0) @binding(0) var<storage, read> inReal: array<f32>;
@group(0) @binding(1) var<storage, read> inImag: array<f32>;
@group(0) @binding(2) var<storage, read_write> outPhase: array<f32>;
struct Params { size: u32, }
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    outPhase[idx] = atan2(inImag[idx], inReal[idx]);
}";

    public static string SplitComplexFromPolar => @"
@group(0) @binding(0) var<storage, read> mag: array<f32>;
@group(0) @binding(1) var<storage, read> phase: array<f32>;
@group(0) @binding(2) var<storage, read_write> outReal: array<f32>;
@group(0) @binding(3) var<storage, read_write> outImag: array<f32>;
struct Params { size: u32, }
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let m = mag[idx]; let p = phase[idx];
    outReal[idx] = m * cos(p);
    outImag[idx] = m * sin(p);
}";

    public static string SplitComplexScale => @"
@group(0) @binding(0) var<storage, read> inReal: array<f32>;
@group(0) @binding(1) var<storage, read> inImag: array<f32>;
@group(0) @binding(2) var<storage, read_write> outReal: array<f32>;
@group(0) @binding(3) var<storage, read_write> outImag: array<f32>;
struct Params { size: u32, scalar: f32, }
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    outReal[idx] = inReal[idx] * params.scalar;
    outImag[idx] = inImag[idx] * params.scalar;
}";

    public static string SplitComplexAdd => @"
@group(0) @binding(0) var<storage, read> aReal: array<f32>;
@group(0) @binding(1) var<storage, read> aImag: array<f32>;
@group(0) @binding(2) var<storage, read> bReal: array<f32>;
@group(0) @binding(3) var<storage, read> bImag: array<f32>;
@group(0) @binding(4) var<storage, read_write> outReal: array<f32>;
@group(0) @binding(5) var<storage, read_write> outImag: array<f32>;
struct Params { size: u32, }
@group(0) @binding(6) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    outReal[idx] = aReal[idx] + bReal[idx];
    outImag[idx] = aImag[idx] + bImag[idx];
}";

    public static string SplitComplexCrossSpectral => @"
@group(0) @binding(0) var<storage, read> xReal: array<f32>;
@group(0) @binding(1) var<storage, read> xImag: array<f32>;
@group(0) @binding(2) var<storage, read> yReal: array<f32>;
@group(0) @binding(3) var<storage, read> yImag: array<f32>;
@group(0) @binding(4) var<storage, read_write> outReal: array<f32>;
@group(0) @binding(5) var<storage, read_write> outImag: array<f32>;
struct Params { size: u32, }
@group(0) @binding(6) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let xr = xReal[idx]; let xi = xImag[idx];
    let yr = yReal[idx]; let yi = yImag[idx];
    outReal[idx] = xr * yr + xi * yi;
    outImag[idx] = xi * yr - xr * yi;
}";
}
