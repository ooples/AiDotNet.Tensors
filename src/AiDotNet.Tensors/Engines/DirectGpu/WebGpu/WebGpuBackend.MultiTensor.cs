#if NET7_0_OR_GREATER
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The global-norm clip as one work-group reduction per tensor (WGSL cannot address a buffer from a table), the WebGPU
// port of CudaBackend's IMultiTensorKernels. acc[0] holds the float sum of squares.
public sealed partial class WebGpuBackend : IMultiTensorKernels
{
    private const string MultiTensorSource = @"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
struct Params { n: u32, maxNorm: f32, _p0: u32, _p1: u32, }
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> partial: array<f32, 256>;

// a = tensor, b = acc. Only work-group 0 runs; thread 0 adds the partials in order, so no atomics are needed.
@compute @workgroup_size(256)
fn tensor_sum_squares_accumulate(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    if (wid.x != 0u) { return; }
    var v: f32 = 0.0;
    for (var i: u32 = lid; i < params.n; i = i + 256u) { let e = a[i]; v = v + e * e; }
    partial[lid] = v;
    workgroupBarrier();
    if (lid == 0u) {
        var s: f32 = 0.0;
        for (var j: u32 = 0u; j < 256u; j = j + 1u) { s = s + partial[j]; }
        b[0] = b[0] + s;
    }
}

// a = acc, b = scale: min(1, maxNorm / (norm + 1e-6)); a non-finite norm leaves the gradients unscaled.
@compute @workgroup_size(256)
fn clip_scale_from_sum_squares(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let norm = sqrt(a[0]);
    var c: f32 = 1.0;
    if ((bitcast<u32>(norm) & 0x7f800000u) != 0x7f800000u) { c = params.maxNorm / (norm + 1e-6); }
    b[0] = min(c, 1.0);
}

// a = scale, b = tensor: b[i] *= a[0].
@compute @workgroup_size(256)
fn scale_by_device_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) { b[i] = b[i] * a[0]; }
}
";

    private static float[] MultiTensorParams(int n, float maxNorm) =>
        new[] { BitConverter.Int32BitsToSingle(n), maxNorm, 0f, 0f };

    public void MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
    {
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        MultiTensorArgs.Validate(tensors, sizes);
        Fill(sumOfSquares, 0f, 2);
        for (int t = 0; t < tensors.Count; t++)
            Dispatch2BufferAsync("MultiTensor", MultiTensorSource, "tensor_sum_squares_accumulate", tensors[t], sumOfSquares,
                MultiTensorParams(sizes[t], 0f), MultiTensorArgs.ReductionGroupSize).GetAwaiter().GetResult();
    }

    public void ClipScaleFromSumOfSquares(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale)
    {
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        Dispatch2BufferAsync("MultiTensor", MultiTensorSource, "clip_scale_from_sum_squares", sumOfSquares, scale,
            MultiTensorParams(1, maxNorm), 1).GetAwaiter().GetResult();
    }

    public void MultiTensorScaleByDeviceScalar(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer scale)
    {
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        MultiTensorArgs.Validate(tensors, sizes);
        for (int t = 0; t < tensors.Count; t++)
            Dispatch2BufferAsync("MultiTensor", MultiTensorSource, "scale_by_device_scalar", scale, tensors[t],
                MultiTensorParams(sizes[t], 0f), sizes[t]).GetAwaiter().GetResult();
    }
}
#endif