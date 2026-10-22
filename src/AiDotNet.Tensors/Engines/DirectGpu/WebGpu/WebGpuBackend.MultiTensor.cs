#if NET7_0_OR_GREATER
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The global-norm clip as one work-group reduction per tensor (WGSL cannot address a buffer from a table), the WebGPU
// port of CudaBackend's IMultiTensorKernels. The sum of squares is the double CUDA writes (WebGpuFiniteGuardKernels).
public sealed partial class WebGpuBackend : IMultiTensorKernels
{
    private const string MultiTensorSource = @"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
struct Params { n: u32, maxNorm: f32, _p0: u32, _p1: u32, }
@group(0) @binding(2) var<uniform> params: Params;

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
        // The double the guard and IDeviceScalarOps read (#1522): a float sum and non-finite mark, then finished.
        for (int t = 0; t < tensors.Count; t++)
            GuardDispatch("guard_sum_squares_accumulate", tensors[t], sumOfSquares, sizes[t], 0f,
                MultiTensorArgs.ReductionGroupSize);
        FinishSumOfSquaresDouble(sumOfSquares);
    }

    public void ClipScaleFromSumOfSquares(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale)
    {
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        GuardDispatch("guard_clip_scale", scale, sumOfSquares, 1, maxNorm, 1);
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