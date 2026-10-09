#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// One-launch rectangular N-d slice, the WebGPU port of CudaBackend.RectSlice. Both buffers are read_write because
// scatter writes the slice back into the full tensor.
public sealed partial class WebGpuBackend : IRectSliceKernels
{
    private const string RectSliceSource = @"
@group(0) @binding(0) var<storage, read_write> full_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> slice_data: array<f32>;

struct RectSliceParams {
    total: u32,
    rank: u32,
    scatter: u32,
    _pad0: u32,
    out_dim0: u32, out_dim1: u32, out_dim2: u32, out_dim3: u32,
    out_dim4: u32, out_dim5: u32, out_dim6: u32, out_dim7: u32,
    full_stride0: u32, full_stride1: u32, full_stride2: u32, full_stride3: u32,
    full_stride4: u32, full_stride5: u32, full_stride6: u32, full_stride7: u32,
    start0: u32, start1: u32, start2: u32, start3: u32,
    start4: u32, start5: u32, start6: u32, start7: u32,
}
@group(0) @binding(2) var<uniform> params: RectSliceParams;

@compute @workgroup_size(256)
fn rect_slice_nd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total) { return; }
    // var, not let: older Naga rejects dynamic indexing of a let-bound array.
    var out_dims = array<u32, 8>(params.out_dim0, params.out_dim1, params.out_dim2, params.out_dim3,
        params.out_dim4, params.out_dim5, params.out_dim6, params.out_dim7);
    var full_strides = array<u32, 8>(params.full_stride0, params.full_stride1, params.full_stride2, params.full_stride3,
        params.full_stride4, params.full_stride5, params.full_stride6, params.full_stride7);
    var starts = array<u32, 8>(params.start0, params.start1, params.start2, params.start3,
        params.start4, params.start5, params.start6, params.start7);
    var remaining = idx;
    var offset: u32 = 0u;
    var d: i32 = i32(params.rank) - 1;
    loop {
        if (d < 0) { break; }
        let c = remaining % out_dims[d];
        remaining = remaining / out_dims[d];
        offset = offset + (starts[d] + c) * full_strides[d];
        d = d - 1;
    }
    if (params.scatter != 0u) {
        full_data[offset] = slice_data[idx];
    } else {
        slice_data[idx] = full_data[offset];
    }
}
";

    public void RectSlice(IGpuBuffer full, IGpuBuffer slice, int[] fullShape, int[] start, int[] length, bool scatter)
    {
        var outDims = new int[RectSliceLimits.MaxRank];
        var fullStrides = new int[RectSliceLimits.MaxRank];
        var starts = new int[RectSliceLimits.MaxRank];
        RectSliceGeometry.Build(full, slice, fullShape, start, length, outDims, fullStrides, starts,
            out int rank, out int total);
        var uniforms = new float[4 + 3 * RectSliceLimits.MaxRank];
        uniforms[0] = BitConverter.Int32BitsToSingle(total);
        uniforms[1] = BitConverter.Int32BitsToSingle(rank);
        uniforms[2] = BitConverter.Int32BitsToSingle(scatter ? 1 : 0);
        for (int i = 0; i < RectSliceLimits.MaxRank; i++)
        {
            uniforms[4 + i] = BitConverter.Int32BitsToSingle(outDims[i]);
            uniforms[4 + RectSliceLimits.MaxRank + i] = BitConverter.Int32BitsToSingle(fullStrides[i]);
            uniforms[4 + 2 * RectSliceLimits.MaxRank + i] = BitConverter.Int32BitsToSingle(starts[i]);
        }
        Dispatch2BufferAsync("RectSlice", RectSliceSource, "rect_slice_nd", full, slice, uniforms, total)
            .GetAwaiter().GetResult();
    }
}
#endif