using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// One-launch rectangular N-d slice, the Vulkan port of CudaBackend.RectSlice. The metadata rides in push constants
// (27 uints = 108 bytes, inside Vulkan's guaranteed 128), so one pipeline serves every slice shape and window.
public sealed partial class VulkanBackend : IRectSliceKernels
{
    private const string RectSliceGlsl = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer Full { float fullData[]; };
layout(set = 0, binding = 1) buffer Slice { float sliceData[]; };
layout(push_constant) uniform Params {
    uint total;
    uint rank;
    uint scatter;
    uint outDims[8];
    uint fullStrides[8];
    uint starts[8];
};
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= total) return;
    uint remaining = idx;
    uint offset = 0u;
    for (int d = int(rank) - 1; d >= 0; d--) {
        uint c = remaining % outDims[d];
        remaining /= outDims[d];
        offset += (starts[d] + c) * fullStrides[d];
    }
    if (scatter != 0u) fullData[offset] = sliceData[idx];
    else sliceData[idx] = fullData[offset];
}";

    public void RectSlice(IGpuBuffer full, IGpuBuffer slice, int[] fullShape, int[] start, int[] length, bool scatter)
    {
        EnsureInitialized();
        var outDims = new int[RectSliceLimits.MaxRank];
        var fullStrides = new int[RectSliceLimits.MaxRank];
        var starts = new int[RectSliceLimits.MaxRank];
        RectSliceGeometry.Build(full, slice, fullShape, start, length, outDims, fullStrides, starts,
            out int rank, out int total);
        var push = new uint[3 + 3 * RectSliceLimits.MaxRank];
        push[0] = (uint)total;
        push[1] = (uint)rank;
        push[2] = scatter ? 1u : 0u;
        for (int i = 0; i < RectSliceLimits.MaxRank; i++)
        {
            push[3 + i] = (uint)outDims[i];
            push[3 + RectSliceLimits.MaxRank + i] = (uint)fullStrides[i];
            push[3 + 2 * RectSliceLimits.MaxRank + i] = (uint)starts[i];
        }
        GlslUnaryOp(RectSliceGlsl, full, slice, total, push, (uint)(push.Length * sizeof(uint)));
    }
}