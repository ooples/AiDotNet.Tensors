// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend : IArgExtremeRowLaunchCounter
{
    // argmax / argmin on a row at least ArgExtremeRows.MinWidth wide: one 256-invocation workgroup per row
    // (CUDA's arg_extreme_axis_rows) instead of one invocation per row scanning it serially. Bit-exact with the scan: a
    // leading NaN wins; otherwise the first index of the extreme over the non-NaN values by exact bit key. Each
    // invocation keeps the first extreme of its strided slice, and a merge takes the strictly better key, else the
    // smaller index, which is associative, so the tree gives the scan's answer. No invocation returns before a barrier.
    private const string ArgExtremeAxisRowsGlsl = @"#version 450
layout(local_size_x = 256) in;
" + ExactFloatKernels.Glsl + @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; uint findMax; };
shared int sKey[256];
shared uint sIdx[256];
void main() {
    uint outer = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    bool live = outer < outerSize;
    uint offset = outer * reduceSize;
    // The max of key' = findMax ? key : -key. |xf_key| <= 0x7F800000, so the negation cannot overflow.
    bool have = false;
    int bestKey = 0;
    uint bestIdx = 0u;
    for (uint i = tid; live && i < reduceSize; i += 256u) {
        uint v = floatBitsToUint(a[offset + i]);
        if (xf_isnan(v)) continue;
        int key = findMax != 0u ? xf_key(v) : -xf_key(v);
        if (!have || key > bestKey) { have = true; bestKey = key; bestIdx = i; }
    }
    sKey[tid] = have ? bestKey : int(0x80000000u);   // INT_MIN: below every finite or infinite key
    sIdx[tid] = have ? bestIdx : 0xFFFFFFFFu;
    memoryBarrierShared();
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            int k2 = sKey[tid + stride];
            uint i2 = sIdx[tid + stride];
            if (k2 > sKey[tid] || (k2 == sKey[tid] && i2 < sIdx[tid])) { sKey[tid] = k2; sIdx[tid] = i2; }
        }
        memoryBarrierShared();
        barrier();
    }
    if (tid == 0u && live)
        b[outer] = xf_isnan(floatBitsToUint(a[offset])) ? 0.0 : float(sIdx[0]);
}";

    private long _argExtremeRowLaunches;
    long IArgExtremeRowLaunchCounter.ArgExtremeRowLaunches => System.Threading.Interlocked.Read(ref _argExtremeRowLaunches);

    private bool TryLaunchArgExtremeRows(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize, uint findMax)
    {
        if (reduceSize < ArgExtremeRows.MinWidth || outerSize <= 0 || (long)outerSize * 256 > int.MaxValue)
            return false;
        EnsureInitialized();
        // One workgroup per row: past the device's X workgroup limit (65,535 on some drivers) keep the serial kernel.
        if ((uint)outerSize > _device.Limits.maxComputeWorkGroupCountX) return false;
        GlslUnaryOp(ArgExtremeAxisRowsGlsl, A, indices, outerSize * 256,
            new uint[] { (uint)outerSize, (uint)reduceSize, findMax }, 3 * sizeof(uint));
        System.Threading.Interlocked.Increment(ref _argExtremeRowLaunches);
        return true;
    }
}
