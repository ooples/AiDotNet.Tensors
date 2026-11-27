// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend : IArgExtremeRowLaunchCounter
{
    // argmax / argmin on a row at least ArgExtremeRows.MinWidth wide: one 256-invocation workgroup per row
    // (CUDA's arg_extreme_axis_rows) instead of one invocation per row scanning it serially. Bit-exact with the scan: a
    // leading NaN wins; otherwise the first index of the extreme over the non-NaN values by exact bit key. Each
    // invocation keeps the first extreme of its strided slice, and a merge takes the strictly better key, else the
    // smaller index, which is associative, so the tree gives the scan's answer. No invocation returns before a barrier.
    // The body is an embedded resource, kept out of the user-string heap (CS8103).
    private static string ArgExtremeAxisRowsGlsl => "#version 450\nlayout(local_size_x = 256) in;\n"
        + ExactFloatKernels.Glsl + KernelSourceResources.Read("ArgExtremeRows.glsl");

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
