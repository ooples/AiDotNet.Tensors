// Copyright (c) CrowdTrainer LLC. All rights reserved.

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IArgExtremeRowLaunchCounter
{
    // argmax / argmin on a row at least ArgExtremeRows.MinWidth wide: one 256-invocation workgroup per row
    // (CUDA's arg_extreme_axis_rows), bit-exact with the serial scan; see VulkanBackend.ArgExtremeRows. No invocation
    // returns before a barrier, and a dispatch folded onto a 2-D grid (WebGpuDispatchFold) keeps workgroup_id flat.
    private const string ArgExtremeRowsModule = "ArgExtremeRows";
    // The body is an embedded resource, kept out of the user-string heap (CS8103).
    private static string ArgExtremeRowsSource => ExactFloatKernels.Wgsl + KernelSourceResources.Read("ArgExtremeRows.wgsl");

    private long _argExtremeRowLaunches;
    long IArgExtremeRowLaunchCounter.ArgExtremeRowLaunches => System.Threading.Interlocked.Read(ref _argExtremeRowLaunches);

    private bool TryLaunchArgExtremeRows(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize, bool findMax)
    {
        if (reduceSize < ArgExtremeRows.MinWidth || outerSize <= 0 || (long)outerSize * 256 > int.MaxValue)
            return false;
        Dispatch2BufferAsync(ArgExtremeRowsModule, ArgExtremeRowsSource, findMax ? "argmax_axis_rows" : "argmin_axis_rows",
            A, indices, MakeUniformInts2(outerSize, reduceSize), outerSize * 256).GetAwaiter().GetResult();
        System.Threading.Interlocked.Increment(ref _argExtremeRowLaunches);
        return true;
    }
}
#endif
