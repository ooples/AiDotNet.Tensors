// Copyright (c) CrowdTrainer LLC. All rights reserved.

using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IArgExtremeRowLaunchCounter
{
    private long _argExtremeRowLaunches;
    long IArgExtremeRowLaunchCounter.ArgExtremeRowLaunches => System.Threading.Interlocked.Read(ref _argExtremeRowLaunches);

    // A row at least ArgExtremeRows.MinWidth wide: one 256-thread threadgroup per row (arg_extreme_axis_rows).
    private bool TryLaunchArgExtremeRows(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize, uint maximum)
    {
        if (reduceSize < ArgExtremeRows.MinWidth || outerSize <= 0 || _residentLibrary == System.IntPtr.Zero)
            return false;
        if (A is not MetalGpuBuffer aBuffer || indices is not MetalGpuBuffer idxBuffer)
            throw new System.ArgumentException("Buffers must be MetalGpuBuffer.");
        var pipeline = GetPipeline("Resident", _residentLibrary, "arg_extreme_axis_rows");
        if (pipeline.MaxTotalThreadsPerThreadgroup < 256) return false;
        using (var encoder = _commandQueue.CreateScopedComputeEncoder())
        {
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer(aBuffer, 0);
            encoder.SetBuffer(idxBuffer, 1);
            encoder.SetBytes((uint)outerSize, 2);
            encoder.SetBytes((uint)reduceSize, 3);
            encoder.SetBytes(maximum, 4);
            encoder.DispatchThreadgroups(new MTLSize((ulong)outerSize, 1, 1), new MTLSize(256, 1, 1));
        }
        System.Threading.Interlocked.Increment(ref _argExtremeRowLaunches);
        return true;
    }
}
