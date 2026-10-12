using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Completes <see cref="IScatterRowsKernels"/> (and so <see cref="IExtendedConvKernels"/>) on this backend.
/// Scatter-mean and scatter-max without a caller-visible counts or argmax buffer run the backend's own
/// resident kernels with a device scratch buffer, so nothing leaves the device (#1142).
/// </summary>
public sealed partial class MetalBackend : IExtendedConvKernels
{
    public void ScatterMeanRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int srcDimSize, int innerSize, int outDimSize)
    {
        if (outDimSize <= 0 || innerSize <= 0) return;
        using var counts = AllocateBuffer(outDimSize);
        ScatterMeanRowsWithCounts(source, indices, output, counts, srcDimSize, innerSize, outDimSize);
    }

    public void ScatterMaxRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int srcDimSize, int innerSize, int outDimSize)
    {
        if (outDimSize <= 0 || innerSize <= 0) return;
        using var argmax = AllocateBuffer(checked(outDimSize * innerSize));
        ScatterMaxWithArgmaxRows(source, indices, output, argmax, srcDimSize, innerSize, outDimSize);
    }
}
