using System;
using System.Buffers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    [ThreadStatic] private static int t_shortWaveInputGradientRuns;
    internal static int ShortWaveInputGradientRunsOnThisThread => t_shortWaveInputGradientRuns;

    // A high-channel, short waveform's dX needs at most4MiB of columns. Multiplying W^T*dY
    // directly avoids the full transposed/flipped weight tensor and the second forward
    // convolution. Scatter only within each input channel, preserving its reduction order.
    private static bool TryShortWaveInputGradient(Tensor<float> destination, Tensor<float> upstream, Tensor<float> kernel,
        int batch, int inChannels, int height, int width, int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        int outputHeight, int outputWidth, bool accumulate)
    {
        if (batch != 1 || height != 1 || kernelHeight != 1 || outputHeight != 1 || padH != 0
            || strideH != 1 || strideW != 1 || dilationH != 1 || dilationW != 1 || kernelWidth <= 1
            || inChannels < 256 || outChannels < 256 || outputWidth > 256
            || (long)inChannels * kernelWidth * outputWidth > 1_048_576
            || kernel.HasPendingGpuData || upstream.HasPendingGpuData) return false;
        using var sourceKeep = new KeepAliveScope(upstream);
        using var weightKeep = new KeepAliveScope(kernel);
        using var targetKeep = new KeepAliveScope(destination);
        var weights = kernel.GetCpuBackingForStridedRead(out int weightOffset);
        var gradient = upstream.GetCpuBackingForStridedRead(out int gradientOffset);
        var target = destination.GetCpuBackingForContiguousWrite(out int targetOffset);
        if (weights is null || gradient is null || target is null) return false;
        int rows = checked(inChannels * kernelWidth), length = checked(rows * outputWidth);
        var targetSpan = target.AsSpan(targetOffset, destination.Length);
        if (weights.AsSpan(weightOffset, kernel.Length).Overlaps(targetSpan)
            || gradient.AsSpan(gradientOffset, upstream.Length).Overlaps(targetSpan)) return false;
        var columns = CpuStorageAllocationDiagnostics.RentWorkspace<float>(length, "workspace.short_wave_dx_columns");
        try
        {
            BlasProvider.GemmEx(rows, outputWidth, outChannels, weights, weightOffset, rows, true,
                gradient, gradientOffset, outputWidth, false, columns, 0, outputWidth);
            if (!accumulate) targetSpan.Clear();
            CpuParallelSettings.ParallelForOrSerial(0, inChannels, length, channel =>
                Im2ColHelper.Col2ImAccumulateChannelRange(columns.AsSpan(0, length),
                    target.AsSpan(targetOffset, destination.Length), channel, channel + 1, height, width,
                    kernelHeight, kernelWidth, strideH, strideW, padH, padW, dilationH, dilationW,
                    outputHeight, outputWidth), deterministicSafe: true);
            t_shortWaveInputGradientRuns++;
            return true;
        }
        finally { ArrayPool<float>.Shared.Return(columns); }
    }
}
