using System;
using System.Buffers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    [ThreadStatic]
    private static int t_conv1DColumnPanelRuns;
    internal static int Conv1DColumnPanelRunsOnThisThread => t_conv1DColumnPanelRuns;

    // A waveform has only one output row, so row-blocked im2col cannot bound its scratch or distribute its columns.
    // Split its time axis instead. At most one panel per worker is live, independently of waveform length and batch.
    private static bool TryConv1DColumnPanels(
        float[] input, float[] kernel, float[] output,
        int batch, int inChannels, int height, int width, int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        int outputHeight, int outputWidth, bool accumulate = false, int inputStorageOffset = 0, int outputStorageOffset = 0,
        int kernelStorageOffset = 0)
    {
        if (height != 1 || kernelHeight != 1 || outputHeight != 1 || padH != 0 || strideH != 1
            || kernelWidth <= 1 || (long)inChannels * kernelWidth * outputWidth < 1_048_576)
            return false;

        int k = checked(inChannels * kernelWidth);
        // About 1 MiB per panel, rounded to a GEMM column tile. A 32-column floor keeps wide kernels useful.
        int columns = Math.Min(outputWidth, Math.Max(32, Math.Min(256, (262_144 / k) / 16 * 16)));
        int blocksPerImage = (outputWidth + columns - 1) / columns;
        int blocks = checked(batch * blocksPerImage);
        int workers = Math.Max(1, Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, blocks));
        int panelFloats = checked(k * columns);
        long work = (long)batch * outChannels * k * outputWidth;
        CpuParallelSettings.ParallelForOrSerial(0, workers, work, worker =>
        {
            var options = new BlasManaged.BlasOptions<float>
            {
                NumThreads = 1,
                PackingMode = BlasManaged.PackingMode.ForcePackBoth,
                Mc = 96, Nc = 256, Kc = 256,
                BetaZero = true,
            };
            var panel = CpuStorageAllocationDiagnostics.RentWorkspace<float>(panelFloats, "workspace.conv1d_panel");
            var product = accumulate ? CpuStorageAllocationDiagnostics.RentWorkspace<float>(checked(outChannels * columns), "workspace.conv1d_product") : null;
            try
            {
                for (int block = worker; block < blocks; block += workers)
                {
                    int image = block / blocksPerImage;
                    int from = (block % blocksPerImage) * columns;
                    int count = Math.Min(columns, outputWidth - from);
                    for (int channel = 0; channel < inChannels; channel++)
                    {
                        int inputRow = inputStorageOffset + (image * inChannels + channel) * width;
                        for (int tap = 0; tap < kernelWidth; tap++)
                        {
                            var target = panel.AsSpan((channel * kernelWidth + tap) * count, count);
                            int first = from * strideW + tap * dilationW - padW;
                            if (strideW == 1)
                            {
                                int left = Math.Min(count, Math.Max(0, -first));
                                int right = Math.Max(left, Math.Min(count, width - first));
                                target.Slice(0, left).Clear();
                                if (right > left)
                                    input.AsSpan(inputRow + first + left, right - left).CopyTo(target.Slice(left));
                                target.Slice(right).Clear();
                            }
                            else
                            {
                                for (int j = 0, position = first; j < count; j++, position += strideW)
                                    target[j] = (uint)position < (uint)width ? input[inputRow + position] : 0f;
                            }
                        }
                    }

                    int outputOffset = checked(outputStorageOffset + image * outChannels * outputWidth + from);
                    var result = product is null ? output.AsSpan(outputOffset) : product.AsSpan();
                    BlasManaged.BlasManaged.Gemm<float>(
                        kernel.AsSpan(kernelStorageOffset, outChannels * k), k, false,
                        panel.AsSpan(0, k * count), count, false,
                        result, product is null ? outputWidth : count,
                        outChannels, count, k, options);
                    if (product is not null)
                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            var destination = output.AsSpan(outputOffset + oc * outputWidth, count);
                            var source = product.AsSpan(oc * count, count);
                            for (int j = 0; j < count; j++) destination[j] += source[j];
                        }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(panel);
                if (product is not null) ArrayPool<float>.Shared.Return(product);
            }
        }, deterministicSafe: true);
        t_conv1DColumnPanelRuns++;
        return true;
    }

    // dX for a stride-one waveform convolution is another dilated convolution, with channel axes swapped and taps
    // reversed. The generic col2im route materializes Cin*kW*outputWidth floats before scattering them back to dX.
    private static bool TryConv1DBackwardInputColumnPanels(
        LinearAlgebra.Tensor<float> destination, LinearAlgebra.Tensor<float> gradient, LinearAlgebra.Tensor<float> kernel,
        int batch, int inChannels, int height, int width, int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        int outputHeight, int outputWidth, bool accumulate)
    {
        long transposedPadding = (long)dilationW * (kernelWidth - 1) - padW;
        if (height != 1 || kernelHeight != 1 || outputHeight != 1 || strideH != 1 || strideW != 1 || padH != 0
            || kernelWidth <= 1 || transposedPadding < 0 || transposedPadding > int.MaxValue
            || !gradient.IsContiguous || !kernel.IsContiguous
            || (long)outChannels * kernelWidth * width < 1_048_576)
            return false;
        var g = gradient.GetCpuBackingForStridedRead(out int gOffset);
        var k = kernel.GetCpuBackingForStridedRead(out int kOffset);
        var d = destination.GetCpuBackingForContiguousWrite(out int dOffset);
        if (g is null || k is null || d is null) return false;

        var flipped = CpuStorageAllocationDiagnostics.RentWorkspace<float>(checked(inChannels * outChannels * kernelWidth), "workspace.conv1d_flipped");
        try
        {
            CpuParallelSettings.ParallelForOrSerial(0, inChannels, (long)inChannels * outChannels * kernelWidth, ic =>
            {
                for (int oc = 0; oc < outChannels; oc++)
                    for (int tap = 0; tap < kernelWidth; tap++)
                        flipped[(ic * outChannels + oc) * kernelWidth + tap] =
                            k[kOffset + (oc * inChannels + ic) * kernelWidth + kernelWidth - 1 - tap];
            }, deterministicSafe: true);
            return TryConv1DColumnPanels(g, flipped, d, batch, outChannels, 1, outputWidth, inChannels, 1, kernelWidth,
                1, 1, 0, (int)transposedPadding, 1, dilationW, 1, width, accumulate, gOffset, dOffset);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(flipped);
        }
    }
}
