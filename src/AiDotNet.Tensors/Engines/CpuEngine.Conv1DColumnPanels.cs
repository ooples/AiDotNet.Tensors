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
        int outputHeight, int outputWidth)
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
            };
            var panel = ArrayPool<float>.Shared.Rent(panelFloats);
            try
            {
                for (int block = worker; block < blocks; block += workers)
                {
                    int image = block / blocksPerImage;
                    int from = (block % blocksPerImage) * columns;
                    int count = Math.Min(columns, outputWidth - from);
                    for (int channel = 0; channel < inChannels; channel++)
                    {
                        int inputRow = (image * inChannels + channel) * width;
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

                    int outputOffset = checked(image * outChannels * outputWidth + from);
                    BlasManaged.BlasManaged.Gemm<float>(
                        kernel.AsSpan(0, outChannels * k), k, false,
                        panel.AsSpan(0, k * count), count, false,
                        output.AsSpan(outputOffset), outputWidth,
                        outChannels, count, k, options);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(panel);
            }
        }, deterministicSafe: true);
        t_conv1DColumnPanelRuns++;
        return true;
    }
}
