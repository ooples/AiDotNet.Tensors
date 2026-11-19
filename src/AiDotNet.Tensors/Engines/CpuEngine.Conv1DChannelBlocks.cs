using System;
using System.Buffers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    [ThreadStatic] private static int t_conv1DChannelBlockRuns;
    internal static int Conv1DChannelBlockRunsOnThisThread => t_conv1DChannelBlockRuns;

    // A fixed channel-block/tap/channel reduction order exposes adjacent input channels to the
    // GEMM packer. Scratch is independent of waveform length and batch: one shared <=16MiB
    // filter tile, and <=2MiB column/product panels per joined worker. No activation is retained.
    internal static bool TryConv1DChannelBlocks(float[] input, int inputOffset, float[] kernel, int kernelOffset,
        float[] output, int outputOffset, int batch, int inChannels, int width, int outChannels,
        int kernelWidth, int stride, int padding, int dilation, int outputWidth, bool splitChannels = true)
    {
        const int channelBlock = 16, aBudget = 4_194_304, workerBudget = 524_288;
        if (inChannels < channelBlock || kernelWidth <= 1 || width <= 0 || outputWidth <= 0
            || batch <= 0 || outChannels <= 0 || stride <= 0 || dilation <= 0 || padding < 0)
            return false;
        int reduction = checked(inChannels * kernelWidth);
        if (reduction > workerBudget) return false;
        int inputCount = checked(batch * inChannels * width);
        int kernelCount = checked(outChannels * reduction);
        int outputCount = checked(batch * outChannels * outputWidth);
        var destination = output.AsSpan(outputOffset, outputCount);
        if ((ReferenceEquals(input, output) && input.AsSpan(inputOffset, inputCount).Overlaps(destination))
            || (ReferenceEquals(kernel, output) && kernel.AsSpan(kernelOffset, kernelCount).Overlaps(destination)))
            return false;
        int rows = Math.Min(outChannels, Math.Max(1, (aBudget / reduction) / 16 * 16));
        int totalColumns = checked(batch * outputWidth);
        int columns = Math.Min(256, Math.Min(workerBudget / reduction, workerBudget / rows));
        if (totalColumns > columns && columns >= 16) columns = columns / 16 * 16;
        columns = Math.Min(columns, totalColumns);
        int blocks = (totalColumns - 1) / columns + 1;
        int workerBudgetCount = Math.Max(1, CpuParallelSettings.MaxDegreeOfParallelism);
        var packedKernel = CpuStorageAllocationDiagnostics.RentWorkspace<float>(checked(rows * reduction), "workspace.conv1d_channel_block_filter");
        try
        {
            for (int rowFrom = 0; rowFrom < outChannels; rowFrom += rows)
            {
                int rowCount = Math.Min(rows, outChannels - rowFrom);
                for (int oc = 0; oc < rowCount; oc++)
                {
                    int packedRow = 0;
                    for (int block = 0; block < inChannels; block += channelBlock)
                        for (int tap = 0; tap < kernelWidth; tap++)
                            for (int channel = block; channel < Math.Min(inChannels, block + channelBlock); channel++, packedRow++)
                                packedKernel[oc * reduction + packedRow] = kernel[kernelOffset + ((rowFrom + oc) * inChannels + channel) * kernelWidth + tap];
                }
                int currentRowFrom = rowFrom;
                // Short time axes expose too few spatial tasks. Partition independent output
                // rows too; six-row boundaries preserve the GEMM microkernel's row grouping.
                // The reduction axis and the pre-existing channel/tap order remain unchanged.
                int taskRows = rowCount;
                // A single spatial task already delegates the full worker budget to GEMM.
                // Preserve that dispatch; only the multi-task/serial-inner gap needs this grid.
                if (splitChannels && blocks > 1 && blocks < workerBudgetCount && rowCount >= 192)
                {
                    int partitions = Math.Min((workerBudgetCount + blocks - 1) / blocks, rowCount / 96);
                    taskRows = Math.Min(rowCount, ((rowCount + partitions * 6 - 1) / (partitions * 6)) * 6);
                }
                int rowTasks = (rowCount + taskRows - 1) / taskRows;
                int tasks = checked(blocks * rowTasks);
                int workers = Math.Min(workerBudgetCount, tasks);
                CpuParallelSettings.ParallelForOrSerial(0, workers, (long)rowCount * reduction * totalColumns, worker =>
                {
                    // Match the regular deterministic TensorMatMul dispatch. A logical N-tail
                    // keeps its packed reduction across every bounded panel, including aligned ones.
                    var options = new BlasManaged.BlasOptions<float>
                    {
                        NumThreads = workers == 1 ? 0 : 1,
                        PackingMode = totalColumns % 16 == 0 ? BlasManaged.PackingMode.DisableAutotune : BlasManaged.PackingMode.ForcePackBoth,
                        BetaZero = true,
                    };
                    var panel = CpuStorageAllocationDiagnostics.RentWorkspace<float>(checked(reduction * columns), "workspace.conv1d_channel_block_columns");
                    float[]? product = null;
                    try
                    {
                        product = CpuStorageAllocationDiagnostics.RentWorkspace<float>(checked(taskRows * columns), "workspace.conv1d_channel_block_product");
                        for (int task = worker; task < tasks; task += workers)
                        {
                            int rowStart = task / blocks * taskRows;
                            int countRows = Math.Min(taskRows, rowCount - rowStart);
                            int tile = task % blocks;
                            int from = tile * columns, count = Math.Min(columns, totalColumns - from), row = 0;
                            for (int block = 0; block < inChannels; block += channelBlock)
                                for (int tap = 0; tap < kernelWidth; tap++)
                                    for (int channel = block; channel < Math.Min(inChannels, block + channelBlock); channel++, row++)
                                        FillConv1DChannelBlockRow(input, inputOffset, panel.AsSpan(row * count, count),
                                            from, outputWidth, inChannels, channel, width, tap, stride, padding, dilation);
                            BlasManaged.BlasManaged.Gemm<float>(packedKernel.AsSpan(rowStart * reduction, countRows * reduction), reduction, false,
                                panel.AsSpan(0, reduction * count), count, false, product.AsSpan(0, countRows * count), count,
                                countRows, count, reduction, options);
                            for (int oc = 0; oc < countRows; oc++)
                                for (int column = 0; column < count;)
                                {
                                    int flat = from + column, image = flat / outputWidth;
                                    int time = flat - image * outputWidth;
                                    int take = Math.Min(count - column, outputWidth - time);
                                    product.AsSpan(oc * count + column, take).CopyTo(output.AsSpan(
                                        outputOffset + (image * outChannels + currentRowFrom + rowStart + oc) * outputWidth + time, take));
                                    column += take;
                                }
                        }
                    }
                    finally
                    {
                        if (product is not null) ArrayPool<float>.Shared.Return(product);
                        ArrayPool<float>.Shared.Return(panel);
                    }
                }, deterministicSafe: true);
            }
            t_conv1DChannelBlockRuns++;
            // The existing counter covers waveform column-panel executions, including this ordering.
            t_conv1DColumnPanelRuns++;
            return true;
        }
        finally { ArrayPool<float>.Shared.Return(packedKernel); }
    }

    // Each batch segment has a single contiguous valid interval. Divide only at batch/padding
    // boundaries; the stride-one body copies the input plane without per-element index arithmetic.
    private static void FillConv1DChannelBlockRow(float[] input, int inputOffset, Span<float> destination,
        int from, int outputWidth, int inChannels, int channel, int width, int tap, int stride, int padding, int dilation)
    {
        for (int column = 0; column < destination.Length;)
        {
            int flat = from + column, image = flat / outputWidth;
            int time = flat - image * outputWidth;
            int take = Math.Min(destination.Length - column, outputWidth - time);
            long first = (long)time * stride - padding + (long)tap * dilation;
            int left = BoundConv1DCeiling(-first, stride, take);
            int right = Math.Max(left, BoundConv1DCeiling(width - first, stride, take));
            destination.Slice(column, left).Clear();
            destination.Slice(column + right, take - right).Clear();
            if (right > left)
            {
                int source = inputOffset + (image * inChannels + channel) * width + (int)(first + (long)left * stride);
                var valid = destination.Slice(column + left, right - left);
                if (stride == 1) input.AsSpan(source, valid.Length).CopyTo(valid);
                else
                    for (int index = 0; index < valid.Length; index++, source += stride)
                        valid[index] = input[source];
            }
            column += take;
        }
    }

    private static int BoundConv1DCeiling(long numerator, int denominator, int upper)
    {
        if (numerator <= 0) return 0;
        long quotient = (numerator + denominator - 1) / denominator;
        return quotient >= upper ? upper : (int)quotient;
    }
}
