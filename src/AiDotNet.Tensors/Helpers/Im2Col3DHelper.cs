using System;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Builds the row-major receptive-field matrix used by the CPU Conv3D GEMM path.
/// Kept in the tensor engine so every Conv3D consumer shares the same optimized
/// implementation instead of rebuilding seven nested loops in individual layers.
/// </summary>
internal static class CpuIm2Col3DHelper
{
    /// <summary>
    /// Converts NCDHW input into [B*OD*OH*OW, C*KD*KH*KW] rows. The destination
    /// is cleared first so padded positions remain numeric zero.
    /// </summary>
    internal static void BuildColumns<T>(
        T[] input,
        T[] columns,
        int batch,
        int channels,
        int depth,
        int height,
        int width,
        int kernelDepth,
        int kernelHeight,
        int kernelWidth,
        int outputDepth,
        int outputHeight,
        int outputWidth,
        int strideDepth,
        int strideHeight,
        int strideWidth,
        int padDepth,
        int padHeight,
        int padWidth,
        int dilationDepth,
        int dilationHeight,
        int dilationWidth)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (columns is null) throw new ArgumentNullException(nameof(columns));

        int columnsPerRow = checked(channels * kernelDepth * kernelHeight * kernelWidth);
        int rowsPerDepth = checked(outputHeight * outputWidth);
        int rowsPerBatch = checked(outputDepth * rowsPerDepth);
        int expectedColumns = checked(batch * rowsPerBatch * columnsPerRow);
        if (columns.Length < expectedColumns)
            throw new ArgumentException(
                $"Im2Col3D destination is too short: expected at least {expectedColumns}, got {columns.Length}.",
                nameof(columns));

        Array.Clear(columns, 0, expectedColumns);

        // Split by batch/output-depth. Each worker owns complete destination rows, so
        // writes are disjoint and require no synchronization. Keeping OH/OW inside the
        // worker amortizes dispatch overhead on small volumes.
        long work = (long)expectedColumns;
        CpuParallelSettings.ParallelForOrSerial(0, batch * outputDepth, work, batchDepth =>
        {
            int b = batchDepth / outputDepth;
            int od = batchDepth % outputDepth;
            int sourceDepthOrigin = od * strideDepth - padDepth;
            int batchInputBase = b * channels * depth * height * width;
            int batchRowBase = b * rowsPerBatch + od * rowsPerDepth;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                int sourceHeightOrigin = oh * strideHeight - padHeight;
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    int sourceWidthOrigin = ow * strideWidth - padWidth;
                    int row = batchRowBase + oh * outputWidth + ow;
                    int destinationRowBase = row * columnsPerRow;

                    for (int channel = 0; channel < channels; channel++)
                    {
                        int channelInputBase = batchInputBase + channel * depth * height * width;
                        int channelColumnBase = destinationRowBase
                            + channel * kernelDepth * kernelHeight * kernelWidth;

                        for (int kd = 0; kd < kernelDepth; kd++)
                        {
                            int sourceDepth = sourceDepthOrigin + kd * dilationDepth;
                            if ((uint)sourceDepth >= (uint)depth) continue;
                            int depthInputBase = channelInputBase + sourceDepth * height * width;
                            int depthColumnBase = channelColumnBase + kd * kernelHeight * kernelWidth;

                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                int sourceHeight = sourceHeightOrigin + kh * dilationHeight;
                                if ((uint)sourceHeight >= (uint)height) continue;
                                int sourceRowBase = depthInputBase + sourceHeight * width;
                                int destinationKernelRow = depthColumnBase + kh * kernelWidth;

                                // The overwhelmingly common dilation-W=1 case is a contiguous
                                // row slice. Copy it in bulk, clipping only the padded edges.
                                if (dilationWidth == 1)
                                {
                                    int firstKernelWidth = Math.Max(0, -sourceWidthOrigin);
                                    int endKernelWidth = Math.Min(kernelWidth, width - sourceWidthOrigin);
                                    int copyLength = endKernelWidth - firstKernelWidth;
                                    if (copyLength > 0)
                                    {
                                        Array.Copy(
                                            input,
                                            sourceRowBase + sourceWidthOrigin + firstKernelWidth,
                                            columns,
                                            destinationKernelRow + firstKernelWidth,
                                            copyLength);
                                    }
                                    continue;
                                }

                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int sourceWidth = sourceWidthOrigin + kw * dilationWidth;
                                    if ((uint)sourceWidth < (uint)width)
                                        columns[destinationKernelRow + kw] = input[sourceRowBase + sourceWidth];
                                }
                            }
                        }
                    }
                }
            }
        });
    }
}
