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

        BuildColumnsRange(
            input, columns, 0, batch * rowsPerBatch,
            batch, channels, depth, height, width,
            kernelDepth, kernelHeight, kernelWidth,
            outputDepth, outputHeight, outputWidth,
            strideDepth, strideHeight, strideWidth,
            padDepth, padHeight, padWidth,
            dilationDepth, dilationHeight, dilationWidth);
    }

    /// <summary>
    /// Builds a contiguous row range of the im2col matrix. This is the bounded-workspace primitive
    /// used by Conv3D: paper-scale volumes can be tiled through GEMM without allocating the full
    /// [B*OD*OH*OW, C*KD*KH*KW] matrix at once.
    /// </summary>
    internal static void BuildColumnsRange<T>(
        T[] input,
        T[] columns,
        int rowStart,
        int rowCount,
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
        int totalRows = checked(batch * rowsPerBatch);
        if (rowStart < 0 || rowCount < 0 || rowStart > totalRows - rowCount)
            throw new ArgumentOutOfRangeException(nameof(rowStart),
                $"Im2Col3D row range [{rowStart}, {rowStart + (long)rowCount}) exceeds {totalRows} rows.");

        int expectedColumns = checked(rowCount * columnsPerRow);
        if (columns.Length < expectedColumns)
            throw new ArgumentException(
                $"Im2Col3D destination is too short: expected at least {expectedColumns}, got {columns.Length}.",
                nameof(columns));

        Array.Clear(columns, 0, expectedColumns);

        // Each worker owns a complete destination row, so writes are disjoint. Global row decoding
        // preserves the exact NCDHW traversal order used by the full-matrix implementation.
        CpuParallelSettings.ParallelForOrSerial(0, rowCount, expectedColumns, localRow =>
        {
            int globalRow = rowStart + localRow;
            int b = globalRow / rowsPerBatch;
            int rowWithinBatch = globalRow - b * rowsPerBatch;
            int od = rowWithinBatch / rowsPerDepth;
            int rowWithinDepth = rowWithinBatch - od * rowsPerDepth;
            int oh = rowWithinDepth / outputWidth;
            int ow = rowWithinDepth - oh * outputWidth;
            int sourceDepthOrigin = od * strideDepth - padDepth;
            int sourceHeightOrigin = oh * strideHeight - padHeight;
            int sourceWidthOrigin = ow * strideWidth - padWidth;
            int batchInputBase = b * channels * depth * height * width;
            int destinationRowBase = localRow * columnsPerRow;

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

                        // The overwhelmingly common dilation-W=1 case is a contiguous row slice.
                        // Copy it in bulk, clipping only the padded edges.
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
        });
    }
}
