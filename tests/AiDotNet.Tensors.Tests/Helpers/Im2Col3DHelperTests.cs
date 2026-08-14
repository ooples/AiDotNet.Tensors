// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class Im2Col3DHelperTests
{
    [Fact]
    public void BuildColumnsRange_TilesAreByteEquivalentToFullMatrix()
    {
        const int batch = 2, channels = 2, depth = 4, height = 5, width = 6;
        const int kernelDepth = 2, kernelHeight = 3, kernelWidth = 2;
        const int outputDepth = 3, outputHeight = 3, outputWidth = 5;
        const int columnsPerRow = channels * kernelDepth * kernelHeight * kernelWidth;
        const int totalRows = batch * outputDepth * outputHeight * outputWidth;

        var input = new float[batch * channels * depth * height * width];
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)Math.Sin(i * 0.13);

        var full = new float[totalRows * columnsPerRow];
        CpuIm2Col3DHelper.BuildColumns(
            input, full,
            batch, channels, depth, height, width,
            kernelDepth, kernelHeight, kernelWidth,
            outputDepth, outputHeight, outputWidth,
            1, 1, 1,
            0, 0, 0,
            1, 1, 1);

        var tiled = new float[full.Length];
        const int tileCapacity = 17;
        for (int rowStart = 0; rowStart < totalRows; rowStart += tileCapacity)
        {
            int rowCount = Math.Min(tileCapacity, totalRows - rowStart);
            var tile = new float[rowCount * columnsPerRow];
            CpuIm2Col3DHelper.BuildColumnsRange(
                input, tile, rowStart, rowCount,
                batch, channels, depth, height, width,
                kernelDepth, kernelHeight, kernelWidth,
                outputDepth, outputHeight, outputWidth,
                1, 1, 1,
                0, 0, 0,
                1, 1, 1);
            Array.Copy(tile, 0, tiled, rowStart * columnsPerRow, tile.Length);
        }

        Assert.Equal(full, tiled);
    }

    [Fact]
    public void BuildColumnsRange_RejectsRangesOutsideTheOutputVolume()
    {
        var input = new float[2 * 2 * 2];
        var columns = new float[2];

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            CpuIm2Col3DHelper.BuildColumnsRange(
                input, columns, 7, 2,
                1, 1, 2, 2, 2,
                1, 1, 1,
                2, 2, 2,
                1, 1, 1,
                0, 0, 0,
                1, 1, 1));
    }
}
