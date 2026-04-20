using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

public class SortKernelsTests
{
    [Fact]
    public void SortFloatAscending_SmallBitonic()
    {
        var data = new float[] { 3, 1, 4, 1, 5, 9, 2, 6 };
        SortKernels.SortFloatAscending(data);
        Assert.Equal(new float[] { 1, 1, 2, 3, 4, 5, 6, 9 }, data);
    }

    [Fact]
    public void SortFloatAscending_ShorterThanBitonicWidth()
    {
        var data = new float[] { 5, 2, 8, 1 };
        SortKernels.SortFloatAscending(data);
        Assert.Equal(new float[] { 1, 2, 5, 8 }, data);
    }

    [Fact]
    public void SortFloatAscending_LongerFallback()
    {
        var data = new float[100];
        for (int i = 0; i < 100; i++) data[i] = 100 - i;
        SortKernels.SortFloatAscending(data);
        for (int i = 0; i < 100; i++) Assert.Equal((float)(i + 1), data[i]);
    }

    [Fact]
    public void SortFloatWithIndices_ReturnsPermutation()
    {
        var vals = new float[] { 3, 1, 4, 1 };
        var idx = new int[] { 0, 1, 2, 3 };
        SortKernels.SortFloatWithIndicesAscending(vals, idx);
        Assert.Equal(new float[] { 1, 1, 3, 4 }, vals);
        // Indices point back to original positions of the sorted values.
        // Both 1s came from positions 1 and 3; the specific order isn't
        // guaranteed (intro-sort isn't stable), just that the mapping is valid.
        Assert.Contains(1, idx);
        Assert.Contains(3, idx);
        Assert.Equal(0, idx[2]);
        Assert.Equal(2, idx[3]);
    }

    [Fact]
    public void LowerBoundFloat_ShortArraysUseAvxPath()
    {
        var seq = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        // lower_bound: first index where seq[i] >= value
        Assert.Equal(0, SortKernels.LowerBoundFloat(seq, 0.5f));
        Assert.Equal(0, SortKernels.LowerBoundFloat(seq, 1.0f));
        Assert.Equal(4, SortKernels.LowerBoundFloat(seq, 5.0f));
        Assert.Equal(5, SortKernels.LowerBoundFloat(seq, 5.5f));
        Assert.Equal(8, SortKernels.LowerBoundFloat(seq, 10.0f));
    }

    [Fact]
    public void UpperBoundFloat_RightBiased()
    {
        var seq = new float[] { 1, 2, 3, 3, 3, 4, 5 };
        // Upper bound for 3 is the first index > 3 (so past the 3-run).
        Assert.Equal(5, SortKernels.UpperBoundFloat(seq, 3.0f));
        // Lower bound for 3 should sit at the start of the 3-run.
        Assert.Equal(2, SortKernels.LowerBoundFloat(seq, 3.0f));
    }

    [Fact]
    public void TopKFloat_SmallK_HeapPath()
    {
        var vals = new float[] { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8 };
        var topV = new float[3];
        var topI = new int[3];
        SortKernels.TopKFloat(vals, 3, topV, topI);
        // Largest three descending: 9, 8, 6 at positions 5, 11, 7
        Assert.Equal(new float[] { 9, 8, 6 }, topV);
        Assert.Equal(new int[] { 5, 11, 7 }, topI);
    }

    [Fact]
    public void TopKFloat_LargeK_FullSortPath()
    {
        var vals = new float[] { 3, 1, 4, 1, 5, 9 };
        var topV = new float[5];
        var topI = new int[5];
        SortKernels.TopKFloat(vals, 5, topV, topI);
        Assert.Equal(new float[] { 9, 5, 4, 3, 1 }, topV);
    }
}
