using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

public class ScanKernelsTests
{
    [Fact]
    public void PrefixSum_MatchesScalarReference_Length8()
    {
        var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var output = new float[8];
        ScanKernels.PrefixSumFloat(input, output);
        Assert.Equal(new float[] { 1, 3, 6, 10, 15, 21, 28, 36 }, output);
    }

    [Fact]
    public void PrefixSum_MatchesScalarReference_Length16()
    {
        // Two AVX blocks with cross-block carry.
        var input = new float[16];
        for (int i = 0; i < 16; i++) input[i] = i + 1;
        var output = new float[16];
        ScanKernels.PrefixSumFloat(input, output);

        float expected = 0;
        for (int i = 0; i < 16; i++)
        {
            expected += input[i];
            Assert.Equal(expected, output[i]);
        }
    }

    [Fact]
    public void PrefixSum_ShortSpan_ScalarPath()
    {
        var input = new float[] { 10, 20, 30 };
        var output = new float[3];
        ScanKernels.PrefixSumFloat(input, output);
        Assert.Equal(new float[] { 10, 30, 60 }, output);
    }

    [Fact]
    public void PrefixSum_TailBeyondBlocks_MatchesScalar()
    {
        // Length 13 = one block + 5 tail elements.
        var input = new float[13];
        for (int i = 0; i < 13; i++) input[i] = 1.0f;
        var output = new float[13];
        ScanKernels.PrefixSumFloat(input, output);
        for (int i = 0; i < 13; i++) Assert.Equal((float)(i + 1), output[i]);
    }

    [Fact]
    public void RunningMax_MatchesScalar()
    {
        var input = new float[] { 3, 1, 4, 1, 5, 9, 2, 6, 5 };
        var output = new float[9];
        ScanKernels.RunningMaxFloat(input, output);
        Assert.Equal(new float[] { 3, 3, 4, 4, 5, 9, 9, 9, 9 }, output);
    }

    [Fact]
    public void RunningMin_MatchesScalar()
    {
        var input = new float[] { 3, 1, 4, 1, 5, 9, 2, 6, 5 };
        var output = new float[9];
        ScanKernels.RunningMinFloat(input, output);
        Assert.Equal(new float[] { 3, 1, 1, 1, 1, 1, 1, 1, 1 }, output);
    }

    [Fact]
    public void PrefixProduct_MatchesScalar()
    {
        var input = new float[] { 2, 3, 4, 0.5f };
        var output = new float[4];
        ScanKernels.PrefixProductFloat(input, output);
        Assert.Equal(new float[] { 2, 6, 24, 12 }, output);
    }
}
