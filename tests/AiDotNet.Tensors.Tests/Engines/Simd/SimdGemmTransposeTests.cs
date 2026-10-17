using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// <see cref="SimdGemm.Transpose"/> (8x8 AVX tiles plus scalar remainders) against the scalar definition
/// dst[dOff + c * dstStride + r] = src[sOff + r * srcStride + c]: shapes below, at and past a tile in each dimension,
/// strided rows wider than the block (a head slice of an interleaved [seq, heads * dim] row), offsets on both sides,
/// and every element outside the written block left untouched.
/// </summary>
public sealed class SimdGemmTransposeTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(7, 9)]
    [InlineData(8, 8)]
    [InlineData(16, 32)]   // attention key tile at head dim 16, seq 32
    [InlineData(32, 16)]
    [InlineData(17, 13)]
    [InlineData(128, 64)]  // a split-k chunk of dW
    [InlineData(64, 130)]
    public void Transpose_MatchesTheScalarDefinition(int rows, int cols)
    {
        const int sOff = 5, dOff = 3;
        int srcStride = cols + 11, dstStride = rows + 7;
        var src = new float[sOff + rows * srcStride + 4];
        var rng = RandomHelper.CreateSeededRandom(rows * 1000 + cols);
        for (int i = 0; i < src.Length; i++) src[i] = (float)(rng.NextDouble() - 0.5);
        const float Sentinel = -12345.5f;
        var dst = new float[dOff + cols * dstStride + 4];
        for (int i = 0; i < dst.Length; i++) dst[i] = Sentinel;
        var expected = (float[])dst.Clone();
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                expected[dOff + c * dstStride + r] = src[sOff + r * srcStride + c];

        SimdGemm.Transpose(src, sOff, srcStride, dst, dOff, dstStride, rows, cols);

        for (int i = 0; i < dst.Length; i++)
            Assert.True(BitConverter.ToInt32(BitConverter.GetBytes(expected[i]), 0) == BitConverter.ToInt32(BitConverter.GetBytes(dst[i]), 0),
                $"rows={rows} cols={cols} [{i}]: {dst[i]:R} != {expected[i]:R}");

        // The scaled variant: the same placement, each element times the scale (one float multiply), the rest untouched.
        const float Scale = 0.25f * 1.0371f;
        var scaledDst = new float[dst.Length];
        for (int i = 0; i < scaledDst.Length; i++) scaledDst[i] = Sentinel;
        var scaledExpected = (float[])scaledDst.Clone();
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                scaledExpected[dOff + c * dstStride + r] = src[sOff + r * srcStride + c] * Scale;
        SimdGemm.TransposeScaled(src, sOff, srcStride, scaledDst, dOff, dstStride, rows, cols, Scale);
        for (int i = 0; i < scaledDst.Length; i++)
            Assert.True(BitConverter.ToInt32(BitConverter.GetBytes(scaledExpected[i]), 0) == BitConverter.ToInt32(BitConverter.GetBytes(scaledDst[i]), 0),
                $"scaled rows={rows} cols={cols} [{i}]: {scaledDst[i]:R} != {scaledExpected[i]:R}");
    }

    [Fact]
    public void Transpose_RejectsAnExtentPastTheSource()
    {
        var src = new float[8 * 8 - 1];
        var dst = new float[8 * 8];
        Assert.ThrowsAny<ArgumentException>(() => SimdGemm.Transpose(src, 0, 8, dst, 0, 8, 8, 8));
    }
}
