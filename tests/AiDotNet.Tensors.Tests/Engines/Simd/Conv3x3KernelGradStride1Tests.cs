// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// The direct 3x3 stride-1 weight gradient (<see cref="SimdConvHelper.Conv3x3KernelGradStride1"/>) against the plain
/// sum, over odd output heights (the last row runs without a partner), widths that leave a masked column chunk,
/// padding 0 and 1, and the accumulate path.
/// </summary>
public class Conv3x3KernelGradStride1Tests
{
    [SkippableTheory]
    [InlineData(2, 3, 8, 32, 32, 1)]
    [InlineData(3, 1, 4, 7, 9, 1)]
    [InlineData(2, 2, 3, 11, 13, 0)]
    [InlineData(1, 4, 2, 5, 17, 1)]
    [InlineData(2, 3, 5, 3, 3, 0)]
    public unsafe void MatchesThePlainSum(int batch, int inChannels, int outChannels, int height, int width, int pad)
    {
        Skip.IfNot(SimdConvHelper.CanUseDirectKernelGrad, "AVX2 + FMA kernel");
        int oH = height + 2 * pad - 2, oW = width + 2 * pad - 2;
        var rng = new Random(batch * 31 + height * 7 + width);
        var x = new float[batch * inChannels * height * width];
        var dy = new float[batch * outChannels * oH * oW];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < dy.Length; i++) dy[i] = (float)(rng.NextDouble() * 2 - 1);
        foreach (bool accumulate in new[] { false, true })
        {
            var dw = new float[outChannels * inChannels * 9];
            for (int i = 0; i < dw.Length; i++) dw[i] = accumulate ? 0.5f * (i % 3) : float.NaN;
            var seed = (float[])dw.Clone();
            fixed (float* px = x, pg = dy, pd = dw)
                SimdConvHelper.Conv3x3KernelGradStride1(px, pg, pd, batch, inChannels, height, width, outChannels,
                    pad, pad, oH, oW, accumulate);
            for (int o = 0; o < outChannels; o++)
                for (int c = 0; c < inChannels; c++)
                    for (int kh = 0; kh < 3; kh++)
                        for (int kw = 0; kw < 3; kw++)
                        {
                            double sum = 0;
                            for (int b = 0; b < batch; b++)
                                for (int i = 0; i < oH; i++)
                                    for (int j = 0; j < oW; j++)
                                    {
                                        int yi = i + kh - pad, xj = j + kw - pad;
                                        if (yi < 0 || yi >= height || xj < 0 || xj >= width) continue;
                                        sum += (double)dy[((b * outChannels + o) * oH + i) * oW + j]
                                            * x[((b * inChannels + c) * height + yi) * width + xj];
                                    }
                            int at = (o * inChannels + c) * 9 + kh * 3 + kw;
                            double expected = (accumulate ? seed[at] : 0) + sum;
                            Assert.True(Math.Abs(dw[at] - expected) <= 1e-4 * (1 + Math.Abs(expected)),
                                $"[{o},{c},{kh},{kw}] acc={accumulate}: expected {expected}, got {dw[at]}");
                        }
        }
    }
}
#endif
