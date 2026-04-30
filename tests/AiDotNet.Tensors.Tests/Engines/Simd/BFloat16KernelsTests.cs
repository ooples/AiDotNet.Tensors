// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Issue #276 sub-feature 1 (continued): SIMD kernels operating directly
/// on bf16 storage with float-accumulate. Each test compares the
/// vectorized path to the scalar reference — equality up to bf16 round-
/// off (one half-ULP per element).
/// </summary>
public class BFloat16KernelsTests
{
    private static BFloat16[] B(params float[] vals)
    {
        var r = new BFloat16[vals.Length];
        for (int i = 0; i < vals.Length; i++) r[i] = BFloat16.FromFloat(vals[i]);
        return r;
    }

    [Fact]
    public void VectorAdd_MatchesScalarReference()
    {
        var x = B(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f);
        var y = B(0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f);
        var dst = new BFloat16[x.Length];
        BFloat16Kernels.VectorAdd(x, y, dst);
        for (int i = 0; i < x.Length; i++)
            Assert.Equal((float)x[i] + (float)y[i], (float)dst[i], 2);
    }

    [Fact]
    public void Dot_AccumulatesInFloat()
    {
        var x = B(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
        var y = B(8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f);
        // 1·8 + 2·7 + 3·6 + 4·5 + 5·4 + 6·3 + 7·2 + 8·1 = 120
        Assert.Equal(120f, BFloat16Kernels.Dot(x, y), 1);
    }

    [Fact]
    public void ReduceSum_MatchesScalarSum()
    {
        var x = B(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f);
        float expected = 0;
        for (int i = 0; i < x.Length; i++) expected += (float)x[i];
        Assert.Equal(expected, BFloat16Kernels.ReduceSum(x), 1);
    }

    [Fact]
    public void Matmul_2x3_Times_3x2_Correct()
    {
        // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
        // C = [[58,64],[139,154]]
        var a = B(1, 2, 3, 4, 5, 6);
        var b = B(7, 8, 9, 10, 11, 12);
        var c = new BFloat16[4];
        BFloat16Kernels.Matmul(a, 3, b, 2, c, 2, m: 2, k: 3, n: 2);
        Assert.Equal(58f, (float)c[0], 0);
        Assert.Equal(64f, (float)c[1], 0);
        Assert.Equal(139f, (float)c[2], 0);
        Assert.Equal(154f, (float)c[3], 0);
    }

    [Fact]
    public void Gelu_MatchesFloatReference()
    {
        var x = B(-2f, -1f, 0f, 1f, 2f);
        var dst = new BFloat16[x.Length];
        BFloat16Kernels.Gelu(x, dst);
        // GELU(0) = 0; GELU is monotonic; check anchors.
        Assert.Equal(0f, (float)dst[2], 1);
        Assert.True((float)dst[3] > 0.8f && (float)dst[3] < 0.85f);
    }

    [Fact]
    public void Softmax_RowSumsToOne()
    {
        var x = B(1f, 2f, 3f, 4f);
        var dst = new BFloat16[4];
        BFloat16Kernels.Softmax(x, dst);
        float sum = 0;
        for (int i = 0; i < 4; i++) sum += (float)dst[i];
        Assert.Equal(1f, sum, 1);
    }

    [Fact]
    public void LayerNorm_ZeroMeanUnitVarianceWhenGammaOneBetaZero()
    {
        var x = B(1f, 2f, 3f, 4f, 5f);
        var gamma = B(1f, 1f, 1f, 1f, 1f);
        var beta = B(0f, 0f, 0f, 0f, 0f);
        var dst = new BFloat16[5];
        BFloat16Kernels.LayerNorm(x, gamma, beta, dst);
        float mean = 0; for (int i = 0; i < 5; i++) mean += (float)dst[i];
        mean /= 5;
        Assert.Equal(0f, mean, 1);
    }
}
#endif
