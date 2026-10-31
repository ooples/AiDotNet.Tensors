using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// An empty reduction (k = 0) is still C = A · B = 0. BlasManaged overwrites C, so a reused destination
/// (FusedLinearInto writes into the caller's buffer) must read zeros plus the epilogue, not what it held before.
/// </summary>
public class BlasManagedEmptyReductionTests
{
    [Fact]
    public void Gemm_EmptyReduction_OverwritesDestinationWithZeros()
    {
        const int m = 3, n = 4;
        var c = new float[m * n];
        c.AsSpan().Fill(7f);

        BlasManagedLib.Gemm<float>(ReadOnlySpan<float>.Empty, 0, false, ReadOnlySpan<float>.Empty, n, false,
            c, n, m, n, 0);

        Assert.All(c, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Gemm_EmptyReduction_AppliesTheEpilogueToTheZeroProduct()
    {
        const int m = 2, n = 3;
        var c = new float[m * n];
        c.AsSpan().Fill(7f);
        var bias = new[] { -1f, 2f, 0.5f };

        BlasManagedLib.Gemm<float>(ReadOnlySpan<float>.Empty, 0, false, ReadOnlySpan<float>.Empty, n, false,
            c, n, m, n, 0,
            new BlasOptions<float> { Epilogue = new Epilogue<float> { BiasN = bias, Activation = FusedActivationType.ReLU } });

        for (int row = 0; row < m; row++)
        {
            Assert.Equal(0f, c[row * n + 0]);
            Assert.Equal(2f, c[row * n + 1]);
            Assert.Equal(0.5f, c[row * n + 2]);
        }
    }

    [Fact]
    public void Gemm_EmptyReduction_LeavesTheStrideGapUntouched()
    {
        const int m = 2, n = 2, ldc = 3;
        var c = new float[m * ldc];
        c.AsSpan().Fill(7f);

        BlasManagedLib.Gemm<float>(ReadOnlySpan<float>.Empty, 0, false, ReadOnlySpan<float>.Empty, n, false,
            c, ldc, m, n, 0);

        Assert.Equal(new[] { 0f, 0f, 7f, 0f, 0f, 7f }, c);
    }
}
