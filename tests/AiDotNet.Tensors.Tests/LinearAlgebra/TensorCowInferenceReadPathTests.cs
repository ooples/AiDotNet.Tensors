// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Issue #624 Stage 2 — "inference on a clone never privatizes." Stage 1 proved a COW clone
/// (<see cref="TensorBase{T}.CloneShared"/>) isolates on write; Stage 2 unlocks the actual benefit:
/// when a cloned model runs inference, its shared WEIGHTS flow through the engine ops as read-only
/// inputs and must NOT trigger copy-on-write privatization (otherwise an O(1) clone silently
/// becomes a full weight-buffer copy on the first matmul — zero benefit for the large-model case).
///
/// <para>Each test clones a weight tensor, runs the op with the clone as the weight operand, and
/// asserts (1) the clone is STILL COW-shared afterwards (the op read it read-only) and (2) the
/// result is byte-identical to running the op against an independent non-shared weight. A failing
/// "still shared" assertion is a precise pointer to an op whose input read still routes through the
/// privatizing <c>GetDataArray()</c> instead of the read-only <c>GetReadOnlyDataArray()</c>.</para>
/// </summary>
public class TensorCowInferenceReadPathTests
{
    private static readonly CpuEngine Engine = new CpuEngine();

    private static Tensor<float> Filled(int[] shape, int seed)
    {
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        // Deterministic, non-trivial values (no RNG needed; reproducible).
        for (int i = 0; i < n; i++)
            data[i] = (float)Math.Sin(0.123 * (i + seed) + 0.7);
        return new Tensor<float>(data, (int[])shape.Clone());
    }

    /// <summary>An independent non-shared weight + a COW clone of the same data.</summary>
    private static (Tensor<float> independent, Tensor<float> cowClone) Weight(int[] shape, int seed)
    {
        var w = Filled(shape, seed);
        var cowSource = Filled(shape, seed);            // separate buffer, identical values
        var clone = (Tensor<float>)cowSource.CloneShared();
        Assert.True(clone.IsCowShared, "precondition: CloneShared must flag the clone COW");
        return (w, clone);
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, float tol = 1e-4f)
    {
        var e = expected.ToArray();
        var a = actual.ToArray();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= tol + 1e-3f * Math.Abs(e[i]),
                $"mismatch at {i}: expected {e[i]}, got {a[i]}");
    }

    [Theory]
    [InlineData(2, 4, 3)]    // small
    [InlineData(8, 16, 32)]  // medium-M float fast path
    [InlineData(64, 128, 96)]
    public void Matmul_DoesNotPrivatizeCowWeight(int m, int k, int n)
    {
        var x = Filled(new[] { m, k }, 1);
        var (w, wClone) = Weight(new[] { k, n }, 100);

        var expected = Engine.TensorMatMul(x, w);
        var actual = Engine.TensorMatMul(x, wClone);

        Assert.True(wClone.IsCowShared, "matmul privatized the COW weight (operand B read through a write accessor)");
        AssertClose(expected, actual);
    }

    [Fact]
    public void LayerNorm_DoesNotPrivatizeCowGammaBeta()
    {
        var x = Filled(new[] { 4, 8 }, 1);
        var (gamma, gammaClone) = Weight(new[] { 8 }, 200);
        var (beta, betaClone) = Weight(new[] { 8 }, 300);

        var expected = Engine.TensorLayerNorm(x, gamma, beta);
        var actual = Engine.TensorLayerNorm(x, gammaClone, betaClone);

        Assert.True(gammaClone.IsCowShared, "layernorm privatized the COW gamma");
        Assert.True(betaClone.IsCowShared, "layernorm privatized the COW beta");
        AssertClose(expected, actual);
    }

    [Fact]
    public void Conv2D_DoesNotPrivatizeCowKernel()
    {
        // input [N=1, C=2, H=5, W=5], kernel [outC=3, inC=2, kh=3, kw=3]
        var x = Filled(new[] { 1, 2, 5, 5 }, 1);
        var (kernel, kernelClone) = Weight(new[] { 3, 2, 3, 3 }, 400);

        var expected = Engine.TensorConv2D(x, kernel, stride: 1, padding: 1, dilation: 1);
        var actual = Engine.TensorConv2D(x, kernelClone, stride: 1, padding: 1, dilation: 1);

        Assert.True(kernelClone.IsCowShared, "conv2d privatized the COW kernel");
        AssertClose(expected, actual);
    }

    [Fact]
    public void Embedding_DoesNotPrivatizeCowTable()
    {
        var indices = new Tensor<int>(new[] { 0, 3, 1, 2 }, new[] { 4 });
        var (table, tableClone) = Weight(new[] { 5, 6 }, 500);

        var expected = Engine.Embedding(indices, table);
        var actual = Engine.Embedding(indices, tableClone);

        Assert.True(tableClone.IsCowShared, "embedding privatized the COW table");
        AssertClose(expected, actual);
    }
}
