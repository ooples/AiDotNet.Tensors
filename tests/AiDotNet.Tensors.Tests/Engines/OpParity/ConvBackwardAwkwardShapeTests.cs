// Copyright (c) AiDotNet. All rights reserved.
#if !NETFRAMEWORK
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Awkward-width coverage for the conv backward-kernel ops, whose two tensor operands are
/// geometrically coupled and so cannot be reached by <see cref="OpTailShapeSweepTests"/>.
/// </summary>
/// <remarks>
/// <para>
/// The sweep rewrites the inner dimension of every input independently. That is exactly right for
/// an op whose operands are free of one another, and exactly wrong here: <c>gradOutput</c>'s spatial
/// size is a FUNCTION of the input's, the kernel and the stride. Widening the input to 13 while also
/// widening gradOutput to 13 describes a convolution that cannot exist — a 3-wide valid kernel over
/// 13 columns produces 11, not 13.
/// </para>
/// <para>
/// The sweep flagged all three of these, and all three were its own fault. Each is verified below at
/// a CONSISTENT awkward width and agrees with the double oracle to ~1e-6, so the kernels handle a
/// non-multiple-of-vector width correctly; the ops are excluded from the perturbation sweep and
/// covered here instead, rather than simply dropped.
/// </para>
/// <para>
/// What the sweep did find is real and separate: these ops validate batch and channel agreement but
/// NOT the spatial geometry, so the impossible shapes were accepted and garbage was returned instead
/// of an <c>ArgumentException</c>. Fourteen conv backward ops share that gap and none has a shared
/// output-dimension helper to fix it in one place. Tracked separately; if it is fixed, the sweep
/// will begin recording these ops as not-applicable on its own and these tests stay as the coverage.
/// </para>
/// </remarks>
public class ConvBackwardAwkwardShapeTests
{
    /// <summary>Float and double must agree far more closely than a broken tail could.</summary>
    private const double GrossBreakageAbs = 1e-3;

    [Fact]
    public void Conv2DBackwardKernel_AwkwardWidth_AgreesWithOracle()
    {
        var e = new CpuEngine();
        // input width 13, kernel 3, stride 1, pad 0, dilation 1  =>  gradOutput width 11.
        AssertAgrees(
            e.Conv2DBackwardKernel(RandF(1920, 1, 4, 6, 11), RandF(1921, 1, 3, 8, 13),
                new[] { 4, 3, 3, 3 }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }).ToArray(),
            e.Conv2DBackwardKernel(RandD(1920, 1, 4, 6, 11), RandD(1921, 1, 3, 8, 13),
                new[] { 4, 3, 3, 3 }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }).ToArray(),
            "Conv2DBackwardKernel");
    }

    [Fact]
    public void Conv1DBackwardKernel_AwkwardLength_AgreesWithOracle()
    {
        var e = new CpuEngine();
        // input length 13, kernel 3, stride 1, pad 0, dilation 1  =>  gradOutput length 11.
        AssertAgrees(
            e.Conv1DBackwardKernel(RandF(1930, 1, 3, 11), RandF(1931, 1, 2, 13),
                new[] { 3, 2, 3 }, 1, 0, 1).ToArray(),
            e.Conv1DBackwardKernel(RandD(1930, 1, 3, 11), RandD(1931, 1, 2, 13),
                new[] { 3, 2, 3 }, 1, 0, 1).ToArray(),
            "Conv1DBackwardKernel");
    }

    [Fact]
    public void ConvTranspose2DBackwardKernel_AwkwardWidth_AgreesWithOracle()
    {
        var e = new CpuEngine();
        // Transposed geometry runs the other way: out = (in - 1) * stride - 2 * pad + kernel.
        // input width 13, kernel 2, stride 2, pad 0  =>  gradOutput width 26.
        AssertAgrees(
            e.ConvTranspose2DBackwardKernel(RandF(1940, 1, 3, 8, 26), RandF(1941, 1, 2, 4, 13),
                new[] { 2, 3, 2, 2 }, new[] { 2, 2 }, new[] { 0, 0 }).ToArray(),
            e.ConvTranspose2DBackwardKernel(RandD(1940, 1, 3, 8, 26), RandD(1941, 1, 2, 4, 13),
                new[] { 2, 3, 2, 2 }, new[] { 2, 2 }, new[] { 0, 0 }).ToArray(),
            "ConvTranspose2DBackwardKernel");
    }

    private static void AssertAgrees(float[] f, double[] d, string op)
    {
        Assert.True(f.Length == d.Length, $"{op}: float/double lengths differ ({f.Length} vs {d.Length})");

        double worst = 0;
        int at = -1;
        for (int i = 0; i < f.Length; i++)
        {
            double diff = Math.Abs(f[i] - d[i]);
            if (diff > worst) { worst = diff; at = i; }
        }

        Assert.True(
            worst < GrossBreakageAbs,
            $"{op} at an awkward inner dimension: float and the double oracle differ by {worst:R} at "
                + $"index {at} of {f.Length}. A vectorized column tail that is computed or stored at "
                + "full width would show up exactly here.");
    }

    private static Tensor<float> RandF(int seed, params int[] shape)
    {
        int n = 1;
        foreach (var d in shape) n *= d;
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(a, shape);
    }

    private static Tensor<double> RandD(int seed, params int[] shape)
    {
        int n = 1;
        foreach (var d in shape) n *= d;
        var rng = new Random(seed);
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = rng.NextDouble() * 2 - 1;
        return new Tensor<double>(a, shape);
    }
}
#endif
