// Copyright (c) AiDotNet. All rights reserved.
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Validates the fused grouped/depthwise deformable conv (DCNv3) CPU reference kernel (#1691) against
/// the per-group composition over the existing groups=1 DeformableConv2D — the correctness oracle the
/// future GPU backends must also match.
/// </summary>
public class GroupedDeformableConv2DTests
{
    private static Tensor<float> Rand(int seed, params int[] shape)
    {
        int n = 1; foreach (var s in shape) n *= s;
        var d = new float[n];
        var rng = new Random(seed);
        for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() - 0.5);
        return new Tensor<float>(shape, new Vector<float>(d));
    }

    [Theory]
    [InlineData(2, 2)]   // groups=2, deformGroups=2
    [InlineData(4, 4)]   // depthwise-ish
    [InlineData(2, 1)]   // shared deformable group across output groups
    public void FusedGrouped_MatchesPerGroupComposition(int groups, int deformGroups)
    {
        var engine = new CpuEngine();
        const int B = 1, inC = 8, outC = 8, H = 8, W = 8, k = 3;
        int inCpg = inC / groups, outCpg = outC / groups, kk = k * k;

        var input = Rand(1, B, inC, H, W);
        var kernel = Rand(2, outC, inCpg, k, k);                 // [outC, inC/groups, k, k]
        var offset = Rand(3, B, 2 * kk * deformGroups, H, W);    // stride 1, pad 1 -> out = H,W
        int[] stride = [1, 1], pad = [1, 1], dil = [1, 1];

        var fused = engine.DeformableConv2DGrouped(input, kernel, offset, null, stride, pad, dil, groups, deformGroups);

        // Reference: per-group groups=1 deformable conv over channel slices, concatenated.
        var groupOuts = new Tensor<float>[groups];
        for (int g = 0; g < groups; g++)
        {
            int dg = deformGroups == 1 ? 0 : g * deformGroups / groups;
            var inSlice = input.Slice(1, g * inCpg, (g + 1) * inCpg);
            var wSlice = kernel.Slice(0, g * outCpg, (g + 1) * outCpg);
            var offSlice = offset.Slice(1, dg * 2 * kk, (dg + 1) * 2 * kk);
            groupOuts[g] = engine.DeformableConv2D(inSlice, wSlice, offSlice, null, stride, pad, dil);
        }
        var reference = Tensor<float>.Concatenate(groupOuts, axis: 1);

        Assert.Equal(reference.Shape, fused.Shape);
        for (int i = 0; i < reference.Length; i++)
            Assert.True(MathF.Abs(reference[i] - fused[i]) < 1e-4f,
                $"groups={groups} dg={deformGroups}: mismatch at {i}: ref={reference[i]} fused={fused[i]}");
    }

    [Fact]
    public void Groups1_ReducesToBaseDeformableConv()
    {
        var engine = new CpuEngine();
        const int B = 1, C = 4, H = 6, W = 6, k = 3, kk = 9;
        var input = Rand(5, B, C, H, W);
        var kernel = Rand(6, C, C, k, k);
        var offset = Rand(7, B, 2 * kk, H, W);
        int[] stride = [1, 1], pad = [1, 1], dil = [1, 1];
        var baseOut = engine.DeformableConv2D(input, kernel, offset, null, stride, pad, dil);
        var grouped = engine.DeformableConv2DGrouped(input, kernel, offset, null, stride, pad, dil, 1, 1);
        for (int i = 0; i < baseOut.Length; i++) Assert.Equal(baseOut[i], grouped[i]);
    }
}

// Finite-difference verification of the grouped backward (double precision for numerical accuracy).
public class GroupedDeformableConv2DBackwardTests
{
    private static Tensor<double> Rand(int seed, double scale, params int[] shape)
    {
        int n = 1; foreach (var s in shape) n *= s;
        var d = new double[n];
        var rng = new System.Random(seed);
        for (int i = 0; i < n; i++) d[i] = (rng.NextDouble() - 0.5) * scale;
        return new Tensor<double>(shape, new Vector<double>(d));
    }

    private static double Loss(CpuEngine e, Tensor<double> input, Tensor<double> kernel, Tensor<double> offset,
        int[] s, int[] p, int[] dl, int groups, int dg)
    {
        var o = e.DeformableConv2DGrouped(input, kernel, offset, null, s, p, dl, groups, dg);
        double sum = 0; for (int i = 0; i < o.Length; i++) sum += o[i];
        return sum;
    }

    [Fact]
    public void GroupedBackward_MatchesFiniteDifference()
    {
        var e = new CpuEngine();
        const int groups = 2, dg = 2, inC = 4, outC = 4, H = 5, W = 5, k = 3, kk = 9, inCpg = 2;
        int[] s = [1, 1], p = [1, 1], dl = [1, 1];
        var input = Rand(1, 1.0, 1, inC, H, W);
        var kernel = Rand(2, 1.0, outC, inCpg, k, k);
        var offset = Rand(3, 0.6, 1, 2 * kk * dg, H, W);
        var gradOut = new Tensor<double>([1, outC, H, W], new Vector<double>(System.Linq.Enumerable.Repeat(1.0, outC * H * W).ToArray()));

        var gI = e.DeformableConv2DGroupedBackwardInput(gradOut, input, kernel, offset, null, [1, inC, H, W], s, p, dl, groups, dg);
        var gK = e.DeformableConv2DGroupedBackwardKernel(gradOut, input, offset, null, [outC, inCpg, k, k], s, p, dl, groups, dg);
        var gO = e.DeformableConv2DGroupedBackwardOffset(gradOut, input, kernel, offset, null, s, p, dl, groups, dg);

        const double eps = 1e-5;
        void Check(Tensor<double> t, Tensor<double> analytic, string name, double tol)
        {
            for (int i = 0; i < t.Length; i++)
            {
                double orig = t[i];
                t[i] = orig + eps; double lp = Loss(e, input, kernel, offset, s, p, dl, groups, dg);
                t[i] = orig - eps; double lm = Loss(e, input, kernel, offset, s, p, dl, groups, dg);
                t[i] = orig;
                double num = (lp - lm) / (2 * eps);
                Assert.True(System.Math.Abs(num - analytic[i]) <= tol + 1e-2 * System.Math.Abs(num),
                    $"{name}[{i}]: analytic={analytic[i]:F5} numeric={num:F5}");
            }
        }
        Check(input, gI, "gradInput", 1e-3);
        Check(kernel, gK, "gradKernel", 1e-3);
        Check(offset, gO, "gradOffset", 5e-3);
    }
}
