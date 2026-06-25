// Copyright (c) AiDotNet. All rights reserved.
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
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

    private static double LossM(CpuEngine e, Tensor<double> input, Tensor<double> kernel, Tensor<double> offset,
        Tensor<double> mask, int[] s, int[] p, int[] dl, int groups, int dg)
    {
        var o = e.DeformableConv2DGrouped(input, kernel, offset, mask, s, p, dl, groups, dg);
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

    /// <summary>
    /// The autodiff wiring (#1691): the grouped forward records on the tape and back-props through the
    /// grouped backward kernels. With loss = sum(output), the upstream gradient is all-ones, so the tape
    /// gradients must equal the manual grouped backward called with gradOut = ones (already finite-diff
    /// verified above). This proves DeformableConv2DGrouped is differentiable via GradientTape.
    /// </summary>
    [Fact]
    public void GroupedForward_OnTape_BackpropsThroughGroupedBackward()
    {
        var e = new CpuEngine();
        const int groups = 2, dg = 2, inC = 4, outC = 4, H = 5, W = 5, k = 3, kk = 9, inCpg = 2;
        int[] s = [1, 1], p = [1, 1], dl = [1, 1];
        var input = Rand(11, 1.0, 1, inC, H, W);
        var kernel = Rand(12, 1.0, outC, inCpg, k, k);
        var offset = Rand(13, 0.6, 1, 2 * kk * dg, H, W);

        using var tape = new GradientTape<double>();
        var output = e.DeformableConv2DGrouped(input, kernel, offset, null, s, p, dl, groups, dg);
        var loss = e.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new[] { input, kernel, offset });

        var gradOut = new Tensor<double>([1, outC, H, W], new Vector<double>(System.Linq.Enumerable.Repeat(1.0, outC * H * W).ToArray()));
        var gI = e.DeformableConv2DGroupedBackwardInput(gradOut, input, kernel, offset, null, [1, inC, H, W], s, p, dl, groups, dg);
        var gK = e.DeformableConv2DGroupedBackwardKernel(gradOut, input, offset, null, [outC, inCpg, k, k], s, p, dl, groups, dg);
        var gO = e.DeformableConv2DGroupedBackwardOffset(gradOut, input, kernel, offset, null, s, p, dl, groups, dg);

        void Same(Tensor<double> tape_, Tensor<double> manual, string name)
        {
            Assert.NotNull(tape_);
            Assert.Equal(manual.Length, tape_.Length);
            for (int i = 0; i < manual.Length; i++)
                Assert.True(System.Math.Abs(tape_[i] - manual[i]) < 1e-9,
                    $"{name}[{i}]: tape={tape_[i]:F9} manual={manual[i]:F9}");
        }
        Same(grads[input], gI, "gradInput");
        Same(grads[kernel], gK, "gradKernel");
        Same(grads[offset], gO, "gradOffset");
    }

    /// <summary>
    /// DCN v2/v3 modulation-mask backward (#1691): finite-difference checks gradInput/gradKernel/
    /// gradOffset/gradMask for a non-null modulation mask against the manual grouped backward kernels.
    /// </summary>
    [Fact]
    public void GroupedBackward_WithMask_MatchesFiniteDifference()
    {
        var e = new CpuEngine();
        const int groups = 2, dg = 2, inC = 4, outC = 4, H = 5, W = 5, k = 3, kk = 9, inCpg = 2;
        int[] s = [1, 1], p = [1, 1], dl = [1, 1];
        var input = Rand(21, 1.0, 1, inC, H, W);
        var kernel = Rand(22, 1.0, outC, inCpg, k, k);
        var offset = Rand(23, 0.6, 1, 2 * kk * dg, H, W);
        var mask = Rand(24, 0.5, 1, kk * dg, H, W);  // modulation mask [B, kk*dg, H, W]
        var gradOut = new Tensor<double>([1, outC, H, W], new Vector<double>(System.Linq.Enumerable.Repeat(1.0, outC * H * W).ToArray()));

        var gI = e.DeformableConv2DGroupedBackwardInput(gradOut, input, kernel, offset, mask, [1, inC, H, W], s, p, dl, groups, dg);
        var gK = e.DeformableConv2DGroupedBackwardKernel(gradOut, input, offset, mask, [outC, inCpg, k, k], s, p, dl, groups, dg);
        var gO = e.DeformableConv2DGroupedBackwardOffset(gradOut, input, kernel, offset, mask, s, p, dl, groups, dg);
        var gM = e.DeformableConv2DGroupedBackwardMask(gradOut, input, kernel, offset, mask, s, p, dl, groups, dg);

        const double eps = 1e-5;
        void Check(Tensor<double> t, Tensor<double> analytic, string name, double tol)
        {
            for (int i = 0; i < t.Length; i++)
            {
                double orig = t[i];
                t[i] = orig + eps; double lp = LossM(e, input, kernel, offset, mask, s, p, dl, groups, dg);
                t[i] = orig - eps; double lm = LossM(e, input, kernel, offset, mask, s, p, dl, groups, dg);
                t[i] = orig;
                double num = (lp - lm) / (2 * eps);
                Assert.True(System.Math.Abs(num - analytic[i]) <= tol + 1e-2 * System.Math.Abs(num),
                    $"{name}[{i}]: analytic={analytic[i]:F5} numeric={num:F5}");
            }
        }
        Check(input, gI, "gradInput", 1e-3);
        Check(kernel, gK, "gradKernel", 1e-3);
        Check(offset, gO, "gradOffset", 5e-3);
        Check(mask, gM, "gradMask", 1e-3);
    }

    /// <summary>
    /// DCN v2/v3 autograd wiring (#1691): the masked grouped forward records all four inputs on the tape
    /// and back-props to input/kernel/offset/mask. With loss=sum(output) the tape grads must equal the
    /// manual masked grouped backward called with gradOut=ones (verified above by finite difference).
    /// </summary>
    [Fact]
    public void MaskedGroupedForward_OnTape_BackpropsToAllFour()
    {
        var e = new CpuEngine();
        const int groups = 2, dg = 2, inC = 4, outC = 4, H = 5, W = 5, k = 3, kk = 9, inCpg = 2;
        int[] s = [1, 1], p = [1, 1], dl = [1, 1];
        var input = Rand(31, 1.0, 1, inC, H, W);
        var kernel = Rand(32, 1.0, outC, inCpg, k, k);
        var offset = Rand(33, 0.6, 1, 2 * kk * dg, H, W);
        var mask = Rand(34, 0.5, 1, kk * dg, H, W);

        using var tape = new GradientTape<double>();
        var output = e.DeformableConv2DGrouped(input, kernel, offset, mask, s, p, dl, groups, dg);
        var loss = e.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new[] { input, kernel, offset, mask });

        var gradOut = new Tensor<double>([1, outC, H, W], new Vector<double>(System.Linq.Enumerable.Repeat(1.0, outC * H * W).ToArray()));
        var gI = e.DeformableConv2DGroupedBackwardInput(gradOut, input, kernel, offset, mask, [1, inC, H, W], s, p, dl, groups, dg);
        var gK = e.DeformableConv2DGroupedBackwardKernel(gradOut, input, offset, mask, [outC, inCpg, k, k], s, p, dl, groups, dg);
        var gO = e.DeformableConv2DGroupedBackwardOffset(gradOut, input, kernel, offset, mask, s, p, dl, groups, dg);
        var gM = e.DeformableConv2DGroupedBackwardMask(gradOut, input, kernel, offset, mask, s, p, dl, groups, dg);

        void Same(Tensor<double> tape_, Tensor<double> manual, string name)
        {
            Assert.NotNull(tape_);
            Assert.Equal(manual.Length, tape_.Length);
            for (int i = 0; i < manual.Length; i++)
                Assert.True(System.Math.Abs(tape_[i] - manual[i]) < 1e-9,
                    $"{name}[{i}]: tape={tape_[i]:F9} manual={manual[i]:F9}");
        }
        Same(grads[input], gI, "gradInput");
        Same(grads[kernel], gK, "gradKernel");
        Same(grads[offset], gO, "gradOffset");
        Same(grads[mask], gM, "gradMask");
    }

    /// <summary>
    /// Base (non-grouped) DeformableConv2D DCN v2 autograd (#1691): closes the long-standing
    /// "modulation mask not wired" gap. The masked forward now records input/kernel/offset/mask and
    /// back-props to all four; tape grads equal the manual DCN v2 backward called with gradOut=ones.
    /// </summary>
    [Fact]
    public void BaseDeformableConv2D_WithMask_OnTape_BackpropsToAllFour()
    {
        var e = new CpuEngine();
        const int inC = 4, outC = 4, H = 5, W = 5, k = 3, kk = 9;
        int[] s = [1, 1], p = [1, 1], dl = [1, 1];
        var input = Rand(41, 1.0, 1, inC, H, W);
        var kernel = Rand(42, 1.0, outC, inC, k, k);   // base kernel [outC, inC, k, k]
        var offset = Rand(43, 0.6, 1, 2 * kk, H, W);   // deformGroups=1
        var mask = Rand(44, 0.5, 1, kk, H, W);

        using var tape = new GradientTape<double>();
        var output = e.DeformableConv2D(input, kernel, offset, mask, s, p, dl);
        var loss = e.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new[] { input, kernel, offset, mask });

        var gradOut = new Tensor<double>([1, outC, H, W], new Vector<double>(System.Linq.Enumerable.Repeat(1.0, outC * H * W).ToArray()));
        var gI = e.DeformableConv2DBackwardInput(gradOut, input, kernel, offset, mask, [1, inC, H, W], s, p, dl);
        var gK = e.DeformableConv2DBackwardKernel(gradOut, input, offset, mask, [outC, inC, k, k], s, p, dl);
        var gO = e.DeformableConv2DBackwardOffset(gradOut, input, kernel, offset, mask, s, p, dl);
        var gM = e.DeformableConv2DBackwardMask(gradOut, input, kernel, offset, mask, s, p, dl);

        void Same(Tensor<double> tape_, Tensor<double> manual, string name)
        {
            Assert.NotNull(tape_);
            Assert.Equal(manual.Length, tape_.Length);
            for (int i = 0; i < manual.Length; i++)
                Assert.True(System.Math.Abs(tape_[i] - manual[i]) < 1e-9,
                    $"{name}[{i}]: tape={tape_[i]:F9} manual={manual[i]:F9}");
        }
        Same(grads[input], gI, "gradInput");
        Same(grads[kernel], gK, "gradKernel");
        Same(grads[offset], gO, "gradOffset");
        Same(grads[mask], gM, "gradMask");
    }
}
