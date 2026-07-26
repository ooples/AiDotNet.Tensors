// Copyright (c) AiDotNet. All rights reserved.
// Splitting a reduction must not change the answer.
//
// The split exists because depthwise_conv2d_3x3_bwd_weights ran 4052.6 us against a
// 3.8 us roofline -- 1081x off -- with 3 blocks on a 68-SM device, and no tile could fix
// it. A transform that buys a thousandfold and quietly changes the result is worthless,
// so the bar is the same one every other kernel is held to: exact agreement with the
// unsplit operator's own fp64 interpretation.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenSplitReductionTests
{
    private static double[] Fill(long count, int salt)
    {
        var v = new double[count];
        for (long i = 0; i < count; i++) v[i] = (((i * 37 + salt * 101) % 97) - 48) / 64.0;
        return v;
    }

    private static long Elements(IReadOnlyList<int> shape)
    {
        long total = 1;
        foreach (int d in shape) total *= d;
        return total;
    }

    /// <summary>A weight gradient: small output, enormous reduction. The motivating case.</summary>
    private static CodegenKernelSpec DepthwiseWeightGradient(int n, int c, int h, int w)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", n), CodegenAxis.Parallel("c", c),
            CodegenAxis.Parallel("oh", h), CodegenAxis.Parallel("ow", w),
            CodegenAxis.Reduce("kh", 3), CodegenAxis.Reduce("kw", 3));
        const int N = 0, C = 1, OH = 2, OW = 3, KH = 4, KW = 5;

        var input = new CodegenTensorBinding(0, "input", new[] { n, c, h, w },
            new[]
            {
                CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
                CodegenAffineExpr.Window(OH, KH, 1, 1), CodegenAffineExpr.Window(OW, KW, 1, 1)
            });
        var weights = new CodegenTensorBinding(1, "weights", new[] { c, 3, 3 },
            new[] { CodegenAffineExpr.Axis(C), CodegenAffineExpr.Axis(KH), CodegenAffineExpr.Axis(KW) });
        var output = new CodegenTensorBinding(2, "output", new[] { n, c, h, w },
            new[]
            {
                CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
                CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
            }, isOutput: true);

        var forward = new CodegenKernelSpec("dw3x3", space, new[] { input, weights }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum);

        return CodegenAdjoint.BackwardWeights(forward, 1);
    }

    /// <summary>
    /// The split must reproduce the unsplit result exactly: run the partial pass, sum it
    /// with the combine pass, and compare against interpreting the original.
    /// </summary>
    [Fact]
    public void SplitThenCombine_ReproducesTheUnsplitResult()
    {
        var spec = DepthwiseWeightGradient(2, 4, 8, 8);
        int axis = CodegenSplitReduction.ChooseAxis(spec);
        Assert.True(axis >= 0, "a 4x3x3 output with a long reduction must be worth splitting");

        var (partial, combine) = CodegenSplitReduction.Split(spec, axis);

        var operands = new double[spec.Inputs.Count][];
        for (int i = 0; i < spec.Inputs.Count; i++)
            operands[i] = Fill(Elements(spec.Inputs[i].Shape), i + 1);

        double[] direct = spec.Interpret(operands);
        double[] staged = combine.Interpret(new[] { partial.Interpret(operands) });

        Assert.Equal(direct.Length, staged.Length);
        for (int i = 0; i < direct.Length; i++)
            Assert.Equal(direct[i], staged[i], 9);
    }

    /// <summary>
    /// Splitting must raise the thread count by the promoted axis's extent -- that is
    /// the entire point, and it is what no tile choice could achieve.
    /// </summary>
    [Fact]
    public void Split_MultipliesTheAvailableParallelism()
    {
        var spec = DepthwiseWeightGradient(2, 4, 8, 8);
        int axis = CodegenSplitReduction.ChooseAxis(spec);
        var (partial, _) = CodegenSplitReduction.Split(spec, axis);

        long before = spec.Output.ElementCount;
        long after = partial.Output.ElementCount;
        Assert.Equal(before * spec.Space.Axes[axis].Extent, after);
        Assert.True(after > before);
    }

    /// <summary>
    /// Promoting SEVERAL axes must also reproduce the unsplit result. One axis took the
    /// motivating kernel from 4079.6 us to 240.8 us and left it 63x off its roofline
    /// with 126 blocks, so the chooser keeps promoting until the device is full -- and
    /// the multi-axis form has to be as exact as the single-axis one.
    /// </summary>
    [Fact]
    public void PromotingSeveralAxes_ReproducesTheUnsplitResult()
    {
        var spec = DepthwiseWeightGradient(2, 4, 8, 8);
        var promoted = CodegenSplitReduction.ChooseAxes(spec);
        Assert.True(promoted.Count >= 2, "a 576-thread kernel needs more than one axis to fill 68 SMs");

        var (partial, combine) = CodegenSplitReduction.Split(spec, promoted);

        var operands = new double[spec.Inputs.Count][];
        for (int i = 0; i < spec.Inputs.Count; i++)
            operands[i] = Fill(Elements(spec.Inputs[i].Shape), i + 1);

        double[] direct = spec.Interpret(operands);
        double[] staged = combine.Interpret(new[] { partial.Interpret(operands) });

        Assert.Equal(direct.Length, staged.Length);
        for (int i = 0; i < direct.Length; i++)
            Assert.Equal(direct[i], staged[i], 9);
    }

    /// <summary>
    /// The chooser must stop once the device is full: every extra promoted axis
    /// multiplies the temporary and the combine pass's read volume by its extent.
    /// </summary>
    [Fact]
    public void ChooseAxes_StopsOnceTheDeviceIsFull()
    {
        var spec = DepthwiseWeightGradient(8, 64, 56, 56);
        var promoted = CodegenSplitReduction.ChooseAxes(spec);
        var (partial, _) = CodegenSplitReduction.Split(spec, promoted);

        long blocks = (partial.Output.ElementCount + 255) / 256;
        Assert.True(blocks >= 68L * 4, "must promote enough to fill the device: " + blocks);

        // ...and not one axis more than that took. Axes are promoted largest-first, so
        // the smallest-extent one is the last that was needed; without it the kernel
        // must fall short, or it should never have been promoted.
        int smallest = int.MaxValue;
        foreach (int a in promoted) smallest = Math.Min(smallest, spec.Space.Axes[a].Extent);
        long without = partial.Output.ElementCount / smallest;
        Assert.True((without + 255) / 256 < 68L * 4,
            "dropping the smallest promoted axis should fall short, or it was not needed");
    }

    /// <summary>
    /// The default plan must be the MEASURED choice, one axis, not the modelled one.
    /// Two axes were predicted to win and measured 334.7 us against 236.4 us.
    /// </summary>
    [Fact]
    public void TryPlan_PromotesTheMeasuredWinnerAndSizesItsTemporary()
    {
        var spec = DepthwiseWeightGradient(2, 4, 8, 8);
        var plan = CodegenSplitReduction.TryPlan(spec);

        Assert.NotNull(plan);
        Assert.Single(plan!.PromotedAxes);
        Assert.Equal(plan.Partial.Output.ElementCount, plan.TempElements);
        Assert.Equal(Elements(plan.Combine.Inputs[0].Shape), plan.TempElements);
        Assert.Equal(spec.Output.ElementCount, plan.Combine.Output.ElementCount);
    }

    /// <summary>A kernel that fills the device must get no plan at all.</summary>
    [Fact]
    public void TryPlan_ReturnsNullForAKernelThatDoesNotNeedIt()
    {
        Assert.Null(CodegenSplitReduction.TryPlan(
            CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(32, 64, 56, 56)));
    }

    /// <summary>Both halves must be emittable, not merely expressible.</summary>
    [Fact]
    public void BothPasses_Emit()
    {
        var spec = DepthwiseWeightGradient(2, 4, 8, 8);
        var (partial, combine) = CodegenSplitReduction.Split(spec, CodegenSplitReduction.ChooseAxis(spec));

        foreach (var half in new[] { partial, combine })
        {
            string ptx = new PtxAffineEmitter().Emit(half, 8, 6);
            Assert.Contains(".visible .entry", ptx, StringComparison.Ordinal);
            Assert.Contains(half.Name, ptx, StringComparison.Ordinal);
        }
    }

    /// <summary>
    /// A kernel that already fills the device must NOT be split: the extra launch and
    /// temporary would be pure cost.
    /// </summary>
    [Fact]
    public void KernelThatAlreadyFillsTheDevice_IsNotSplit()
    {
        // A forward depthwise convolution has a large output, so it is already parallel.
        var big = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(32, 64, 56, 56);
        Assert.Equal(-1, CodegenSplitReduction.ChooseAxis(big));
    }

    /// <summary>
    /// An epilogue cannot be split, and must be refused rather than applied once per
    /// partial -- which would add the bias S times.
    /// </summary>
    [Fact]
    public void EpilogueOperator_IsRefused()
    {
        var withBias = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(2, 4, 8, 8);
        Assert.Throws<NotSupportedException>(
            () => CodegenSplitReduction.Split(withBias, withBias.Space.ReductionAxes[0]));
    }

    /// <summary>A max reduction has no summed-partial combine and must be refused.</summary>
    [Fact]
    public void NonSumReduction_IsRefused()
    {
        var entry = CodegenKernelCatalog.Find("maxpool2d_2x2");
        Assert.NotNull(entry);
        Assert.Throws<NotSupportedException>(
            () => CodegenSplitReduction.Split(entry!.Bench, entry.Bench.Space.ReductionAxes[0]));
    }
}
