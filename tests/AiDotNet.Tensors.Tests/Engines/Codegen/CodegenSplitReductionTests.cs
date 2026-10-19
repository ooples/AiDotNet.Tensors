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

    /// <summary>
    /// Chunking a reduction axis must reproduce the unsplit result exactly. The chunk
    /// index is folded into the operand maps as an extra term, so getting the stride
    /// wrong reads a shifted slice and still emits.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public void SplitChunked_ReproducesTheUnsplitResult(int factor)
    {
        // A matmul has exactly one reduction axis, which is the case that must chunk.
        const int M = 6, K = 16, N = 5;
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("m", M), CodegenAxis.Parallel("n", N), CodegenAxis.Reduce("k", K));
        var a = new CodegenTensorBinding(0, "a", new[] { M, K },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(2) });
        var b = new CodegenTensorBinding(1, "b", new[] { K, N },
            new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) });
        var outBinding = new CodegenTensorBinding(2, "out", new[] { M, N },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);
        var spec = new CodegenKernelSpec("mm", space, new[] { a, b }, outBinding,
            new[] { 0, 1 }, CodegenReduceKind.Sum);

        var (partial, combine) = CodegenSplitReduction.SplitChunked(spec, 2, factor);

        // The partial must still REDUCE -- that is the whole point of chunking rather
        // than promoting, and it is what the losing measurements lacked.
        Assert.Equal(CodegenReduceKind.Sum, partial.Reduce);

        double[] av = Fill((long)M * K, 1), bv = Fill((long)K * N, 2);
        double[] direct = spec.Interpret(new[] { av, bv });
        double[] staged = combine.Interpret(new[] { partial.Interpret(new[] { av, bv }) });

        Assert.Equal(direct.Length, staged.Length);
        for (int i = 0; i < direct.Length; i++) Assert.Equal(direct[i], staged[i], 9);
    }

    /// <summary>
    /// A chunk count that does not divide the extent would need a bounds guard on a
    /// reduction axis, which reads outside the operand, so it must be refused.
    /// </summary>
    [Fact]
    public void SplitChunked_RefusesAFactorThatDoesNotDivide()
    {
        var spec = DepthwiseWeightGradient(2, 4, 8, 8);
        int axis = CodegenSplitReduction.ChooseAxes(spec)[0];
        int extent = spec.Space.Axes[axis].Extent;
        Assert.Throws<ArgumentOutOfRangeException>(
            () => CodegenSplitReduction.SplitChunked(spec, axis, extent - 1));
    }

    /// <summary>
    /// When the chosen axis IS the whole reduction, the plan must chunk it rather than
    /// promote it whole -- promoting leaves the combine doing the entire reduction with
    /// the original thread count, which measured up to 3.80x SLOWER than not splitting.
    /// </summary>
    [Fact]
    public void TryPlan_ChunksWhenTheAxisIsTheEntireReduction()
    {
        const int M = 64, K = 512, N = 8;
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("m", M), CodegenAxis.Parallel("n", N), CodegenAxis.Reduce("k", K));
        var a = new CodegenTensorBinding(0, "a", new[] { M, K },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(2) });
        var b = new CodegenTensorBinding(1, "b", new[] { K, N },
            new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) });
        var outBinding = new CodegenTensorBinding(2, "out", new[] { M, N },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);
        var spec = new CodegenKernelSpec("mm", space, new[] { a, b }, outBinding,
            new[] { 0, 1 }, CodegenReduceKind.Sum);

        var plan = CodegenSplitReduction.TryPlan(spec);
        Assert.NotNull(plan);

        // Chunked, not promoted: the partial still reduces, and the combine's reduction
        // is strictly shorter than the original's.
        Assert.Equal(CodegenReduceKind.Sum, plan!.Partial.Reduce);
        int combineTrips = plan.Combine.Space.Axes[plan.Combine.Space.Axes.Count - 1].Extent;
        Assert.True(combineTrips < K,
            "the combine must reduce fewer than the original " + K + " terms, got " + combineTrips);

        double[] av = Fill((long)M * K, 3), bv = Fill((long)K * N, 4);
        double[] want = spec.Interpret(new[] { av, bv });
        double[] got = plan.Combine.Interpret(new[] { plan.Partial.Interpret(new[] { av, bv }) });
        for (int i = 0; i < want.Length; i++) Assert.Equal(want[i], got[i], 9);
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
    /// An epilogue must MOVE to the combine pass, not be applied once per partial -- which
    /// would add the bias once for every promoted position -- and not be refused, since
    /// refusing it would exclude every linear layer from the split.
    /// </summary>
    [Fact]
    public void Epilogue_MovesToTheCombinePassAndKeepsTheAnswer()
    {
        var spec = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(2, 4, 8, 8);
        var (partial, combine) = CodegenSplitReduction.Split(spec, spec.Space.ReductionAxes[0]);

        // The partial pass computes the raw sum: no bias, no scale, no activation.
        Assert.False(partial.BiasInput.HasValue);
        Assert.False(partial.ScaleInput.HasValue);
        Assert.Equal(CodegenActivationKind.None, partial.Activation);

        // The combine carries them instead.
        Assert.True(combine.BiasInput.HasValue);
        Assert.Equal(spec.Activation, combine.Activation);

        // And the two-pass answer still matches the one-pass answer.
        var operands = new double[spec.Inputs.Count][];
        for (int i = 0; i < spec.Inputs.Count; i++)
            operands[i] = Fill(Elements(spec.Inputs[i].Shape), i + 1);

        var partialOperands = new double[partial.Inputs.Count][];
        for (int i = 0; i < partial.ProductInputs.Count; i++)
            partialOperands[i] = operands[spec.ProductInputs[i]];

        var combineOperands = new double[combine.Inputs.Count][];
        combineOperands[0] = partial.Interpret(partialOperands);
        combineOperands[combine.BiasInput!.Value] = operands[spec.BiasInput!.Value];

        double[] direct = spec.Interpret(operands);
        double[] staged = combine.Interpret(combineOperands);

        Assert.Equal(direct.Length, staged.Length);
        for (int i = 0; i < direct.Length; i++)
            Assert.Equal(direct[i], staged[i], 9);
    }

    /// <summary>
    /// Every kernel the autotuner can win on must be reachable through the split, and the
    /// three weight gradients are exactly those -- measured at 17.12x, 35.09x and 2.03x.
    /// </summary>
    [Theory]
    [InlineData("depthwise_conv2d_3x3_bwd_weights")]
    [InlineData("conv2d_1x1_bwd_weights")]
    [InlineData("conv2d_3x3_bwd_weights")]
    public void WeightGradientCatalogKernels_AllPlanASplit(string kernelName)
    {
        var entry = CodegenKernelCatalog.Find(kernelName);
        Assert.NotNull(entry);

        var plan = CodegenSplitReduction.TryPlan(entry!.Bench);
        Assert.NotNull(plan);
        Assert.True(plan!.TempElements > entry.Bench.Output.ElementCount);
    }

    /// <summary>
    /// The split requires ASSOCIATIVITY, and both reduce kinds that exist have it.
    /// </summary>
    /// <remarks>
    /// This test used to assert that MAX was refused, which encoded the bug rather than the
    /// requirement: max(max(a,b),c) = max(a,max(b,c)) associates exactly as addition does.
    ///
    /// The throwing branch is currently UNREACHABLE and is asserted through the predicate
    /// instead of by constructing a spec. CodegenReduceKind has only Sum, Max and None; None
    /// cannot coexist with a reduction axis -- the spec constructor refuses it -- so no
    /// non-associative reduction can be built today. The guard is there for the next reduce
    /// kind added, and saying so is more honest than a test that fabricates an impossible
    /// spec to reach it.
    /// </remarks>
    [Fact]
    public void SplitRequiresAnAssociativeReduction()
    {
        Assert.True(CodegenSplitReduction.IsSplittable(CodegenReduceKind.Sum));
        Assert.True(CodegenSplitReduction.IsSplittable(CodegenReduceKind.Max));
        Assert.False(CodegenSplitReduction.IsSplittable(CodegenReduceKind.None));
    }

    // ---- splitting a MAXIMUM ---------------------------------------------------------------
    //
    // The splitter refused Max, on the stated grounds that "the combine pass has to be the same
    // associative operation, and Max is not summed partials". The first half is right; the
    // second does not follow. max(max(a,b),c) = max(a,max(b,c)) exactly as addition associates,
    // so the combine over Max partials is a Max.
    //
    // The cost of that mistaken premise was concrete: a row-wise maximum ran one thread per
    // row, which profiles at 32 sectors per global request against an ideal of 4 -- an
    // eightfold traffic waste -- and 0.04 waves per multiprocessor. It was the only reduction
    // shape with no available fix, for a reason that was not a property of Max.

    private static CodegenKernelSpec RowReduce(int rows, int inner, CodegenReduceKind reduce)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("k", inner));

        var x = new CodegenTensorBinding(0, "x", new[] { rows, inner },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var y = new CodegenTensorBinding(1, "y", new[] { rows },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        return new CodegenKernelSpec("rowreduce", space, new[] { x }, y, new[] { 0 }, reduce);
    }

    private static CodegenKernelSpec RowMaxWithIndices(int rows, int inner)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("k", inner));
        var x = new CodegenTensorBinding(0, "x", new[] { rows, inner },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var values = new CodegenTensorBinding(1, "values", new[] { rows },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var indices = new CodegenTensorBinding(2, "indices", new[] { rows },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        return new CodegenKernelSpec(
            "rowmax_indices", space, new[] { x }, values, new[] { 0 }, CodegenReduceKind.Max,
            secondaryOutput: indices, secondaryIndexExpr: CodegenAffineExpr.Axis(1));
    }

    private static double[] MaxSplitValues(int count)
    {
        var data = new double[count];
        for (int i = 0; i < count; i++) data[i] = (((i * 37) % 101) - 50) / 8.0;
        return data;
    }

    /// <summary>Max is associative, so it splits.</summary>
    [Fact]
    public void MaxIsSplittable()
    {
        Assert.True(CodegenSplitReduction.IsSplittable(CodegenReduceKind.Max));
        Assert.True(CodegenSplitReduction.IsSplittable(CodegenReduceKind.Sum));
        Assert.False(CodegenSplitReduction.IsSplittable(CodegenReduceKind.None));
    }

    /// <summary>Argmax metadata stays single-pass until split temporaries carry positions.</summary>
    [Fact]
    public void MaxWithArgmaxOutput_IsRefusedByEverySplitEntryPoint()
    {
        var spec = RowMaxWithIndices(64, 128);

        Assert.True(CodegenSplitReduction.IsSplittable(CodegenReduceKind.Max));
        Assert.False(CodegenSplitReduction.IsSplittable(spec));
        Assert.Empty(CodegenSplitReduction.ChooseAxes(spec));
        Assert.Null(CodegenSplitReduction.TryPlan(spec));

        var whole = Assert.Throws<NotSupportedException>(
            () => CodegenSplitReduction.Split(spec, 1));
        var chunked = Assert.Throws<NotSupportedException>(
            () => CodegenSplitReduction.SplitChunked(spec, 1, 8));
        Assert.Contains("argmax secondary output", whole.Message, StringComparison.Ordinal);
        Assert.Contains("argmax secondary output", chunked.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// The split maximum must equal the unsplit one, checked against a hand-written scan
    /// rather than against the single-pass spec: a shared interpreter can share a mistake.
    /// </summary>
    [Theory]
    [InlineData(64, 128, 8)]
    [InlineData(32, 256, 16)]
    [InlineData(16, 64, 4)]
    public void SplitMax_MatchesAHandWrittenScan(int rows, int inner, int factor)
    {
        var spec = RowReduce(rows, inner, CodegenReduceKind.Max);
        var (partial, combine) = CodegenSplitReduction.SplitChunked(spec, 1, factor);

        double[] x = MaxSplitValues(rows * inner);
        double[] got = combine.Interpret(new[] { partial.Interpret(new[] { x }) });

        for (int i = 0; i < rows; i++)
        {
            double best = double.NegativeInfinity;
            for (int k = 0; k < inner; k++) best = Math.Max(best, x[i * inner + k]);
            Assert.Equal(best, got[i], 9);
        }
    }

    /// <summary>The combine must reduce with the SAME operation, not with a sum.</summary>
    [Fact]
    public void CombineUsesTheSameOperation()
    {
        var (_, combine) = CodegenSplitReduction.SplitChunked(
            RowReduce(64, 128, CodegenReduceKind.Max), 1, 8);

        Assert.Equal(CodegenReduceKind.Max, combine.Reduce);
    }

    /// <summary>A sum still splits into a sum -- the fix must not have swapped the operation.</summary>
    [Fact]
    public void SumStillSplitsIntoASum()
    {
        var spec = RowReduce(64, 128, CodegenReduceKind.Sum);
        var (partial, combine) = CodegenSplitReduction.SplitChunked(spec, 1, 8);

        Assert.Equal(CodegenReduceKind.Sum, combine.Reduce);

        double[] x = MaxSplitValues(64 * 128);
        double[] got = combine.Interpret(new[] { partial.Interpret(new[] { x }) });

        for (int i = 0; i < 64; i++)
        {
            double sum = 0;
            for (int k = 0; k < 128; k++) sum += x[i * 128 + k];
            Assert.Equal(sum, got[i], 6);
        }
    }

    /// <summary>The planner now offers a plan for a maximum, where it previously offered none.</summary>
    [Fact]
    public void PlannerHandlesMax()
    {
        Assert.NotNull(CodegenSplitReduction.TryPlan(RowReduce(4096, 1024, CodegenReduceKind.Max)));
    }

    // ---- per-term transforms -------------------------------------------------------------
    //
    // A pre-bias and a pre-reduction act on each TERM before it is summed. The split used to
    // build its partial pass from the product operands alone, so a softmax denominator or a
    // variance that took the split summed raw x instead of exp(x - max) or (x - mean)^2 --
    // a plausible number, silently wrong.

    /// <summary>
    /// Runs a split through <see cref="CodegenSplitReduction.PartialOperandSources"/> and
    /// <see cref="CodegenSplitReduction.CombineOperandSources"/>, the mapping every launcher
    /// binds with, so the test exercises the same binding the device paths use.
    /// </summary>
    private static double[] RunSplit(
        CodegenKernelSpec spec, CodegenKernelSpec partial, CodegenKernelSpec combine, double[][] operands)
    {
        int[] partialSources = CodegenSplitReduction.PartialOperandSources(spec);
        Assert.Equal(partial.Inputs.Count, partialSources.Length);
        var partialOperands = new double[partialSources.Length][];
        for (int i = 0; i < partialSources.Length; i++) partialOperands[i] = operands[partialSources[i]];

        int[] combineSources = CodegenSplitReduction.CombineOperandSources(spec);
        Assert.Equal(combine.Inputs.Count, 1 + combineSources.Length);
        var combineOperands = new double[combine.Inputs.Count][];
        combineOperands[0] = partial.Interpret(partialOperands);
        for (int i = 0; i < combineSources.Length; i++) combineOperands[1 + i] = operands[combineSources[i]];
        return combine.Interpret(combineOperands);
    }

    private static CodegenKernelSpec RowVariance(int rows, int inner, double epsilon)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("k", inner));
        var x = new CodegenTensorBinding(0, "x", new[] { rows, inner },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var negMean = new CodegenTensorBinding(1, "negmean", new[] { rows },
            new[] { CodegenAffineExpr.Axis(0) });
        var y = new CodegenTensorBinding(2, "rstd", new[] { rows },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        return new CodegenKernelSpec("rowvar", space, new[] { x, negMean }, y, new[] { 0 },
            CodegenReduceKind.Sum, activation: CodegenActivationKind.Rsqrt,
            reduceScale: 1.0 / inner, preReduce: CodegenPreReduceOp.Square, preBiasInput: 1,
            reduceOffset: epsilon);
    }

    /// <summary>
    /// rsqrt(mean((x - mean)^2) + eps), split, must equal a hand-written scan. Holding it to
    /// an independent reference rather than to the unsplit spec matters here: the bug was in
    /// what the split CARRIED, and both passes interpret through the same spec code.
    /// </summary>
    [Theory]
    [InlineData(16, 128, 8)]
    [InlineData(8, 96, 4)]
    public void SplitKeepsPreBiasAndPreReduceOnThePartial(int rows, int inner, int factor)
    {
        const double Epsilon = 1e-3;
        var spec = RowVariance(rows, inner, Epsilon);
        var (partial, combine) = CodegenSplitReduction.SplitChunked(spec, 1, factor);

        Assert.Equal(CodegenPreReduceOp.Square, partial.PreReduce);
        Assert.Equal(CodegenPreReduceOp.None, combine.PreReduce);
        Assert.True(partial.PreBiasInput.HasValue);
        Assert.Equal(Epsilon, combine.ReduceOffset);
        Assert.Equal(0.0, partial.ReduceOffset);

        double[] x = Fill(rows * inner, 3);
        var negMean = new double[rows];
        var want = new double[rows];
        for (int i = 0; i < rows; i++)
        {
            double mean = 0;
            for (int k = 0; k < inner; k++) mean += x[i * inner + k];
            mean /= inner;
            negMean[i] = -mean;
            double sum = 0;
            for (int k = 0; k < inner; k++) sum += (x[i * inner + k] - mean) * (x[i * inner + k] - mean);
            want[i] = 1.0 / Math.Sqrt(sum / inner + Epsilon);
        }

        double[] got = RunSplit(spec, partial, combine, new[] { x, negMean });
        for (int i = 0; i < rows; i++) Assert.Equal(want[i], got[i], 9);
    }

    /// <summary>The same through the promoting split, which builds its partial separately.</summary>
    [Fact]
    public void PromotingSplitKeepsPreReduce()
    {
        // Two reduction axes, so promoting one leaves the partial real work.
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", 6), CodegenAxis.Reduce("a", 8), CodegenAxis.Reduce("b", 16));
        var x = new CodegenTensorBinding(0, "x", new[] { 6, 8, 16 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1), CodegenAffineExpr.Axis(2) });
        var y = new CodegenTensorBinding(1, "y", new[] { 6 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var spec = new CodegenKernelSpec("expsum", space, new[] { x }, y, new[] { 0 },
            CodegenReduceKind.Sum, preReduce: CodegenPreReduceOp.Exp);

        var (partial, combine) = CodegenSplitReduction.Split(spec, 1);
        double[] xv = Fill(6 * 8 * 16, 5);
        double[] got = RunSplit(spec, partial, combine, new[] { xv });

        for (int i = 0; i < 6; i++)
        {
            double sum = 0;
            for (int k = 0; k < 8 * 16; k++) sum += Math.Exp(xv[i * 128 + k]);
            Assert.Equal(sum, got[i], 9);
        }
    }

    /// <summary>
    /// A post-scale and post-bias move to the combine with the rest of the epilogue, in the
    /// order the mapping reports them.
    /// </summary>
    [Fact]
    public void ExtendedEpilogueMovesToTheCombine()
    {
        const int Rows = 8, Inner = 64;
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", Rows), CodegenAxis.Reduce("k", Inner));
        CodegenTensorBinding Vector(int index, string name) => new(index, name, new[] { Rows },
            new[] { CodegenAffineExpr.Axis(0) });
        var x = new CodegenTensorBinding(0, "x", new[] { Rows, Inner },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var y = new CodegenTensorBinding(5, "y", new[] { Rows },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var spec = new CodegenKernelSpec("affine_sum", space,
            new[] { x, Vector(1, "bias"), Vector(2, "scale"), Vector(3, "pscale"), Vector(4, "pbias") },
            y, new[] { 0 }, CodegenReduceKind.Sum,
            biasInput: 1, scaleInput: 2, reduceOffset: 0.25, postScaleInput: 3, postBiasInput: 4);

        Assert.Equal(new[] { 1, 2, 3, 4 }, CodegenSplitReduction.CombineOperandSources(spec));
        var (partial, combine) = CodegenSplitReduction.SplitChunked(spec, 1, 8);
        Assert.False(partial.HasExtendedEpilogue);

        var operands = new double[5][];
        operands[0] = Fill(Rows * Inner, 1);
        for (int i = 1; i < 5; i++) operands[i] = Fill(Rows, i + 7);
        double[] got = RunSplit(spec, partial, combine, operands);

        for (int r = 0; r < Rows; r++)
        {
            double sum = 0;
            for (int k = 0; k < Inner; k++) sum += operands[0][r * Inner + k];
            double want = ((sum + 0.25 + operands[1][r]) * operands[2][r]) * operands[3][r] + operands[4][r];
            Assert.Equal(want, got[r], 9);
        }
    }

    /// <summary>
    /// A spec whose extra outputs the combine cannot rebuild is not splittable, rather than
    /// split with its extra outputs dropped.
    /// </summary>
    [Fact]
    public void ExtraOutputsAreRefusedNotDropped()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", 8), CodegenAxis.Reduce("k", 64));
        var x = new CodegenTensorBinding(0, "x", new[] { 8, 64 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var y = new CodegenTensorBinding(1, "y", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var twice = new CodegenTensorBinding(2, "twice", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var spec = new CodegenKernelSpec("sum_and_twice", space, new[] { x }, y, new[] { 0 },
            CodegenReduceKind.Sum,
            extraOutputs: new[] { new CodegenExtraOutput(twice, CodegenExtraOutputKind.AffineOfPrimary, null, 2.0) });

        Assert.False(CodegenSplitReduction.IsSplittable(spec));
        Assert.Null(CodegenSplitReduction.TryPlan(spec));
        Assert.Throws<NotSupportedException>(() => CodegenSplitReduction.SplitChunked(spec, 1, 8));
    }
}
