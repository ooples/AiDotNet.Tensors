// Copyright (c) AiDotNet. All rights reserved.
// The warp-per-output reduction lowering (#890 phase 4a): which specs take it, that the
// lane-interleaved rewrite reads exactly the terms the original does, and that the split
// planner counts the 32 threads per output it launches.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenWarpReductionTests
{
    private static CodegenKernelSpec RowReduce(int rows, int inner, CodegenReduceKind reduce,
        CodegenPreReduceOp preReduce = CodegenPreReduceOp.None)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("k", inner));
        var x = new CodegenTensorBinding(0, "x", new[] { rows, inner },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var y = new CodegenTensorBinding(1, "y", new[] { rows },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        return new CodegenKernelSpec("rowreduce", space, new[] { x }, y, new[] { 0 }, reduce,
            preReduce: preReduce);
    }

    private static CodegenKernelSpec MatMul(int m, int k, int n)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("m", m), CodegenAxis.Parallel("n", n), CodegenAxis.Reduce("k", k));
        var a = new CodegenTensorBinding(0, "a", new[] { m, k },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(2) });
        var b = new CodegenTensorBinding(1, "b", new[] { k, n },
            new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) });
        var c = new CodegenTensorBinding(2, "c", new[] { m, n },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);
        return new CodegenKernelSpec("mm", space, new[] { a, b }, c, new[] { 0, 1 }, CodegenReduceKind.Sum);
    }

    [Fact]
    public void ARowStatisticOverTheContiguousAxisIsEligible()
    {
        Assert.True(CodegenWarpReduction.IsEligible(RowReduce(128, 1024, CodegenReduceKind.Sum)));
        Assert.True(CodegenWarpReduction.IsEligible(RowReduce(128, 64, CodegenReduceKind.Max)));
        foreach (var pass in CodegenFusedStatistics.LayerNorm(64, 1024).Passes)
            if (pass.Reduce != CodegenReduceKind.None)
                Assert.True(CodegenWarpReduction.IsEligible(pass), pass.Name);
    }

    [Fact]
    public void WhatTheLoweringCannotCarryIsNotEligible()
    {
        // Too short to give each lane two terms.
        Assert.False(CodegenWarpReduction.IsEligible(RowReduce(128, 32, CodegenReduceKind.Sum)));
        // A contiguous extent no warp divides.
        Assert.False(CodegenWarpReduction.IsEligible(RowReduce(128, 56 * 2, CodegenReduceKind.Sum)));
        // Pointwise: nothing to reduce.
        var pointwise = CodegenFusedStatistics.LayerNorm(64, 1024).Passes[2];
        Assert.False(CodegenWarpReduction.IsEligible(pointwise));
    }

    /// <summary>
    /// A[m,k] does not read n, so one load of it serves a tile of outputs. A warp per output
    /// would give that reuse up, so a contraction keeps the tiled lowering.
    /// </summary>
    [Fact]
    public void AContractionWithCrossOutputReuseIsNotEligible()
    {
        Assert.False(CodegenWarpReduction.IsEligible(MatMul(64, 512, 64)));
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(MatMul(64, 512, 64), 8, 6);
        Assert.False(emitter.WarpReduced);
        Assert.DoesNotContain("shfl.sync", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// The rewrite must read exactly the original's terms: summing the 32 lane partials of
    /// the interleaved spec reproduces the original reduction. Checked by giving the
    /// interleaved spec an output with a lane dimension, so each partial is kept.
    /// </summary>
    [Theory]
    [InlineData(CodegenReduceKind.Sum, CodegenPreReduceOp.None)]
    [InlineData(CodegenReduceKind.Sum, CodegenPreReduceOp.Square)]
    [InlineData(CodegenReduceKind.Max, CodegenPreReduceOp.None)]
    public void LanePartialsCombineToTheOriginal(CodegenReduceKind reduce, CodegenPreReduceOp preReduce)
    {
        const int Rows = 6, Inner = 128;
        var spec = RowReduce(Rows, Inner, reduce, preReduce);
        Assert.True(CodegenWarpReduction.TryLaneInterleave(spec, out var lane));

        int laneAxis = lane.Space.Axes.Count - 1;
        Assert.Equal(CodegenWarpReduction.Lanes, lane.Space.Axes[laneAxis].Extent);
        Assert.Equal(Inner / CodegenWarpReduction.Lanes, lane.Space.Axes[1].Extent);

        var perLane = new CodegenTensorBinding(1, "partials", new[] { Rows, CodegenWarpReduction.Lanes },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(laneAxis) }, isOutput: true);
        var keep = new CodegenKernelSpec("partials", lane.Space, new[] { lane.Inputs[0] }, perLane,
            new[] { 0 }, reduce, preReduce: preReduce);

        var x = new double[Rows * Inner];
        for (int i = 0; i < x.Length; i++) x[i] = ((i * 37) % 101 - 50) / 25.0;
        double[] want = spec.Interpret(new[] { x });
        double[] partials = keep.Interpret(new[] { x });

        for (int r = 0; r < Rows; r++)
        {
            double combined = reduce == CodegenReduceKind.Max ? double.NegativeInfinity : 0.0;
            for (int l = 0; l < CodegenWarpReduction.Lanes; l++)
            {
                double p = partials[r * CodegenWarpReduction.Lanes + l];
                combined = reduce == CodegenReduceKind.Max ? Math.Max(combined, p) : combined + p;
            }
            Assert.Equal(want[r], combined, 9);
        }
    }

    /// <summary>
    /// Lane l on trip t reads term t*32 + l, so a warp's 32 lanes read 32 CONSECUTIVE
    /// elements -- the property the lowering exists for.
    /// </summary>
    [Fact]
    public void ConsecutiveLanesReadConsecutiveElements()
    {
        Assert.True(CodegenWarpReduction.TryLaneInterleave(
            RowReduce(4, 256, CodegenReduceKind.Sum), out var lane));
        var map = lane.Inputs[0].Map[1];
        int laneAxis = lane.Space.Axes.Count - 1;
        int laneCoefficient = 0, tripCoefficient = 0;
        foreach (var term in map.Terms)
        {
            if (term.Axis == laneAxis) laneCoefficient = term.Coefficient;
            if (term.Axis == 1) tripCoefficient = term.Coefficient;
        }
        Assert.Equal(1, laneCoefficient);
        Assert.Equal(CodegenWarpReduction.Lanes, tripCoefficient);
    }

    [Fact]
    public void EligibleReductionsEmitAButterflyAndALeaderStore()
    {
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(RowReduce(512, 1024, CodegenReduceKind.Sum), 7, 5);
        Assert.True(emitter.WarpReduced);
        Assert.Equal(5, CountOccurrences(ptx, "shfl.sync.bfly.b32"));
        Assert.Equal(0, emitter.LaunchBlockThreads % CodegenWarpReduction.Lanes);
        // 512 outputs x 32 lanes, in 256-thread blocks.
        Assert.Equal(512u * 32u / 256u, emitter.LaunchBlocks);

        var off = new PtxAffineEmitter { EnableWarpReduction = false };
        string single = off.Emit(RowReduce(512, 1024, CodegenReduceKind.Sum), 7, 5);
        Assert.False(off.WarpReduced);
        Assert.DoesNotContain("shfl.sync", single, StringComparison.Ordinal);
    }

    /// <summary>
    /// The planner counts 32 threads per warp-reduced output, so a row statistic over
    /// enough rows is left alone, while a per-channel statistic -- 64 outputs -- still splits.
    /// </summary>
    [Fact]
    public void ThePlannerCountsTheWarpsThreads()
    {
        Assert.Null(CodegenSplitReduction.TryPlan(RowReduce(8192, 1024, CodegenReduceKind.Sum)));
        Assert.NotNull(CodegenSplitReduction.TryPlan(CodegenFusedStatistics.BatchNorm2D(32, 64, 56, 56).Passes[1]));
    }

    private static int CountOccurrences(string text, string token)
    {
        int count = 0;
        for (int at = text.IndexOf(token, StringComparison.Ordinal); at >= 0;
             at = text.IndexOf(token, at + token.Length, StringComparison.Ordinal))
            count++;
        return count;
    }
}
