// Copyright (c) AiDotNet. All rights reserved.
// Softmax and LayerNorm, and the numerical trap that makes them worth testing.
//
// These were blocked on the spec's BODY, not on the number of passes: the activation runs
// once on the finished accumulator, while softmax's denominator needs exp applied to every
// TERM before summing. A pre-reduction slot supplies that; sequencing two kernels does not.
//
// The stability test is the important one. exp(x) overflows fp32 above about 88, so a
// softmax that skipped the max subtraction would return NaN on perfectly ordinary logits
// while still agreeing with any reference that made the same mistake.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenFusedStatisticsTests
{
    /// <summary>Runs a program's passes in order and returns the final output.</summary>
    private static double[] RunProgram(CodegenProgram program, double[] x)
    {
        var produced = new double[program.Passes.Count][];
        for (int p = 0; p < program.Passes.Count; p++)
        {
            var pass = program.Passes[p];
            var operands = new double[pass.Inputs.Count][];
            for (int i = 0; i < pass.Inputs.Count; i++)
            {
                // Parameter 0 is always the source tensor; later parameters are the
                // statistics produced by earlier passes, in order.
                operands[i] = i == 0 ? x : produced[i - 1];
            }
            produced[p] = pass.Interpret(operands);
        }
        return produced[program.Passes.Count - 1];
    }

    private static double[] Fill(int rows, int columns, double scale, int salt)
    {
        var v = new double[rows * columns];
        for (int i = 0; i < v.Length; i++)
            v[i] = scale * ((((i * 37 + salt * 101) % 97) - 48) / 48.0);
        return v;
    }

    /// <summary>Softmax must match an independently written row softmax.</summary>
    [Theory]
    [InlineData(4, 8)]
    [InlineData(16, 33)]
    [InlineData(3, 128)]
    public void Softmax_MatchesAHandWrittenReference(int rows, int columns)
    {
        double[] x = Fill(rows, columns, 3.0, 1);
        double[] got = RunProgram(CodegenFusedStatistics.Softmax(rows, columns), x);

        for (int i = 0; i < rows; i++)
        {
            double max = double.NegativeInfinity;
            for (int j = 0; j < columns; j++) max = Math.Max(max, x[i * columns + j]);

            double sum = 0;
            for (int j = 0; j < columns; j++) sum += Math.Exp(x[i * columns + j] - max);

            for (int j = 0; j < columns; j++)
                Assert.Equal(Math.Exp(x[i * columns + j] - max) / sum, got[i * columns + j], 9);
        }
    }

    /// <summary>Every row must sum to one — the defining property.</summary>
    [Fact]
    public void Softmax_RowsSumToOne()
    {
        const int Rows = 6, Columns = 40;
        double[] got = RunProgram(CodegenFusedStatistics.Softmax(Rows, Columns),
                                  Fill(Rows, Columns, 5.0, 2));

        for (int i = 0; i < Rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < Columns; j++) sum += got[i * Columns + j];
            Assert.Equal(1.0, sum, 9);
        }
    }

    /// <summary>
    /// THE STABILITY TEST. Logits around 200 overflow fp32's exp, and even in fp64 a
    /// softmax without the max subtraction produces inf/inf = NaN. The subtraction is why
    /// pass 1 exists, so this is what proves it is doing its job.
    /// </summary>
    [Fact]
    public void Softmax_SurvivesLargeLogits()
    {
        const int Rows = 4, Columns = 16;
        var x = new double[Rows * Columns];
        for (int i = 0; i < x.Length; i++) x[i] = 700.0 + (i % 5) * 3.0;   // exp(700) overflows fp64

        double[] got = RunProgram(CodegenFusedStatistics.Softmax(Rows, Columns), x);

        foreach (double v in got)
        {
            Assert.False(double.IsNaN(v), "a stable softmax cannot produce NaN on large logits");
            Assert.InRange(v, 0.0, 1.0);
        }
        for (int i = 0; i < Rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < Columns; j++) sum += got[i * Columns + j];
            Assert.Equal(1.0, sum, 9);
        }
    }

    /// <summary>LayerNorm statistics must match a hand-written mean and variance.</summary>
    [Theory]
    [InlineData(5, 12)]
    [InlineData(9, 64)]
    public void LayerNormStatistics_MatchAHandWrittenReference(int rows, int columns)
    {
        double[] x = Fill(rows, columns, 2.0, 3);
        var program = CodegenFusedStatistics.LayerNormStatistics(rows, columns);

        double[] negMean = program.Passes[0].Interpret(new[] { x });
        double[] variance = program.Passes[1].Interpret(new[] { x, negMean });

        for (int i = 0; i < rows; i++)
        {
            double mean = 0;
            for (int j = 0; j < columns; j++) mean += x[i * columns + j];
            mean /= columns;

            Assert.Equal(-mean, negMean[i], 9);

            double var = 0;
            for (int j = 0; j < columns; j++)
            {
                double d = x[i * columns + j] - mean;
                var += d * d;
            }
            var /= columns;

            Assert.Equal(var, variance[i], 9);
        }
    }

    /// <summary>Variance is never negative, however the terms are ordered.</summary>
    [Fact]
    public void LayerNormVariance_IsNonNegative()
    {
        var program = CodegenFusedStatistics.LayerNormStatistics(8, 50);
        double[] x = Fill(8, 50, 7.0, 4);
        double[] negMean = program.Passes[0].Interpret(new[] { x });
        foreach (double v in program.Passes[1].Interpret(new[] { x, negMean }))
            Assert.True(v >= 0.0, "a sum of squares cannot be negative; got " + v);
    }

    /// <summary>Every pass of both programs must emit.</summary>
    [Fact]
    public void AllPasses_Emit()
    {
        foreach (var program in new[]
                 {
                     CodegenFusedStatistics.Softmax(32, 64),
                     CodegenFusedStatistics.LayerNormStatistics(32, 64),
                 })
        {
            foreach (var pass in program.Passes)
            {
                string ptx = new PtxAffineEmitter().Emit(pass, 8, 6);
                Assert.Contains(".visible .entry", ptx, StringComparison.Ordinal);
                Assert.Contains(pass.Name, ptx, StringComparison.Ordinal);
            }
        }
    }

    /// <summary>
    /// The exponential must be inside the reduction, not in the epilogue. If it moved to
    /// the epilogue the kernel would compute exp(sum) rather than sum(exp).
    /// </summary>
    [Fact]
    public void SoftmaxDenominator_AppliesExpPerTerm()
    {
        var recipSum = CodegenFusedStatistics.Softmax(16, 32).Passes[1];

        Assert.Equal(CodegenPreReduceOp.Exp, recipSum.PreReduce);
        Assert.Equal(CodegenReduceKind.Sum, recipSum.Reduce);
        Assert.True(recipSum.PreBiasInput.HasValue, "the max shift must be a pre-reduction bias");
        Assert.Equal(CodegenActivationKind.Reciprocal, recipSum.Activation);
    }

    /// <summary>
    /// Pass 1 produces the NEGATED maximum, because the spec can add a bias but cannot
    /// subtract one. Losing the sign would shift the wrong way and silently change nothing
    /// about the row sums, which still come to one.
    /// </summary>
    [Fact]
    public void SoftmaxPassOne_ProducesTheNegatedMaximum()
    {
        var negMax = CodegenFusedStatistics.Softmax(4, 8).Passes[0];
        Assert.Equal(CodegenReduceKind.Max, negMax.Reduce);
        Assert.Equal(-1.0, negMax.ReduceScale, 12);

        double[] x = Fill(4, 8, 3.0, 5);
        double[] got = negMax.Interpret(new[] { x });
        for (int i = 0; i < 4; i++)
        {
            double max = double.NegativeInfinity;
            for (int j = 0; j < 8; j++) max = Math.Max(max, x[i * 8 + j]);
            Assert.Equal(-max, got[i], 9);
        }
    }
}
