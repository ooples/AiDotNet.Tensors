// Copyright (c) AiDotNet. All rights reserved.
// Expression bodies and the normalisation backward programs (#890 phase 4a).
//
// The backward programs are held to gradients derived independently here -- by finite
// differences of a hand-written forward, not by re-stating the closed form the programs use --
// so a sign or a missing term in a declared body cannot agree with a reference that shares it.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenBackwardProgramTests
{
    private static double[] Fill(long count, double scale, int salt)
    {
        var v = new double[count];
        for (long i = 0; i < count; i++) v[i] = scale * ((((i * 37 + salt * 101) % 97) - 48) / 48.0) + 0.05 * salt;
        return v;
    }

    // ---- expression bodies ---------------------------------------------------------------

    [Fact]
    public void ExpressionEvaluatesRemapsAndListsItsOperands()
    {
        // (x0 + x2) * -(x1) + 0.5
        var e = CodegenExpr.Add(
            CodegenExpr.Multiply(CodegenExpr.Add(CodegenExpr.Operand(0), CodegenExpr.Operand(2)),
                                 CodegenExpr.Negate(CodegenExpr.Operand(1))),
            CodegenExpr.Constant(0.5));
        double[] v = { 2.0, 3.0, 5.0 };
        Assert.Equal((2.0 + 5.0) * -3.0 + 0.5, e.Evaluate(i => v[i]), 12);
        Assert.Equal(new[] { 0, 1, 2 }, e.Operands());
        Assert.Equal(new[] { 10, 11, 12 }, e.Remap(i => i + 10).Operands());
        Assert.Throws<ArgumentOutOfRangeException>(() => CodegenExpr.Constant(double.NaN));
    }

    [Fact]
    public void ABodyMustListExactlyTheInputsItReads()
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", 8));
        CodegenTensorBinding In(int p, string n) => new(p, n, new[] { 8 }, new[] { CodegenAffineExpr.Axis(0) });
        var y = new CodegenTensorBinding(2, "y", new[] { 8 }, new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var body = CodegenExpr.Subtract(CodegenExpr.Operand(0), CodegenExpr.Operand(1));

        var ok = new CodegenKernelSpec("sub", space, new[] { In(0, "a"), In(1, "b") }, y,
            new[] { 0, 1 }, CodegenReduceKind.None, body: body);
        double[] got = ok.Interpret(new[] { Fill(8, 1, 1), Fill(8, 1, 2) });
        double[] a = Fill(8, 1, 1), b = Fill(8, 1, 2);
        for (int i = 0; i < 8; i++) Assert.Equal(a[i] - b[i], got[i], 12);

        Assert.Throws<ArgumentException>(() => new CodegenKernelSpec("sub", space,
            new[] { In(0, "a"), In(1, "b") }, y, new[] { 0 }, CodegenReduceKind.None, body: body));
        Assert.Throws<ArgumentException>(() => new CodegenKernelSpec("sub", space,
            new[] { In(0, "a"), In(1, "b") }, y, new[] { 0, 1 }, CodegenReduceKind.None,
            preBiasInput: 1, body: body));
    }

    /// <summary>A product-shaped lowering must refuse a body it cannot carry.</summary>
    [Fact]
    public void ProductLoweringsRefuseAnExpressionBody()
    {
        var spec = CodegenFusedStatistics.LayerNormBackward(64, 64).Passes[3];
        Assert.True(spec.HasExpressionBody);
        Assert.Throws<NotSupportedException>(() => CodegenAdjoint.BackwardData(spec, 0));
    }

    [Fact]
    public void TheEmitterLowersABodyOneRoundedInstructionPerNode()
    {
        var dx = CodegenFusedStatistics.SoftmaxBackward(64, 256).Passes[1];
        string ptx = new PtxAffineEmitter { EnableStreamingRows = false }.Emit(dx, 7, 5);
        Assert.Contains("neg.f32", ptx, StringComparison.Ordinal);
        Assert.Contains("add.rn.f32", ptx, StringComparison.Ordinal);
        Assert.Contains("mul.rn.f32", ptx, StringComparison.Ordinal);
    }

    // ---- backward programs against finite differences --------------------------------------

    private static double[] LayerNormForward(double[] x, double[] g, double[] b, int rows, int cols, double eps)
    {
        var y = new double[rows * cols];
        for (int i = 0; i < rows; i++)
        {
            double mean = 0, var = 0;
            for (int j = 0; j < cols; j++) mean += x[i * cols + j];
            mean /= cols;
            for (int j = 0; j < cols; j++) var += (x[i * cols + j] - mean) * (x[i * cols + j] - mean);
            double rstd = 1.0 / Math.Sqrt(var / cols + eps);
            for (int j = 0; j < cols; j++) y[i * cols + j] = (x[i * cols + j] - mean) * rstd * g[j] + b[j];
        }
        return y;
    }

    private static double Dot(double[] a, double[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    /// <summary>
    /// Every gradient of LayerNorm backward against a central difference of the loss
    /// L = sum(dy * LayerNorm(x)), perturbing each x, gamma and beta element in turn.
    /// </summary>
    [Fact]
    public void LayerNormBackward_MatchesFiniteDifferences()
    {
        const int Rows = 3, Cols = 8;
        const double Eps = 1e-5, H = 1e-5;
        double[] x = Fill(Rows * Cols, 2.0, 1), dy = Fill(Rows * Cols, 1.0, 2);
        double[] g = Fill(Cols, 0.7, 3), b = Fill(Cols, 0.3, 4);

        var forward = CodegenFusedStatistics.LayerNorm(Rows, Cols, Eps)
            .Interpret(new Dictionary<string, double[]> { ["x"] = x, ["gamma"] = g, ["beta"] = b });
        var grads = CodegenFusedStatistics.LayerNormBackward(Rows, Cols).Interpret(new Dictionary<string, double[]>
        {
            ["x"] = x, ["dy"] = dy, ["gamma"] = g,
            ["negmean"] = forward["negmean"], ["rstd"] = forward["rstd"],
        });

        double Loss(double[] xv, double[] gv, double[] bv) => Dot(dy, LayerNormForward(xv, gv, bv, Rows, Cols, Eps));
        void Check(double[] param, double[] analytic, Func<double[], double> loss)
        {
            for (int k = 0; k < param.Length; k++)
            {
                double keep = param[k];
                param[k] = keep + H; double up = loss(param);
                param[k] = keep - H; double down = loss(param);
                param[k] = keep;
                Assert.Equal((up - down) / (2 * H), analytic[k], 5);
            }
        }
        Check(x, grads["dx"], xv => Loss(xv, g, b));
        Check(g, grads["dgamma"], gv => Loss(x, gv, b));
        Check(b, grads["dbeta"], bv => Loss(x, g, bv));
    }

    [Fact]
    public void RmsNormBackward_MatchesFiniteDifferences()
    {
        const int Rows = 3, Cols = 8;
        const double Eps = 1e-6, H = 1e-5;
        double[] x = Fill(Rows * Cols, 2.0, 5), dy = Fill(Rows * Cols, 1.0, 6), g = Fill(Cols, 0.7, 7);

        double[] Forward(double[] xv, double[] gv)
        {
            var y = new double[Rows * Cols];
            for (int i = 0; i < Rows; i++)
            {
                double ms = 0;
                for (int j = 0; j < Cols; j++) ms += xv[i * Cols + j] * xv[i * Cols + j];
                double rstd = 1.0 / Math.Sqrt(ms / Cols + Eps);
                for (int j = 0; j < Cols; j++) y[i * Cols + j] = xv[i * Cols + j] * rstd * gv[j];
            }
            return y;
        }

        var forward = CodegenFusedStatistics.RmsNorm(Rows, Cols, Eps)
            .Interpret(new Dictionary<string, double[]> { ["x"] = x, ["gamma"] = g });
        var grads = CodegenFusedStatistics.RmsNormBackward(Rows, Cols).Interpret(new Dictionary<string, double[]>
        {
            ["x"] = x, ["dy"] = dy, ["gamma"] = g, ["rstd"] = forward["rstd"],
        });

        for (int k = 0; k < x.Length; k++)
        {
            double keep = x[k];
            x[k] = keep + H; double up = Dot(dy, Forward(x, g));
            x[k] = keep - H; double down = Dot(dy, Forward(x, g));
            x[k] = keep;
            Assert.Equal((up - down) / (2 * H), grads["dx"][k], 5);
        }
        for (int k = 0; k < g.Length; k++)
        {
            double keep = g[k];
            g[k] = keep + H; double up = Dot(dy, Forward(x, g));
            g[k] = keep - H; double down = Dot(dy, Forward(x, g));
            g[k] = keep;
            Assert.Equal((up - down) / (2 * H), grads["dgamma"][k], 5);
        }
    }

    [Fact]
    public void SoftmaxBackward_MatchesFiniteDifferences()
    {
        const int Rows = 3, Cols = 8;
        const double H = 1e-6;
        double[] x = Fill(Rows * Cols, 2.0, 8), dy = Fill(Rows * Cols, 1.0, 9);
        var softmax = CodegenFusedStatistics.Softmax(Rows, Cols);
        double[] Forward(double[] xv) => softmax.Interpret(new Dictionary<string, double[]> { ["x"] = xv })["y"];

        var grads = CodegenFusedStatistics.SoftmaxBackward(Rows, Cols).Interpret(
            new Dictionary<string, double[]> { ["y"] = Forward(x), ["dy"] = dy });
        for (int k = 0; k < x.Length; k++)
        {
            double keep = x[k];
            x[k] = keep + H; double up = Dot(dy, Forward(x));
            x[k] = keep - H; double down = Dot(dy, Forward(x));
            x[k] = keep;
            Assert.Equal((up - down) / (2 * H), grads["dx"][k], 6);
        }
    }

    /// <summary>The input-gradient passes fuse into one row kernel; the column reductions do not.</summary>
    [Fact]
    public void TheInputGradientFusesAndTheColumnReductionsStayApart()
    {
        var program = CodegenFusedStatistics.LayerNormBackward(64, 1024);
        var dx = new CodegenProgram(new[] { program.Passes[0], program.Passes[1], program.Passes[2] },
            new long[] { 64, 64, 64 * 1024 }, "dx");
        Assert.NotNull(CodegenRowFusion.TryPlan(dx, out _));
        Assert.Null(CodegenRowFusion.TryPlan(program, out _));
    }

    // ---- column reductions -----------------------------------------------------------------

    /// <summary>
    /// A reduction DOWN columns splits with the chunk index outermost, so consecutive threads
    /// are consecutive columns; the two-pass answer still equals the one-pass answer.
    /// </summary>
    [Fact]
    public void AColumnReductionSplitsWithTheChunkOutermost()
    {
        const int Rows = 256, Cols = 64;
        var spec = CodegenFusedStatistics.LayerNormBackward(Rows, Cols).Passes[4];   // dbeta
        var (partial, combine) = CodegenSplitReduction.SplitChunked(spec, 1, 8);

        Assert.False(partial.Space.Axes[0].IsReduction);
        Assert.Equal(8, partial.Space.Axes[0].Extent);
        Assert.Equal(8, partial.Output.Shape[0]);

        double[] dy = Fill(Rows * Cols, 1.0, 10);
        double[] want = spec.Interpret(new[] { dy });
        double[] got = combine.Interpret(new[] { partial.Interpret(new[] { dy }) });
        for (int j = 0; j < Cols; j++) Assert.Equal(want[j], got[j], 9);
    }

    /// <summary>A body moves to the partial pass, renumbered onto its inputs.</summary>
    [Fact]
    public void ASplitCarriesTheBodyOntoThePartial()
    {
        const int Rows = 256, Cols = 64;
        var spec = CodegenFusedStatistics.LayerNormBackward(Rows, Cols).Passes[3];   // dgamma
        var (partial, combine) = CodegenSplitReduction.SplitChunked(spec, 1, 8);
        Assert.True(partial.HasExpressionBody);
        Assert.False(combine.HasExpressionBody);

        var operands = new[] { Fill(Rows * Cols, 2, 1), Fill(Rows * Cols, 1, 2), Fill(Rows, 1, 3), Fill(Rows, 1, 4) };
        int[] sources = CodegenSplitReduction.PartialOperandSources(spec);
        var partialOperands = new double[sources.Length][];
        for (int i = 0; i < sources.Length; i++) partialOperands[i] = operands[sources[i]];
        double[] want = spec.Interpret(operands);
        double[] got = combine.Interpret(new[] { partial.Interpret(partialOperands) });
        for (int j = 0; j < Cols; j++) Assert.Equal(want[j], got[j], 9);
    }
}
