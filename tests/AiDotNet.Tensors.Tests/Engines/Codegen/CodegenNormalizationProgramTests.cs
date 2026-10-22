// Copyright (c) AiDotNet. All rights reserved.
// The normalisation family as declared programs (#890 phase 4a), held to hand-written fp64
// references rather than to the spec interpreter alone: the programs and their passes are
// built from the same IR, so agreeing with the interpreter would only prove the program
// agrees with itself.
//
// What made these expressible is the epilogue: rsqrt(var + eps) needs a constant inside the
// activation (ReduceOffset), and (x - mean) * rstd * gamma + beta needs a second scale and
// a bias that lands AFTER the scales (PostScaleInput, PostBiasInput).

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenNormalizationProgramTests
{
    private static double[] Fill(long count, double scale, int salt)
    {
        var v = new double[count];
        for (long i = 0; i < count; i++)
            v[i] = scale * ((((i * 37 + salt * 101) % 97) - 48) / 48.0) + 0.1 * salt;
        return v;
    }

    private static Dictionary<string, double[]> Sources(CodegenProgram program)
    {
        var sources = new Dictionary<string, double[]>(StringComparer.Ordinal);
        int salt = 1;
        foreach (var source in program.Sources())
            sources[source.Name] = Fill(source.ElementCount, source.Name == "x" ? 3.0 : 0.5, salt++);
        return sources;
    }

    [Theory]
    [InlineData(5, 12, 1e-5)]
    [InlineData(9, 64, 1e-3)]
    public void LayerNorm_MatchesAHandWrittenReference(int rows, int columns, double epsilon)
    {
        var program = CodegenFusedStatistics.LayerNorm(rows, columns, epsilon);
        var sources = Sources(program);
        var outputs = program.Interpret(sources);
        double[] x = sources["x"], gamma = sources["gamma"], beta = sources["beta"];

        for (int i = 0; i < rows; i++)
        {
            double mean = 0;
            for (int j = 0; j < columns; j++) mean += x[i * columns + j];
            mean /= columns;
            double variance = 0;
            for (int j = 0; j < columns; j++)
                variance += (x[i * columns + j] - mean) * (x[i * columns + j] - mean);
            variance /= columns;
            double rstd = 1.0 / Math.Sqrt(variance + epsilon);

            Assert.Equal(-mean, outputs["negmean"][i], 10);
            Assert.Equal(rstd, outputs["rstd"][i], 10);
            for (int j = 0; j < columns; j++)
                Assert.Equal((x[i * columns + j] - mean) * rstd * gamma[j] + beta[j],
                             outputs["y"][i * columns + j], 9);
        }
    }

    /// <summary>
    /// Epsilon must reach the result. The statistics-only program accepted an epsilon and
    /// never read it, so this pins the difference a non-zero one makes.
    /// </summary>
    [Fact]
    public void LayerNorm_EpsilonChangesTheResult()
    {
        var small = CodegenFusedStatistics.LayerNorm(4, 16, 1e-5);
        var large = CodegenFusedStatistics.LayerNorm(4, 16, 1.0);
        var sources = Sources(small);
        double a = small.Interpret(sources)["rstd"][0];
        double b = large.Interpret(sources)["rstd"][0];
        Assert.True(a > b * 1.01, "a larger epsilon must shrink rstd; got " + a + " vs " + b);
    }

    [Theory]
    [InlineData(6, 32, 1e-6)]
    [InlineData(3, 80, 1e-2)]
    public void RmsNorm_MatchesAHandWrittenReference(int rows, int columns, double epsilon)
    {
        var program = CodegenFusedStatistics.RmsNorm(rows, columns, epsilon);
        var sources = Sources(program);
        var outputs = program.Interpret(sources);
        double[] x = sources["x"], gamma = sources["gamma"];

        for (int i = 0; i < rows; i++)
        {
            double meanSquare = 0;
            for (int j = 0; j < columns; j++) meanSquare += x[i * columns + j] * x[i * columns + j];
            double rstd = 1.0 / Math.Sqrt(meanSquare / columns + epsilon);
            Assert.Equal(rstd, outputs["rstd"][i], 10);
            for (int j = 0; j < columns; j++)
                Assert.Equal(x[i * columns + j] * rstd * gamma[j], outputs["y"][i * columns + j], 9);
        }
    }

    /// <summary>
    /// The statistics run over (n, h, w) for each channel -- n separate planes, not one
    /// contiguous run -- which is the index map a hand-written kernel has to re-derive.
    /// </summary>
    [Fact]
    public void BatchNorm2D_MatchesAHandWrittenReference()
    {
        const int N = 3, C = 4, H = 5, W = 6;
        const double Epsilon = 1e-5;
        var program = CodegenFusedStatistics.BatchNorm2D(N, C, H, W, Epsilon);
        var sources = Sources(program);
        var outputs = program.Interpret(sources);
        double[] x = sources["x"], gamma = sources["gamma"], beta = sources["beta"];

        for (int c = 0; c < C; c++)
        {
            double mean = 0, count = N * H * W;
            for (int n = 0; n < N; n++)
                for (int s = 0; s < H * W; s++) mean += x[(n * C + c) * H * W + s];
            mean /= count;
            double variance = 0;
            for (int n = 0; n < N; n++)
                for (int s = 0; s < H * W; s++)
                {
                    double d = x[(n * C + c) * H * W + s] - mean;
                    variance += d * d;
                }
            double rstd = 1.0 / Math.Sqrt(variance / count + Epsilon);

            Assert.Equal(-mean, outputs["negmean"][c], 10);
            Assert.Equal(rstd, outputs["rstd"][c], 10);
            for (int n = 0; n < N; n++)
                for (int s = 0; s < H * W; s++)
                {
                    int at = (n * C + c) * H * W + s;
                    Assert.Equal((x[at] - mean) * rstd * gamma[c] + beta[c], outputs["y"][at], 9);
                }
        }
    }

    [Theory]
    [InlineData(2, 8, 2, 9)]
    [InlineData(2, 6, 6, 4)]   // groups == channels: InstanceNorm
    public void GroupNorm_MatchesAHandWrittenReference(int batch, int channels, int groups, int spatial)
    {
        const double Epsilon = 1e-5;
        var program = CodegenFusedStatistics.GroupNorm(batch, channels, groups, spatial, Epsilon);
        var sources = Sources(program);
        var outputs = program.Interpret(sources);
        double[] x = sources["x"], gamma = sources["gamma"], beta = sources["beta"];
        int perGroup = channels / groups;

        for (int n = 0; n < batch; n++)
            for (int g = 0; g < groups; g++)
            {
                int first = (n * channels + g * perGroup) * spatial;
                int count = perGroup * spatial;
                double mean = 0;
                for (int e = 0; e < count; e++) mean += x[first + e];
                mean /= count;
                double variance = 0;
                for (int e = 0; e < count; e++) variance += (x[first + e] - mean) * (x[first + e] - mean);
                double rstd = 1.0 / Math.Sqrt(variance / count + Epsilon);

                Assert.Equal(rstd, outputs["rstd"][n * groups + g], 10);
                for (int e = 0; e < count; e++)
                {
                    int c = g * perGroup + e / spatial;
                    Assert.Equal((x[first + e] - mean) * rstd * gamma[c] + beta[c],
                                 outputs["y"][first + e], 9);
                }
            }
    }

    [Fact]
    public void GroupNorm_RefusesAGroupCountThatDoesNotDivideTheChannels()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => CodegenFusedStatistics.GroupNorm(1, 6, 4, 8));
    }

    /// <summary>Chaining is by name, so every intermediate must agree with its producer.</summary>
    [Fact]
    public void Sources_ExcludeIntermediatesAndListParametersOnce()
    {
        var names = new List<string>();
        foreach (var source in CodegenFusedStatistics.LayerNorm(4, 8).Sources()) names.Add(source.Name);
        Assert.Equal(new[] { "x", "gamma", "beta" }, names);
    }

    [Fact]
    public void Interpret_RefusesAMissingSource()
    {
        var program = CodegenFusedStatistics.RmsNorm(2, 4);
        var sources = new Dictionary<string, double[]> { ["x"] = new double[8] };
        Assert.Throws<ArgumentException>(() => program.Interpret(sources));
    }

    [Fact]
    public void EveryPass_Emits()
    {
        foreach (var program in new[]
                 {
                     CodegenFusedStatistics.LayerNorm(32, 64),
                     CodegenFusedStatistics.RmsNorm(32, 64),
                     CodegenFusedStatistics.BatchNorm2D(2, 8, 4, 4),
                     CodegenFusedStatistics.GroupNorm(2, 8, 4, 16),
                 })
        {
            foreach (var pass in program.Passes)
            {
                string ptx = new PtxAffineEmitter().Emit(pass, 7, 5);
                Assert.Contains(".visible .entry", ptx, StringComparison.Ordinal);
                Assert.Contains(pass.Name, ptx, StringComparison.Ordinal);
            }
        }
    }

    /// <summary>
    /// The emitted epilogue must contain the offset add and BOTH post operands. A lowering
    /// that dropped one would still emit and still run.
    /// </summary>
    [Fact]
    public void AffinePass_EmitsTheExtendedEpilogue()
    {
        var program = CodegenFusedStatistics.LayerNorm(32, 64, 0.5);
        string rstd = new PtxAffineEmitter().Emit(program.Passes[1], 7, 5);
        Assert.Contains("add.rn.f32", rstd, StringComparison.Ordinal);
        Assert.Contains("0f3F000000", rstd, StringComparison.Ordinal);   // 0.5, the epsilon

        string affine = new PtxAffineEmitter().Emit(program.Passes[2], 7, 5);
        int multiplies = CountOccurrences(affine, "mul.rn.f32");
        Assert.True(multiplies >= 2, "rstd and gamma are two multiplies; found " + multiplies);
    }

    /// <summary>The extended epilogue is real-valued only; another algebra must refuse it.</summary>
    [Fact]
    public void ExtendedEpilogue_IsRefusedOverAnotherAlgebra()
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", 4));
        var x = new CodegenTensorBinding(0, "x", new[] { 4 }, new[] { CodegenAffineExpr.Axis(0) });
        var y = new CodegenTensorBinding(1, "y", new[] { 4 }, new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        // The same spec without the offset is valid, so the refusal below is about the offset.
        _ = new CodegenKernelSpec("c", space, new[] { x }, y,
            new[] { 0 }, CodegenReduceKind.None, algebra: CodegenAlgebra.Complex);
        Assert.Throws<ArgumentException>(() => new CodegenKernelSpec("c", space, new[] { x }, y,
            new[] { 0 }, CodegenReduceKind.None, algebra: CodegenAlgebra.Complex, reduceOffset: 1.0));
    }

    /// <summary>
    /// Specs that do not use the new fields keep their autotune identity, so winners
    /// recorded before the fields existed still apply; specs that do use them get a
    /// different one.
    /// </summary>
    [Fact]
    public void AutotuneIdentity_IsStableWithoutTheNewFields()
    {
        var plain = CodegenFusedStatistics.LayerNormStatistics(32, 64).Passes[1];
        string text = CodegenAutotuneIdentity.CanonicalSpec(plain);
        Assert.EndsWith(";algebra=Real;", text, StringComparison.Ordinal);
        Assert.DoesNotContain("reduceOffset", text, StringComparison.Ordinal);

        var a = CodegenFusedStatistics.LayerNorm(32, 64, 1e-5).Passes[1];
        var b = CodegenFusedStatistics.LayerNorm(32, 64, 1e-3).Passes[1];
        Assert.Contains("reduceOffset=", CodegenAutotuneIdentity.CanonicalSpec(a), StringComparison.Ordinal);
        Assert.NotEqual(
            CodegenAutotuneIdentity.Create(a, "device", 7, 5).SpecFingerprint,
            CodegenAutotuneIdentity.Create(b, "device", 7, 5).SpecFingerprint);
    }

    /// <summary>
    /// An affine-of-primary extra output scales the FINISHED value, as the emitter does. The
    /// interpreter used to apply the activation a second time, which only an activation that
    /// is not idempotent (here sigmoid) can expose.
    /// </summary>
    [Fact]
    public void ExtraOutput_ScalesTheActivatedValueOnce()
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", 4));
        var x = new CodegenTensorBinding(0, "x", new[] { 4 }, new[] { CodegenAffineExpr.Axis(0) });
        var y = new CodegenTensorBinding(1, "y", new[] { 4 }, new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var twice = new CodegenTensorBinding(2, "twice", new[] { 4 }, new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var spec = new CodegenKernelSpec("sig", space, new[] { x }, y, new[] { 0 },
            CodegenReduceKind.None, activation: CodegenActivationKind.Sigmoid,
            extraOutputs: new[] { new CodegenExtraOutput(twice, CodegenExtraOutputKind.AffineOfPrimary, null, 2.0) });

        double[] input = { -1.0, 0.0, 0.5, 2.0 };
        double[][] all = spec.InterpretAll(new[] { input });
        for (int i = 0; i < input.Length; i++)
        {
            double sigmoid = 1.0 / (1.0 + Math.Exp(-input[i]));
            Assert.Equal(sigmoid, all[0][i], 12);
            Assert.Equal(2.0 * sigmoid, all[1][i], 12);
        }
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
