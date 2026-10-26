// Copyright (c) AiDotNet. All rights reserved.
// The rest of the row-wise scatter family as declared programs (#1441): each program on the fp64 interpreter against
// the hand-written kernel's own loop, including empty groups, out-of-range indices, ties and non-finite values.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenScatterRowsFamilyTests
{
    private static readonly CodegenLibraryFunction NotEqual =
        new("not_equal_scalar", "test", 2, a => a[0] != a[1] ? 1.0 : 0.0);

    private static readonly CodegenLibraryFunction Divide = new("divide_vectors", "test", 2, a => a[0] / a[1]);

    private static double[] Ids(params int[] ids) => Array.ConvertAll(ids, v => (double)v);

    [Fact]
    public void ScatterMaxArgmax_KeepsTheFirstMaximumAndMarksEmptyGroups()
    {
        // 5 rows, 2 columns, 4 groups: group 3 is empty, row 4 is outside the groups, rows 0 and 2 tie in column 0.
        int[] ids = { 1, 0, 1, 0, 7 };
        double[] source = { 2, -1, 5, 3, 2, 9, 4, 4, 100, 100 };
        var outputs = CodegenBackendPrograms.ScatterMaxArgmaxRows(5, 2, 4).Interpret(new Dictionary<string, double[]>
        {
            ["indices"] = Ids(ids),
            ["source"] = source,
        });
        double[] expectedMax = { 5, 4, 2, 9, double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity };
        double[] expectedArg = { 1, 3, 0, 2, -1, -1, -1, -1 };
        // Group 1 column 0: rows 0 and 2 both hold 2, the first (row 0) wins.
        Assert.Equal(expectedMax, outputs["out"]);
        Assert.Equal(expectedArg, outputs["argmax"]);
    }

    [Fact]
    public void ScatterMaxArgmax_EmitsAFloatArgmaxStartingAtMinusOne()
    {
        var pass = CodegenBackendPrograms.ScatterMaxArgmaxRows(64, 8, 5).Passes[0];
        string ptx = new PtxAffineEmitter().Emit(pass, 8, 6);
        Assert.Contains("mov.u32 %r", ptx);
        Assert.Contains(", -1;", ptx);
        Assert.Contains("cvt.rn.f32.s32", ptx);
        Assert.DoesNotContain("st.global.u32", ptx);
    }

    /// <summary>
    /// A rolled reduction loop carries the argmax register from trip to trip, so the select must write it in place.
    /// Writing a fresh register per trip left every looped argmax at its no-winner value on the device.
    /// </summary>
    [Fact]
    public void ALoopedArgmaxIsUpdatedInPlace()
    {
        string ptx = new PtxAffineEmitter().Emit(CodegenBackendPrograms.ScatterMaxArgmaxRows(300, 33, 7).Passes[0], 8, 6);
        Assert.Contains("LOOP", ptx);
        var select = System.Text.RegularExpressions.Regex.Match(ptx, @"selp\.b32 (%r\d+), %r\d+, (%r\d+), %p\d+;");
        Assert.True(select.Success, "no argmax select in the loop");
        Assert.Equal(select.Groups[1].Value, select.Groups[2].Value);
    }

    [Fact]
    public void ScatterMaxBackward_RoutesEachGradientToItsArgmaxRow()
    {
        // argmax [3 groups, 2 columns]; -1 marks an empty group; row 3 was chosen by no group.
        int[] argmax = { 0, 2, -1, -1, 1, 1 };
        double[] gradOutput = { 10, 20, 30, 40, 50, 60 };
        var outputs = CodegenBackendPrograms.ScatterMaxBackwardRows(4, 2, 3).Interpret(new Dictionary<string, double[]>
        {
            ["argmax"] = Ids(argmax),
            ["gradOutput"] = gradOutput,
        });
        var expected = new double[4 * 2];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++)
                for (int g = 0; g < 3; g++)
                    if (argmax[g * 2 + j] == i) expected[i * 2 + j] = gradOutput[g * 2 + j];
        Assert.Equal(expected, outputs["gradSource"]);
    }

    [Fact]
    public void ScatterMeanBackward_DividesByTheCountAndZeroesRowsOutsideTheGroups()
    {
        int[] ids = { 0, 2, 2, -1, 5, 1 };
        int[] counts = { 1, 0, 2 };   // group 1's count is 0: divided by 1
        double[] gradOutput = { 3, 6, 9, 12, 15, 18 };
        var outputs = CodegenBackendPrograms.ScatterMeanBackwardRows(6, 2, 3, NotEqual, Divide).Interpret(
            new Dictionary<string, double[]>
            {
                ["counts"] = Ids(counts),
                ["indices"] = Ids(ids),
                ["gradOutput"] = gradOutput,
            });
        var expected = new double[6 * 2];
        for (int i = 0; i < 6; i++)
        {
            int g = ids[i];
            if (g < 0 || g >= 3) continue;
            double divisor = counts[g] > 0 ? counts[g] : 1;
            for (int j = 0; j < 2; j++) expected[i * 2 + j] = gradOutput[g * 2 + j] / divisor;
        }
        Assert.Equal(expected, outputs["gradSource"]);
    }

    [Fact]
    public void ScatterSoftmaxBackward_MatchesTheKernelsLoopAndZeroesOutOfGroupRows()
    {
        int[] ids = { 0, 1, 0, -1, 1, 0 };
        double[] output = { 0.2, 0.5, 0.3, 0.1, 0.5, 0.4, double.PositiveInfinity, double.NaN, 0.5, 0.5, 0.5, 0.5 };
        double[] gradOutput = { 1, -2, 0.5, 3, -1, 2, 4, 4, 0.25, -0.75, 1.5, 1 };
        var outputs = CodegenBackendPrograms.ScatterSoftmaxBackwardRows(6, 2, 2, NotEqual).Interpret(
            new Dictionary<string, double[]>
            {
                ["indices"] = Ids(ids),
                ["output"] = output,
                ["gradOutput"] = gradOutput,
            });
        var expected = new double[6 * 2];
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 2; j++)
            {
                int g = ids[i];
                if (g < 0 || g >= 2) continue;
                double sum = 0;
                for (int r = 0; r < 6; r++)
                    if (ids[r] == g) sum += output[r * 2 + j] * gradOutput[r * 2 + j];
                expected[i * 2 + j] = output[i * 2 + j] * (gradOutput[i * 2 + j] - sum);
            }
        // Row 3 holds +Inf and NaN but lies outside the groups: exactly 0, not NaN.
        Assert.Equal(0.0, outputs["gradSource"][6]);
        Assert.Equal(0.0, outputs["gradSource"][7]);
        for (int k = 0; k < expected.Length; k++) Assert.Equal(expected[k], outputs["gradSource"][k], 12);
    }

    [Fact]
    public void ScatterMeanActivation_CountsAsIntegersAndDividesOnlyNonEmptyGroups()
    {
        int[] ids = { 2, 0, 2, 2, 0 };
        double[] source = { 1, 2, 3, 4, 5, 6, 7, 8, double.NaN, 10 };
        var outputs = CodegenBackendPrograms.ScatterMeanActivation(5, 4, 2, NotEqual, Divide).Interpret(
            new Dictionary<string, double[]>
            {
                ["source"] = source,
                ["indices"] = Ids(ids),
            });
        Assert.Equal(new double[] { 2, 0, 3, 0 }, outputs["counts"]);
        var expected = new double[4 * 2];
        var counts = new int[4];
        foreach (int g in ids) counts[g]++;
        for (int i = 0; i < 5; i++)
            for (int f = 0; f < 2; f++) expected[ids[i] * 2 + f] += source[i * 2 + f];
        for (int k = 0; k < expected.Length; k++) if (counts[k / 2] > 0) expected[k] /= counts[k / 2];
        // Row 4's first column is NaN: still counted, and its group's column 0 mean is NaN.
        Assert.True(double.IsNaN(outputs["output"][0]));
        for (int k = 1; k < expected.Length; k++) Assert.Equal(expected[k], outputs["output"][k], 12);
    }

    [Fact]
    public void IntegerValues_LoadAndStoreThroughExactConversions()
    {
        var program = CodegenBackendPrograms.ScatterMeanActivation(64, 8, 4, NotEqual, Divide);
        string counts = new PtxAffineEmitter().Emit(program.Passes[1], 8, 6);
        Assert.Contains("cvt.rni.s32.f32", counts);
        Assert.Contains("st.global.s32", counts);
        string divisor = new PtxAffineEmitter().Emit(program.Passes[3], 8, 6);
        Assert.Contains("ld.global.nc.s32", divisor);
        Assert.Contains("cvt.rn.f32.s32", divisor);
        Assert.Null(CodegenRowFusion.TryPlan(new CodegenProgram(new[] { program.Passes[3], program.Passes[4] },
            new long[] { 8, 32 }, "divisor, out"), out string reason));
        Assert.Contains("integer values", reason);
    }

    [Fact]
    public void AnIntegerValueTensorCannotAddressAnother()
    {
        var counts = new CodegenTensorBinding(0, "counts", new[] { 4 }, new[] { CodegenAffineExpr.Axis(0) },
            elementType: CodegenElementType.Int32, integerValue: true);
        Assert.False(counts.IsIndexTensor);
        var table = new CodegenTensorBinding(1, "table", new[] { 4 }, new[] { CodegenAffineExpr.Const(0) },
            indirect: new CodegenIndirectIndex?[] { new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), 4) });
        Assert.Throws<ArgumentException>(() => new CodegenKernelSpec("bad",
            new CodegenIterationSpace(CodegenAxis.Parallel("i", 4)), new[] { counts, table },
            new CodegenTensorBinding(2, "out", new[] { 4 }, new[] { CodegenAffineExpr.Axis(0) }, isOutput: true),
            new[] { 1 }, CodegenReduceKind.None));
        Assert.Throws<ArgumentException>(() => new CodegenTensorBinding(0, "x", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, integerValue: true));
    }
}