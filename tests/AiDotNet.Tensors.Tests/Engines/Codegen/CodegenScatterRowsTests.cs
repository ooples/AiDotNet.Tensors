// Copyright (c) AiDotNet. All rights reserved.
// The row-wise scatter mean and softmax as declared programs (#890 phase 4e, #1441): the interpreter against the
// hand-written resident_scatter_mean_rows_counts / resident_scatter_softmax_rows arithmetic, and the emitted PTX
// against the determinism contract (no atomic).

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenScatterRowsTests
{
    // The derived not_equal_scalar's semantics: 1 when the operands differ (NaN differs from everything), else 0.
    private static readonly CodegenLibraryFunction NotEqual =
        new("not_equal_scalar", "test", 2, a => a[0] != a[1] ? 1.0 : 0.0);

    // Groups 0..3; group 2 receives nothing, and two rows carry the -1 / past-the-end sentinels.
    private static readonly int[] Ids = { 1, 0, 3, 1, -1, 0, 1, 4, 3 };
    private const int Groups = 4, Inner = 3;

    private static Dictionary<string, double[]> Inputs(double[] source)
    {
        var ids = new double[Ids.Length];
        for (int i = 0; i < ids.Length; i++) ids[i] = Ids[i];
        return new Dictionary<string, double[]> { ["indices"] = ids, ["source"] = source };
    }

    private static double[] Source()
    {
        var source = new double[Ids.Length * Inner];
        for (int i = 0; i < source.Length; i++) source[i] = ((i * 7) % 13) - 6 + 0.125 * i;
        return source;
    }

    [Fact]
    public void ScatterMean_CountsAndAveragesEachGroup_AndLeavesAnEmptyGroupAtZero()
    {
        var source = Source();
        var outputs = CodegenBackendPrograms.ScatterMeanRows(Ids.Length, Inner, Groups, NotEqual).Interpret(Inputs(source));

        for (int g = 0; g < Groups; g++)
        {
            int count = 0;
            for (int i = 0; i < Ids.Length; i++) if (Ids[i] == g) count++;
            Assert.Equal(count, outputs["counts"][g]);
            for (int j = 0; j < Inner; j++)
            {
                double sum = 0;
                for (int i = 0; i < Ids.Length; i++) if (Ids[i] == g) sum += source[i * Inner + j];
                double expected = count > 0 ? sum * (1.0 / count) : 0.0;
                Assert.Equal(expected, outputs["out"][g * Inner + j], 12);
            }
        }
        // Group 2 is empty: count 0 and a mean of exactly 0, not NaN from 0 * (1/0).
        Assert.Equal(0.0, outputs["counts"][2]);
        for (int j = 0; j < Inner; j++) Assert.Equal(0.0, outputs["out"][2 * Inner + j]);
    }

    /// <summary>
    /// The counts come from a tensor of ones derived from the source's first column; a NaN or infinite value there
    /// must still count as one row, or a group's mean would divide by the wrong count.
    /// </summary>
    [Fact]
    public void ScatterMean_CountsRowsWhoseFirstColumnIsNaNOrInfinite()
    {
        var source = Source();
        source[0 * Inner] = double.NaN;                 // row 0, group 1
        source[3 * Inner] = double.PositiveInfinity;    // row 3, group 1
        source[5 * Inner] = double.NegativeInfinity;    // row 5, group 0
        var outputs = CodegenBackendPrograms.ScatterMeanRows(Ids.Length, Inner, Groups, NotEqual).Interpret(Inputs(source));
        Assert.Equal(2.0, outputs["counts"][0]);
        Assert.Equal(3.0, outputs["counts"][1]);
        Assert.Equal(2.0, outputs["counts"][3]);
        // Column 1 has no special values, so its mean is the ordinary one.
        double sum = source[1 * Inner + 1] + source[5 * Inner + 1];
        Assert.Equal(sum * 0.5, outputs["out"][0 * Inner + 1], 12);
    }

    [Fact]
    public void ScatterSoftmax_NormalisesEachGroupsColumns_AndZeroesARowOutsideTheGroups()
    {
        var source = Source();
        var outputs = CodegenBackendPrograms.ScatterSoftmaxRows(Ids.Length, Inner, Groups).Interpret(Inputs(source));

        for (int i = 0; i < Ids.Length; i++)
            for (int j = 0; j < Inner; j++)
            {
                int g = Ids[i];
                double expected = 0.0;
                if (g >= 0 && g < Groups)
                {
                    double max = double.NegativeInfinity, sum = 0;
                    for (int o = 0; o < Ids.Length; o++) if (Ids[o] == g) max = Math.Max(max, source[o * Inner + j]);
                    for (int o = 0; o < Ids.Length; o++) if (Ids[o] == g) sum += Math.Exp(source[o * Inner + j] - max);
                    expected = Math.Exp(source[i * Inner + j] - max) / sum;
                }
                Assert.Equal(expected, outputs["out"][i * Inner + j], 12);
            }

        // Each group's column sums to 1.
        for (int g = 0; g < Groups; g++)
            for (int j = 0; j < Inner; j++)
            {
                double total = 0;
                bool any = false;
                for (int i = 0; i < Ids.Length; i++)
                    if (Ids[i] == g) { total += outputs["out"][i * Inner + j]; any = true; }
                if (any) Assert.Equal(1.0, total, 12);
            }
    }

    [Fact]
    public void ScatterRows_EmitNoAtomic()
    {
        foreach (var program in new[]
                 {
                     CodegenBackendPrograms.ScatterMeanRows(64, 8, 16, NotEqual),
                     CodegenBackendPrograms.ScatterSoftmaxRows(64, 8, 16),
                 })
            foreach (var pass in program.Passes)
            {
                var emitter = new PtxAffineEmitter();
                string ptx = emitter.Emit(pass, 7, 5);
                Assert.Equal(0, emitter.AtomicStores);
                Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal);
                Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
            }
    }
}
