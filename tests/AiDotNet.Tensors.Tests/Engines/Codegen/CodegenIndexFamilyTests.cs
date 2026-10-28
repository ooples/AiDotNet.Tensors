// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The index family as declared programs (#1441) on the fp64 interpreter: index write / copy / fill (last index wins),
// index add, scatter reduce (sum, max, min), int32 take-along-dim and float-id gathers, against the kernels' loops.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenIndexFamilyTests
{
    private static readonly CodegenLibraryFunction NotEqual =
        new("not_equal_scalar", "test", 2, a => a[0] != a[1] ? 1.0 : 0.0);
    private static readonly CodegenLibraryFunction Select = new("where_select", "test", 3, a => a[0] != 0 ? a[1] : a[2]);
    private static readonly CodegenLibraryFunction Max = new("max_vectors", "test", 2, a => a[0] > a[1] ? a[0] : a[1]);
    private static readonly CodegenLibraryFunction Min = new("min_vectors", "test", 2, a => a[0] < a[1] ? a[0] : a[1]);

    private static double[] D(params int[] v) => Array.ConvertAll(v, x => (double)x);

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void IndexWrite_TheLastMatchingIndexWinsAndUnnamedSlicesKeepTheirValue(bool copy)
    {
        const int outer = 2, idxLen = 5, inner = 3, dst = 4;
        int[] indices = { 2, 0, 2, 9, -1 };   // slice 2 named twice (j = 0 and 2: j = 2 wins); 1 and 3 unnamed
        var seed = new double[outer * dst * inner];
        for (int k = 0; k < seed.Length; k++) seed[k] = k == 4 ? double.NaN : 100 + k;
        var source = new double[outer * idxLen * inner];
        for (int k = 0; k < source.Length; k++) source[k] = -k;
        var sources = new Dictionary<string, double[]> { ["seed"] = seed, ["indices"] = D(indices) };
        if (copy) sources["source"] = source;
        var outputs = CodegenBackendPrograms.IndexWrite(outer, idxLen, inner, dst, copy, 7.5f, NotEqual, Select)
            .Interpret(sources);
        var expected = (double[])seed.Clone();
        for (int o = 0; o < outer; o++)
            for (int t = 0; t < dst; t++)
            {
                int last = -1;
                for (int j = 0; j < idxLen; j++) if (indices[j] == t) last = j;
                if (last < 0) continue;
                for (int i = 0; i < inner; i++)
                    expected[(o * dst + t) * inner + i] = copy ? source[(o * idxLen + last) * inner + i] : 7.5;
            }
        var actual = outputs["out"];
        for (int k = 0; k < expected.Length; k++)
        {
            if (double.IsNaN(expected[k])) Assert.True(double.IsNaN(actual[k]), "index " + k);
            else Assert.Equal(expected[k], actual[k]);
        }
    }

    [Fact]
    public void IndexAdd_AccumulatesOntoTheSeedInIndexOrder()
    {
        const int outer = 2, dst = 3, inner = 2, idxLen = 4;
        int[] indices = { 1, 1, 5, 0 };
        var seed = new double[outer * dst * inner];
        for (int k = 0; k < seed.Length; k++) seed[k] = k;
        var source = new double[outer * idxLen * inner];
        for (int k = 0; k < source.Length; k++) source[k] = 0.5 * k - 2;
        var outputs = CodegenBackendPrograms.IndexAdd(outer, dst, inner, idxLen).Interpret(new Dictionary<string, double[]>
        {
            ["indices"] = D(indices), ["source"] = source, ["seed"] = seed,
        });
        var expected = (double[])seed.Clone();
        for (int o = 0; o < outer; o++)
            for (int j = 0; j < idxLen; j++)
            {
                int t = indices[j];
                if (t < 0 || t >= dst) continue;
                for (int i = 0; i < inner; i++) expected[(o * dst + t) * inner + i] += source[(o * idxLen + j) * inner + i];
            }
        Assert.Equal(expected, outputs["out"]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void ScatterReduce_MatchesTheSequentialReduction(int mode)
    {
        const int outer = 2, src = 4, dst = 3, inner = 2;
        var index = new int[outer * src * inner];
        for (int k = 0; k < index.Length; k++) index[k] = (k * 7 + 1) % 5 - 1;   // includes -1 and 3 (out of range)
        var source = new double[index.Length];
        for (int k = 0; k < source.Length; k++) source[k] = k % 3 == 2 ? double.NaN : (k * 13 % 11) - 5;
        var seed = new double[outer * dst * inner];
        for (int k = 0; k < seed.Length; k++) seed[k] = k - 4;
        var outputs = CodegenBackendPrograms.ScatterReduce(outer, src, dst, inner, mode, mode == 2 ? Max : Min)
            .Interpret(new Dictionary<string, double[]> { ["index"] = D(index), ["source"] = source, ["seed"] = seed });
        var expected = (double[])seed.Clone();
        for (int o = 0; o < outer; o++)
            for (int s = 0; s < src; s++)
                for (int i = 0; i < inner; i++)
                {
                    int k = (o * src + s) * inner + i, t = index[k];
                    if (t < 0 || t >= dst) continue;
                    int cell = (o * dst + t) * inner + i;
                    double v = source[k], pf = expected[cell];
                    // fmaxf / fminf: a NaN operand yields the other.
                    expected[cell] = mode == 0 ? pf + v : mode == 1 ? pf * v
                        : double.IsNaN(v) ? pf : double.IsNaN(pf) ? v : mode == 2 ? Math.Max(pf, v) : Math.Min(pf, v);
                }
        var actual = outputs["out"];
        for (int k = 0; k < expected.Length; k++)
        {
            if (double.IsNaN(expected[k])) Assert.True(double.IsNaN(actual[k]), "index " + k);
            else Assert.Equal(expected[k], actual[k]);
        }
    }

    /// <summary>A product's out-of-range term is the identity 1, and an empty product leaves the seed exactly.</summary>
    [Fact]
    public void AProductReductionEmitsItsIdentityAndMultiplies()
    {
        var pass = CodegenBackendPrograms.ScatterReduce(2, 64, 5, 3, 1, null).Passes[0];
        string ptx = new AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx.PtxAffineEmitter().Emit(pass, 8, 6);
        Assert.Contains("0f3F800000", ptx);
        Assert.Contains("mul.rn.f32", ptx);
        Assert.Null(CodegenRowFusion.TryPlan(new CodegenProgram(new[] { pass }, Array.Empty<long>(), "p"), out string reason));
        Assert.False(CodegenSplitReduction.IsSplittable(pass));
    }

    [Fact]
    public void MaskedScatter_WritesMaskedElementsWithAValidPrefixAndKeepsTheRest()
    {
        double[] seed = { 1, 2, double.NaN, 4, 5, 6 };
        double[] mask = { 1, 0, 1, 255, 1, 0 };
        int[] prefix = { 0, 1, 1, 7, -1, 2 };   // element 3's and 4's positions lie outside the 3-long source
        double[] source = { 10, 20, 30 };
        var outputs = CodegenBackendPrograms.MaskedScatter(6, 3, NotEqual, Select).Interpret(new Dictionary<string, double[]>
        {
            ["source"] = source, ["prefix"] = D(prefix), ["seed"] = seed, ["mask"] = mask,
        });
        var actual = outputs["out"];
        double[] expected = { 10, 2, 20, 4, 5, 6 };
        for (int k = 0; k < 6; k++) Assert.Equal(expected[k], actual[k]);
    }

    [Fact]
    public void AByteMaskLoadsAsUnsignedBytes()
    {
        var pass = CodegenBackendPrograms.MaskedScatter(64, 16, NotEqual, Select).Passes[2];
        string ptx = new AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx.PtxAffineEmitter().Emit(pass, 8, 6);
        Assert.Contains("ld.global.nc.u8", ptx);
        Assert.Contains("cvt.rn.f32.u32", ptx);
        Assert.Throws<ArgumentException>(() => new CodegenTensorBinding(0, "m", new[] { 4 }, new[] { CodegenAffineExpr.Axis(0) },
            elementType: CodegenElementType.UInt8));
    }

    [Fact]
    public void TakeAlongDimInt_AndFloatIdGathers_ZeroOutOfRangeIndices()
    {
        var take = CodegenBackendPrograms.TakeAlongDimInt(1, 3, 2, 4).Interpret(new Dictionary<string, double[]>
        {
            ["indices"] = D(0, 3, 2, -1, 4, 1), ["input"] = new double[] { 10, 11, 12, 13, 14, 15, 16, 17 },
        })["out"];
        Assert.Equal(new double[] { 10, 17, 14, 0, 0, 13 }, take);
        var select = CodegenBackendPrograms.EmbeddingFloatIds(3, 2, 2).Interpret(new Dictionary<string, double[]>
        {
            ["ids"] = new double[] { 1.9, -0.5, 7 }, ["table"] = new double[] { 1, 2, 3, 4 },
        })["out"];
        Assert.Equal(new double[] { 3, 4, 1, 2, 0, 0 }, select);
    }
}