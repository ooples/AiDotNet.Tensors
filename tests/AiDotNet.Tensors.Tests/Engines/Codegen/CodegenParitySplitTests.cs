// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The parity split of exact-division windows (#890 phase 4b): a strided convolution's data gradient becomes one pass per
// output parity class, each a plain affine reduction over only the taps that class keeps, and together they compute
// exactly what the unsplit spec computes.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenParitySplitTests
{
    private static double[] Fill(long count, int salt)
    {
        var values = new double[count];
        for (long i = 0; i < count; i++) values[i] = (((i * 37 + salt * 101) % 97) - 48) / 64.0;
        return values;
    }

    private static long Work(CodegenKernelSpec spec)
    {
        long work = 1;
        foreach (var axis in spec.Space.Axes) work *= axis.Extent;
        return work;
    }

    /// <summary>
    /// Stride 2 and 3, odd and even windows, padding, a dilation coprime with the stride, odd and even extents: the
    /// passes together equal the unsplit spec, read no division, and visit fewer (position, tap) pairs.
    /// </summary>
    [Theory]
    [InlineData(2, 3, 4, 9, 8, 3, 2, 1, 1, 4)]    // the ResNet downsample's data gradient: 4 classes
    [InlineData(2, 4, 3, 10, 11, 3, 2, 1, 1, 4)]
    [InlineData(1, 3, 2, 13, 12, 5, 3, 2, 1, 9)]   // stride 3, 5x5: 9 classes
    [InlineData(1, 2, 3, 10, 10, 4, 2, 1, 1, 4)]   // an even window
    [InlineData(1, 2, 2, 15, 14, 3, 2, 3, 3, 4)]   // dilation 3, coprime with stride 2
    public void ThePassesTogetherEqualTheUnsplitSpec(int n, int c, int k, int h, int w, int kernel, int stride,
        int pad, int dilation, int passes)
    {
        var spec = CodegenAdjoint.BackwardData(
            CodegenKernelCatalog.Conv2D(n, c, k, h, w, kernel, stride, pad, dilation, 1, epilogue: false), 0);
        var program = CodegenParitySplit.TrySplit(spec, out string reason);
        if (program is null)
        {
            Assert.Fail(reason);
            return;
        }
        Assert.True(program.PassesWriteDisjointSlices);
        Assert.Equal(passes, program.Passes.Count);

        long work = 0;
        foreach (var pass in program.Passes)
        {
            work += Work(pass);
            foreach (var input in pass.Inputs)
                foreach (var expr in input.Map)
                    Assert.Equal(1, expr.Divisor);
        }
        Assert.True(work < Work(spec), work + " (position, tap) pairs split against " + Work(spec) + " unsplit");

        var inputs = new[] { Fill(spec.Inputs[0].ElementCount, 1), Fill(spec.Inputs[1].ElementCount, 2) };
        double[] expected = spec.Interpret(inputs);
        var sources = new Dictionary<string, double[]>
        {
            [spec.Inputs[0].Name] = inputs[0],
            [spec.Inputs[1].Name] = inputs[1],
        };
        double[] actual = program.Interpret(sources)[spec.Output.Name];
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12, $"element {i}: {expected[i]} vs {actual[i]}");
    }

    /// <summary>
    /// A class with positions but no taps (a 1x1 window at stride 2: odd positions receive nothing), a tap step
    /// sharing a factor with the stride, and a spec with no exact-division window are refused.
    /// </summary>
    [Theory]
    [InlineData(1, 0, 1, 2, "no taps")]
    [InlineData(3, 2, 2, 2, "sharing a factor")]
    [InlineData(3, 1, 1, 1, "no operand")]
    public void WhatCannotSplitIsRefused(int kernel, int pad, int dilation, int stride, string because)
    {
        var spec = CodegenAdjoint.BackwardData(
            CodegenKernelCatalog.Conv2D(1, 2, 2, 9, 9, kernel, stride, pad, dilation, 1, epilogue: false), 0);
        Assert.Null(CodegenParitySplit.TrySplit(spec, out string reason));
        Assert.Contains(because, reason, StringComparison.Ordinal);
    }

    /// <summary>Without the declaration a second writer of one output is still an error.</summary>
    [Fact]
    public void ASecondWriterNeedsTheDeclaration()
    {
        var spec = CodegenAdjoint.BackwardData(
            CodegenKernelCatalog.Conv2D(1, 2, 2, 9, 8, 3, 2, 1, 1, 1, epilogue: false), 0);
        var program = CodegenParitySplit.TrySplit(spec, out _);
        Assert.NotNull(program);
        var undeclared = program with { PassesWriteDisjointSlices = false };
        Assert.Throws<InvalidOperationException>(() => undeclared.Sources());
    }
}
