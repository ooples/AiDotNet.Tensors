// Copyright (c) AiDotNet. All rights reserved.
// The one defect class a shared oracle cannot catch.
//
// Every other correctness gate on this project compares the emitted kernel against the fp64
// interpretation of the SAME spec. That catches anything where the two disagree -- and it is
// blind, by construction, to a mistake both sides make in the same direction.
//
// This was such a mistake. An out-of-range tap was filled with 0.0 in the interpreter and
// with 0f00000000 in the emitter, on the reasoning that zero is the padding value. Zero is
// the identity of ADDITION. Under a maximum it is a real candidate, so a padded max-pool
// over all-negative inputs returns 0 rather than the largest negative value -- in both
// implementations, in agreement, at 0.000E+000.
//
// It stayed latent only because the catalog's max-pool has no padding. These tests pin the
// behaviour against arithmetic written out by hand, which is the only reference that does
// not share the assumption.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenPaddedMaxTests
{
    /// <summary>A padded 3x3 max-pool, stride 1, so every border output has padded taps.</summary>
    private static CodegenKernelSpec PaddedMaxPool(int n, int c, int h, int w)
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
        var output = new CodegenTensorBinding(1, "output", new[] { n, c, h, w },
            new[]
            {
                CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
                CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
            }, isOutput: true);

        return new CodegenKernelSpec("padded_maxpool_3x3", space, new[] { input }, output,
            new[] { 0 }, CodegenReduceKind.Max);
    }

    /// <summary>
    /// With every input negative, no output may be zero. Zero can only appear by treating
    /// padding as a candidate.
    /// </summary>
    [Fact]
    public void PaddedMax_OverAllNegativeInput_NeverReturnsZero()
    {
        const int N = 1, C = 2, H = 5, W = 5;
        var spec = PaddedMaxPool(N, C, H, W);

        long count = (long)N * C * H * W;
        var input = new double[count];
        for (long i = 0; i < count; i++) input[i] = -1.0 - (i % 7);      // all strictly negative

        double[] got = spec.Interpret(new[] { input });

        foreach (double v in got)
            Assert.True(v < 0.0,
                "a maximum over strictly negative inputs cannot be " + v +
                "; zero appears only if padding was treated as a candidate");
    }

    /// <summary>
    /// The full result, against a hand-written max that skips out-of-range taps rather than
    /// filling them.
    /// </summary>
    [Fact]
    public void PaddedMax_MatchesAHandWrittenMaxThatSkipsPadding()
    {
        const int N = 1, C = 2, H = 5, W = 5;
        var spec = PaddedMaxPool(N, C, H, W);

        long count = (long)N * C * H * W;
        var input = new double[count];
        for (long i = 0; i < count; i++) input[i] = ((i * 37) % 97) - 96.0;   // all negative

        double[] got = spec.Interpret(new[] { input });

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        for (int oh = 0; oh < H; oh++)
        for (int ow = 0; ow < W; ow++)
        {
            double want = double.NegativeInfinity;
            for (int kh = 0; kh < 3; kh++)
            for (int kw = 0; kw < 3; kw++)
            {
                int ih = oh + kh - 1, iw = ow + kw - 1;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;   // SKIP, do not fill
                want = Math.Max(want, input[((n * C + c) * H + ih) * W + iw]);
            }
            Assert.Equal(want, got[((n * C + c) * H + oh) * W + ow], 9);
        }
    }

    /// <summary>
    /// An unpadded max-pool must be unaffected, so the fix cannot have moved the case that
    /// already worked.
    /// </summary>
    [Fact]
    public void UnpaddedMaxPool_IsUnchanged()
    {
        var entry = CodegenKernelCatalog.Find("maxpool2d_2x2");
        Assert.NotNull(entry);

        var spec = entry!.Verify;
        long count = 1;
        foreach (int d in spec.Inputs[0].Shape) count *= d;

        var input = new double[count];
        for (long i = 0; i < count; i++) input[i] = (((i * 37) % 97) - 48) / 64.0;

        double[] got = spec.Interpret(new[] { input });
        Assert.Contains(got, v => v != 0.0);
        foreach (double v in got) Assert.True(!double.IsNaN(v) && !double.IsInfinity(v));
    }

    /// <summary>
    /// The emitted kernel must carry negative infinity as its out-of-range fill, not zero.
    /// This is the emitter half of the same bug.
    /// </summary>
    [Fact]
    public void EmittedMaxKernel_FillsOutOfRangeWithNegativeInfinity()
    {
        string ptx = new PtxAffineEmitter().Emit(PaddedMaxPool(1, 2, 8, 8), 8, 6);

        Assert.Contains("0fFF800000", ptx, StringComparison.Ordinal);   // -inf
        Assert.Contains("max.f32", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// A sum keeps zero, because zero IS the additive identity. The fix must be specific to
    /// the reduction, not applied everywhere.
    /// </summary>
    [Fact]
    public void EmittedSumKernel_StillFillsOutOfRangeWithZero()
    {
        var entry = CodegenKernelCatalog.Find("depthwise_conv2d_3x3");
        Assert.NotNull(entry);

        string ptx = new PtxAffineEmitter().Emit(entry!.Verify, 8, 6);
        Assert.DoesNotContain("0fFF800000", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// A maximum over a PRODUCT of operands has no well-defined padding identity, since
    /// -inf times zero is NaN. It must be refused rather than emitted.
    /// </summary>
    [Fact]
    public void MaxOverAProductOfOperands_IsRefused()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", 4), CodegenAxis.Reduce("k", 3));
        var a = new CodegenTensorBinding(0, "a", new[] { 4, 3 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var b = new CodegenTensorBinding(1, "b", new[] { 4, 3 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(2, "out", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        var spec = new CodegenKernelSpec("max_of_product", space, new[] { a, b }, output,
            new[] { 0, 1 }, CodegenReduceKind.Max);

        var ex = Assert.Throws<NotSupportedException>(() => new PtxAffineEmitter().Emit(spec, 8, 6));
        Assert.Contains("padding identity", ex.Message, StringComparison.Ordinal);
    }
}
