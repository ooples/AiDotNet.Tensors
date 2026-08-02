// Copyright (c) AiDotNet. All rights reserved.
// A max-pool that also reports WHERE the maximum was.
//
// CodegenKernelSpec had one output binding. Max-pool is promoted at 1.41x and its generated
// kernel verifies at 0.000E+000, but the engine's MaxPool2D also writes an INDICES buffer
// that MaxPool2DBackward consumes -- so dispatching without it would not merely lose a
// speedup, it would silently break training.
//
// Two conventions have to be exactly right, and both are easy to get plausibly wrong:
//
//   the VALUE stored is the SPATIAL index ih*inWidth + iw, not a flat input offset and not
//   a tap index, because the backward kernel recovers ih = idx / inWidth and iw = idx %
//   inWidth and adds the batch and channel offset itself;
//
//   the TIE-BREAK keeps the FIRST maximum, because that is what the established kernel
//   does and the gradient is routed by this index.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenSecondaryOutputTests
{
    /// <summary>2x2 stride-2 max pool that also writes the argmax spatial index.</summary>
    private static CodegenKernelSpec MaxPoolWithIndices(int n, int c, int h, int w)
    {
        int outH = h / 2, outW = w / 2;
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", n), CodegenAxis.Parallel("c", c),
            CodegenAxis.Parallel("oh", outH), CodegenAxis.Parallel("ow", outW),
            CodegenAxis.Reduce("kh", 2), CodegenAxis.Reduce("kw", 2));
        const int N = 0, C = 1, OH = 2, OW = 3, KH = 4, KW = 5;

        var input = new CodegenTensorBinding(0, "input", new[] { n, c, h, w },
            new[]
            {
                CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
                CodegenAffineExpr.Window(OH, KH, 2, 0), CodegenAffineExpr.Window(OW, KW, 2, 0)
            });
        var output = new CodegenTensorBinding(1, "output", new[] { n, c, outH, outW },
            new[]
            {
                CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
                CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
            }, isOutput: true);
        var indices = new CodegenTensorBinding(2, "indices", new[] { n, c, outH, outW },
            new[]
            {
                CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
                CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
            }, isOutput: true);

        // ih * inWidth + iw, with ih = oh*2 + kh and iw = ow*2 + kw.
        var spatial = new CodegenAffineExpr(
            new[]
            {
                new CodegenAffineTerm(OH, 2 * w),
                new CodegenAffineTerm(KH, w),
                new CodegenAffineTerm(OW, 2),
                new CodegenAffineTerm(KW, 1),
            },
            constant: 0, divisor: 1, requiresExactDivision: false);

        return new CodegenKernelSpec("maxpool2d_2x2_indices", space,
            new[] { input }, output, new[] { 0 }, CodegenReduceKind.Max,
            secondaryOutput: indices, secondaryIndexExpr: spatial);
    }

    /// <summary>
    /// The indices must match a hand-written argmax using the BACKWARD kernel's convention:
    /// the spatial index, first-maximum on a tie.
    /// </summary>
    [Theory]
    [InlineData(2, 3, 8, 8)]
    [InlineData(1, 4, 16, 12)]
    public void MaxPoolIndices_MatchTheBackwardKernelsConvention(int n, int c, int h, int w)
    {
        var spec = MaxPoolWithIndices(n, c, h, w);
        int outH = h / 2, outW = w / 2;

        long count = (long)n * c * h * w;
        var x = new double[count];
        for (long i = 0; i < count; i++) x[i] = (((i * 37) % 97) - 48) / 64.0;

        double[] values = spec.Interpret(new[] { x }, out double[]? indices);
        Assert.NotNull(indices);

        for (int b = 0; b < n; b++)
        for (int ch = 0; ch < c; ch++)
        for (int oh = 0; oh < outH; oh++)
        for (int ow = 0; ow < outW; ow++)
        {
            double best = double.NegativeInfinity;
            int bestSpatial = 0;
            for (int kh = 0; kh < 2; kh++)
            for (int kw = 0; kw < 2; kw++)
            {
                int ih = oh * 2 + kh, iw = ow * 2 + kw;
                double v = x[((b * c + ch) * h + ih) * w + iw];
                if (v > best)                       // strictly greater: FIRST maximum wins
                {
                    best = v;
                    bestSpatial = ih * w + iw;      // the backward kernel's convention
                }
            }

            long at = ((b * c + ch) * outH + oh) * outW + ow;
            Assert.Equal(best, values[at], 9);
            Assert.Equal(bestSpatial, (int)indices![at]);
        }
    }

    /// <summary>
    /// Every index must decode to a position that really holds the pooled value. This is
    /// what a flat-offset or tap-index convention would fail.
    /// </summary>
    [Fact]
    public void EveryIndex_DecodesToThePooledValue()
    {
        const int N = 2, C = 2, H = 8, W = 8;
        var spec = MaxPoolWithIndices(N, C, H, W);

        var x = new double[(long)N * C * H * W];
        for (long i = 0; i < x.Length; i++) x[i] = (((i * 53) % 89) - 44) / 32.0;

        double[] values = spec.Interpret(new[] { x }, out double[]? indices);

        int outH = H / 2, outW = W / 2;
        for (int b = 0; b < N; b++)
        for (int ch = 0; ch < C; ch++)
        for (int oh = 0; oh < outH; oh++)
        for (int ow = 0; ow < outW; ow++)
        {
            long at = ((b * C + ch) * outH + oh) * outW + ow;
            int spatial = (int)indices![at];
            int ih = spatial / W, iw = spatial % W;
            Assert.Equal(values[at], x[((b * C + ch) * H + ih) * W + iw], 9);
        }
    }

    /// <summary>The emitted kernel must store the index as an INTEGER, not a float.</summary>
    [Fact]
    public void EmittedKernel_StoresTheIndexAsAnInteger()
    {
        string ptx = new PtxAffineEmitter().Emit(MaxPoolWithIndices(2, 8, 16, 16), 8, 6);

        Assert.Contains("st.global.u32", ptx, StringComparison.Ordinal);
        Assert.Contains("selp.b32", ptx, StringComparison.Ordinal);   // the argmax select
    }

    /// <summary>A secondary output without its index expression is refused, and vice versa.</summary>
    [Fact]
    public void HalfSpecifiedSecondaryOutput_IsRefused()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", 4), CodegenAxis.Reduce("k", 4));
        var x = new CodegenTensorBinding(0, "x", new[] { 4, 4 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(1, "out", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var second = new CodegenTensorBinding(2, "idx", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "half", space, new[] { x }, output, new[] { 0 },
            CodegenReduceKind.Max, secondaryOutput: second));
    }

    /// <summary>
    /// A secondary output on a SUM is refused: it currently means the argmax position, and
    /// a sum has none.
    /// </summary>
    [Fact]
    public void SecondaryOutputOnASum_IsRefused()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", 4), CodegenAxis.Reduce("k", 4));
        var x = new CodegenTensorBinding(0, "x", new[] { 4, 4 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(1, "out", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var second = new CodegenTensorBinding(2, "idx", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "sum_with_index", space, new[] { x }, output, new[] { 0 },
            CodegenReduceKind.Sum,
            secondaryOutput: second, secondaryIndexExpr: CodegenAffineExpr.Axis(1)));
    }

    /// <summary>The legacy secondary pair uses the same index-axis validation as extras.</summary>
    [Fact]
    public void SecondaryOutputWithInvalidIndexAxis_IsRefusedAtConstruction()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", 4), CodegenAxis.Reduce("k", 4));
        var x = new CodegenTensorBinding(0, "x", new[] { 4, 4 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(1, "out", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var second = new CodegenTensorBinding(2, "idx", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        var ex = Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "bad_secondary_axis", space, new[] { x }, output, new[] { 0 },
            CodegenReduceKind.Max,
            secondaryOutput: second, secondaryIndexExpr: CodegenAffineExpr.Axis(2)));

        Assert.Contains("index expression references affine axis 2", ex.Message,
            StringComparison.Ordinal);
    }

    /// <summary>The parameter count must include the secondary output, or a launcher misbinds.</summary>
    [Fact]
    public void ParameterCount_IncludesTheSecondaryOutput()
    {
        Assert.Equal(3, MaxPoolWithIndices(2, 4, 8, 8).ParameterCount);   // input, output, indices
    }
}
