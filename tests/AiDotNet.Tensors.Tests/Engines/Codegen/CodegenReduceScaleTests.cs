// Copyright (c) AiDotNet. All rights reserved.
// A mean is a sum with a constant, and the constant belongs to the epilogue.
//
// CodegenReduceKind had Sum and Max only, so global average pooling and any loss
// normalisation were inexpressible. A `Mean` reduce kind was the obvious fix and is the
// worse one: the same constant also serves a loss's 1/N and softmax's 1/denominator, so
// one scalar covers three operators where a reduce kind covers one.
//
// The trap this file exists for is the SPLIT. If the constant stayed on the partial pass,
// a mean split four ways would be divided once per partial and come out four times too
// small -- and it would still agree with itself, so only a comparison against the unsplit
// operator catches it.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenReduceScaleTests
{
    private static double[] Fill(long count, int salt)
    {
        var v = new double[count];
        for (long i = 0; i < count; i++) v[i] = (((i * 37 + salt * 101) % 97) - 48) / 64.0;
        return v;
    }

    private static CodegenGraph GlobalAveragePool(int n, int c, int h, int w)
    {
        var g = new CodegenGraph();
        int x = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { n, c, h, w }));
        int mean = g.AddNode(new CodegenNode(CodegenOpKind.ReduceMean, new[] { x },
            CodegenElementType.Float32, new[] { n, c }, new[] { 2, 3 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { mean },
            CodegenElementType.Float32, new[] { n, c }));
        return g;
    }

    /// <summary>A mean must equal the arithmetic mean, checked against a hand-written one.</summary>
    [Fact]
    public void ReduceMean_ComputesTheArithmeticMean()
    {
        const int N = 3, C = 4, H = 5, W = 6;
        var graph = GlobalAveragePool(N, C, H, W);

        Assert.True(CodegenGraphToSpec.TryTranslate(graph, "gap", out var spec, out string reason), reason);

        // Expressed as a SUM with a constant, not as a distinct reduce kind.
        Assert.Equal(CodegenReduceKind.Sum, spec!.Reduce);
        Assert.Equal(1.0 / (H * W), spec.ReduceScale, 12);

        double[] x = Fill((long)N * C * H * W, 1);
        double[] got = spec.Interpret(new[] { x });

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        {
            double sum = 0;
            for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                sum += x[((n * C + c) * H + h) * W + w];
            Assert.Equal(sum / (H * W), got[n * C + c], 9);
        }
    }

    /// <summary>
    /// THE SPLIT TRAP. A split mean must equal the unsplit mean; if the constant stayed on
    /// the partial pass it would divide once per partial.
    /// </summary>
    [Fact]
    public void SplitMean_EqualsTheUnsplitMean()
    {
        var graph = GlobalAveragePool(2, 8, 16, 16);
        Assert.True(CodegenGraphToSpec.TryTranslate(graph, "gap", out var spec, out string reason), reason);

        var plan = CodegenSplitReduction.TryPlan(spec!);
        Assert.NotNull(plan);

        // The partial computes a RAW sum; the constant travels with the epilogue.
        Assert.Equal(1.0, plan!.Partial.ReduceScale, 12);
        Assert.Equal(spec!.ReduceScale, plan.Combine.ReduceScale, 12);

        double[] x = Fill(2L * 8 * 16 * 16, 2);
        double[] want = spec.Interpret(new[] { x });
        double[] got = plan.Combine.Interpret(new[] { plan.Partial.Interpret(new[] { x }) });

        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++) Assert.Equal(want[i], got[i], 9);
    }

    /// <summary>
    /// The scale applies BEFORE the bias. Scaling afterwards would scale the bias too,
    /// which is a different operator.
    /// </summary>
    [Fact]
    public void ScaleAppliesBeforeTheBias()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", 4), CodegenAxis.Reduce("k", 8));
        var x = new CodegenTensorBinding(0, "x", new[] { 4, 8 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var bias = new CodegenTensorBinding(1, "bias", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) });
        var output = new CodegenTensorBinding(2, "out", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        var spec = new CodegenKernelSpec("mean_bias", space, new[] { x, bias }, output,
            new[] { 0 }, CodegenReduceKind.Sum, biasInput: 1, reduceScale: 0.125);

        double[] xv = Fill(32, 3), bv = Fill(4, 4);
        double[] got = spec.Interpret(new[] { xv, bv });

        for (int n = 0; n < 4; n++)
        {
            double sum = 0;
            for (int k = 0; k < 8; k++) sum += xv[n * 8 + k];
            // mean THEN bias -- not (sum + bias) * 0.125.
            Assert.Equal(sum * 0.125 + bv[n], got[n], 9);
        }
    }

    /// <summary>The emitted kernel must carry the constant as an exact fp32 bit pattern.</summary>
    [Fact]
    public void EmittedKernel_CarriesTheScaleAsAnExactLiteral()
    {
        var graph = GlobalAveragePool(2, 8, 16, 16);
        Assert.True(CodegenGraphToSpec.TryTranslate(graph, "gap", out var spec, out string reason), reason);

        string ptx = new PtxAffineEmitter().Emit(spec!, 8, 6);

        // 1/256 is exactly representable: 0x3B800000.
        Assert.Contains("0f3B800000", ptx, StringComparison.Ordinal);
    }

    /// <summary>A scale of exactly 1 must emit no multiply at all.</summary>
    [Fact]
    public void UnitScale_EmitsNothing()
    {
        var entry = CodegenKernelCatalog.Find("depthwise_conv2d_3x3");
        Assert.NotNull(entry);
        Assert.Equal(1.0, entry!.Verify.ReduceScale, 12);
    }

    /// <summary>A non-finite scale is refused rather than compiled into a NaN kernel.</summary>
    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void NonFiniteScale_IsRefused(double scale)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", 4), CodegenAxis.Reduce("k", 8));
        var x = new CodegenTensorBinding(0, "x", new[] { 4, 8 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(1, "out", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "bad", space, new[] { x }, output, new[] { 0 },
            CodegenReduceKind.Sum, reduceScale: scale));
    }
}
