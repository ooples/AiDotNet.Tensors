// Copyright (c) AiDotNet. All rights reserved.
// The activation set, and what each one costs in accuracy.
//
// Before this the emitter had exactly two activations, None and ReLU, and emitted zero
// transcendental instructions -- no ex2, lg2, rcp or rsqrt anywhere. That blocked four
// open PRs (LayerNorm+GELU, GLU, and the softmax family) on a capability rather than on
// wiring.
//
// These are the first kernels that CANNOT reach the exact 0.000E+000 the affine kernels
// hit, because ex2 and rcp are approximate instructions. So the accuracy is measured and
// asserted rather than assumed, and the oracle's formulas ARE the operator's definition:
// GELU is the tanh approximation, and swapping it for the erf form would move results by
// ~1e-3 near |x| = 2, which is a different operator.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenActivationTests
{
    /// <summary>Every activation must match its textbook definition in fp64.</summary>
    [Theory]
    [InlineData(-8.0)]
    [InlineData(-2.5)]
    [InlineData(-0.5)]
    [InlineData(0.0)]
    [InlineData(0.5)]
    [InlineData(2.5)]
    [InlineData(8.0)]
    public void Oracle_MatchesTheTextbookDefinitions(double x)
    {
        Assert.Equal(Math.Max(0.0, x),
            CodegenKernelSpec.ApplyActivation(CodegenActivationKind.ReLU, x), 12);

        Assert.Equal(1.0 / (1.0 + Math.Exp(-x)),
            CodegenKernelSpec.ApplyActivation(CodegenActivationKind.Sigmoid, x), 12);

        Assert.Equal(Math.Tanh(x),
            CodegenKernelSpec.ApplyActivation(CodegenActivationKind.Tanh, x), 12);

        Assert.Equal(x * (1.0 / (1.0 + Math.Exp(-x))),
            CodegenKernelSpec.ApplyActivation(CodegenActivationKind.Swish, x), 12);

        double gelu = 0.5 * x * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
        Assert.Equal(gelu, CodegenKernelSpec.ApplyActivation(CodegenActivationKind.Gelu, x), 12);
    }

    /// <summary>
    /// GELU here is the TANH form. Pinning the difference from the erf form stops a future
    /// "equivalent formula" edit from quietly changing the operator.
    /// </summary>
    [Fact]
    public void Gelu_IsTheTanhForm_AndDiffersMeasurablyFromErf()
    {
        // erf-based GELU at x = 2: 0.5 * 2 * (1 + erf(2/sqrt(2))) = 1 + erf(sqrt(2))/...
        // computed via the normal CDF, which is what the erf form means.
        static double ErfGelu(double x) => x * 0.5 * (1.0 + Erf(x / Math.Sqrt(2.0)));

        double tanhForm = CodegenKernelSpec.ApplyActivation(CodegenActivationKind.Gelu, 2.0);
        double erfForm = ErfGelu(2.0);

        Assert.True(Math.Abs(tanhForm - erfForm) > 1e-5,
            "the two GELU forms must differ measurably, else this test proves nothing; got " +
            tanhForm + " vs " + erfForm);
        Assert.True(Math.Abs(tanhForm - erfForm) < 1e-2,
            "they should still agree to within 1e-2, or one of them is wrong");
    }

    /// <summary>Abramowitz-Stegun 7.1.26, accurate to ~1.5e-7 — enough to tell the forms apart.</summary>
    private static double Erf(double x)
    {
        double sign = x < 0 ? -1.0 : 1.0;
        x = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                          + 0.254829592) * t * Math.Exp(-x * x);
        return sign * y;
    }

    /// <summary>
    /// Each activation must emit the instructions it claims to use, and tanh must NOT use
    /// the native approximation -- measured at 7.268E-006 against 1.576E-007 for the
    /// ex2-derived form.
    /// </summary>
    [Theory]
    [InlineData(CodegenActivationKind.Sigmoid, "ex2.approx.f32")]
    [InlineData(CodegenActivationKind.Sigmoid, "rcp.approx.f32")]
    [InlineData(CodegenActivationKind.Tanh, "ex2.approx.f32")]
    [InlineData(CodegenActivationKind.Tanh, "copysign.f32")]
    [InlineData(CodegenActivationKind.Swish, "ex2.approx.f32")]
    [InlineData(CodegenActivationKind.Gelu, "ex2.approx.f32")]
    public void EmittedKernel_ContainsTheExpectedInstruction(
        CodegenActivationKind activation, string instruction)
    {
        string ptx = new PtxAffineEmitter().Emit(SpecWith(activation), 8, 6);
        Assert.Contains(instruction, ptx, StringComparison.Ordinal);
    }

    /// <summary>The native tanh approximation must not come back: it is 46x less accurate.</summary>
    [Theory]
    [InlineData(CodegenActivationKind.Tanh)]
    [InlineData(CodegenActivationKind.Gelu)]
    public void TanhIsNotTheNativeApproximation(CodegenActivationKind activation)
    {
        string ptx = new PtxAffineEmitter().Emit(SpecWith(activation), 8, 6);
        Assert.DoesNotContain("tanh.approx", ptx, StringComparison.Ordinal);
    }

    /// <summary>ReLU stays exact — max is not an approximation, and it must not regress.</summary>
    [Fact]
    public void Relu_StillEmitsAPlainMax_AndNoTranscendental()
    {
        string ptx = new PtxAffineEmitter().Emit(SpecWith(CodegenActivationKind.ReLU), 8, 6);
        Assert.Contains("max.f32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("ex2.approx", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("rcp.approx", ptx, StringComparison.Ordinal);
    }

    /// <summary>A graph carrying any of these activations must translate, not decline.</summary>
    [Theory]
    [InlineData(CodegenOpKind.ReLU, CodegenActivationKind.ReLU)]
    [InlineData(CodegenOpKind.Sigmoid, CodegenActivationKind.Sigmoid)]
    [InlineData(CodegenOpKind.Tanh, CodegenActivationKind.Tanh)]
    [InlineData(CodegenOpKind.Swish, CodegenActivationKind.Swish)]
    [InlineData(CodegenOpKind.GELU, CodegenActivationKind.Gelu)]
    public void FrontEnd_MapsTheActivationOps(CodegenOpKind op, CodegenActivationKind expected)
    {
        int[] shape = { 8, 16 };
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, shape));
        int b = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, shape));
        int mul = g.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { a, b },
            CodegenElementType.Float32, shape));
        int act = g.AddNode(new CodegenNode(op, new[] { mul },
            CodegenElementType.Float32, shape));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { act },
            CodegenElementType.Float32, shape));

        Assert.True(CodegenGraphToSpec.TryTranslate(g, "act", out var spec, out string reason), reason);
        Assert.Equal(expected, spec!.Activation);
    }

    /// <summary>
    /// Parameterised activations must NOT be mapped. LeakyReLU and ELU carry a slope the
    /// spec has nowhere to store, so accepting them at a default slope would silently
    /// compile a different operator.
    /// </summary>
    [Theory]
    [InlineData(CodegenOpKind.LeakyReLU)]
    [InlineData(CodegenOpKind.ELU)]
    public void ParameterisedActivations_AreNotMapped(CodegenOpKind op)
    {
        int[] shape = { 8, 16 };
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, shape));
        int b = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, shape));
        int mul = g.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { a, b },
            CodegenElementType.Float32, shape));
        int act = g.AddNode(new CodegenNode(op, new[] { mul },
            CodegenElementType.Float32, shape));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { act },
            CodegenElementType.Float32, shape));

        Assert.False(CodegenGraphToSpec.TryTranslate(g, "param", out _, out string reason));
        Assert.Contains(op.ToString(), reason, StringComparison.Ordinal);
    }

    private static CodegenKernelSpec SpecWith(CodegenActivationKind activation)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", 8), CodegenAxis.Parallel("m", 16),
            CodegenAxis.Reduce("k", 4));
        var a = new CodegenTensorBinding(0, "a", new[] { 8, 4 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(2) });
        var b = new CodegenTensorBinding(1, "b", new[] { 4, 16 },
            new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(2, "out", new[] { 8, 16 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);

        return new CodegenKernelSpec("act_" + activation, space, new[] { a, b }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum, activation: activation);
    }
}
