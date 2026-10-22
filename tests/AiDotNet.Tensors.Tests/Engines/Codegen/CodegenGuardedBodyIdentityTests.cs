// Copyright (c) AiDotNet. All rights reserved.
// The guarded-body half of the padding-identity contract (see CodegenPaddedMaxTests).
//
// CodegenKernelSpec.Interpret gives an out-of-range tap the identity OF THE REDUCTION and
// skips the pre-bias and the pre-reduce for that term. For a bare product of operands under
// Sum the emitter can leave the select out, because the guarded load already fills with the
// identity and 0 * x == 0 carries it to the accumulator untouched. An EXPRESSION BODY breaks
// that argument: a - b maps the 0 fill to -b, which is a real contribution, so the emitter
// has to put the identity back after the body.
//
// These pin both halves: the oracle against hand-written arithmetic that skips rather than
// fills, and the emitted text against the three cases that must differ from each other.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenGuardedBodyIdentityTests
{
    private const int Outputs = 8, Taps = 3;

    /// <summary>sum over k of (a[o + k - 1] - b[k]); a is a padded window, so it is guarded.</summary>
    private static CodegenKernelSpec GuardedBody()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("o", Outputs), CodegenAxis.Reduce("k", Taps));
        var a = new CodegenTensorBinding(0, "a", new[] { Outputs },
            new[] { CodegenAffineExpr.Window(0, 1, 1, 1) });
        var b = new CodegenTensorBinding(1, "b", new[] { Taps },
            new[] { CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(2, "out", new[] { Outputs },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var body = CodegenExpr.Subtract(CodegenExpr.Operand(0), CodegenExpr.Operand(1));
        return new CodegenKernelSpec("guarded_body_sum", space, new[] { a, b }, output,
            body.Operands(), CodegenReduceKind.Sum, body: body);
    }

    /// <summary>The same body with no window, so no tap is ever out of range.</summary>
    private static CodegenKernelSpec UnguardedBody()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("o", Outputs), CodegenAxis.Reduce("k", Taps));
        var a = new CodegenTensorBinding(0, "a", new[] { Outputs, Taps },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var b = new CodegenTensorBinding(1, "b", new[] { Taps },
            new[] { CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(2, "out", new[] { Outputs },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var body = CodegenExpr.Subtract(CodegenExpr.Operand(0), CodegenExpr.Operand(1));
        return new CodegenKernelSpec("unguarded_body_sum", space, new[] { a, b }, output,
            body.Operands(), CodegenReduceKind.Sum, body: body);
    }

    /// <summary>
    /// The oracle, against a hand-written sum that SKIPS an out-of-range tap. Filling the tap
    /// and then evaluating the body would contribute -b[k] for every border tap instead of 0.
    /// </summary>
    [Fact]
    public void GuardedBody_OutOfRangeTapContributesTheAdditiveIdentity()
    {
        var a = new double[Outputs];
        for (int i = 0; i < Outputs; i++) a[i] = 1.0 + i;
        var b = new double[Taps];
        for (int k = 0; k < Taps; k++) b[k] = 10.0 * (k + 1);     // large, so a wrong term cannot hide

        double[] got = GuardedBody().Interpret(new[] { a, b });

        for (int o = 0; o < Outputs; o++)
        {
            double want = 0.0;
            for (int k = 0; k < Taps; k++)
            {
                int i = o + k - 1;
                if (i < 0 || i >= Outputs) continue;              // SKIP, do not fill and evaluate
                want += a[i] - b[k];
            }
            Assert.Equal(want, got[o], 9);
        }
    }

    /// <summary>
    /// The border outputs must differ from what filling-then-evaluating would give, so the
    /// previous assertion is not satisfied by both readings of the contract.
    /// </summary>
    [Fact]
    public void GuardedBody_FillingThenEvaluatingWouldGiveADifferentAnswer()
    {
        var a = new double[Outputs];
        for (int i = 0; i < Outputs; i++) a[i] = 1.0 + i;
        var b = new double[Taps];
        for (int k = 0; k < Taps; k++) b[k] = 10.0 * (k + 1);

        double[] got = GuardedBody().Interpret(new[] { a, b });

        double filled = 0.0;
        for (int k = 0; k < Taps; k++)
        {
            int i = 0 + k - 1;
            filled += (i < 0 || i >= Outputs ? 0.0 : a[i]) - b[k];
        }
        Assert.NotEqual(filled, got[0], 9);
    }

    /// <summary>
    /// The emitter half: the guarded body's term is selected back to the fill before it reaches
    /// the accumulator. A Sum kernel emits no selp.f32 for any other reason -- the fill is a
    /// plain zero store and the argmax select is selp.b32 under Max -- so its presence here is
    /// the predicate being applied.
    /// </summary>
    [Fact]
    public void EmittedGuardedBodyKernel_SelectsTheIdentityForAnOutOfRangeTerm()
    {
        string ptx = new PtxAffineEmitter().Emit(GuardedBody(), 8, 6);
        Assert.Contains("selp.f32", ptx, StringComparison.Ordinal);
    }

    /// <summary>An unguarded body has no predicate, so it must not pay for a select.</summary>
    [Fact]
    public void EmittedUnguardedBodyKernel_HasNoIdentitySelect()
    {
        string ptx = new PtxAffineEmitter().Emit(UnguardedBody(), 8, 6);
        Assert.DoesNotContain("selp.f32", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// A guarded PLAIN PRODUCT under Sum keeps its instruction count: 0 * x == 0 already carries
    /// the identity, and dense convolution's input is guarded on every tap, so a select per term
    /// here would be a real cost for no correctness gain.
    /// </summary>
    [Fact]
    public void EmittedGuardedProductKernel_StillNeedsNoIdentitySelect()
    {
        var entry = CodegenKernelCatalog.Find("depthwise_conv2d_3x3");
        Assert.NotNull(entry);
        Assert.Null(entry!.Verify.Body);

        string ptx = new PtxAffineEmitter().Emit(entry.Verify, 8, 6);
        Assert.DoesNotContain("selp.f32", ptx, StringComparison.Ordinal);
    }
}
