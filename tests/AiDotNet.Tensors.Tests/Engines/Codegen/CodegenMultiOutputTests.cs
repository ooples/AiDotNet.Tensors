// Copyright (c) AiDotNet. All rights reserved.
// Three or more outputs written from one iteration point.
//
// The generator could express exactly two: a primary and one argmax. That capped the whole
// optimizer family, because an Adam step writes THREE things -- the first moment, the second
// moment, and the updated parameter -- and splitting it across kernels re-reads every operand
// once per kernel, which is precisely the cost the fusion argument is about.
//
// The dangerous failure here is two outputs landing on the same parameter. Both stores go to
// one buffer with no ordering between them, so the result depends on warp scheduling: it
// produces plausible values and a different answer per run. That is refused at construction.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenMultiOutputTests
{
    /// <summary>
    /// An SGD-with-momentum step, which is the smallest honest three-output shape:
    /// <c>v' = mu*v + g</c> as the primary, and <c>p' = p - lr*v'</c> as an extra.
    /// </summary>
    private static CodegenKernelSpec MomentumStep(int count, double momentum, double learningRate)
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", count));
        var map = new[] { CodegenAffineExpr.Axis(0) };

        var velocity = new CodegenTensorBinding(0, "v", new[] { count }, map);
        var gradient = new CodegenTensorBinding(1, "g", new[] { count }, map);
        var parameter = new CodegenTensorBinding(2, "p", new[] { count }, map);

        var newVelocity = new CodegenTensorBinding(3, "v_out", new[] { count }, map, isOutput: true);
        var newParameter = new CodegenTensorBinding(4, "p_out", new[] { count }, map, isOutput: true);

        // primary = mu*v + g   (velocity scaled by ReduceScale, gradient added as bias)
        return new CodegenKernelSpec("momentum", space, new[] { velocity, gradient, parameter },
            newVelocity, new[] { 0 }, CodegenReduceKind.None,
            biasInput: 1, reduceScale: momentum,
            extraOutputs: new[]
            {
                // p' = p - lr*v'  ==  (-lr)*primary + 1.0*p
                new CodegenExtraOutput(newParameter, CodegenExtraOutputKind.AffineOfPrimary,
                    Scale: -learningRate, BiasInput: 2, BiasScale: 1.0),
            });
    }

    /// <summary>Four outputs from one point, to show the count is genuinely not capped.</summary>
    private static CodegenKernelSpec FourOutputs(int count)
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", count));
        var map = new[] { CodegenAffineExpr.Axis(0) };

        var x = new CodegenTensorBinding(0, "x", new[] { count }, map);
        var primary = new CodegenTensorBinding(1, "a", new[] { count }, map, isOutput: true);
        var b = new CodegenTensorBinding(2, "b", new[] { count }, map, isOutput: true);
        var c = new CodegenTensorBinding(3, "c", new[] { count }, map, isOutput: true);
        var d = new CodegenTensorBinding(4, "d", new[] { count }, map, isOutput: true);

        return new CodegenKernelSpec("four", space, new[] { x }, primary,
            new[] { 0 }, CodegenReduceKind.None,
            extraOutputs: new[]
            {
                new CodegenExtraOutput(b, CodegenExtraOutputKind.AffineOfPrimary, Scale: 2.0),
                new CodegenExtraOutput(c, CodegenExtraOutputKind.AffineOfPrimary, Scale: -1.0),
                new CodegenExtraOutput(d, CodegenExtraOutputKind.AffineOfPrimary, Scale: 0.5),
            });
    }

    private static double[] BuildRamp(int count, double scale)
    {
        var data = new double[count];
        for (int i = 0; i < count; i++) data[i] = ((i % 11) - 5) * scale;
        return data;
    }

    /// <summary>The momentum step, against arithmetic written out by hand.</summary>
    [Fact]
    public void MomentumStep_MatchesAHandWrittenReference()
    {
        const int Count = 16;
        const double Momentum = 0.9, LearningRate = 0.01;

        var spec = MomentumStep(Count, Momentum, LearningRate);
        double[] v = BuildRamp(Count, 0.25), g = BuildRamp(Count, 0.5), p = BuildRamp(Count, 1.0);

        double[][] all = spec.InterpretAll(new[] { v, g, p });

        Assert.Equal(2, all.Length);
        for (int i = 0; i < Count; i++)
        {
            double expectedVelocity = Momentum * v[i] + g[i];
            Assert.Equal(expectedVelocity, all[0][i], 9);
            Assert.Equal(p[i] - LearningRate * expectedVelocity, all[1][i], 9);
        }
    }

    /// <summary>
    /// The extra is computed from the FINISHED primary, not a partial one. An optimizer's
    /// parameter update steps by the state it just computed; using the pre-epilogue value
    /// would step by the wrong quantity and still look like a plausible training curve.
    /// </summary>
    [Fact]
    public void ExtraOutput_UsesThePrimaryAfterItsEpilogue()
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", 4));
        var map = new[] { CodegenAffineExpr.Axis(0) };

        var x = new CodegenTensorBinding(0, "x", new[] { 4 }, map);
        var primary = new CodegenTensorBinding(1, "out", new[] { 4 }, map, isOutput: true);
        var doubled = new CodegenTensorBinding(2, "twice", new[] { 4 }, map, isOutput: true);

        // ReLU on the primary; the extra is twice that, so a negative input must give 0 and
        // not -2x.
        var spec = new CodegenKernelSpec("relu_pair", space, new[] { x }, primary,
            new[] { 0 }, CodegenReduceKind.None,
            activation: CodegenActivationKind.ReLU,
            extraOutputs: new[]
            {
                new CodegenExtraOutput(doubled, CodegenExtraOutputKind.AffineOfPrimary, Scale: 2.0),
            });

        double[][] all = spec.InterpretAll(new[] { new double[] { -3, -1, 2, 5 } });

        Assert.Equal(new double[] { 0, 0, 2, 5 }, all[0]);
        Assert.Equal(new double[] { 0, 0, 4, 10 }, all[1]);
    }

    /// <summary>Four outputs, all from one pass over the input.</summary>
    [Fact]
    public void FourOutputs_AreAllWritten()
    {
        var spec = FourOutputs(8);
        double[] x = BuildRamp(8, 1.0);

        double[][] all = spec.InterpretAll(new[] { x });

        Assert.Equal(4, all.Length);
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(x[i], all[0][i], 9);
            Assert.Equal(2.0 * x[i], all[1][i], 9);
            Assert.Equal(-1.0 * x[i], all[2][i], 9);
            Assert.Equal(0.5 * x[i], all[3][i], 9);
        }
    }

    /// <summary>The parameter count must cover every output, or the launch passes too few buffers.</summary>
    [Fact]
    public void ParameterCount_CoversEveryOutput()
    {
        Assert.Equal(1 + 1 + 3, FourOutputs(8).ParameterCount);          // 1 input, 4 outputs
        Assert.Equal(3 + 1 + 1, MomentumStep(8, 0.9, 0.01).ParameterCount);
    }

    /// <summary>Every extra output must produce its own store.</summary>
    [Fact]
    public void EmittedKernel_StoresEveryExtra()
    {
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(FourOutputs(1024), 8, 6);

        Assert.Equal(3, emitter.ExtraOutputStores);
        Assert.Contains("st.global.f32", ptx, StringComparison.Ordinal);
    }

    /// <summary>The momentum step must emit the fused multiply-add its extra describes.</summary>
    [Fact]
    public void MomentumStep_EmitsAFusedUpdate()
    {
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(MomentumStep(1024, 0.9, 0.01), 8, 6);

        Assert.Equal(1, emitter.ExtraOutputStores);
        Assert.Contains("fma.rn.f32", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// The legacy secondary-output pair still works, and is now the SAME mechanism -- it
    /// folds into the extras list rather than living beside it.
    /// </summary>
    [Fact]
    public void LegacySecondaryOutput_IsAnExtraOutput()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", 8), CodegenAxis.Reduce("k", 4));

        var x = new CodegenTensorBinding(0, "x", new[] { 8, 4 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var values = new CodegenTensorBinding(1, "out", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        var indices = new CodegenTensorBinding(2, "idx", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        var spec = new CodegenKernelSpec("maxpool_like", space, new[] { x }, values,
            new[] { 0 }, CodegenReduceKind.Max,
            secondaryOutput: indices, secondaryIndexExpr: CodegenAffineExpr.Axis(1));

        Assert.Single(spec.ExtraOutputs);
        Assert.Equal(CodegenExtraOutputKind.ArgMaxIndex, spec.ExtraOutputs[0].Kind);
        Assert.Same(indices, spec.SecondaryOutput);
        Assert.Equal(3, spec.ParameterCount);
    }

    /// <summary>An argmax extra alongside two affine extras: all three coexist.</summary>
    [Fact]
    public void ArgMaxAndAffineExtras_Coexist()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", 8), CodegenAxis.Reduce("k", 4));
        var rowMap = new[] { CodegenAffineExpr.Axis(0) };

        var x = new CodegenTensorBinding(0, "x", new[] { 8, 4 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var values = new CodegenTensorBinding(1, "out", new[] { 8 }, rowMap, isOutput: true);
        var indices = new CodegenTensorBinding(2, "idx", new[] { 8 }, rowMap, isOutput: true);
        var scaled = new CodegenTensorBinding(3, "scaled", new[] { 8 }, rowMap, isOutput: true);

        var spec = new CodegenKernelSpec("max_plus", space, new[] { x }, values,
            new[] { 0 }, CodegenReduceKind.Max,
            extraOutputs: new[]
            {
                new CodegenExtraOutput(scaled, CodegenExtraOutputKind.AffineOfPrimary, Scale: 3.0),
            },
            secondaryOutput: indices, secondaryIndexExpr: CodegenAffineExpr.Axis(1));

        var data = new double[8 * 4];
        for (int i = 0; i < data.Length; i++) data[i] = ((i * 7) % 13) - 6;

        double[][] all = spec.InterpretAll(new[] { data });
        Assert.Equal(3, all.Length);

        for (int i = 0; i < 8; i++)
        {
            double best = double.NegativeInfinity;
            int bestIndex = 0;
            for (int k = 0; k < 4; k++)
                if (data[i * 4 + k] > best) { best = data[i * 4 + k]; bestIndex = k; }

            Assert.Equal(best, all[0][i], 9);
            Assert.Equal(bestIndex, all[1][i], 9);
            Assert.Equal(3.0 * best, all[2][i], 9);
        }
    }

    // ---- What must be refused ------------------------------------------------------------

    /// <summary>
    /// Two outputs on one parameter is a silent data race: both stores land on the same
    /// buffer with no ordering, so the answer depends on warp scheduling.
    /// </summary>
    [Fact]
    public void TwoOutputsOnOneParameter_AreRefused()
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", 8));
        var map = new[] { CodegenAffineExpr.Axis(0) };

        var x = new CodegenTensorBinding(0, "x", new[] { 8 }, map);
        var primary = new CodegenTensorBinding(1, "out", new[] { 8 }, map, isOutput: true);
        var clash = new CodegenTensorBinding(1, "clash", new[] { 8 }, map, isOutput: true);

        var ex = Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "racy", space, new[] { x }, primary, new[] { 0 }, CodegenReduceKind.None,
            extraOutputs: new[]
            {
                new CodegenExtraOutput(clash, CodegenExtraOutputKind.AffineOfPrimary),
            }));

        Assert.Contains("race", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>Two extras colliding with each other are caught too, not just with the primary.</summary>
    [Fact]
    public void TwoExtrasOnOneParameter_AreRefused()
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", 8));
        var map = new[] { CodegenAffineExpr.Axis(0) };

        var x = new CodegenTensorBinding(0, "x", new[] { 8 }, map);
        var primary = new CodegenTensorBinding(1, "out", new[] { 8 }, map, isOutput: true);
        var a = new CodegenTensorBinding(2, "a", new[] { 8 }, map, isOutput: true);
        var b = new CodegenTensorBinding(2, "b", new[] { 8 }, map, isOutput: true);

        Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "racy", space, new[] { x }, primary, new[] { 0 }, CodegenReduceKind.None,
            extraOutputs: new[]
            {
                new CodegenExtraOutput(a, CodegenExtraOutputKind.AffineOfPrimary),
                new CodegenExtraOutput(b, CodegenExtraOutputKind.AffineOfPrimary),
            }));
    }

    /// <summary>An argmax extra without a Max reduction has no winning term to report.</summary>
    [Fact]
    public void ArgMaxExtraOnASum_IsRefused()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", 8), CodegenAxis.Reduce("k", 4));
        var rowMap = new[] { CodegenAffineExpr.Axis(0) };

        var x = new CodegenTensorBinding(0, "x", new[] { 8, 4 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var values = new CodegenTensorBinding(1, "out", new[] { 8 }, rowMap, isOutput: true);
        var indices = new CodegenTensorBinding(2, "idx", new[] { 8 }, rowMap, isOutput: true);

        var ex = Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "bad", space, new[] { x }, values, new[] { 0 }, CodegenReduceKind.Sum,
            extraOutputs: new[]
            {
                new CodegenExtraOutput(indices, CodegenExtraOutputKind.ArgMaxIndex,
                    CodegenAffineExpr.Axis(1)),
            }));

        Assert.Contains("Max reduction", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>An extra binding not marked IsOutput would be read-addressed.</summary>
    [Fact]
    public void ExtraNotMarkedAsOutput_IsRefused()
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", 8));
        var map = new[] { CodegenAffineExpr.Axis(0) };

        var x = new CodegenTensorBinding(0, "x", new[] { 8 }, map);
        var primary = new CodegenTensorBinding(1, "out", new[] { 8 }, map, isOutput: true);
        var notAnOutput = new CodegenTensorBinding(2, "nope", new[] { 8 }, map);

        var ex = Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "bad", space, new[] { x }, primary, new[] { 0 }, CodegenReduceKind.None,
            extraOutputs: new[]
            {
                new CodegenExtraOutput(notAnOutput, CodegenExtraOutputKind.AffineOfPrimary),
            }));

        Assert.Contains("IsOutput", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>Kernels with no extras must be untouched.</summary>
    [Fact]
    public void KernelsWithoutExtras_AreUnchanged()
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", 256));
        var map = new[] { CodegenAffineExpr.Axis(0) };
        var x = new CodegenTensorBinding(0, "x", new[] { 256 }, map);
        var output = new CodegenTensorBinding(1, "out", new[] { 256 }, map, isOutput: true);

        var spec = new CodegenKernelSpec("plain", space, new[] { x }, output,
            new[] { 0 }, CodegenReduceKind.None);

        var emitter = new PtxAffineEmitter();
        emitter.Emit(spec, 8, 6);

        Assert.Empty(spec.ExtraOutputs);
        Assert.Equal(0, emitter.ExtraOutputStores);
        Assert.Equal(2, spec.ParameterCount);
    }
}
