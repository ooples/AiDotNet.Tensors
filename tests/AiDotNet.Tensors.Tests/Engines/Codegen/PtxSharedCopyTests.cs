// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The tiled emitters' global->shared copy: cp.async on sm_80+, a synchronous register round trip on Turing (sm_75).
// Ampere output must not change by a byte -- cubins and autotune identities are keyed on it -- and Turing output must
// contain no cp.async, which its PTX target does not have.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class PtxSharedCopyTests
{
    private static (List<string> Lines, Func<string> F, Func<string> P) Sink()
    {
        var lines = new List<string>();
        int f = 0, p = 0;
        return (lines, () => "%f" + f++, () => "%p" + p++);
    }

    private static CodegenKernelSpec Catalog(string name) =>
        CodegenKernelCatalog.Find(name)?.Bench ?? throw new InvalidOperationException(name + " is not in the catalog.");

    [Fact]
    public void Ampere_EmitsTheCpAsyncTextUnchanged()
    {
        var (lines, f, p) = Sink();
        PtxSharedCopy.Emit(true, lines.Add, f, p, "%rd5+128", "%rd6", 16, "%p3", "%p4");
        PtxSharedCopy.Emit(true, lines.Add, f, p, "%rd7", "%rd8", 4, "%p3");
        PtxSharedCopy.Commit(true, lines.Add);
        PtxSharedCopy.Wait(true, lines.Add);
        Assert.Equal(new[]
        {
            "@%p3 cp.async.ca.shared.global [%rd5+128], [%rd6], 16, %p4;",
            "@%p3 cp.async.ca.shared.global [%rd7], [%rd8], 4;",
            "cp.async.commit_group;",
            "cp.async.wait_group 0;",
        }, lines);
    }

    /// <summary>
    /// Turing: load under (valid and not ignored), store under valid, zeros otherwise -- the semantics of cp.async's
    /// ignore-source form -- and no group boundaries.
    /// </summary>
    [Fact]
    public void Turing_LoadsThroughRegistersAndZeroFillsTheIgnoredSource()
    {
        var (lines, f, p) = Sink();
        PtxSharedCopy.Emit(false, lines.Add, f, p, "%rd5+128", "%rd6", 16, "%p3", "%p4");
        PtxSharedCopy.Commit(false, lines.Add);
        PtxSharedCopy.Wait(false, lines.Add);
        Assert.Equal(new[]
        {
            "mov.f32 %f0, 0f00000000;",
            "mov.f32 %f1, 0f00000000;",
            "mov.f32 %f2, 0f00000000;",
            "mov.f32 %f3, 0f00000000;",
            "not.pred %p0, %p4;",
            "and.pred %p1, %p3, %p0;",
            "@%p1 ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd6];",
            "@%p3 st.shared.v4.f32 [%rd5+128], {%f0, %f1, %f2, %f3};",
        }, lines);
    }

    [Fact]
    public void Turing_AScalarCopyWithoutIgnoreIsOneLoadAndOneStore()
    {
        var (lines, f, p) = Sink();
        PtxSharedCopy.Emit(false, lines.Add, f, p, "%rd7", "%rd8", 4, "%p3");
        Assert.Equal(new[] { "@%p3 ld.global.f32 %f0, [%rd8];", "@%p3 st.shared.f32 [%rd7], %f0;" }, lines);
    }

    [Fact]
    public void ACopyWidthOtherThanOneTwoOrFourWordsIsRefused()
    {
        var (lines, f, p) = Sink();
        Assert.Throws<ArgumentOutOfRangeException>(
            () => PtxSharedCopy.Emit(false, lines.Add, f, p, "%rd1", "%rd2", 12, "%p0"));
    }

    /// <summary>The dense-convolution tile now emits for sm_75, with no cp.async; sm_86 keeps it.</summary>
    [Fact]
    public void TheDenseConvolutionTileEmitsForTuringWithoutCpAsync()
    {
        var spec = Catalog("conv2d_3x3_bias_relu");
        string turing = new PtxTiledConv2DEmitter().Emit(spec, 7, 5);
        string ampere = new PtxTiledConv2DEmitter().Emit(spec, 8, 6);
        Assert.Contains(".target sm_75", turing, StringComparison.Ordinal);
        Assert.DoesNotContain("cp.async", turing, StringComparison.Ordinal);
        Assert.Contains("st.shared", turing, StringComparison.Ordinal);
        Assert.Contains("cp.async.ca.shared.global", ampere, StringComparison.Ordinal);
    }

    [Fact]
    public void TheContractionTileEmitsForTuringWithoutCpAsync()
    {
        var spec = Catalog("conv2d_1x1_bias_relu");
        string turing = new PtxTiledContractionEmitter().Emit(spec, 7, 5);
        Assert.DoesNotContain("cp.async", turing, StringComparison.Ordinal);
        Assert.Contains("ld.global.v4.f32", turing, StringComparison.Ordinal);
    }
}
