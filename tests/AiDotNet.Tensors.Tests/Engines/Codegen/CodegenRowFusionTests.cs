// Copyright (c) AiDotNet. All rights reserved.
// Row fusion (#890 phase 4a): a row-shaped program runs as ONE warp-per-row kernel that reads
// its sources once. These pin which programs are accepted, and the property fusion exists
// for -- every source element is loaded once, however many passes read it.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenRowFusionTests
{
    [Fact]
    public void RowShapedProgramsAreAccepted()
    {
        foreach (var program in new[]
                 {
                     CodegenFusedStatistics.Softmax(64, 1024),
                     CodegenFusedStatistics.LayerNorm(64, 1024),
                     CodegenFusedStatistics.LayerNorm(64, 64),
                     CodegenFusedStatistics.RmsNorm(64, 512),
                 })
        {
            var plan = CodegenRowFusion.TryPlan(program, out string reason);
            Assert.True(plan is not null, program.Description + ": " + reason);
        }
    }

    [Fact]
    public void ProgramsTheRowKernelCannotHoldAreRefused()
    {
        // Statistics over (n, h*w) per channel: not a row of contiguous elements.
        Assert.Null(CodegenRowFusion.TryPlan(CodegenFusedStatistics.BatchNorm2D(2, 8, 4, 8), out _));
        // Four-axis passes.
        Assert.Null(CodegenRowFusion.TryPlan(CodegenFusedStatistics.GroupNorm(2, 8, 4, 32), out _));
        // A width no warp divides.
        Assert.Null(CodegenRowFusion.TryPlan(CodegenFusedStatistics.LayerNorm(16, 56), out _));
        // Wider than the registers hold.
        Assert.Null(CodegenRowFusion.TryPlan(
            CodegenFusedStatistics.LayerNorm(16, CodegenRowFusion.MaxWidth * 2), out _));
    }

    /// <summary>
    /// LayerNorm's three passes read x three times as separate kernels. Fused, each thread
    /// loads its Width/32 elements of x ONCE, plus gamma and beta: three loads per element
    /// position, not five.
    /// </summary>
    [Fact]
    public void FusedLayerNormLoadsEachSourceOnce()
    {
        var plan = CodegenRowFusion.TryPlan(CodegenFusedStatistics.LayerNorm(128, 1024), out string reason);
        Assert.True(plan is not null, reason);
        // Scalar form: one load per element position.
        var scalar = new PtxAffineEmitter { EnableVectorLoads = false };
        string ptx = scalar.EmitRowProgram(plan!, "rowfused_layernorm", 7, 5);
        Assert.Equal(3 * plan!.PerLane, scalar.EmittedLoads);
        // Two statistics, five butterfly steps each.
        Assert.Equal(10, CountOccurrences(ptx, "shfl.sync.bfly.b32"));
        // y stored by every lane; negmean and rstd by the leader only.
        Assert.Equal(plan.PerLane + 2, CountOccurrences(ptx, "st.global.f32"));
        Assert.Equal(2, CountOccurrences(ptx, "@%p1 st.global.f32"));
        Assert.Equal(128u * 32u / 256u, scalar.LaunchBlocks);

        // 128-bit form: four elements per load and per store, still each loaded once.
        var vector = new PtxAffineEmitter();
        string wide = vector.EmitRowProgram(plan, "rowfused_layernorm", 7, 5);
        Assert.Equal(3 * plan.PerLane / 4, vector.EmittedLoads);
        Assert.Equal(3 * plan.PerLane / 4, CountOccurrences(wide, "ld.global.nc.v4.f32"));
        Assert.Equal(plan.PerLane / 4, CountOccurrences(wide, "st.global.v4.f32"));
        Assert.Equal(2, CountOccurrences(wide, "@%p1 st.global.f32"));
    }

    /// <summary>A row that is not a whole number of 128-element chunks stays scalar.</summary>
    [Fact]
    public void ANarrowRowStaysScalar()
    {
        var plan = CodegenRowFusion.TryPlan(CodegenFusedStatistics.LayerNorm(128, 64), out _);
        string ptx = new PtxAffineEmitter().EmitRowProgram(plan!, "rowfused_d64", 7, 5);
        Assert.DoesNotContain("v4.f32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void ParametersAreSourcesThenOutputs()
    {
        var plan = CodegenRowFusion.TryPlan(CodegenFusedStatistics.LayerNorm(8, 64), out _);
        Assert.NotNull(plan);
        Assert.Equal(new[] { "x", "gamma", "beta", "negmean", "rstd", "y" }, plan!.Parameters);
        Assert.Equal(CodegenRowOperandKind.Matrix, plan.Kinds["x"]);
        Assert.Equal(CodegenRowOperandKind.Feature, plan.Kinds["gamma"]);
        Assert.Equal(CodegenRowOperandKind.Row, plan.Kinds["rstd"]);
    }

    /// <summary>The fused kernel must use the same exp the multi-pass kernels do.</summary>
    [Fact]
    public void FusedSoftmaxUsesTheSharedExpSequence()
    {
        var plan = CodegenRowFusion.TryPlan(CodegenFusedStatistics.Softmax(32, 256), out _);
        string fused = new PtxAffineEmitter().EmitRowProgram(plan!, "rowfused_softmax", 7, 5);
        string pass = new PtxAffineEmitter().Emit(CodegenFusedStatistics.Softmax(32, 256).Passes[1], 7, 5);
        Assert.Contains("ex2.approx.f32", fused, StringComparison.Ordinal);
        Assert.Contains("0f3FB8AA3B", fused, StringComparison.Ordinal);
        Assert.Contains("0f3FB8AA3B", pass, StringComparison.Ordinal);
        Assert.Contains("rcp.approx.f32", fused, StringComparison.Ordinal);
    }

    private static int CountOccurrences(string text, string token)
    {
        int count = 0;
        for (int at = text.IndexOf(token, StringComparison.Ordinal); at >= 0;
             at = text.IndexOf(token, at + token.Length, StringComparison.Ordinal))
            count++;
        return count;
    }
}
