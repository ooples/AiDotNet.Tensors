// Copyright (c) AiDotNet. All rights reserved.
// Declared kernels composed into one resident (persistent cooperative) kernel (#890 phase 4 on #1430). These pin the
// composition's structure -- what it renames, what it rewrites, where the grid barriers go, what it refuses -- on the
// CPU; DeclaredProgramRouteTests runs the composed chains on a device against the stages launched separately.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class PtxResidentComposerTests
{
    private static List<PtxProgramStep> Steps(params CodegenProgram[] programs)
    {
        var steps = new List<PtxProgramStep>();
        foreach (var program in programs) steps.AddRange(PtxProgramLowering.Create(program, "chain", 7, 5).Steps);
        return steps;
    }

    private static int Count(string text, string token)
    {
        int count = 0;
        for (int at = text.IndexOf(token, StringComparison.Ordinal); at >= 0; at = text.IndexOf(token, at + token.Length, StringComparison.Ordinal))
            count++;
        return count;
    }

    /// <summary>
    /// LayerNorm then softmax: one entry, each stage's labels under its own prefix, a grid barrier
    /// between the stages and none after the last, and the arguments every stage's then the barrier.
    /// </summary>
    [Fact]
    public void TwoStagesComposeIntoOneEntryWithOneGridBarrier()
    {
        var steps = Steps(CodegenBackendPrograms.LayerNorm(64, 1024, 1e-5), CodegenFusedStatistics.Softmax(64, 1024));
        Assert.Equal(2, steps.Count);
        var composed = PtxResidentComposer.Compose(steps, "resident_chain", 7, 5);

        Assert.Equal(1, Count(composed.Ptx, ".entry "));
        Assert.Contains("s0_BLOCK:", composed.Ptx, StringComparison.Ordinal);
        Assert.Contains("s1_BLOCK:", composed.Ptx, StringComparison.Ordinal);
        Assert.Equal(1, Count(composed.Ptx, "atom.global.add.u32 %resarr"));   // one barrier: between the two stages
        Assert.DoesNotContain("%ctaid.x;", composed.Ptx.Replace("mov.u32 %resvb, %ctaid.x;", ""), StringComparison.Ordinal);
        Assert.DoesNotContain("ld.global.nc", composed.Ptx, StringComparison.Ordinal);
        Assert.Equal(steps[0].Arguments.Count + steps[1].Arguments.Count + 1, composed.Arguments.Count);
        Assert.Equal(PtxResidentComposer.BarrierArgument, composed.Arguments.Last());
        Assert.Equal(Math.Max(steps[0].Blocks, steps[1].Blocks), composed.MaxStageBlocks);
        Assert.Equal(steps[0].BlockX, composed.BlockX);
    }

    /// <summary>A stage's own END label and ret become the end of its virtual block, never the kernel's return.</summary>
    [Fact]
    public void AStageReturnEndsItsVirtualBlock()
    {
        var steps = Steps(CodegenFusedStatistics.Softmax(16, 256), CodegenFusedStatistics.Softmax(16, 256));
        var composed = PtxResidentComposer.Compose(steps, "resident_chain", 7, 5);
        Assert.Equal(1, Count(composed.Ptx, "ret;"));                       // the composed kernel's own
        Assert.Contains("bra s0_NEXTBLOCK;", composed.Ptx, StringComparison.Ordinal);
        Assert.Contains("bra s1_NEXTBLOCK;", composed.Ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// A stage whose body calls a math-library function: the call's extern declaration is carried into the composed
    /// module once, ahead of the entry, and the call's own { } scope does not end the stage's body early.
    /// </summary>
    [Fact]
    public void AStageThatCallsALibraryFunctionComposes()
    {
        var twice = new CodegenLibraryFunction("aidn_twice", "test_module", 1);
        var spec = new CodegenKernelSpec("twice", new CodegenIterationSpace(CodegenAxis.Parallel("i", 4096)),
            new[] { new CodegenTensorBinding(0, "x", new[] { 4096 }, new[] { CodegenAffineExpr.Axis(0) }) },
            new CodegenTensorBinding(1, "y", new[] { 4096 }, new[] { CodegenAffineExpr.Axis(0) }, isOutput: true),
            new[] { 0 }, CodegenReduceKind.None, body: CodegenExpr.Call(twice, CodegenExpr.Operand(0)));
        var program = new CodegenProgram(new[] { spec }, new long[] { 4096 }, "twice");
        var steps = Steps(program, program);
        Assert.Equal(new[] { twice }, steps[0].Calls);
        var composed = PtxResidentComposer.Compose(steps, "resident_calls", 7, 5);

        Assert.Equal(1, Count(composed.Ptx, ".extern .func"));
        Assert.True(composed.Ptx.IndexOf(".extern .func", StringComparison.Ordinal) < composed.Ptx.IndexOf(".entry", StringComparison.Ordinal));
        Assert.Equal(new[] { twice }, composed.Calls);
        // Both stages' bodies survive whole: each still reaches its own end-of-block branch after the call.
        int call0 = composed.Ptx.IndexOf("call.uni", StringComparison.Ordinal);
        Assert.True(call0 >= 0 && composed.Ptx.IndexOf("s0_NEXTBLOCK:", StringComparison.Ordinal) > call0);
        Assert.Contains("s1_NEXTBLOCK:", composed.Ptx, StringComparison.Ordinal);
        // Every call of both stages survives (a held row calls once per element it holds).
        Assert.Equal(2 * Count(steps[0].Ptx, "call.uni"), Count(composed.Ptx, "call.uni"));
    }

    /// <summary>
    /// Three implicit-GEMM stages, each staging its tiles through 18.5 KB of shared memory: their arrays are overlaid on
    /// one buffer the size of one stage's (summed, the chain was refused at 55.5 KB), and every stage addresses it.
    /// </summary>
    [Fact]
    public void StagesOverlayTheirSharedMemory()
    {
        static PtxPassLowering? Lower(CodegenKernelSpec pass)
        {
            var emitter = new PtxAffineEmitter { ImplicitGemm = true };
            string ptx = emitter.Emit(pass, 7, 5);
            return new PtxPassLowering(new[]
            {
                new PtxPassKernel(ptx, pass.Name, emitter.LaunchBlocks, (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY),
            }, null, "implicit-gemm");
        }

        var steps = new List<PtxProgramStep>();
        for (int i = 0; i < 3; i++)
            steps.AddRange(PtxProgramLowering.Create(CodegenBackendPrograms.Gemm(1, 64, 96, 80, false, false), "g" + i, 7, 5,
                Lower).Steps);
        var declaration = System.Text.RegularExpressions.Regex.Match(steps[0].Ptx, @"\.shared\s+\.align\s+\d+\s+\.b8\s+\w+\[(\d+)\]");
        Assert.True(declaration.Success, "an implicit-GEMM stage stages through shared memory");
        int stageBytes = int.Parse(declaration.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
        Assert.True(3 * stageBytes > 48 * 1024, "summed, the three stages would not fit");

        var composed = PtxResidentComposer.Compose(steps, "resident_gemms", 7, 5);
        Assert.Equal(stageBytes, composed.SharedBytes);
        Assert.Equal(1, Count(composed.Ptx, ".shared "));
        foreach (string prefix in new[] { "s0_", "s1_", "s2_" })
            Assert.Contains(prefix + "BLOCK:", composed.Ptx, StringComparison.Ordinal);
        Assert.True(Count(composed.Ptx, "resident_shared") >= 4, "the declaration and every stage's use");
    }

    /// <summary>Stages that do not share a block size, or that use a 2-D block, are refused.</summary>
    [Fact]
    public void WhatACompositionCannotCarryIsRefused()
    {
        var softmax = Steps(CodegenFusedStatistics.Softmax(16, 256))[0];
        var narrower = softmax with { BlockX = softmax.BlockX / 2 };
        Assert.Throws<NotSupportedException>(() => PtxResidentComposer.Compose(new[] { softmax, narrower }, "x", 7, 5));
        var twoDimensional = softmax with { BlockY = 2 };
        Assert.Throws<NotSupportedException>(() => PtxResidentComposer.Compose(new[] { twoDimensional }, "x", 7, 5));
    }
}
