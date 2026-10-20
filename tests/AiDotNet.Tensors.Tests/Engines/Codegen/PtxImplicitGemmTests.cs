// Copyright (c) AiDotNet. All rights reserved.
// The tiled implicit-GEMM lowering (#890 phase 4b): which axes become M, N and the batch slices,
// what it refuses, and the structure of the kernel it emits. Its numerical agreement with the
// fp64 oracle is the conveyor's --kernel-verify and the autotuner's candidate check, on device.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class PtxImplicitGemmTests
{
    private static int CountOccurrences(string text, string token)
    {
        int count = 0;
        for (int at = text.IndexOf(token, StringComparison.Ordinal); at >= 0;
             at = text.IndexOf(token, at + token.Length, StringComparison.Ordinal))
            count++;
        return count;
    }

    /// <summary>
    /// A strided convolution: M is the output channel (read only by the weights), N the
    /// batch and output positions (read only by the input), and every reduction axis is R.
    /// </summary>
    [Fact]
    public void AConvolutionSplitsIntoChannelsPositionsAndTaps()
    {
        var spec = CodegenKernelCatalog.Conv2D(2, 8, 16, 12, 12, 3, 2, 1, 1, 1, epilogue: true);
        var roles = PtxAffineEmitter.PlanImplicitGemm(spec, out string reason)
            ?? throw new InvalidOperationException(reason);
        Assert.Equal(new[] { 1 }, roles.M);
        Assert.Equal(new[] { 0, 2, 3 }, roles.N);
        Assert.Empty(roles.Batch);
        Assert.Equal(spec.Space.ReductionAxes, roles.R);
    }

    /// <summary>A grouped convolution's group axis is read by both operands: a batch slice.</summary>
    [Fact]
    public void AGroupIsABatchSlice()
    {
        var spec = CodegenKernelCatalog.Conv2D(2, 16, 16, 8, 8, 3, 1, 1, 1, 4, epilogue: false);
        var roles = PtxAffineEmitter.PlanImplicitGemm(spec, out string reason)
            ?? throw new InvalidOperationException(reason);
        Assert.Equal(new[] { 1 }, roles.Batch);
    }

    [Fact]
    public void WhatTheLoweringCannotCarryIsRefused()
    {
        // A row softmax pass has one operand and a transform inside the reduction.
        Assert.Null(PtxAffineEmitter.PlanImplicitGemm(CodegenFusedStatistics.Softmax(8, 64).Passes[1], out _));
        // An expression body is not a product of two operands.
        Assert.Null(PtxAffineEmitter.PlanImplicitGemm(CodegenFusedStatistics.LayerNormBackward(8, 64).Passes[1], out _));
    }

    /// <summary>
    /// The emitted kernel stages both operands to shared memory, accumulates a 4x4 register
    /// block per thread with fma (16 per staged reduction term, 8 terms per step), and has no
    /// cp.async, so it runs on Turing. Off by default, so other emissions are unchanged.
    /// </summary>
    [Fact]
    public void TheKernelStagesBothOperandsAndAccumulatesARegisterBlock()
    {
        var spec = CodegenKernelCatalog.Conv2D(8, 64, 128, 56, 56, 3, 2, 1, 1, 1, epilogue: true);
        var emitter = new PtxAffineEmitter { ImplicitGemm = true };
        string ptx = emitter.Emit(spec, 7, 5);
        Assert.Contains("IG_STEP:", ptx, StringComparison.Ordinal);
        Assert.Equal(4, CountOccurrences(ptx, "st.shared.f32"));   // 2 values of each operand per thread
        Assert.Equal(16 * 8, CountOccurrences(ptx, "fma.rn.f32"));
        Assert.DoesNotContain("cp.async", ptx, StringComparison.Ordinal);
        Assert.Equal(256, emitter.LaunchBlockThreads);
        // 128 channels / 64 x 8*28*28 positions / 64.
        Assert.Equal((uint)(2 * ((8 * 28 * 28 + 63) / 64)), emitter.LaunchBlocks);

        string plain = new PtxAffineEmitter().Emit(spec, 7, 5);
        Assert.DoesNotContain("IG_STEP:", plain, StringComparison.Ordinal);
    }

    [Fact]
    public void A32TileHalvesTheRegisterBlock()
    {
        var spec = CodegenKernelCatalog.Conv2D(2, 32, 32, 28, 28, 3, 1, 2, 2, 1, epilogue: false);
        string ptx = new PtxAffineEmitter { ImplicitGemm = true, ImplicitGemmTile = 32 }.Emit(spec, 7, 5);
        Assert.Equal(4 * 8, CountOccurrences(ptx, "fma.rn.f32"));
    }
}
