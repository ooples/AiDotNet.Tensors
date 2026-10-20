// Copyright (c) AiDotNet. All rights reserved.
// The tiled implicit-GEMM lowering (#890 phase 4b): which axes become M, N and the batch slices,
// what it refuses, the structure of the kernel it emits, and -- on device -- every tile shape
// against the fp64 interpreter on ragged tiles and a partial reduction chunk.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;
using static AiDotNet.Tensors.Tests.Engines.Codegen.TiledPtxTestHelper;

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
    /// A strided convolution: M is what only the FIRST product operand reads -- the input, so
    /// the batch and output positions -- N what only the weights read (the output channel),
    /// and every reduction axis is R.
    /// </summary>
    [Fact]
    public void AConvolutionSplitsIntoChannelsPositionsAndTaps()
    {
        var spec = CodegenKernelCatalog.Conv2D(2, 8, 16, 12, 12, 3, 2, 1, 1, 1, epilogue: true);
        var roles = PtxAffineEmitter.PlanImplicitGemm(spec, out string reason)
            ?? throw new InvalidOperationException(reason);
        Assert.Equal("input", spec.Inputs[roles.A].Name);
        Assert.Equal(new[] { 0, 2, 3 }, roles.M);
        Assert.Equal(new[] { 1 }, roles.N);
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
    /// block per thread with fma (16 per staged reduction term, 16 terms per step) read by
    /// vector loads, double-buffers the staging (the first chunk before the loop, the next one
    /// inside it), and has no cp.async, so it runs on Turing. Off by default, so other
    /// emissions are unchanged.
    /// </summary>
    [Fact]
    public void TheKernelStagesBothOperandsAndAccumulatesARegisterBlock()
    {
        var spec = CodegenKernelCatalog.Conv2D(8, 64, 128, 56, 56, 3, 2, 1, 1, 1, epilogue: true);
        var emitter = new PtxAffineEmitter { ImplicitGemm = true };
        string ptx = emitter.Emit(spec, 7, 5);
        Assert.Contains("IG_STEP:", ptx, StringComparison.Ordinal);
        // 4 values of each operand per thread, stored once before the loop and once inside it.
        Assert.Equal(2 * (4 + 4), CountOccurrences(ptx, "st.shared.f32"));
        Assert.Equal(16 * 16, CountOccurrences(ptx, "fma.rn.f32"));
        Assert.Equal(2 * 16, CountOccurrences(ptx, "ld.shared.v4.f32"));
        Assert.Equal(2, CountOccurrences(ptx, "bar.sync"));   // after the first chunk, then one a step (double buffer)
        Assert.Contains("double-buffered, staged mr-fastest", emitter.TileDescription, StringComparison.Ordinal);
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
        Assert.Equal(4 * 16, CountOccurrences(ptx, "fma.rn.f32"));
    }

    [Fact]
    public void ARectangularTileGivesEachThreadAnEightByFourBlock()
    {
        var spec = CodegenKernelCatalog.Conv2D(8, 64, 128, 56, 56, 3, 2, 1, 1, 1, epilogue: true);
        var emitter = new PtxAffineEmitter { ImplicitGemm = true, ImplicitGemmTileM = 128 };
        string ptx = emitter.Emit(spec, 7, 5);
        Assert.Equal(8 * 4 * 16, CountOccurrences(ptx, "fma.rn.f32"));
        Assert.Contains(".minnctapersm 2", ptx, StringComparison.Ordinal);
        // 128 channels / 64 x 8*28*28 positions / 128.
        Assert.Equal((uint)(2 * ((8 * 28 * 28 + 127) / 128)), emitter.LaunchBlocks);
    }

    public static TheoryData<int, int, int> TileShapes => new()
    {
        // (shape, ImplicitGemmTile, ImplicitGemmTileM)
        { 0, 64, 0 }, { 0, 32, 0 }, { 0, 64, 96 }, { 0, 64, 128 }, { 0, 128, 64 },
        { 1, 64, 0 }, { 1, 64, 128 },
        { 2, 64, 0 }, { 2, 32, 0 },
        { 3, 64, 0 }, { 3, 64, 96 },
        { 4, 64, 0 }, { 4, 128, 64 },
    };

    private static CodegenKernelSpec Shape(int which) => which switch
    {
        // Stride 2 with bias + ReLU: 40 positions, 10 channels and 54 reduction terms, so both
        // tile edges are ragged and the last 16-term chunk is partial.
        0 => CodegenKernelCatalog.Conv2D(2, 6, 10, 9, 8, 3, 2, 1, 1, 1, epilogue: true),
        1 => CodegenKernelCatalog.Conv2D(3, 8, 8, 11, 11, 3, 1, 2, 2, 1, epilogue: false),   // dilation 2
        2 => CodegenKernelCatalog.Conv2D(2, 8, 8, 7, 7, 3, 1, 1, 1, 4, epilogue: false),     // groups: batch slices
        // The derived data gradient of a stride-2 conv (exact-division windows) and a weight gradient.
        3 => CodegenAdjoint.BackwardData(CodegenKernelCatalog.Conv2D(2, 6, 4, 9, 8, 3, 2, 1, 1, 1, epilogue: false), 0),
        _ => CodegenAdjoint.BackwardWeights(CodegenKernelCatalog.Conv2D(2, 6, 4, 9, 8, 3, 2, 1, 1, 1, epilogue: false), 1),
    };

    [SkippableTheory]
    [MemberData(nameof(TileShapes))]
    public unsafe void EveryTileShapeMatchesTheInterpreterOnDevice(int shape, int tile, int tileM)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var spec = Shape(shape);
        double[][] inputs = CreateInputs(spec, out var host);
        double[] expected = spec.Interpret(inputs);

        using var runtime = new DirectPtxRuntime();
        var emitter = new PtxAffineEmitter { ImplicitGemm = true, ImplicitGemmTile = tile, ImplicitGemmTileM = tileM };
        string ptx = emitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        Assert.StartsWith("implicit-gemm", emitter.TileDescription, StringComparison.Ordinal);
        using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
        IntPtr function = module.GetFunction(spec.Name, out _);
        var buffers = new DirectPtxBuffer[host.Length];
        try
        {
            for (int i = 0; i < host.Length; i++)
            {
                buffers[i] = runtime.AllocateBytes((nuint)(host[i].Length * sizeof(float)));
                buffers[i].Upload<float>(host[i]);
            }
            using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
            uint blocks = emitter.LaunchBlocks, threads = checked((uint)emitter.LaunchBlockThreads);
            if (host.Length == 3)
                LaunchFour(module, function, buffers[0].Pointer, buffers[1].Pointer, buffers[2].Pointer,
                    output.Pointer, blocks, threads, 1);
            else
                LaunchThree(module, function, buffers[0].Pointer, buffers[1].Pointer, output.Pointer,
                    blocks, threads, 1);
            runtime.Synchronize();
            var actual = new float[expected.Length];
            output.Download<float>(actual);
            AssertClose(expected, actual, 1e-5, emitter.TileDescription + " on shape " + shape, relative: true);
        }
        finally
        {
            foreach (var buffer in buffers) buffer?.Dispose();
        }
    }
}
