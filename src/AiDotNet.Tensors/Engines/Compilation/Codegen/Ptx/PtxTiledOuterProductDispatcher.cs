// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

/// <summary>The emitted tiled split partial and the metadata shared by its consumers.</summary>
internal sealed class PtxTiledOuterProductProgram
{
    internal PtxTiledOuterProductProgram(
        string text, int blocks, int blockThreads, int steps,
        int innerReduction, int tileM, int tileN, int sharedMemoryBytes, string stagedLabel,
        string? outerProductRefusal)
    {
        Text = text;
        Blocks = blocks;
        BlockThreads = blockThreads;
        Steps = steps;
        InnerReduction = innerReduction;
        TileM = tileM;
        TileN = tileN;
        SharedMemoryBytes = sharedMemoryBytes;
        StagedLabel = stagedLabel;
        OuterProductRefusal = outerProductRefusal;
    }

    internal string Text { get; }
    internal int Blocks { get; }
    internal uint LaunchBlocks => checked((uint)Blocks);
    internal int BlockThreads { get; }
    internal int Steps { get; }
    internal int InnerReduction { get; }
    internal int TileM { get; }
    internal int TileN { get; }
    internal int SharedMemoryBytes { get; }
    internal string StagedLabel { get; }

    /// <summary>Why the generic emitter refused when the Conv2D fallback won.</summary>
    internal string? OuterProductRefusal { get; }
}

/// <summary>Both tiled split emitters refused the specification.</summary>
internal sealed class PtxTiledOuterProductDispatchException : NotSupportedException
{
    internal PtxTiledOuterProductDispatchException(
        string outerProductRefusal, NotSupportedException fallback)
        : base(fallback.Message, fallback)
    {
        OuterProductRefusal = outerProductRefusal;
    }

    internal string OuterProductRefusal { get; }
}

/// <summary>Single authority for selecting a tiled split-partial emitter.</summary>
internal static class PtxTiledOuterProductDispatcher
{
    internal static PtxTiledOuterProductProgram Emit(
        CodegenKernelSpec spec, int computeMajor, int computeMinor)
    {
        try
        {
            var emitter = new PtxTiledOuterProductEmitter();
            string text = emitter.Emit(spec, computeMajor, computeMinor);
            CodegenTiledOuterProductPlan plan = emitter.Plan!;
            return new PtxTiledOuterProductProgram(
                text, plan.Blocks, emitter.LaunchBlockThreads,
                plan.Steps, plan.InnerReduction, plan.TileM, plan.TileN,
                emitter.SharedMemoryBytes,
                "left+right rows", outerProductRefusal: null);
        }
        catch (NotSupportedException outerProduct)
        {
            try
            {
                var emitter = new PtxTiledConv2DOuterProductEmitter();
                string text = emitter.Emit(spec, computeMajor, computeMinor);
                CodegenTiledConv2DOuterProductPlan plan = emitter.Plan!;
                return new PtxTiledOuterProductProgram(
                    text, plan.Blocks, emitter.LaunchBlockThreads,
                    plan.Steps, plan.InnerReduction, plan.TileM, plan.TileN,
                    emitter.SharedMemoryBytes,
                    "output+input rows", outerProduct.Message);
            }
            catch (NotSupportedException fallback)
            {
                throw new PtxTiledOuterProductDispatchException(
                    outerProduct.Message, fallback);
            }
        }
    }
}
