// Copyright (c) AiDotNet. All rights reserved.
// The PTX emitter, wearing the interface every other emitter already wears.
//
// PtxAffineEmitter was referenced by nothing in the library except itself. It consumed
// CodegenKernelSpec, which only a hand-written catalog produced, so the kernels it
// generated were measured carefully and could never actually run: a compiler with no
// front end and no consumer.
//
// This closes that. It implements IKernelEmitter, so CodegenDispatcher can select it the
// same way it selects the CPU emitters, and it translates the CodegenGraph that the
// ordinary lowering path already builds. Graphs it cannot express are DECLINED with a
// reason rather than approximated, which is the contract the interface documents and the
// same discipline the index-map layer follows.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

/// <summary>
/// Emits direct PTX for a <see cref="CodegenGraph"/> via the index-map layer.
/// </summary>
public sealed class PtxGraphEmitter : IKernelEmitter
{
    /// <summary>Compute capability to emit for. Defaults to the sm_86 the catalog targets.</summary>
    public int ComputeMajor { get; set; } = 8;

    /// <summary>Compute capability minor version.</summary>
    public int ComputeMinor { get; set; } = 6;

    /// <inheritdoc/>
    public CodegenTarget Target => CodegenTarget.DirectPtx;

    /// <summary>
    /// Blocks the host must launch for the last successful emission, and the block
    /// shape to launch them with. Read together with <see cref="LastSpec"/>.
    /// </summary>
    public uint LastLaunchBlocks { get; private set; }

    /// <summary>Block width of the last successful emission.</summary>
    public uint LastLaunchBlockX { get; private set; }

    /// <summary>Block height of the last successful emission.</summary>
    public uint LastLaunchBlockY { get; private set; } = 1;

    /// <summary>The spec the last successful emission was built from.</summary>
    public CodegenKernelSpec? LastSpec { get; private set; }

    /// <inheritdoc/>
    public CodegenEmitResult Emit(CodegenGraph graph, CodegenElementType dtype)
    {
        if (graph is null) return CodegenEmitResult.Decline("graph was null");

        // The index-map layer and every released cubin are fp32. Declining rather than
        // silently emitting f32 for an f64 graph is the point of the decline contract.
        if (dtype != CodegenElementType.Float32)
            return CodegenEmitResult.Decline(
                "direct PTX emits fp32; graph asks for " + dtype);

        if (!CodegenGraphToSpec.TryTranslate(graph, KernelName(graph), out var spec, out string reason))
            return CodegenEmitResult.Decline("cannot express this graph as a kernel spec: " + reason);

        string ptx;
        var emitter = new PtxAffineEmitter();
        try
        {
            ptx = emitter.Emit(spec!, ComputeMajor, ComputeMinor);
        }
        catch (NotSupportedException ex)
        {
            // The emitter refuses specs it cannot lower correctly. That refusal is a
            // decline at this layer, not a crash.
            return CodegenEmitResult.Decline("emitter refused the spec: " + ex.Message);
        }

        LastSpec = spec;
        LastLaunchBlocks = emitter.LaunchBlocks;
        LastLaunchBlockX = (uint)emitter.LaunchBlockX;
        LastLaunchBlockY = (uint)emitter.LaunchBlockY;

        // No CodegenKernel is produced here: building one means owning a CUDA context,
        // module load and buffer binding, which belongs to the GPU execution path rather
        // than to an emitter. The PTX is returned as source, which is what a caller
        // needs to load it, and what the conveyor's release stage already consumes.
        return CodegenEmitResult.Decline(
            "PTX emitted; loading it needs a device context this emitter does not own", ptx);
    }

    /// <summary>Stable name derived from the graph's content, so identical graphs agree.</summary>
    private static string KernelName(CodegenGraph graph)
    {
        long hash = graph.ComputeContentHash();
        return "ptx_graph_" + ((ulong)hash).ToString("x16", System.Globalization.CultureInfo.InvariantCulture);
    }
}
