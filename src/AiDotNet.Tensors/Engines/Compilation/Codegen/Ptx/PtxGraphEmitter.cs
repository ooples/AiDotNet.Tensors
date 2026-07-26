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
/// A two-kernel program for a reduction that could not fill the device as one kernel.
/// </summary>
/// <remarks>
/// The single kernel in <see cref="CodegenEmitResult.Source"/> is always correct; this is
/// the faster route when it exists. A caller runs it by allocating
/// <paramref name="TempElements"/> floats, launching the partial pass with the product
/// operands followed by the temporary, then the combine pass with the temporary, any
/// epilogue operands, and the real output.
/// </remarks>
/// <param name="PartialSource">PTX for the partial pass.</param>
/// <param name="PartialName">Entry point name of the partial pass.</param>
/// <param name="PartialBlocks">Blocks to launch the partial pass with.</param>
/// <param name="PartialBlockX">Block width of the partial pass.</param>
/// <param name="PartialBlockY">Block height of the partial pass.</param>
/// <param name="CombineSource">PTX for the combine pass.</param>
/// <param name="CombineName">Entry point name of the combine pass.</param>
/// <param name="CombineBlocks">Blocks to launch the combine pass with.</param>
/// <param name="CombineBlockX">Block width of the combine pass.</param>
/// <param name="CombineBlockY">Block height of the combine pass.</param>
/// <param name="TempElements">Floats the caller must allocate between the passes.</param>
/// <param name="Plan">The IR-level plan the two kernels were emitted from.</param>
public sealed record PtxSplitProgram(
    string PartialSource, string PartialName,
    uint PartialBlocks, uint PartialBlockX, uint PartialBlockY,
    string CombineSource, string CombineName,
    uint CombineBlocks, uint CombineBlockX, uint CombineBlockY,
    long TempElements, CodegenSplitPlan Plan);

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

    /// <summary>
    /// A faster two-kernel route for the last emission, when the single kernel could not
    /// fill the device; null when it could, or when the split did not emit.
    /// </summary>
    /// <remarks>
    /// This is an optimisation, never a requirement. A reduction whose output is small and
    /// whose reduction is long runs at a few percent of one wave as a single kernel --
    /// measured at 1081x off roofline on a weight gradient -- and no tile can fix it,
    /// because tiling redistributes work among threads that exist. See
    /// <c>docs/SPLIT_K_REDUCTION.md</c>.
    /// </remarks>
    public PtxSplitProgram? LastSplitProgram { get; private set; }

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
        LastSplitProgram = TryEmitSplit(spec!);

        // No CodegenKernel is produced here: building one means owning a CUDA context,
        // module load and buffer binding, which belongs to the GPU execution path rather
        // than to an emitter. The PTX is returned as source, which is what a caller
        // needs to load it, and what the conveyor's release stage already consumes.
        return CodegenEmitResult.Decline(
            "PTX emitted; loading it needs a device context this emitter does not own", ptx);
    }

    /// <summary>
    /// Emits the two-kernel route when the spec would leave the device idle, or null.
    /// </summary>
    /// <remarks>
    /// Every failure here is silent and returns null, which is correct: the single kernel
    /// has already emitted and is right. A split that cannot be built costs performance,
    /// not correctness, and turning it into a decline would throw away a working kernel to
    /// report a missed optimisation.
    /// </remarks>
    private PtxSplitProgram? TryEmitSplit(CodegenKernelSpec spec)
    {
        CodegenSplitPlan? plan;
        try { plan = CodegenSplitReduction.TryPlan(spec); }
        catch (NotSupportedException) { return null; }
        if (plan is null) return null;

        try
        {
            var partialEmitter = new PtxAffineEmitter();
            string partialPtx = partialEmitter.Emit(plan.Partial, ComputeMajor, ComputeMinor);

            var combineEmitter = new PtxAffineEmitter();
            string combinePtx = combineEmitter.Emit(plan.Combine, ComputeMajor, ComputeMinor);

            return new PtxSplitProgram(
                partialPtx, plan.Partial.Name,
                partialEmitter.LaunchBlocks, (uint)partialEmitter.LaunchBlockX,
                (uint)partialEmitter.LaunchBlockY,
                combinePtx, plan.Combine.Name,
                combineEmitter.LaunchBlocks, (uint)combineEmitter.LaunchBlockX,
                (uint)combineEmitter.LaunchBlockY,
                plan.TempElements, plan);
        }
        catch (NotSupportedException) { return null; }
    }

    /// <summary>Stable name derived from the graph's content, so identical graphs agree.</summary>
    private static string KernelName(CodegenGraph graph)
    {
        long hash = graph.ComputeContentHash();
        return "ptx_graph_" + ((ulong)hash).ToString("x16", System.Globalization.CultureInfo.InvariantCulture);
    }
}
