// Copyright (c) AiDotNet. All rights reserved.
// Shared contract for every codegen emitter — CPU (AVX-512 C#),
// Triton, HIP, MSL, WGSL, GLSL compute. Each emitter consumes the
// same CodegenGraph (see Codegen/Ir/) and produces a kernel
// specialised to its target.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen;

/// <summary>
/// Outcome of an emitter's run over a <see cref="CodegenGraph"/> —
/// either a compiled, callable <see cref="CodegenKernel"/> or a
/// structured failure explaining why the emitter can't handle this
/// graph (so a fallback emitter / the engine's composed-ops path
/// can pick up without retrying).
/// </summary>
public readonly struct CodegenEmitResult
{
    /// <summary>The compiled kernel — null when the emitter declined.</summary>
    public CodegenKernel? Kernel { get; }

    /// <summary>
    /// Human-readable source form of the emitted kernel (e.g. a C# method,
    /// a Triton function, a WGSL shader). Always present when
    /// <see cref="Kernel"/> is non-null; present best-effort when the
    /// emitter declines but wants to surface the partial emission for
    /// debugging.
    /// </summary>
    public string? Source { get; }

    /// <summary>
    /// True when this emitter declined to handle the graph (e.g.
    /// contains an <see cref="CodegenOpKind.Opaque"/> node, or a dtype
    /// the target language doesn't support). The caller should either
    /// try the next emitter in the preference chain or fall back to
    /// the composed-ops path.
    /// </summary>
    public bool Declined { get; }

    /// <summary>
    /// When <see cref="Declined"/> is true, a short human-readable
    /// reason the guard system (Phase D) logs so operators can see
    /// why a graph refused to compile.
    /// </summary>
    public string? DeclineReason { get; }

    private CodegenEmitResult(CodegenKernel? kernel, string? source, bool declined, string? reason)
    {
        Kernel = kernel;
        Source = source;
        Declined = declined;
        DeclineReason = reason;
    }

    /// <summary>Constructs a successful emit result.</summary>
    public static CodegenEmitResult Succeeded(CodegenKernel kernel, string? source = null)
        => new CodegenEmitResult(
            kernel ?? throw new ArgumentNullException(nameof(kernel)),
            source,
            declined: false,
            reason: null);

    /// <summary>Constructs a decline result — no kernel produced, reason logged.</summary>
    public static CodegenEmitResult Decline(string reason, string? partialSource = null)
    {
        if (reason is null) throw new ArgumentNullException(nameof(reason));
        return new CodegenEmitResult(kernel: null, source: partialSource, declined: true, reason: reason);
    }
}

/// <summary>
/// A compiled kernel returned by an emitter. The kernel is callable
/// directly (<see cref="Execute{T}"/>) or through the codegen fusion
/// pass's wrapping into a <see cref="CompiledStep{T}"/>. It is
/// deliberately dtype-erased — every emitter knows at construction
/// time which <see cref="CodegenElementType"/> it specialised for,
/// and <see cref="Execute{T}"/> throws if the caller's T doesn't
/// match.
/// </summary>
public abstract class CodegenKernel
{
    /// <summary>The element type this kernel was specialised for.</summary>
    public CodegenElementType Dtype { get; }

    /// <summary>The codegen graph the kernel was emitted from.</summary>
    public CodegenGraph Graph { get; }

    /// <summary>The target this kernel runs on.</summary>
    public CodegenTarget Target { get; }

    /// <summary>
    /// Binding order of the kernel's inputs — matches the order of
    /// <see cref="CodegenGraph.InputNodes"/> at emit time. Callers
    /// pass input tensors in this order.
    /// </summary>
    public int InputCount => Graph.InputNodes.Count;

    /// <summary>
    /// Binding order of the kernel's outputs. Matches
    /// <see cref="CodegenGraph.OutputNodes"/>.
    /// </summary>
    public int OutputCount => Graph.OutputNodes.Count;

    /// <summary>
    /// Initialises a kernel that knows its dtype + graph + target.
    /// </summary>
    protected CodegenKernel(CodegenElementType dtype, CodegenGraph graph, CodegenTarget target)
    {
        Dtype = dtype;
        Graph = graph ?? throw new ArgumentNullException(nameof(graph));
        Target = target;
    }

    /// <summary>
    /// Runs the kernel over caller-supplied arrays. Inputs and
    /// outputs are passed in binding order; every array must have
    /// length ≥ the element count computed from the corresponding
    /// node's shape.
    /// </summary>
    /// <typeparam name="T">Element type — must match
    /// <see cref="Dtype"/>. Passing the wrong T throws
    /// <see cref="ArgumentException"/> rather than silently reading
    /// mis-typed memory.</typeparam>
    /// <param name="inputs">Input arrays in binding order.</param>
    /// <param name="outputs">Output arrays in binding order.</param>
    /// <exception cref="ArgumentException">Thrown if
    /// <c>typeof(T)</c> does not match <see cref="Dtype"/>, or input
    /// / output counts mismatch.</exception>
    public abstract void Execute<T>(T[][] inputs, T[][] outputs) where T : unmanaged;
}

/// <summary>
/// Compilation target a kernel was emitted for.
/// </summary>
public enum CodegenTarget
{
    /// <summary>
    /// Expression-tree → JIT-compiled .NET delegate running on the
    /// CPU. The Phase B reference emitter. Produces real native
    /// code (the JIT applies vectorization where applicable); it's
    /// the emitter that proves the IR contract without requiring
    /// the Roslyn/external-toolchain dependency of Phase C emitters.
    /// </summary>
    CpuDotNetJit,
    /// <summary>
    /// Hand-emitted AVX-512 intrinsic kernel — uses
    /// <see cref="System.Runtime.Intrinsics.X86.Avx512F"/> directly via
    /// Expression trees rather than relying on RyuJIT auto-vectorization.
    /// Produces a 16-wide loop on Float32 (8-wide on Float64) with a
    /// scalar tail. Selected over <see cref="CpuDotNetJit"/> by the
    /// fusion pass when AVX-512F is available and the graph is
    /// pointwise — gives a deterministic vectorisation guarantee that
    /// the generic JIT path doesn't.
    /// </summary>
    CpuAvx512,
    /// <summary>OpenAI Triton (CUDA).</summary>
    Triton,
    /// <summary>HIP kernel source (AMD ROCm).</summary>
    Hip,
    /// <summary>Metal Shading Language (Apple).</summary>
    Msl,
    /// <summary>WebGPU Shading Language.</summary>
    Wgsl,
    /// <summary>GLSL compute (Vulkan).</summary>
    Glsl,
}

/// <summary>
/// The emitter contract. A single emitter handles one target
/// language; the fusion pass may hold several emitters and pick per
/// graph based on the running engine's backend.
/// </summary>
public interface IKernelEmitter
{
    /// <summary>Which target this emitter produces.</summary>
    CodegenTarget Target { get; }

    /// <summary>
    /// Emits a kernel for <paramref name="graph"/> specialised for
    /// <paramref name="dtype"/>. Returns a success or decline result.
    /// </summary>
    CodegenEmitResult Emit(CodegenGraph graph, CodegenElementType dtype);
}
