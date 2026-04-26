// Copyright (c) AiDotNet. All rights reserved.
// Shared helpers for every GPU source-string emitter (Triton, HIP,
// MSL, WGSL, GLSL). Each emitter owns its kernel shell + local-
// variable declaration syntax (Triton has no type annotations, WGSL
// spells "let v : f32 = ...", GLSL spells "float v = ..."). The
// pieces that are identical across dialects — op-expression
// formatting, support checks, kernel-naming conventions — live here.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen;

/// <summary>
/// Per-target source-language dialect — the minimum set of knobs
/// each GPU emitter has to supply so the shared helpers in
/// <see cref="GpuEmitterCommon"/> can format expressions correctly.
/// </summary>
public readonly struct GpuTargetDialect
{
    /// <summary>Function name for exp.</summary>
    public string Exp { get; }
    /// <summary>Function name for log.</summary>
    public string Log { get; }
    /// <summary>Function name for sqrt.</summary>
    public string Sqrt { get; }
    /// <summary>Function name for sin.</summary>
    public string Sin { get; }
    /// <summary>Function name for cos.</summary>
    public string Cos { get; }
    /// <summary>Function name for tan.</summary>
    public string Tan { get; }
    /// <summary>Function name for tanh.</summary>
    public string Tanh { get; }
    /// <summary>Function name for abs.</summary>
    public string Abs { get; }
    /// <summary>Function name for floor.</summary>
    public string Floor { get; }
    /// <summary>Function name for ceil.</summary>
    public string Ceil { get; }
    /// <summary>Function name for round.</summary>
    public string Round { get; }
    /// <summary>Two-argument max — usually <c>max</c> or <c>fmax</c>.</summary>
    public string Max { get; }
    /// <summary>Two-argument min.</summary>
    public string Min { get; }
    /// <summary>Literal for floating-point zero (e.g. <c>0.0</c>, <c>0.0f</c>).</summary>
    public string FloatZeroLiteral { get; }
    /// <summary>Literal for floating-point one.</summary>
    public string FloatOneLiteral { get; }

    /// <summary>Constructs a dialect.</summary>
    public GpuTargetDialect(
        string exp, string log, string sqrt,
        string sin, string cos, string tan, string tanh,
        string abs, string floor, string ceil, string round,
        string max, string min,
        string floatZeroLiteral, string floatOneLiteral)
    {
        Exp = exp; Log = log; Sqrt = sqrt;
        Sin = sin; Cos = cos; Tan = tan; Tanh = tanh;
        Abs = abs; Floor = floor; Ceil = ceil; Round = round;
        Max = max; Min = min;
        FloatZeroLiteral = floatZeroLiteral;
        FloatOneLiteral = floatOneLiteral;
    }
}

/// <summary>
/// Shared helpers for the GPU source emitters.
/// </summary>
public static class GpuEmitterCommon
{
    /// <summary>
    /// Gate-check: can the supplied dialect represent this
    /// <paramref name="graph"/> and <paramref name="dtype"/>?
    /// Returns a decline reason string on failure, null on success.
    /// </summary>
    public static string? CheckSupport(CodegenGraph graph, CodegenElementType dtype, HashSet<CodegenElementType> supportedDtypes)
    {
        if (!supportedDtypes.Contains(dtype))
            return $"Dtype {dtype} not supported by this emitter.";

        foreach (var node in graph.Nodes)
        {
            var cat = CodegenOpKinds.Categorize(node.Op);
            if (cat != CodegenOpCategory.Pointwise && cat != CodegenOpCategory.Movement)
                return $"GPU pointwise emitter does not yet handle {cat} ops (found {node.Op}).";
        }

        // Phase C emits per-thread element-wise kernels — every node
        // must have the same element count so a single global thread
        // index suffices.
        if (graph.Nodes.Count == 0) return "Graph is empty.";
        long refCount = 1;
        for (int i = 0; i < graph.Nodes[0].Shape.Length; i++) refCount *= graph.Nodes[0].Shape[i];
        foreach (var node in graph.Nodes)
        {
            long c = 1;
            for (int i = 0; i < node.Shape.Length; i++) c *= node.Shape[i];
            if (c != refCount)
                return $"Phase C pointwise emitter requires uniform element count across nodes; "
                     + $"found {refCount} vs {c} at op {node.Op}.";
        }
        return null;
    }

    /// <summary>
    /// Formats a single pointwise op node as an infix/prefix/function-call
    /// expression in the supplied dialect. Callers place the result on
    /// the right-hand side of a local variable declaration.
    /// </summary>
    public static string FormatOpExpression(CodegenNode node, GpuTargetDialect d) => node.Op switch
    {
        CodegenOpKind.Add => $"v{node.Inputs[0]} + v{node.Inputs[1]}",
        CodegenOpKind.Sub => $"v{node.Inputs[0]} - v{node.Inputs[1]}",
        CodegenOpKind.Mul => $"v{node.Inputs[0]} * v{node.Inputs[1]}",
        CodegenOpKind.Div => $"v{node.Inputs[0]} / v{node.Inputs[1]}",
        CodegenOpKind.Negate => $"-v{node.Inputs[0]}",
        CodegenOpKind.Exp => $"{d.Exp}(v{node.Inputs[0]})",
        CodegenOpKind.Log => $"{d.Log}(v{node.Inputs[0]})",
        CodegenOpKind.Sqrt => $"{d.Sqrt}(v{node.Inputs[0]})",
        CodegenOpKind.Sin => $"{d.Sin}(v{node.Inputs[0]})",
        CodegenOpKind.Cos => $"{d.Cos}(v{node.Inputs[0]})",
        CodegenOpKind.Tan => $"{d.Tan}(v{node.Inputs[0]})",
        CodegenOpKind.Tanh => $"{d.Tanh}(v{node.Inputs[0]})",
        CodegenOpKind.Abs => $"{d.Abs}(v{node.Inputs[0]})",
        CodegenOpKind.Floor => $"{d.Floor}(v{node.Inputs[0]})",
        CodegenOpKind.Ceil => $"{d.Ceil}(v{node.Inputs[0]})",
        CodegenOpKind.Round => $"{d.Round}(v{node.Inputs[0]})",
        CodegenOpKind.ReLU => $"{d.Max}(v{node.Inputs[0]}, {d.FloatZeroLiteral})",
        CodegenOpKind.Sigmoid => $"{d.FloatOneLiteral} / ({d.FloatOneLiteral} + {d.Exp}(-v{node.Inputs[0]}))",
        CodegenOpKind.Max => $"{d.Max}(v{node.Inputs[0]}, v{node.Inputs[1]})",
        CodegenOpKind.Min => $"{d.Min}(v{node.Inputs[0]}, v{node.Inputs[1]})",
        _ => throw new ArgumentException($"FormatOpExpression: unsupported op {node.Op}"),
    };

    /// <summary>
    /// Element count implied by the graph (uniform across nodes by
    /// Phase C support check). Used by emitters to parameterise the
    /// launch geometry in the kernel shell.
    /// </summary>
    public static int GetElementCount(CodegenGraph graph)
    {
        long c = 1;
        for (int i = 0; i < graph.Nodes[0].Shape.Length; i++) c *= graph.Nodes[0].Shape[i];
        if (c > int.MaxValue)
            throw new InvalidOperationException($"Element count {c} exceeds int.MaxValue.");
        return (int)c;
    }
}

/// <summary>
/// Kernel returned by Phase C source-emitting emitters — carries
/// the emitted source plus the dialect's kernel entry point name.
/// Phase C doesn't yet wire the source into backend dispatch (that
/// integration lives with each GPU backend's native compilation
/// surface); <see cref="Execute{T}"/> throws
/// <see cref="NotSupportedException"/> until the backend-dispatch
/// wiring lands.
/// </summary>
public sealed class GpuSourceKernel : CodegenKernel
{
    /// <summary>The generated source as a string.</summary>
    public string Source { get; }

    /// <summary>Entry-point function name within <see cref="Source"/>.</summary>
    public string EntryPoint { get; }

    internal GpuSourceKernel(
        CodegenGraph graph,
        CodegenElementType dtype,
        CodegenTarget target,
        string source,
        string entryPoint)
        : base(dtype, graph, target)
    {
        Source = source;
        EntryPoint = entryPoint;
    }

    /// <inheritdoc/>
    public override void Execute<T>(T[][] inputs, T[][] outputs)
        => throw new NotSupportedException(
            $"{Target} kernel source is emitted but backend-dispatch wiring is a follow-up PR. "
          + "Use Source / EntryPoint to feed into the target's compiler externally, "
          + "or switch to CodegenTarget.CpuDotNetJit for local execution.");
}
