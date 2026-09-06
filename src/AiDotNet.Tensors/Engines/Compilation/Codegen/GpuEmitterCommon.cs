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
        if (graph is null) return "Graph is null.";
        if (supportedDtypes is null) return "Supported dtype set is null.";
        if (!supportedDtypes.Contains(dtype))
            return $"Dtype {dtype} not supported by this emitter.";
        if (graph.Nodes.Count == 0) return "Graph is empty.";
        if (graph.InputNodes.Count == 0) return "GPU pointwise graph has no input.";
        if (graph.OutputNodes.Count == 0) return "GPU pointwise graph has no output.";

        foreach (var node in graph.Nodes)
        {
            if (!CanEmitPointwiseNode(node.Op))
                return $"GPU pointwise emitter does not yet handle {node.Op}.";
            int expectedArity = GetPointwiseArity(node.Op);
            if (node.Inputs.Length != expectedArity)
            {
                return $"GPU pointwise op {node.Op} requires {expectedArity} input(s), " +
                       $"but the graph supplies {node.Inputs.Length}.";
            }
        }

        // Phase C emits per-thread element-wise kernels — every node
        // must have the same element count so a single global thread
        // index suffices.
        if (!TryGetElementCount(graph.Nodes[0].Shape, out long refCount, out string? reason))
            return reason;
        foreach (var node in graph.Nodes)
        {
            if (!TryGetElementCount(node.Shape, out long c, out reason))
                return reason;
            if (c != refCount)
                return $"Phase C pointwise emitter requires uniform element count across nodes; "
                     + $"found {refCount} vs {c} at op {node.Op}.";
        }
        return null;
    }

    private static bool CanEmitPointwiseNode(CodegenOpKind op) => op is
        CodegenOpKind.LoadInput or
        CodegenOpKind.StoreOutput or
        CodegenOpKind.Add or
        CodegenOpKind.Sub or
        CodegenOpKind.Mul or
        CodegenOpKind.Div or
        CodegenOpKind.Max or
        CodegenOpKind.Min or
        CodegenOpKind.Negate or
        CodegenOpKind.Sqrt or
        CodegenOpKind.Exp or
        CodegenOpKind.Log or
        CodegenOpKind.Sin or
        CodegenOpKind.Cos or
        CodegenOpKind.Tan or
        CodegenOpKind.Tanh or
        CodegenOpKind.Abs or
        CodegenOpKind.Floor or
        CodegenOpKind.Ceil or
        CodegenOpKind.Round or
        CodegenOpKind.ReLU or
        CodegenOpKind.Sigmoid;

    private static int GetPointwiseArity(CodegenOpKind op) => op switch
    {
        CodegenOpKind.LoadInput => 0,
        CodegenOpKind.StoreOutput => 1,
        CodegenOpKind.Add or CodegenOpKind.Sub or CodegenOpKind.Mul or CodegenOpKind.Div or
        CodegenOpKind.Max or CodegenOpKind.Min => 2,
        _ => 1
    };

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
        if (graph is null) throw new ArgumentNullException(nameof(graph));
        if (graph.Nodes.Count == 0) throw new InvalidOperationException("Graph is empty.");
        if (!TryGetElementCount(graph.Nodes[0].Shape, out long c, out string? reason))
            throw new InvalidOperationException(reason);
        if (c > int.MaxValue) throw new InvalidOperationException($"Element count {c} exceeds int.MaxValue.");
        return (int)c;
    }

    /// <summary>
    /// Enforces that a source-kernel launch covers the exact uniform extent emitted from the graph.
    /// </summary>
    internal static void ValidateLaunchElementCount(GpuSourceKernel kernel, int elementCount)
    {
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (elementCount <= 0) throw new ArgumentOutOfRangeException(nameof(elementCount));
        int emittedElementCount = GetElementCount(kernel.Graph);
        if (elementCount != emittedElementCount)
        {
            throw new ArgumentException(
                $"Launch element count {elementCount} does not match emitted extent {emittedElementCount}.",
                nameof(elementCount));
        }
    }

    private static bool TryGetElementCount(
        IReadOnlyList<int> shape,
        out long elementCount,
        out string? reason)
    {
        elementCount = 1;
        for (int i = 0; i < shape.Count; i++)
        {
            int dimension = shape[i];
            if (dimension <= 0)
            {
                reason = $"GPU pointwise emission requires positive dimensions; found {dimension}.";
                return false;
            }

            try
            {
                elementCount = checked(elementCount * dimension);
            }
            catch (OverflowException)
            {
                reason = "GPU pointwise element count exceeds Int64 capacity.";
                return false;
            }
        }

        reason = null;
        return true;
    }
}

/// <summary>
/// Kernel returned by Phase C source-emitting emitters — carries
/// the emitted source plus the dialect's kernel entry point name.
/// Execution is intentionally owned by each backend because compilation, native artifact
/// caching, queues, and buffers are backend-specific. Every supported GPU backend exposes that
/// lifecycle through <see cref="INativeGpuCodegenExecutor"/>. Direct array execution remains
/// unsupported.
/// </summary>
public sealed class GpuSourceKernel : CodegenKernel
{
    /// <summary>The generated source as a string.</summary>
    public string Source { get; }

    /// <summary>Entry-point function name within <see cref="Source"/>.</summary>
    public string EntryPoint { get; }

    /// <summary>
    /// Workgroup width embedded in the source language, or null when the backend selects it at launch.
    /// </summary>
    public int? DeclaredWorkgroupSize { get; }

    internal GpuSourceKernel(
        CodegenGraph graph,
        CodegenElementType dtype,
        CodegenTarget target,
        string source,
        string entryPoint,
        int? declaredWorkgroupSize = null)
        : base(dtype, graph, target)
    {
        if (declaredWorkgroupSize.HasValue && declaredWorkgroupSize.Value <= 0)
            throw new ArgumentOutOfRangeException(nameof(declaredWorkgroupSize));
        Source = source;
        EntryPoint = entryPoint;
        DeclaredWorkgroupSize = declaredWorkgroupSize;
    }

    /// <inheritdoc/>
    public override void Execute<T>(T[][] inputs, T[][] outputs)
        => throw new NotSupportedException(
            $"{Target} source kernels require their native GPU backend and device buffers. "
          + "Use the matching backend execution API, or switch to CodegenTarget.CpuDotNetJit "
          + "for direct managed-array execution.");
}
