// Copyright (c) AiDotNet. All rights reserved.
// CompiledStep → CodegenGraph lowering. Walks a contiguous run of
// fusible pointwise steps and emits an equivalent CodegenGraph that
// the emitter layer turns into a single kernel.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// Translates a sequence of <see cref="CompiledStep{T}"/> objects
/// produced by <see cref="CompiledInferencePlan{T}"/> / the
/// optimization passes into a <see cref="CodegenGraph"/> suitable
/// for emitter consumption.
/// </summary>
/// <remarks>
/// <para><b>Scope of this Phase A lowering:</b></para>
/// <para>
/// This pass recognises pointwise op chains (contiguous <c>Add</c>,
/// <c>Mul</c>, activation, etc. sequences producing a single output
/// tensor) and lowers them into a single-output CodegenGraph. It
/// does NOT lower matmul, reductions, or attention — those land
/// in later phases as dedicated lowering helpers keyed on
/// <see cref="CodegenOpCategory"/>. Phase A deliberately ships
/// just the pointwise subset so we can prove the IR shape is right
/// before investing in the tricky fused-matmul epilogue logic.
/// </para>
/// <para><b>Design — a stateless builder:</b></para>
/// <para>
/// Each lowering call allocates a fresh builder, so the class is
/// safe to call from multiple fusion passes concurrently. The
/// <see cref="CompiledStep{T}"/> objects being lowered are not
/// mutated — the builder only reads their <c>OpName</c>, <c>Inputs</c>,
/// and <c>OutputBuffer</c>.
/// </para>
/// </remarks>
public static class CodegenLowering
{
    /// <summary>
    /// Maps an internal <see cref="LazyNodeType"/> to its codegen op
    /// kind. Returns <see cref="CodegenOpKind.Opaque"/> for anything
    /// the Phase A pointwise-only lowering can't represent — the
    /// fusion pass treats those as boundaries. Internal because
    /// <see cref="LazyNodeType"/> is not part of the public surface.
    /// </summary>
    internal static CodegenOpKind MapLazyNodeType(LazyNodeType t) => t switch
    {
        LazyNodeType.Add => CodegenOpKind.Add,
        LazyNodeType.Subtract => CodegenOpKind.Sub,
        LazyNodeType.Multiply => CodegenOpKind.Mul,
        LazyNodeType.Divide => CodegenOpKind.Div,
        LazyNodeType.Negate => CodegenOpKind.Negate,

        LazyNodeType.ReLU => CodegenOpKind.ReLU,
        LazyNodeType.Sigmoid => CodegenOpKind.Sigmoid,
        LazyNodeType.GELU => CodegenOpKind.GELU,
        LazyNodeType.Tanh => CodegenOpKind.Tanh,
        LazyNodeType.Swish => CodegenOpKind.Swish,
        LazyNodeType.LeakyReLU => CodegenOpKind.LeakyReLU,
        LazyNodeType.ELU => CodegenOpKind.ELU,

        LazyNodeType.MatMul => CodegenOpKind.MatMul,
        LazyNodeType.BatchMatMul => CodegenOpKind.BatchMatMul,
        LazyNodeType.Softmax => CodegenOpKind.Softmax,
        LazyNodeType.ReduceSum => CodegenOpKind.ReduceSum,
        LazyNodeType.ReduceMean => CodegenOpKind.ReduceMean,
        LazyNodeType.Sum => CodegenOpKind.ReduceSum,
        LazyNodeType.Mean => CodegenOpKind.ReduceMean,

        LazyNodeType.Transpose => CodegenOpKind.Transpose,
        LazyNodeType.Reshape => CodegenOpKind.Reshape,
        LazyNodeType.Concat => CodegenOpKind.Concat,

        _ => CodegenOpKind.Opaque,
    };

    /// <summary>
    /// Maps a lazy-op name string to its codegen op kind.
    /// Used by external callers that have only the op name (as in
    /// <see cref="CompiledStep{T}"/>.OpName) and no enum handle.
    /// </summary>
    public static CodegenOpKind MapOpName(string opName) => opName switch
    {
        "TensorAdd" or "Add" => CodegenOpKind.Add,
        "TensorSubtract" or "Subtract" => CodegenOpKind.Sub,
        "TensorMultiply" or "Multiply" => CodegenOpKind.Mul,
        "TensorDivide" or "Divide" => CodegenOpKind.Div,
        "TensorNegate" or "Negate" => CodegenOpKind.Negate,
        "ReLU" => CodegenOpKind.ReLU,
        "Sigmoid" => CodegenOpKind.Sigmoid,
        "GELU" => CodegenOpKind.GELU,
        "Tanh" => CodegenOpKind.Tanh,
        "Swish" => CodegenOpKind.Swish,
        "LeakyReLU" => CodegenOpKind.LeakyReLU,
        "ELU" => CodegenOpKind.ELU,
        "TensorMatMul" or "MatMul" => CodegenOpKind.MatMul,
        "TensorTranspose" or "Transpose" => CodegenOpKind.Transpose,
        "Reshape" => CodegenOpKind.Reshape,
        _ => CodegenOpKind.Opaque,
    };

    /// <summary>
    /// Resolves the <see cref="CodegenElementType"/> for a .NET
    /// generic parameter. Emitters pick their scalar type from this.
    /// </summary>
    /// <exception cref="NotSupportedException">Thrown when T is not
    /// in the codegen-supported set.</exception>
    public static CodegenElementType ResolveElementType<T>()
    {
        if (typeof(T) == typeof(float)) return CodegenElementType.Float32;
        if (typeof(T) == typeof(double)) return CodegenElementType.Float64;
        if (typeof(T) == typeof(int)) return CodegenElementType.Int32;
        if (typeof(T) == typeof(long)) return CodegenElementType.Int64;
        if (typeof(T) == typeof(sbyte)) return CodegenElementType.Int8;
        if (typeof(T) == typeof(byte)) return CodegenElementType.UInt8;
        if (typeof(T) == typeof(bool)) return CodegenElementType.Bool;
        throw new NotSupportedException(
            $"Codegen does not yet support element type {typeof(T).Name}. "
          + "Extend CodegenElementType and update ResolveElementType if needed.");
    }

    /// <summary>
    /// Lowers a single pointwise unary op into a two-node graph
    /// (LoadInput → op → StoreOutput). This is the smallest useful
    /// unit a Phase A test can exercise end-to-end; the fusion pass
    /// composes larger graphs by chaining these primitives.
    /// </summary>
    /// <typeparam name="T">The element type of the tensors.</typeparam>
    /// <param name="opKind">The pointwise op to apply.</param>
    /// <param name="inputShape">Shape of the input (and output — the
    /// op is elementwise).</param>
    /// <returns>A three-node graph: <c>[LoadInput, op, StoreOutput]</c>.</returns>
    public static CodegenGraph LowerUnaryPointwise<T>(CodegenOpKind opKind, int[] inputShape)
    {
        if (!CodegenOpKinds.IsUnaryPointwise(opKind))
            throw new ArgumentException(
                $"Op {opKind} is not a unary pointwise primitive — cannot lower with LowerUnaryPointwise.",
                nameof(opKind));
        if (inputShape is null) throw new ArgumentNullException(nameof(inputShape));

        var dtype = ResolveElementType<T>();
        var graph = new CodegenGraph();
        int load = graph.AddNode(new CodegenNode(
            CodegenOpKind.LoadInput,
            inputs: Array.Empty<int>(),
            dtype: dtype,
            shape: (int[])inputShape.Clone(),
            attribute: 0));
        int op = graph.AddNode(new CodegenNode(
            opKind,
            inputs: new[] { load },
            dtype: dtype,
            shape: (int[])inputShape.Clone()));
        graph.AddNode(new CodegenNode(
            CodegenOpKind.StoreOutput,
            inputs: new[] { op },
            dtype: dtype,
            shape: (int[])inputShape.Clone(),
            attribute: 0));
        return graph;
    }

    /// <summary>
    /// Lowers a binary elementwise op (<see cref="CodegenOpKind.Add"/>,
    /// <see cref="CodegenOpKind.Sub"/>, <see cref="CodegenOpKind.Mul"/>,
    /// <see cref="CodegenOpKind.Div"/>, etc.) into a four-node graph.
    /// </summary>
    public static CodegenGraph LowerBinaryPointwise<T>(CodegenOpKind opKind, int[] shape)
    {
        var category = CodegenOpKinds.Categorize(opKind);
        if (category != CodegenOpCategory.Pointwise)
            throw new ArgumentException(
                $"Op {opKind} is not pointwise — cannot lower with LowerBinaryPointwise.",
                nameof(opKind));
        if (opKind == CodegenOpKind.LoadInput
         || opKind == CodegenOpKind.StoreOutput
         || opKind == CodegenOpKind.Constant
         || CodegenOpKinds.IsUnaryPointwise(opKind))
            throw new ArgumentException(
                $"Op {opKind} is not a binary pointwise op — use the appropriate lowering helper.",
                nameof(opKind));
        if (shape is null) throw new ArgumentNullException(nameof(shape));

        var dtype = ResolveElementType<T>();
        var graph = new CodegenGraph();
        int a = graph.AddNode(new CodegenNode(
            CodegenOpKind.LoadInput, Array.Empty<int>(), dtype, (int[])shape.Clone(), 0));
        int b = graph.AddNode(new CodegenNode(
            CodegenOpKind.LoadInput, Array.Empty<int>(), dtype, (int[])shape.Clone(), 1));
        int op = graph.AddNode(new CodegenNode(
            opKind, new[] { a, b }, dtype, (int[])shape.Clone()));
        graph.AddNode(new CodegenNode(
            CodegenOpKind.StoreOutput, new[] { op }, dtype, (int[])shape.Clone(), 0));
        return graph;
    }

    /// <summary>
    /// Lowers a chain of pointwise unary ops
    /// (e.g. <c>Sigmoid(Exp(Negate(x)))</c>) into a single graph.
    /// The chain walks as a composition, with every intermediate
    /// flowing into the next producer — this is the fusion pattern
    /// the Phase B emitter proves works end-to-end.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="ops">Op chain in application order (<c>ops[0]</c>
    /// is applied first to the input). Must all be unary
    /// pointwise.</param>
    /// <param name="inputShape">Shape of the single input (and
    /// every intermediate / the output).</param>
    /// <returns>A graph with <c>2 + ops.Length</c> nodes:
    /// LoadInput → op₀ → op₁ → … → StoreOutput.</returns>
    public static CodegenGraph LowerUnaryChain<T>(IReadOnlyList<CodegenOpKind> ops, int[] inputShape)
    {
        if (ops is null) throw new ArgumentNullException(nameof(ops));
        if (inputShape is null) throw new ArgumentNullException(nameof(inputShape));

        for (int i = 0; i < ops.Count; i++)
            if (!CodegenOpKinds.IsUnaryPointwise(ops[i]))
                throw new ArgumentException(
                    $"ops[{i}] = {ops[i]} is not a unary pointwise op.", nameof(ops));

        var dtype = ResolveElementType<T>();
        var graph = new CodegenGraph();
        int prev = graph.AddNode(new CodegenNode(
            CodegenOpKind.LoadInput, Array.Empty<int>(), dtype, (int[])inputShape.Clone(), 0));
        for (int i = 0; i < ops.Count; i++)
        {
            prev = graph.AddNode(new CodegenNode(
                ops[i], new[] { prev }, dtype, (int[])inputShape.Clone()));
        }
        graph.AddNode(new CodegenNode(
            CodegenOpKind.StoreOutput, new[] { prev }, dtype, (int[])inputShape.Clone(), 0));
        return graph;
    }
}

