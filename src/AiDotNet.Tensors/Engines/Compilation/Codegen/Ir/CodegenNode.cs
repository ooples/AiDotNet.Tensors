// Copyright (c) AiDotNet. All rights reserved.
// Codegen IR node — a single operation in the codegen graph.

using System;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// One node in the codegen IR graph. Operations reference producers
/// by integer index into the enclosing <see cref="CodegenGraph"/>,
/// not by object reference, so the graph is trivially serializable
/// (for autotune-cache persistence and cross-process build servers)
/// and trivially pattern-matchable (pattern matchers see a flat
/// indexable list, not a pointer graph).
/// </summary>
/// <remarks>
/// <para><b>Why a struct:</b></para>
/// <para>
/// The fusion matcher visits every node multiple times — once to
/// categorise, once per candidate pattern, once to emit. A class
/// would allocate on every IR transformation; a struct stays on the
/// stack / in a flat List&lt;T&gt; backing array. Mutations go through
/// the graph's indexed setter so the "immutable after lowering"
/// contract is preserved.
/// </para>
/// <para><b>Why integer indices into Inputs instead of node refs:</b></para>
/// <para>
/// Same rationale as the struct choice: indices are stable across
/// graph copies (pattern-matcher works on a cloned sub-graph without
/// updating any pointers) and let the emitter allocate a single
/// variable-name-per-index array, not a Dictionary.
/// </para>
/// </remarks>
public readonly struct CodegenNode
{
    /// <summary>The op kind — picks the emitter dispatch branch.</summary>
    public CodegenOpKind Op { get; }

    /// <summary>
    /// Indices into the enclosing graph's node list identifying this
    /// op's producers. Order matches the op's operand convention
    /// (e.g. MatMul: [A, B]; Sub: [lhs, rhs]). Length 0 for
    /// <see cref="CodegenOpKind.LoadInput"/> and
    /// <see cref="CodegenOpKind.Constant"/>.
    /// </summary>
    public int[] Inputs { get; }

    /// <summary>
    /// Output element type. Emitters use this to pick the scalar
    /// type in their target language.
    /// </summary>
    public CodegenElementType Dtype { get; }

    /// <summary>
    /// Output shape (concrete for Phase A; Phase D will extend this
    /// to support symbolic dimensions that participate in the guard
    /// system).
    /// </summary>
    public int[] Shape { get; }

    /// <summary>
    /// Op-specific metadata. For <see cref="CodegenOpKind.LoadInput"/>
    /// this is the graph input index; for
    /// <see cref="CodegenOpKind.StoreOutput"/> the graph output
    /// index; for <see cref="CodegenOpKind.Constant"/> the boxed
    /// scalar value; for
    /// <see cref="CodegenOpKind.ReduceSum"/> / <see cref="CodegenOpKind.ReduceMean"/>
    /// etc. the axes array; for
    /// <see cref="CodegenOpKind.Softmax"/> the softmax axis;
    /// for <see cref="CodegenOpKind.Opaque"/> the delegate or saved
    /// state the original <c>CompiledStep</c> carried.
    /// Null when the op has no metadata.
    /// </summary>
    public object? Attribute { get; }

    /// <summary>
    /// Creates a node. Callers should normally construct nodes
    /// through <see cref="CodegenGraph.AddNode"/> rather than this
    /// ctor directly — the graph assigns the node's index and runs
    /// invariant checks the struct ctor can't.
    /// </summary>
    /// <param name="op">The op kind.</param>
    /// <param name="inputs">Producer indices (can be empty for leaves).</param>
    /// <param name="dtype">Output element type.</param>
    /// <param name="shape">Output shape.</param>
    /// <param name="attribute">Op-specific metadata (see remarks).</param>
    /// <exception cref="ArgumentNullException">Thrown if
    /// <paramref name="inputs"/> or <paramref name="shape"/> is null.</exception>
    public CodegenNode(
        CodegenOpKind op,
        int[] inputs,
        CodegenElementType dtype,
        int[] shape,
        object? attribute = null)
    {
        Op = op;
        Inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
        Dtype = dtype;
        Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        Attribute = attribute;
    }

    /// <summary>
    /// Returns the total number of elements the node produces (the
    /// product of <see cref="Shape"/>). Used by the fusion matcher
    /// to estimate register pressure and by the emitter to pick
    /// tile sizes.
    /// </summary>
    public long ElementCount
    {
        get
        {
            long total = 1;
            for (int i = 0; i < Shape.Length; i++) total *= Shape[i];
            return total;
        }
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        var shapeStr = Shape.Length == 0 ? "[]" : "[" + string.Join(",", Shape) + "]";
        var inputStr = Inputs.Length == 0 ? "" : " ← " + string.Join(",", Inputs);
        return $"{Op} : {Dtype}{shapeStr}{inputStr}";
    }
}
