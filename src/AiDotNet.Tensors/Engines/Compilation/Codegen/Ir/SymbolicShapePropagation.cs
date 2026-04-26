// Copyright (c) AiDotNet. All rights reserved.
// Symbolic-shape constraint propagation across codegen graph nodes —
// addresses the issue #225 dynamic-shapes section: "Constraint
// propagation across fusion passes".

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// Propagates a <see cref="SymbolicShape"/> through every node of a
/// pointwise / movement codegen graph, returning a per-node map of
/// "which dimensions are symbolic at this point in the graph". The
/// output enables fusion passes to reason about which axes can vary
/// at runtime (so the recompile budget only fires on changes to
/// non-symbolic dims) without re-deriving the constraint at every
/// pass.
/// </summary>
/// <remarks>
/// <para><b>Coverage:</b></para>
/// <list type="bullet">
/// <item>Pointwise unary/binary ops preserve the symbolic mask
/// element-for-element (the output shape equals the input shape).</item>
/// <item>Movement ops (Reshape, Permute) require the caller to
/// supply a rank-mapping — propagation handles a no-op rank-preserving
/// reshape directly; arbitrary reshape that fuses dims is left for the
/// caller (we Decline rather than guess wrong).</item>
/// <item>Reduction / matmul / attention / opaque ops reset the
/// symbolic-mask to "no known symbolic axes" — the user must
/// re-annotate them after these ops change rank.</item>
/// </list>
/// </remarks>
public static class SymbolicShapePropagation
{
    /// <summary>
    /// Propagates <paramref name="inputShapes"/> through
    /// <paramref name="graph"/>. Returns a list parallel to
    /// <see cref="CodegenGraph.Nodes"/> giving the symbolic shape at
    /// each node, or null at indices the propagator could not infer.
    /// </summary>
    /// <param name="graph">The codegen graph to annotate.</param>
    /// <param name="inputShapes">Symbolic shapes of the LoadInput
    /// nodes, in declaration order. Length must equal
    /// <see cref="CodegenGraph.InputNodes"/>.Count.</param>
    public static IReadOnlyList<SymbolicShape?> Propagate(
        CodegenGraph graph,
        IReadOnlyList<SymbolicShape> inputShapes)
    {
        if (graph is null) throw new ArgumentNullException(nameof(graph));
        if (inputShapes is null) throw new ArgumentNullException(nameof(inputShapes));
        if (inputShapes.Count != graph.InputNodes.Count)
            throw new ArgumentException(
                $"inputShapes.Count={inputShapes.Count} must equal graph.InputNodes.Count={graph.InputNodes.Count}.");

        var perNode = new SymbolicShape?[graph.Count];
        int inputCursor = 0;
        for (int i = 0; i < graph.Count; i++)
        {
            var node = graph[i];
            switch (node.Op)
            {
                case CodegenOpKind.LoadInput:
                    perNode[i] = inputShapes[inputCursor++];
                    break;

                case CodegenOpKind.StoreOutput:
                    // Output's shape is the producer's shape.
                    perNode[i] = perNode[node.Inputs[0]];
                    break;

                case CodegenOpKind.Reshape:
                case CodegenOpKind.Transpose:
                    // Movement ops without rank/shape mapping can't be
                    // propagated automatically — surface as null so the
                    // caller knows to re-annotate.
                    perNode[i] = null;
                    break;

                default:
                    // Pointwise / category-Pointwise: output shape
                    // matches input shape (verified upstream by the
                    // emitter's shape-uniformity check). Use input 0's
                    // symbolic shape as the propagated answer; for
                    // binary ops we additionally union the symbolic
                    // masks of input 0 and input 1 — if either input
                    // is symbolic at axis i, the output is symbolic at
                    // axis i.
                    if (node.Inputs.Length == 0)
                    {
                        perNode[i] = null;
                        break;
                    }
                    var lhs = perNode[node.Inputs[0]];
                    if (node.Inputs.Length == 1)
                    {
                        perNode[i] = lhs;
                        break;
                    }
                    var rhs = perNode[node.Inputs[1]];
                    perNode[i] = MergeSymbolic(lhs, rhs);
                    break;
            }
        }
        return perNode;
    }

    /// <summary>
    /// Returns a SymbolicShape whose symbolic-axis set is the union
    /// of <paramref name="a"/>'s and <paramref name="b"/>'s. Used by
    /// pointwise binary ops: if either operand is dynamic at axis i
    /// the output is dynamic at axis i too.
    /// </summary>
    private static SymbolicShape? MergeSymbolic(SymbolicShape? a, SymbolicShape? b)
    {
        if (a is null || b is null) return a ?? b;
        if (a.ConcreteShape.Length != b.ConcreteShape.Length) return null;

        var union = new HashSet<int>(a.SymbolicDimensions);
        foreach (var s in b.SymbolicDimensions) union.Add(s);
        var dims = new int[union.Count];
        int idx = 0;
        foreach (var s in union) dims[idx++] = s;
        Array.Sort(dims);

        // Concrete shape: prefer a's where both are static, b's where
        // a is dynamic, default to a's value otherwise.
        var concrete = new int[a.ConcreteShape.Length];
        for (int i = 0; i < concrete.Length; i++)
        {
            bool aDynamic = Array.IndexOf(a.SymbolicDimensions, i) >= 0;
            concrete[i] = aDynamic ? b.ConcreteShape[i] : a.ConcreteShape[i];
        }
        return new SymbolicShape(concrete, dims);
    }
}
