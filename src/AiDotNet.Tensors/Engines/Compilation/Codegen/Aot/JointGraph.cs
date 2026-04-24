// Copyright (c) AiDotNet. All rights reserved.
// Joint forward+backward graph — the representation AOTAutograd
// operates on. Captures both halves of a training step in one
// structure so whole-graph optimisations (DCE, in-place reinsertion,
// min-cut activation partitioning) can span the forward/backward
// boundary.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Aot;

/// <summary>
/// A single joint graph edge — either a forward op producing an
/// activation, or a backward op consuming it. The partitioner
/// decides which activations to retain (cheap to store, expensive
/// to recompute) vs. recompute on-demand at backward time.
/// </summary>
/// <remarks>
/// <para><b>Why "joint" instead of two separate graphs:</b></para>
/// <para>
/// PyTorch's AOTAutograd constructs a single functional graph where
/// forward nodes produce tensors and backward nodes consume them.
/// Optimisations like DCE naturally work across the boundary —
/// eliminating a forward op drops every backward node that only
/// existed to propagate its gradient. Separate forward and backward
/// graphs would need coupled pass infrastructure to achieve the
/// same result.
/// </para>
/// </remarks>
public sealed class JointGraph
{
    private readonly List<JointNode> _nodes = new();

    /// <summary>Total number of forward + backward nodes.</summary>
    public int Count => _nodes.Count;

    /// <summary>Read-only view of the nodes in insertion order.</summary>
    public IReadOnlyList<JointNode> Nodes => _nodes;

    /// <summary>
    /// Indices of nodes that are part of the forward pass, in
    /// topological order. Populated by <see cref="AppendForward"/>.
    /// </summary>
    public IReadOnlyList<int> ForwardNodes => _forwardNodes;
    private readonly List<int> _forwardNodes = new();

    /// <summary>
    /// Indices of nodes that are part of the backward pass, in
    /// topological order (reverse of how gradient-tape walks them).
    /// </summary>
    public IReadOnlyList<int> BackwardNodes => _backwardNodes;
    private readonly List<int> _backwardNodes = new();

    /// <summary>
    /// Appends a forward node and returns its index.
    /// </summary>
    /// <param name="opName">The op name (e.g. "TensorMultiply").</param>
    /// <param name="inputs">Producer node indices (must reference
    /// existing forward nodes or leaves).</param>
    /// <param name="outputShape">Shape of the produced tensor.</param>
    /// <param name="isLeaf">True for input tensors / parameters
    /// (no producer).</param>
    /// <returns>The new node's index.</returns>
    public int AppendForward(string opName, int[] inputs, int[] outputShape, bool isLeaf = false)
    {
        if (opName is null) throw new ArgumentNullException(nameof(opName));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputShape is null) throw new ArgumentNullException(nameof(outputShape));
        foreach (var idx in inputs)
            if (idx < 0 || idx >= _nodes.Count)
                throw new ArgumentException($"Input index {idx} out of range [0, {_nodes.Count}).");

        int id = _nodes.Count;
        _nodes.Add(new JointNode(id, opName, inputs, outputShape, JointNodeKind.Forward, isLeaf));
        _forwardNodes.Add(id);
        return id;
    }

    /// <summary>
    /// Appends a backward node. Backward nodes consume both forward
    /// activations (by index into the forward subgraph) and other
    /// backward nodes (e.g. grad-of-output flowing into grad-of-input).
    /// </summary>
    /// <param name="opName">The backward op name.</param>
    /// <param name="inputs">Producer node indices — can mix forward
    /// activations and prior backward results.</param>
    /// <param name="outputShape">Shape of the produced gradient.</param>
    /// <returns>The new node's index.</returns>
    public int AppendBackward(string opName, int[] inputs, int[] outputShape)
    {
        if (opName is null) throw new ArgumentNullException(nameof(opName));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputShape is null) throw new ArgumentNullException(nameof(outputShape));
        foreach (var idx in inputs)
            if (idx < 0 || idx >= _nodes.Count)
                throw new ArgumentException($"Input index {idx} out of range [0, {_nodes.Count}).");

        int id = _nodes.Count;
        _nodes.Add(new JointNode(id, opName, inputs, outputShape, JointNodeKind.Backward, isLeaf: false));
        _backwardNodes.Add(id);
        return id;
    }

    /// <summary>
    /// Computes the total element count referenced by a node's
    /// output — a proxy for memory use when deciding whether to
    /// retain or recompute the activation.
    /// </summary>
    public long ElementCountAt(int nodeIndex)
    {
        var node = _nodes[nodeIndex];
        long c = 1;
        for (int i = 0; i < node.OutputShape.Length; i++) c *= node.OutputShape[i];
        return c;
    }
}

/// <summary>
/// Which half of the training step a node belongs to.
/// </summary>
public enum JointNodeKind
{
    /// <summary>Forward pass node — produces an activation.</summary>
    Forward,
    /// <summary>Backward pass node — produces a gradient.</summary>
    Backward,
}

/// <summary>
/// One node in a <see cref="JointGraph"/>.
/// </summary>
public readonly struct JointNode
{
    /// <summary>Stable index in the graph.</summary>
    public int Id { get; }
    /// <summary>Engine op name.</summary>
    public string OpName { get; }
    /// <summary>Producer indices (into the same graph).</summary>
    public int[] Inputs { get; }
    /// <summary>Output tensor shape.</summary>
    public int[] OutputShape { get; }
    /// <summary>Forward or backward.</summary>
    public JointNodeKind Kind { get; }
    /// <summary>True for input / parameter leaves.</summary>
    public bool IsLeaf { get; }

    /// <summary>Constructs a node.</summary>
    public JointNode(int id, string opName, int[] inputs, int[] outputShape, JointNodeKind kind, bool isLeaf)
    {
        Id = id;
        OpName = opName;
        Inputs = inputs;
        OutputShape = outputShape;
        Kind = kind;
        IsLeaf = isLeaf;
    }
}
