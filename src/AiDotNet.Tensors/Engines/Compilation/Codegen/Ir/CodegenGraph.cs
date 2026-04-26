// Copyright (c) AiDotNet. All rights reserved.
// Codegen IR graph — the unit a pattern matcher sees and an emitter
// turns into a kernel.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// A directed acyclic graph of <see cref="CodegenNode"/>s ready to
/// be fed into an emitter. One graph = one kernel's worth of work
/// — the fusion pass decides how to split a <see cref="CompiledStep{T}"/>
/// sequence into graphs of this form, after which each graph is
/// lowered independently to a target language.
/// </summary>
/// <remarks>
/// <para><b>Construction model:</b></para>
/// <para>
/// Graphs are built node-by-node via <see cref="AddNode"/>, which
/// returns the new node's index. Inputs must reference nodes that
/// were added earlier (topological order is enforced by
/// construction), so pattern matchers can iterate forward without
/// cycle checks.
/// </para>
/// <para><b>Inputs and outputs:</b></para>
/// <para>
/// A graph has two distinguished subsets of nodes:
/// <list type="bullet">
/// <item><see cref="CodegenOpKind.LoadInput"/> nodes mark places
/// where the generated kernel receives a tensor argument.</item>
/// <item><see cref="CodegenOpKind.StoreOutput"/> nodes mark places
/// where the kernel writes a tensor result.</item>
/// </list>
/// Both reference their host-side tensor by the integer attribute
/// on the node — the emitter binds those indices to function
/// parameters at emission time.
/// </para>
/// </remarks>
public sealed class CodegenGraph
{
    private readonly List<CodegenNode> _nodes = new();

    /// <summary>Total number of nodes in the graph.</summary>
    public int Count => _nodes.Count;

    /// <summary>Read-only view of the node list (in construction / topological order).</summary>
    public IReadOnlyList<CodegenNode> Nodes => _nodes;

    /// <summary>Indexed accessor — nodes are stable at their creation index.</summary>
    public CodegenNode this[int index] => _nodes[index];

    /// <summary>
    /// Indices of nodes that represent host-side inputs to the
    /// generated kernel, in binding order. Constructed as
    /// <see cref="CodegenOpKind.LoadInput"/> nodes are added.
    /// </summary>
    public IReadOnlyList<int> InputNodes => _inputNodes;
    private readonly List<int> _inputNodes = new();

    /// <summary>
    /// Indices of nodes that represent host-side outputs, in
    /// binding order. Constructed as
    /// <see cref="CodegenOpKind.StoreOutput"/> nodes are added.
    /// </summary>
    public IReadOnlyList<int> OutputNodes => _outputNodes;
    private readonly List<int> _outputNodes = new();

    /// <summary>
    /// Adds a node. Returns the new node's stable index.
    /// </summary>
    /// <param name="node">The node to add. Its <see cref="CodegenNode.Inputs"/>
    /// must reference indices strictly less than the current
    /// <see cref="Count"/>.</param>
    /// <returns>The new node's index in <see cref="Nodes"/>.</returns>
    /// <exception cref="ArgumentException">Thrown if an input
    /// index is outside <c>[0, Count)</c>.</exception>
    public int AddNode(CodegenNode node)
    {
        int count = _nodes.Count;
        for (int i = 0; i < node.Inputs.Length; i++)
        {
            int idx = node.Inputs[i];
            if ((uint)idx >= (uint)count)
                throw new ArgumentException(
                    $"CodegenNode input {i} = {idx} is outside [0, {count}) — graph must be added in topological order.",
                    nameof(node));
        }

        int newIndex = count;
        _nodes.Add(node);

        if (node.Op == CodegenOpKind.LoadInput) _inputNodes.Add(newIndex);
        else if (node.Op == CodegenOpKind.StoreOutput) _outputNodes.Add(newIndex);

        return newIndex;
    }

    /// <summary>
    /// Returns the set of node indices that consume <paramref name="producerIndex"/>.
    /// Built on demand — not cached, so don't call this inside a
    /// tight matcher loop. Pass builders needing per-node consumer
    /// lists should call <see cref="BuildConsumerTable"/> once.
    /// </summary>
    public IEnumerable<int> GetConsumers(int producerIndex)
    {
        for (int i = 0; i < _nodes.Count; i++)
        {
            foreach (var inp in _nodes[i].Inputs)
                if (inp == producerIndex)
                {
                    yield return i;
                    break;
                }
        }
    }

    /// <summary>
    /// Builds a flat consumer table for the whole graph in one pass.
    /// Returned array: <c>result[p]</c> is the list of nodes that
    /// consume node <c>p</c>. O(N + E) in graph size — use when
    /// multiple pass phases need the same info.
    /// </summary>
    public List<int>[] BuildConsumerTable()
    {
        var table = new List<int>[_nodes.Count];
        for (int i = 0; i < table.Length; i++) table[i] = new List<int>();
        for (int i = 0; i < _nodes.Count; i++)
        {
            foreach (var inp in _nodes[i].Inputs)
                table[inp].Add(i);
        }
        return table;
    }

    /// <summary>
    /// Computes a stable content hash covering every node's op,
    /// dtype, shape, and input wiring. Used by the autotune cache
    /// to key compiled kernels and by the recompile-guard system
    /// (Phase D) to detect graph identity across runs.
    /// </summary>
    /// <remarks>
    /// Hash collisions are acceptable here — the guard system
    /// resolves them by a full equality check on a cache-miss. FNV-1a
    /// is chosen for speed over distribution quality; if collisions
    /// prove costly in telemetry, swap in xxHash without changing
    /// the caller contract.
    /// </remarks>
    public long ComputeContentHash()
    {
        const long FnvOffset = unchecked((long)0xCBF29CE484222325UL);
        const long FnvPrime = 0x00000100000001B3L;
        long h = FnvOffset;

        for (int i = 0; i < _nodes.Count; i++)
        {
            var n = _nodes[i];
            h = unchecked((h ^ (long)n.Op) * FnvPrime);
            h = unchecked((h ^ (long)n.Dtype) * FnvPrime);
            for (int k = 0; k < n.Shape.Length; k++)
                h = unchecked((h ^ n.Shape[k]) * FnvPrime);
            for (int k = 0; k < n.Inputs.Length; k++)
                h = unchecked((h ^ n.Inputs[k]) * FnvPrime);
        }
        return h;
    }

    /// <summary>
    /// Produces a human-readable dump of the graph — one node per
    /// line, indexed. Used by Phase G observability and for debug.
    /// </summary>
    public string Dump()
    {
        var sb = new System.Text.StringBuilder();
        sb.Append($"CodegenGraph(nodes={_nodes.Count}, inputs=[{string.Join(",", _inputNodes)}], outputs=[{string.Join(",", _outputNodes)}])\n");
        for (int i = 0; i < _nodes.Count; i++)
            sb.Append($"  %{i} = {_nodes[i]}\n");
        return sb.ToString();
    }
}
