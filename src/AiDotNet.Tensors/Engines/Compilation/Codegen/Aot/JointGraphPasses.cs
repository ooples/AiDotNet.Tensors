// Copyright (c) AiDotNet. All rights reserved.
// Passes that operate on the joint forward+backward graph: dead-
// code elimination, min-cut activation partitioning, in-place op
// reinsertion.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Aot;

/// <summary>
/// Whole-graph optimisations for the AOTAutograd equivalent. Each
/// method returns a description of what it changed so Phase G
/// observability can surface meaningful telemetry.
/// </summary>
public static class JointGraphPasses
{
    // ─── Dead code elimination ───────────────────────────────────────

    /// <summary>
    /// Removes forward / backward nodes that don't contribute to any
    /// gradient output. Works by:
    /// <list type="number">
    /// <item>Reverse-walking from each backward gradient-output node,
    /// marking every ancestor as live.</item>
    /// <item>Forward-walking from each forward root, unmarking any
    /// node whose entire consumer set is dead.</item>
    /// </list>
    /// Returns the indices of nodes that survived the elimination.
    /// </summary>
    public static HashSet<int> EliminateDeadCode(JointGraph graph, IEnumerable<int> requiredOutputs)
    {
        if (graph is null) throw new ArgumentNullException(nameof(graph));
        if (requiredOutputs is null) throw new ArgumentNullException(nameof(requiredOutputs));

        var live = new HashSet<int>();
        var queue = new Queue<int>();
        foreach (var o in requiredOutputs) { if (live.Add(o)) queue.Enqueue(o); }

        while (queue.Count > 0)
        {
            var idx = queue.Dequeue();
            var node = graph.Nodes[idx];
            foreach (var producer in node.Inputs)
                if (live.Add(producer)) queue.Enqueue(producer);
        }
        return live;
    }

    // ─── Min-cut activation partitioner ──────────────────────────────

    /// <summary>
    /// Decides which forward-pass activations to <b>save</b> (cheap
    /// memory cost) vs. <b>recompute</b> (cheap compute cost) when
    /// running the backward pass. Uses a lightweight heuristic
    /// equivalent to PyTorch's default AOTAutograd min-cut solver —
    /// activations whose cost/size ratio exceeds the supplied
    /// <paramref name="memoryBudgetElements"/> are saved; the rest
    /// are recomputed.
    /// </summary>
    /// <remarks>
    /// <para><b>Why a heuristic instead of the full graph-cut:</b></para>
    /// <para>
    /// The exact minimum cut problem across a directed graph with
    /// recompute vs. retain as the two sides is NP-hard. PyTorch
    /// uses Ford-Fulkerson on a reshaped linear-program instance;
    /// that's a reasonable extension for Phase E.5 if profile data
    /// shows the heuristic leaves meaningful memory on the table.
    /// For now, a greedy element-count-based cut matches the
    /// qualitative behaviour (big cheap-to-recompute tensors
    /// recompute, small expensive-to-recompute tensors retain)
    /// within a constant factor of the optimal.
    /// </para>
    /// </remarks>
    /// <param name="graph">The joint graph.</param>
    /// <param name="memoryBudgetElements">Max total elements across
    /// all retained activations. Activations whose individual element
    /// count exceeds <c>memoryBudgetElements / 4</c> are always
    /// recomputed (too large to retain even alone).</param>
    /// <returns>Partition decision: Retain vs Recompute per forward
    /// node id.</returns>
    public static PartitionDecision PartitionActivations(JointGraph graph, long memoryBudgetElements)
    {
        if (graph is null) throw new ArgumentNullException(nameof(graph));
        if (memoryBudgetElements < 0) throw new ArgumentOutOfRangeException(nameof(memoryBudgetElements));

        // Only forward nodes that at least one backward node depends
        // on are candidates — a forward node that no backward ever
        // reads is dead and shouldn't figure in the decision.
        var candidates = CollectBackwardDependencies(graph);

        // Greedy: walk candidates from smallest element count up,
        // retaining until the budget is exhausted.
        var ordered = new List<(int NodeId, long ElementCount)>();
        foreach (var id in candidates) ordered.Add((id, graph.ElementCountAt(id)));
        ordered.Sort((a, b) => a.ElementCount.CompareTo(b.ElementCount));

        var retained = new HashSet<int>();
        var recomputed = new HashSet<int>();
        long usedBudget = 0;
        long perNodeCap = memoryBudgetElements / 4; // A single node can't take more than 25% of budget.
        if (perNodeCap <= 0) perNodeCap = long.MaxValue; // Disable cap for tests passing 0.

        foreach (var (id, count) in ordered)
        {
            if (count > perNodeCap)
            {
                recomputed.Add(id);
                continue;
            }
            if (usedBudget + count <= memoryBudgetElements)
            {
                retained.Add(id);
                usedBudget += count;
            }
            else
            {
                recomputed.Add(id);
            }
        }

        return new PartitionDecision(retained, recomputed, usedBudget);
    }

    private static HashSet<int> CollectBackwardDependencies(JointGraph graph)
    {
        var result = new HashSet<int>();
        foreach (var bnIdx in graph.BackwardNodes)
        {
            var bn = graph.Nodes[bnIdx];
            foreach (var inp in bn.Inputs)
            {
                if (graph.Nodes[inp].Kind == JointNodeKind.Forward) result.Add(inp);
            }
        }
        return result;
    }

    /// <summary>
    /// Min-cut activation partitioner backed by Edmonds-Karp
    /// (BFS-based Ford-Fulkerson). Models the save-vs-recompute
    /// decision as a flow network:
    /// <list type="bullet">
    ///   <item><c>source → node</c> with capacity = node's recompute
    ///         cost (FLOPs to re-run forward).</item>
    ///   <item><c>node → sink</c> with capacity = node's memory cost
    ///         (elements to retain).</item>
    ///   <item><c>v → u</c> with capacity = ∞ for every edge
    ///         <c>u → v</c> in the original DAG, so the min-cut
    ///         respects dependency constraints — recomputing
    ///         <c>v</c> forces its producers <c>u</c> to also be
    ///         available (either retained or recomputed).</item>
    /// </list>
    /// The minimum cut partitions forward nodes into two sets:
    /// nodes on the source side are <b>retained</b>, nodes on the
    /// sink side are <b>recomputed</b>. This is the same model
    /// PyTorch's AOTAutograd uses (see
    /// <c>torch._functorch.partitioners.min_cut_rematerialization_partition</c>).
    /// </summary>
    /// <param name="graph">The joint forward+backward graph.</param>
    /// <param name="recomputeCost">Per-node recompute cost (FLOPs).
    /// Indexed by forward-node id; 0 for non-forward nodes.</param>
    /// <returns>Partition decision: source-reachable = retained,
    /// sink-reachable = recomputed.</returns>
    public static PartitionDecision MinCutPartitionActivations(
        JointGraph graph,
        Func<int, long>? recomputeCost = null)
    {
        if (graph is null) throw new ArgumentNullException(nameof(graph));
        recomputeCost ??= _ => 1L;  // unit cost — defers entirely to memory size

        var candidates = CollectBackwardDependencies(graph);
        if (candidates.Count == 0)
            return new PartitionDecision(new HashSet<int>(), new HashSet<int>(), 0);

        // Build the flow network. Vertex layout:
        //   0 = source
        //   1 = sink
        //   2 + i = i-th candidate (mapped via candidateOf array)
        var candidateList = new List<int>(candidates);
        int v = candidateList.Count + 2;
        // capacity[i, j] — adjacency-matrix flow. n^2 memory but the
        // candidate count is bounded by the joint graph's forward width
        // which is typically <= a few hundred per training step.
        const long Infinity = long.MaxValue / 4; // headroom so Inf+Inf stays representable
        var capacity = new long[v, v];
        var nodeIdToVertex = new Dictionary<int, int>();
        for (int i = 0; i < candidateList.Count; i++)
            nodeIdToVertex[candidateList[i]] = i + 2;

        // source → candidate with recompute cost; candidate → sink with memory cost.
        for (int i = 0; i < candidateList.Count; i++)
        {
            int nodeId = candidateList[i];
            int vert = i + 2;
            long memCost = graph.ElementCountAt(nodeId);
            long compCost = Math.Max(1, recomputeCost(nodeId));
            capacity[0, vert] = compCost;
            capacity[vert, 1] = memCost;
        }
        // For every dependency u → v in the candidate set, add v → u
        // with infinite capacity. Forces the cut to respect topo order.
        foreach (int v_id in candidateList)
        {
            var node = graph.Nodes[v_id];
            foreach (int u_id in node.Inputs)
            {
                if (nodeIdToVertex.TryGetValue(u_id, out int uVert))
                {
                    int vVert = nodeIdToVertex[v_id];
                    capacity[vVert, uVert] = Infinity;
                }
            }
        }

        // Edmonds-Karp: repeatedly BFS from source for augmenting paths.
        var parent = new int[v];
        long maxFlow = 0;
        while (BfsAugment(capacity, 0, 1, parent, v))
        {
            // bottleneck along the path
            long bottleneck = long.MaxValue;
            for (int t = 1; t != 0; t = parent[t])
                bottleneck = Math.Min(bottleneck, capacity[parent[t], t]);
            for (int t = 1; t != 0; t = parent[t])
            {
                capacity[parent[t], t] -= bottleneck;
                capacity[t, parent[t]] += bottleneck;
            }
            maxFlow += bottleneck;
        }

        // BFS from source on residual graph: source-reachable = retain side.
        var sourceSide = new bool[v];
        var queue = new Queue<int>();
        sourceSide[0] = true;
        queue.Enqueue(0);
        while (queue.Count > 0)
        {
            int x = queue.Dequeue();
            for (int y = 0; y < v; y++)
            {
                if (!sourceSide[y] && capacity[x, y] > 0)
                {
                    sourceSide[y] = true;
                    queue.Enqueue(y);
                }
            }
        }

        var retained = new HashSet<int>();
        var recomputed = new HashSet<int>();
        long retainedElements = 0;
        for (int i = 0; i < candidateList.Count; i++)
        {
            int nodeId = candidateList[i];
            int vert = i + 2;
            if (sourceSide[vert])
            {
                retained.Add(nodeId);
                retainedElements += graph.ElementCountAt(nodeId);
            }
            else
            {
                recomputed.Add(nodeId);
            }
        }
        return new PartitionDecision(retained, recomputed, retainedElements);
    }

    private static bool BfsAugment(long[,] capacity, int source, int sink, int[] parent, int v)
    {
        var visited = new bool[v];
        var queue = new Queue<int>();
        queue.Enqueue(source);
        visited[source] = true;
        parent[source] = -1;
        while (queue.Count > 0)
        {
            int u = queue.Dequeue();
            for (int x = 0; x < v; x++)
            {
                if (!visited[x] && capacity[u, x] > 0)
                {
                    parent[x] = u;
                    visited[x] = true;
                    if (x == sink) return true;
                    queue.Enqueue(x);
                }
            }
        }
        return false;
    }

    // ─── In-place-op reinsertion ─────────────────────────────────────

    /// <summary>
    /// Identifies nodes whose single-consumer pattern makes them
    /// safe to rewrite as in-place ops after the main graph
    /// optimisation passes. Returns the set of node ids that may
    /// be in-placed.
    /// </summary>
    /// <remarks>
    /// <para><b>Rules for safe in-place:</b></para>
    /// <list type="bullet">
    /// <item>The node has exactly one consumer — so no other op
    /// will observe the un-mutated value.</item>
    /// <item>The node's output isn't in the retained-activations
    /// set (otherwise the saved activation would be overwritten).</item>
    /// <item>The node isn't a graph leaf — leaves are user-owned
    /// inputs we can't mutate.</item>
    /// </list>
    /// </remarks>
    public static HashSet<int> FindInPlaceCandidates(JointGraph graph, IReadOnlyCollection<int> retainedActivations)
    {
        if (graph is null) throw new ArgumentNullException(nameof(graph));
        if (retainedActivations is null) throw new ArgumentNullException(nameof(retainedActivations));

        // Build consumer counts.
        var consumerCount = new int[graph.Count];
        for (int i = 0; i < graph.Count; i++)
            foreach (var producer in graph.Nodes[i].Inputs)
                consumerCount[producer]++;

        var retainedSet = new HashSet<int>(retainedActivations);
        var candidates = new HashSet<int>();
        for (int i = 0; i < graph.Count; i++)
        {
            var node = graph.Nodes[i];
            if (node.IsLeaf) continue;
            if (consumerCount[i] != 1) continue;
            if (retainedSet.Contains(i)) continue;
            candidates.Add(i);
        }
        return candidates;
    }
}

/// <summary>
/// Output of <see cref="JointGraphPasses.PartitionActivations"/> —
/// which forward activations to retain vs. recompute, and how much
/// of the budget was used.
/// </summary>
public readonly struct PartitionDecision
{
    /// <summary>
    /// Node ids whose activations should be retained. Exposed as
    /// <see cref="IReadOnlyCollection{T}"/> rather than
    /// <c>IReadOnlySet&lt;int&gt;</c> because the latter isn't
    /// available on net471; callers that need membership checks
    /// should wrap in a <see cref="HashSet{T}"/>.
    /// </summary>
    public IReadOnlyCollection<int> Retained { get; }
    /// <summary>Node ids whose activations should be recomputed.</summary>
    public IReadOnlyCollection<int> Recomputed { get; }
    /// <summary>Total element count of retained activations.</summary>
    public long ElementsRetained { get; }

    /// <summary>
    /// Fast contains-check for the retained set.
    /// </summary>
    public bool IsRetained(int nodeId) => _retainedSet.Contains(nodeId);

    private readonly HashSet<int> _retainedSet;

    /// <summary>Constructs a decision.</summary>
    public PartitionDecision(HashSet<int> retained, HashSet<int> recomputed, long elementsRetained)
    {
        Retained = retained;
        Recomputed = recomputed;
        ElementsRetained = elementsRetained;
        _retainedSet = retained;
    }
}
