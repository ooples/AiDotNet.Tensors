namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Compiles a lazy computation graph into an optimized execution plan.
/// Orchestrates optimization passes (fusion, dead code elimination, memory planning,
/// operation reordering) and produces a topologically-sorted list of nodes ready for execution.
///
/// Pipeline: Record → Optimize → TopSort → Execute
/// </summary>
internal sealed class LazyGraphCompiler
{
    private readonly List<ILazyGraphOptimizationPass> _passes;

    internal LazyGraphCompiler()
    {
        _passes = new List<ILazyGraphOptimizationPass>
        {
            new CpuFusionPass(),
            new DeadCodeEliminationPass(),
            new OperationReorderingPass(),
            new MemoryPlanningPass()
        };
    }

    /// <summary>
    /// Compiles the graph: runs all optimization passes, assigns topological indices,
    /// and returns the optimized node list in execution order.
    /// </summary>
    internal List<ILazyNode> Compile(IReadOnlyList<ILazyNode> rawNodes)
    {
        var nodes = new List<ILazyNode>(rawNodes);

        // Run optimization passes
        foreach (var pass in _passes)
        {
            nodes = pass.Run(nodes);
        }

        // Assign topological indices
        for (int i = 0; i < nodes.Count; i++)
        {
            nodes[i].TopologicalIndex = i;
        }

        // Count consumers for each node
        foreach (var node in nodes)
        {
            node.ConsumerCount = 0;
        }
        foreach (var node in nodes)
        {
            foreach (var input in node.GetInputNodes())
            {
                input.ConsumerCount++;
            }
        }

        return nodes;
    }

    /// <summary>
    /// Dead code elimination: removes nodes whose output is never consumed
    /// (except the final output nodes).
    /// </summary>
    private sealed class DeadCodeEliminationPass : ILazyGraphOptimizationPass
    {
        public string Name => "DeadCodeElimination";

        public List<ILazyNode> Run(List<ILazyNode> nodes)
        {
            if (nodes.Count <= 1)
                return nodes;

            // Build consumer counts
            var consumers = new Dictionary<ILazyNode, int>();
            foreach (var node in nodes)
            {
                foreach (var input in node.GetInputNodes())
                {
                    if (consumers.ContainsKey(input))
                        consumers[input]++;
                    else
                        consumers[input] = 1;
                }
            }

            // Keep every node that either has a consumer OR is a leaf (no
            // downstream op consumes its output) — every leaf is potentially
            // a graph output and MUST survive DCE. The prior implementation
            // only kept the SINGLE last node in the list as terminal, which
            // worked for single-output scoped traces (TensorCodec's default
            // usage) but silently dropped every-other-graph-output for
            // multi-output compilations like ONNX models. Surfaced by
            // BERT-SQuAD × 100 sample replay: after the OnnxImporter wrapped
            // every declared graph output in TensorAdd(x, 0), DCE kept only
            // the final wrap as compiled, leaving all other output wraps
            // uncompiled — their output tensors' LazySource stayed alive,
            // auto-materialization via AsSpan triggered Realize-cascade at
            // first read but NOT on subsequent executes (IsRealized=true
            // blocks re-realize), so those outputs froze at run-1's values.
            var result = new List<ILazyNode>(nodes.Count);
            foreach (var node in nodes)
            {
                bool hasConsumers = consumers.ContainsKey(node) && consumers[node] > 0;
                bool isLeaf = !hasConsumers;
                if (hasConsumers || isLeaf)
                    result.Add(node);
            }

            return result;
        }
    }

    /// <summary>
    /// Operation reordering: schedules nodes for better cache locality.
    /// Moves producers as close as possible to their consumers to minimize
    /// the time data sits in cache between operations.
    ///
    /// Uses a priority-based topological sort: among ready nodes, prefers
    /// the one whose output will be consumed soonest.
    /// </summary>
    private sealed class OperationReorderingPass : ILazyGraphOptimizationPass
    {
        public string Name => "OperationReordering";

        public List<ILazyNode> Run(List<ILazyNode> nodes)
            => ReorderOperations(nodes);
    }

    /// <summary>
    /// Topologically schedules operations while preferring a producer whose first consumer is
    /// closest to becoming ready.
    /// </summary>
    /// <remarks>
    /// Ready nodes that share a first consumer also share the same priority. Grouping them by that
    /// consumer makes priority changes O(log V) instead of rescanning every ready node after each
    /// scheduled operation. Node identity is resolved once while building the graph; the scheduling
    /// loop itself uses array indices and performs no dictionary lookups.
    /// </remarks>
    internal static List<ILazyNode> ReorderOperations(List<ILazyNode> nodes)
    {
        int nodeCount = nodes.Count;
        if (nodeCount <= 2)
            return nodes;

        var nodeIndices = new Dictionary<ILazyNode, int>(nodeCount);
        var inDegree = new int[nodeCount];
        var dependents = new List<int>?[nodeCount];

        for (int i = 0; i < nodeCount; i++)
            nodeIndices[nodes[i]] = i;

        for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
        {
            foreach (var input in nodes[nodeIndex].GetInputNodes())
            {
                if (!nodeIndices.TryGetValue(input, out int inputIndex))
                    continue;

                inDegree[nodeIndex]++;
                (dependents[inputIndex] ??= new List<int>()).Add(nodeIndex);
            }
        }

        // A ready producer's score is the remaining in-degree of its first consumer. Producers
        // with no consumer use the terminal group, which retains the original int.MaxValue score.
        //
        // Keep the original ready-list positions as the secondary key. The previous implementation
        // removed a selected node by replacing it with the final ready node, so equal-score choices
        // were intentionally not stable input order. Preserving those positions retains the exact
        // historical execution plan while eliminating the full ready-list scan.
        int terminalGroup = nodeCount;
        var firstDependentGroups = new int[nodeCount];
        var readyGroups = new SortedSet<int>?[nodeCount + 1];
        var groupVersions = new int[nodeCount + 1];
        var ready = new List<int>(nodeCount);
        var readyGroupQueue = new ReadyGroupPriorityQueue(nodeCount);

        for (int i = 0; i < nodeCount; i++)
        {
            firstDependentGroups[i] = dependents[i] is { Count: > 0 } nodeDependents
                ? nodeDependents[0]
                : terminalGroup;
        }

        void PublishGroup(int group)
        {
            SortedSet<int>? members = readyGroups[group];
            if (members is null || members.Count == 0)
                return;

            int version = ++groupVersions[group];
            int score = group == terminalGroup ? int.MaxValue : inDegree[group];
            readyGroupQueue.Enqueue(group, version, score, members.Min);
        }

        void AddReadyNode(int nodeIndex)
        {
            int group = firstDependentGroups[nodeIndex];
            SortedSet<int> members = readyGroups[group] ??= new SortedSet<int>();
            bool wasEmpty = members.Count == 0;
            int position = ready.Count;
            ready.Add(nodeIndex);
            members.Add(position);
            if (wasEmpty)
                PublishGroup(group);
        }

        for (int i = 0; i < nodeCount; i++)
        {
            if (inDegree[i] == 0)
                AddReadyNode(i);
        }

        var result = new List<ILazyNode>(nodeCount);
        while (readyGroupQueue.TryDequeue(out int group, out int version))
        {
            SortedSet<int>? members = readyGroups[group];
            if (version != groupVersions[group] || members is null || members.Count == 0)
                continue;

            int chosenPosition = members.Min;
            int chosenIndex = ready[chosenPosition];
            int lastPosition = ready.Count - 1;
            int lastIndex = ready[lastPosition];
            int lastGroup = firstDependentGroups[lastIndex];

            members.Remove(chosenPosition);
            groupVersions[group]++;

            if (chosenPosition != lastPosition)
            {
                SortedSet<int> lastMembers = readyGroups[lastGroup]!;
                lastMembers.Remove(lastPosition);
                lastMembers.Add(chosenPosition);
                ready[chosenPosition] = lastIndex;

                if (lastGroup != group)
                    groupVersions[lastGroup]++;
            }

            ready.RemoveAt(lastPosition);
            PublishGroup(group);
            if (lastGroup != group)
                PublishGroup(lastGroup);

            result.Add(nodes[chosenIndex]);
            List<int>? chosenDependents = dependents[chosenIndex];
            if (chosenDependents is null)
                continue;

            foreach (int dependentIndex in chosenDependents)
            {
                int remaining = --inDegree[dependentIndex];

                // Every ready producer whose first consumer is this dependent belongs to the
                // dependent's group. Republish that group with its new shared priority.
                if (readyGroups[dependentIndex] is { Count: > 0 })
                    PublishGroup(dependentIndex);

                if (remaining == 0)
                    AddReadyNode(dependentIndex);
            }
        }

        return result;
    }

    /// <summary>
    /// Minimal stable min-heap used by operation reordering. Kept local instead of relying on
    /// <c>System.Collections.Generic.PriorityQueue</c>, which is unavailable on the net471 target.
    /// </summary>
    private sealed class ReadyGroupPriorityQueue
    {
        private readonly List<Entry> _entries;

        internal ReadyGroupPriorityQueue(int capacity)
        {
            _entries = new List<Entry>(capacity);
        }

        internal void Enqueue(int group, int version, int score, int order)
        {
            var entry = new Entry(group, version, score, order);
            int index = _entries.Count;
            _entries.Add(entry);

            while (index > 0)
            {
                int parent = (index - 1) / 2;
                if (!ComesBefore(entry, _entries[parent]))
                    break;

                _entries[index] = _entries[parent];
                index = parent;
            }

            _entries[index] = entry;
        }

        internal bool TryDequeue(out int group, out int version)
        {
            if (_entries.Count == 0)
            {
                group = 0;
                version = 0;
                return false;
            }

            Entry first = _entries[0];
            int lastIndex = _entries.Count - 1;
            Entry last = _entries[lastIndex];
            _entries.RemoveAt(lastIndex);

            if (lastIndex > 0)
            {
                int index = 0;
                while (true)
                {
                    int left = index * 2 + 1;
                    if (left >= lastIndex)
                        break;

                    int right = left + 1;
                    int child = right < lastIndex && ComesBefore(_entries[right], _entries[left])
                        ? right
                        : left;
                    if (!ComesBefore(_entries[child], last))
                        break;

                    _entries[index] = _entries[child];
                    index = child;
                }

                _entries[index] = last;
            }

            group = first.Group;
            version = first.Version;
            return true;
        }

        private static bool ComesBefore(Entry left, Entry right)
            => left.Score < right.Score || left.Score == right.Score && left.Order < right.Order;

        private readonly struct Entry
        {
            internal Entry(int group, int version, int score, int order)
            {
                Group = group;
                Version = version;
                Score = score;
                Order = order;
            }

            internal int Group { get; }
            internal int Version { get; }
            internal int Score { get; }
            internal int Order { get; }
        }
    }

    /// <summary>
    /// Memory planning: analyzes buffer lifetimes and marks nodes whose output
    /// buffers can be reused by subsequent operations. This reduces peak memory
    /// usage by allowing buffer recycling within the compiled plan.
    ///
    /// Currently sets ConsumerCount which CompiledStep uses to determine when
    /// a buffer can be returned to the pool. Future: explicit buffer aliasing.
    /// </summary>
    private sealed class MemoryPlanningPass : ILazyGraphOptimizationPass
    {
        public string Name => "MemoryPlanning";

        public List<ILazyNode> Run(List<ILazyNode> nodes)
        {
            // Compute consumer counts (used later for buffer lifetime analysis)
            var consumers = new Dictionary<ILazyNode, int>();
            foreach (var node in nodes)
            {
                foreach (var input in node.GetInputNodes())
                {
                    if (consumers.ContainsKey(input))
                        consumers[input]++;
                    else
                        consumers[input] = 1;
                }
            }

            // Set consumer counts on nodes (used by execution to know when buffers are free)
            foreach (var node in nodes)
            {
                node.ConsumerCount = consumers.ContainsKey(node) ? consumers[node] : 0;
            }

            // Future: Group same-sized buffers for aliasing.
            // Nodes whose ConsumerCount reaches 0 after execution have free buffers
            // that can be recycled for subsequent nodes with matching shape.

            return nodes;
        }
    }
}
