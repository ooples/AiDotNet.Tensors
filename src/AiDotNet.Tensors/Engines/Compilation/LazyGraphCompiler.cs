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
        {
            if (nodes.Count <= 2)
                return nodes;

            // Build dependency graph: for each node, which nodes must come before it
            var nodeSet = new HashSet<ILazyNode>(nodes);
            var inDegree = new Dictionary<ILazyNode, int>();
            var dependents = new Dictionary<ILazyNode, List<ILazyNode>>();

            foreach (var node in nodes)
            {
                inDegree[node] = 0;
                dependents[node] = new List<ILazyNode>();
            }

            foreach (var node in nodes)
            {
                foreach (var input in node.GetInputNodes())
                {
                    if (nodeSet.Contains(input))
                    {
                        inDegree[node]++;
                        dependents[input].Add(node);
                    }
                }
            }

            // Kahn's algorithm with a stable priority queue. The old implementation scanned the
            // entire ready list for every emitted node. Wide autodiff graphs can have tens of
            // thousands of simultaneously-ready parameter/gradient producers, turning compilation
            // into O(V^2) work (SegMamba spent 7.9 of 9.3 seconds in this pass alone).
            //
            // The first consumer's original topological position is a stable locality score: among
            // ready producers, emit the one needed by the earliest downstream operation. This keeps
            // producers close to consumers without a mutable priority or a full ready-list rescan.
            var originalOrder = new Dictionary<ILazyNode, int>(nodes.Count);
            for (int i = 0; i < nodes.Count; i++)
                originalOrder[nodes[i]] = i;

            int GetLocalityScore(ILazyNode node)
            {
                var nodeDependents = dependents[node];
                return nodeDependents.Count == 0
                    ? int.MaxValue
                    : originalOrder[nodeDependents[0]];
            }

            var ready = new SortedSet<ILazyNode>(Comparer<ILazyNode>.Create((left, right) =>
            {
                if (ReferenceEquals(left, right)) return 0;
                int score = GetLocalityScore(left).CompareTo(GetLocalityScore(right));
                return score != 0
                    ? score
                    : originalOrder[left].CompareTo(originalOrder[right]);
            }));
            foreach (var node in nodes)
            {
                if (inDegree[node] == 0)
                    ready.Add(node);
            }

            var result = new List<ILazyNode>(nodes.Count);
            while (ready.Count > 0)
            {
                var chosen = ready.Min!;
                ready.Remove(chosen);
                result.Add(chosen);

                foreach (var dep in dependents[chosen])
                {
                    inDegree[dep]--;
                    if (inDegree[dep] == 0)
                        ready.Add(dep);
                }
            }

            return result;
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
