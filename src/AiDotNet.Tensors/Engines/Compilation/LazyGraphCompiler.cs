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

            // Kahn's algorithm with priority: prefer nodes that feed into
            // operations that are closest to being ready (lowest remaining in-degree)
            var ready = new List<ILazyNode>();
            foreach (var node in nodes)
            {
                if (inDegree[node] == 0)
                    ready.Add(node);
            }

            var result = new List<ILazyNode>(nodes.Count);
            while (ready.Count > 0)
            {
                // Pick the ready node whose first dependent has the lowest remaining in-degree
                // (heuristic: schedule producers right before their consumer becomes ready)
                int bestIdx = 0;
                int bestScore = int.MaxValue;
                for (int i = 0; i < ready.Count; i++)
                {
                    var deps = dependents[ready[i]];
                    int score = deps.Count > 0 ? inDegree[deps[0]] : int.MaxValue;
                    if (score < bestScore)
                    {
                        bestScore = score;
                        bestIdx = i;
                    }
                }

                var chosen = ready[bestIdx];
                ready[bestIdx] = ready[ready.Count - 1];
                ready.RemoveAt(ready.Count - 1);
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
