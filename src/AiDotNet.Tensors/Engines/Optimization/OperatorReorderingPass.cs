using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Reorders independent operations to maximize cache line reuse between
/// producer and consumer. If op A writes tensor X and op C reads X, but
/// independent op B runs between them, X is evicted from cache by B's
/// working set. Pulling C closer to A (A→C→B when B is independent)
/// keeps X hot in L1/L2.
///
/// <para><b>Algorithm:</b></para>
/// <list type="number">
/// <item>Build the DAG of op dependencies from step inputs/outputs.</item>
/// <item>For each tensor, identify its producer step and consumer steps.</item>
/// <item>When a consumer is separated from its producer by independent ops,
/// move it closer while preserving all dependency edges.</item>
/// </list>
///
/// <para>This is a local reordering (greedy, single-pass) — not a full
/// topological-sort-based scheduler. It captures the common case in
/// diffusion UNets where residual-add and skip-connection ops create
/// large gaps between producer and consumer.</para>
/// </summary>
internal sealed class OperatorReorderingPass : ICpuOptimizationPass
{
    public string Name => "OperatorReordering";
    public bool IsEnabled => true;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (steps.Length < 4) return null; // Not worth reordering tiny plans

        // Phase 1: Build dependency info
        // For each step, record which other steps it depends on (transitively
        // through tensor references). Must use reference identity — the earlier
        // implementation keyed on RuntimeHelpers.GetHashCode(t) (an int), which
        // loses tensor identity under hash collisions and can produce invalid
        // schedules (two distinct tensors hashing equal would be treated as the
        // same, dropping a real dependency or inventing a spurious one).
        var refComparer = ReferenceEqualityComparer<Tensor<T>>.Instance;
        var producerOf = new Dictionary<Tensor<T>, int>(refComparer);
        for (int i = 0; i < steps.Length; i++)
            producerOf[steps[i].OutputBuffer] = i;

        // dependsOn[i] = set of step indices that step i reads from
        var dependsOn = new HashSet<int>[steps.Length];
        for (int i = 0; i < steps.Length; i++)
        {
            dependsOn[i] = new HashSet<int>();
            for (int j = 0; j < steps[i].Inputs.Length; j++)
            {
                if (producerOf.TryGetValue(steps[i].Inputs[j], out int prodIdx) && prodIdx != i)
                    dependsOn[i].Add(prodIdx);
            }
        }

        // Phase 2: Greedy pull-forward
        // For each step, check if its producer is far away. If so, try to
        // move this step closer to the producer.
        var reordered = new List<CompiledStep<T>>(steps);
        bool anyMoved = false;

        for (int pass = 0; pass < 2; pass++) // Two passes to catch cascading opportunities
        {
            for (int i = 1; i < reordered.Count; i++)
            {
                var step = reordered[i];
                if (dependsOn[i].Count == 0) continue; // Leaf input — no producer to pull toward

                // Find the latest producer
                int latestProducer = -1;
                foreach (var dep in dependsOn[i])
                    if (dep > latestProducer) latestProducer = dep;

                if (latestProducer < 0 || latestProducer >= i - 1) continue; // Already adjacent

                // Can we move step i to position (latestProducer + 1)?
                // Check that no step between latestProducer+1 and i depends on step i's output.
                int targetPos = latestProducer + 1;
                bool canMove = true;
                var stepOutput = step.OutputBuffer;

                for (int k = targetPos; k < i; k++)
                {
                    // Does step k depend on step i? (step i's output is step k's input)
                    for (int j = 0; j < reordered[k].Inputs.Length; j++)
                    {
                        if (ReferenceEquals(reordered[k].Inputs[j], stepOutput))
                        {
                            canMove = false;
                            break;
                        }
                    }
                    if (!canMove) break;
                }

                if (canMove && targetPos < i)
                {
                    // Move step i to targetPos
                    reordered.RemoveAt(i);
                    reordered.Insert(targetPos, step);

                    // Rebuild dependency indices for the affected range
                    RebuildDependencies(reordered, producerOf, dependsOn);
                    anyMoved = true;
                }
            }
        }

        return anyMoved ? reordered.ToArray() : null;
    }

    private static void RebuildDependencies<T>(
        List<CompiledStep<T>> steps,
        Dictionary<Tensor<T>, int> producerOf,
        HashSet<int>[] dependsOn)
    {
        producerOf.Clear();
        for (int i = 0; i < steps.Count; i++)
            producerOf[steps[i].OutputBuffer] = i;

        for (int i = 0; i < steps.Count; i++)
        {
            dependsOn[i].Clear();
            for (int j = 0; j < steps[i].Inputs.Length; j++)
            {
                if (producerOf.TryGetValue(steps[i].Inputs[j], out int prodIdx) && prodIdx != i)
                    dependsOn[i].Add(prodIdx);
            }
        }
    }

    /// <summary>Reference-equality comparer for tensor identity tracking.</summary>
    private sealed class ReferenceEqualityComparer<TItem> : IEqualityComparer<TItem> where TItem : class
    {
        public static readonly ReferenceEqualityComparer<TItem> Instance = new();
        public bool Equals(TItem? x, TItem? y) => ReferenceEquals(x, y);
        public int GetHashCode(TItem obj) => RuntimeHelpers.GetHashCode(obj);
    }
}
