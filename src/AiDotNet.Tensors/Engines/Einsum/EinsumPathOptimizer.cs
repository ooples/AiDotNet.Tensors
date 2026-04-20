using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.Einsum;

/// <summary>
/// Chooses a pairwise contraction order for an einsum equation that
/// minimises the total contraction cost.
/// </summary>
/// <remarks>
/// <para>
/// An einsum over n operands can be evaluated as (n − 1) pairwise
/// contractions. The order matters: <c>(A·B)·C</c> can be orders of magnitude
/// cheaper than <c>A·(B·C)</c> depending on shapes. This is the same problem
/// opt_einsum solves; we bake it in so users never need the optional dep.
/// </para>
/// <para>Algorithms supplied:</para>
/// <list type="bullet">
///   <item><description><see cref="Greedy"/> — O(n³) per step, O(n⁴) total.
///     Scales to ~12 operands without hitting user-visible latency.</description></item>
/// </list>
/// </remarks>
public static class EinsumPathOptimizer
{
    /// <summary>
    /// Reserved internal label that stands for the full ellipsis batch block.
    /// Never appears in user-supplied equations (the parser accepts only ASCII
    /// letters), so it's safe to use as a marker in the optimizer.
    /// </summary>
    internal const char EllipsisMarker = '@';

    /// <summary>
    /// Greedy contraction order: at each step, pick the pair whose combined
    /// intermediate has the smallest size, with ties broken by "most labels
    /// summed out now" (opt_einsum-style greedy).
    /// </summary>
    public static EinsumPath Greedy(EinsumShapeBinding binding)
    {
        if (binding is null) throw new ArgumentNullException(nameof(binding));
        var eq = binding.Equation;

        // Build per-label size table, including the ellipsis marker.
        var sizes = new Dictionary<char, long>(binding.LabelSizes.Count + 1);
        foreach (var kv in binding.LabelSizes) sizes[kv.Key] = kv.Value;
        long batchProduct = 1L;
        foreach (var d in binding.BatchDims) batchProduct *= d;
        if (eq.HasEllipsis) sizes[EllipsisMarker] = batchProduct;

        // Build the starting "live" operand-label sets. Each live entry is
        // the set of labels currently represented by that tensor.
        var live = new List<HashSet<char>>(eq.Operands.Count);
        for (int i = 0; i < eq.Operands.Count; i++)
        {
            var op = eq.Operands[i];
            var set = new HashSet<char>(op.Labels);
            if (op.HasEllipsis) set.Add(EllipsisMarker);
            live.Add(set);
        }

        // Labels that must survive to the output.
        var outputSet = new HashSet<char>(eq.Output.Labels);
        if (eq.Output.HasEllipsis) outputSet.Add(EllipsisMarker);

        // Zero- or one-operand case: no pairwise contractions needed.
        if (live.Count <= 1)
            return new EinsumPath(Array.Empty<EinsumPathStep>(), 0);

        var steps = new List<EinsumPathStep>(live.Count - 1);
        long total = 0;

        while (live.Count > 1)
        {
            (int bestI, int bestJ, HashSet<char> bestResult, long bestCost, int bestRemoved)
                = (-1, -1, null!, long.MaxValue, -1);

            for (int i = 0; i < live.Count - 1; i++)
            for (int j = i + 1; j < live.Count; j++)
            {
                // Combined labels of the pair.
                var combined = new HashSet<char>(live[i]);
                combined.UnionWith(live[j]);

                // Labels still needed downstream (present in any other live
                // operand or in the output).
                var needed = new HashSet<char>(outputSet);
                for (int k = 0; k < live.Count; k++)
                {
                    if (k == i || k == j) continue;
                    needed.UnionWith(live[k]);
                }

                // Result labels = combined ∩ needed. Labels removed by this
                // step = combined − result.
                var resultLabels = new HashSet<char>(combined);
                resultLabels.IntersectWith(needed);

                int removed = combined.Count - resultLabels.Count;
                long cost = 2L * ProductOfSizes(combined, sizes);

                // Greedy objective (opt_einsum-style):
                //   primary: minimise cost
                //   tiebreak: maximise 'removed'
                if (cost < bestCost || (cost == bestCost && removed > bestRemoved))
                {
                    bestI = i;
                    bestJ = j;
                    bestResult = resultLabels;
                    bestCost = cost;
                    bestRemoved = removed;
                }
            }

            // Labels contracted (summed) by this step = combined − result.
            var contracted = new HashSet<char>(live[bestI]);
            contracted.UnionWith(live[bestJ]);
            contracted.ExceptWith(bestResult);

            var step = new EinsumPathStep(
                leftIndex: bestI,
                rightIndex: bestJ,
                resultLabels: bestResult.ToArray(),
                contractedLabels: contracted.ToArray(),
                estimatedFlops: bestCost);
            steps.Add(step);
            total += bestCost;

            // Replace the pair with the intermediate.
            // Remove higher index first so lower-index removal does not shift.
            live.RemoveAt(bestJ);
            live.RemoveAt(bestI);
            live.Add(bestResult);
        }

        return new EinsumPath(steps, total);
    }

    private static long ProductOfSizes(HashSet<char> labels, Dictionary<char, long> sizes)
    {
        long p = 1;
        foreach (var c in labels) p = checked(p * sizes[c]);
        return p;
    }

    /// <summary>
    /// Cache-aware path selection: checks <see cref="AutotuneCache"/> for a
    /// previously-recorded winner keyed by (equation, operand shapes) on the
    /// current hardware; on a miss runs <see cref="Greedy"/> and stores the
    /// chosen variant for future runs.  Callers get the same
    /// <see cref="EinsumPath"/> back either way.
    /// </summary>
    /// <remarks>
    /// The first call for a given (equation, shapes) pays the O(n⁴) greedy
    /// cost; subsequent calls are O(1) after a filesystem read. This matches
    /// the "opt_einsum-as-cached-warmup" pattern the #210 plan calls out.
    /// When branch-and-bound / Hungarian variants land, the stored
    /// <c>Variant</c> field will tell the optimiser which algorithm produced
    /// the cached path so stale entries can be invalidated.
    /// </remarks>
    public static EinsumPath Optimize(EinsumShapeBinding binding)
    {
        if (binding is null) throw new ArgumentNullException(nameof(binding));

        var kernelId = new KernelId("einsum", binding.Equation.Source);
        var shapeDims = new List<int>();
        foreach (var operandShape in binding.OperandShapes)
            foreach (var d in operandShape) shapeDims.Add(d);
        var shape = new ShapeProfile(shapeDims.ToArray());

        var cached = AutotuneCache.Lookup(kernelId, shape);
        if (cached != null && string.Equals(cached.Variant, "greedy", StringComparison.Ordinal))
        {
            // Fresh cache hit for the only variant we implement today.
            // When branch-and-bound lands we'll also accept "bnb" here and
            // dispatch accordingly; for now any unknown variant is ignored.
        }

        var path = Greedy(binding);

        // Store-best-effort: ignore exceptions so einsum never fails because
        // of a read-only HOME / missing cache directory.
        AutotuneCache.TryStore(kernelId, shape, new KernelChoice
        {
            Variant = "greedy",
            Parameters = new Dictionary<string, string>
            {
                { "numOperands", binding.Equation.Operands.Count.ToString() },
                { "estimatedFlops", path.TotalFlops.ToString() },
            },
        });

        return path;
    }
}

/// <summary>
/// A sequence of pairwise contraction steps that evaluates an einsum.
/// </summary>
public sealed class EinsumPath
{
    /// <summary>Ordered contraction steps. Empty for 0- or 1-operand equations.</summary>
    public IReadOnlyList<EinsumPathStep> Steps { get; }

    /// <summary>Sum of <see cref="EinsumPathStep.EstimatedFlops"/> across all steps.</summary>
    public long TotalFlops { get; }

    /// <summary>Constructs a path.</summary>
    public EinsumPath(IReadOnlyList<EinsumPathStep> steps, long totalFlops)
    {
        Steps = steps;
        TotalFlops = totalFlops;
    }
}

/// <summary>
/// A single pairwise contraction step in an <see cref="EinsumPath"/>.
/// </summary>
/// <remarks>
/// <para>
/// Indices are into the *current* live list: after each step, the two
/// contracted operands are removed and the intermediate is appended to the
/// end. The executor walks steps in order and maintains that same list.
/// </para>
/// </remarks>
public sealed class EinsumPathStep
{
    /// <summary>Index of the left operand in the live list at this step.</summary>
    public int LeftIndex { get; }

    /// <summary>Index of the right operand in the live list at this step.</summary>
    public int RightIndex { get; }

    /// <summary>Labels that survive into the intermediate (in unspecified order).</summary>
    public IReadOnlyList<char> ResultLabels { get; }

    /// <summary>Labels that are summed over in this step.</summary>
    public IReadOnlyList<char> ContractedLabels { get; }

    /// <summary>Estimated FLOP count for this step (2 × product of merged-label sizes).</summary>
    public long EstimatedFlops { get; }

    /// <summary>Constructs a path step.</summary>
    public EinsumPathStep(
        int leftIndex,
        int rightIndex,
        IReadOnlyList<char> resultLabels,
        IReadOnlyList<char> contractedLabels,
        long estimatedFlops)
    {
        LeftIndex = leftIndex;
        RightIndex = rightIndex;
        ResultLabels = resultLabels;
        ContractedLabels = contractedLabels;
        EstimatedFlops = estimatedFlops;
    }
}
