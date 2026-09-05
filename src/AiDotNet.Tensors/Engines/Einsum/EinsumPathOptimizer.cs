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
    /// <summary>Current typed contraction-order search-space version.</summary>
    public const int CurrentSearchSpaceVersion = 1;

    /// <summary>Current correctness and timing protocol for promoted contraction orders.</summary>
    public const int CurrentBenchmarkProtocolVersion = 1;

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
        foreach (var d in binding.BatchDims) batchProduct = checked(batchProduct * d);
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
            return new EinsumPath(
                Array.Empty<EinsumPathStep>(),
                0,
                new EinsumContractionOrder(Array.Empty<EinsumContractionPair>()),
                EinsumPathStrategy.Greedy);

        var steps = new List<EinsumPathStep>(live.Count - 1);
        var choices = new List<EinsumContractionPair>(live.Count - 1);
        long total = 0;

        while (live.Count > 1)
        {
            int bestI = -1;
            int bestJ = -1;
            HashSet<char>? bestResult = null;
            long bestCost = long.MaxValue;
            int bestRemoved = -1;

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
                long cost = checked(2L * ProductOfSizes(combined, sizes));

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

            HashSet<char> selectedResult = bestResult ?? throw new InvalidOperationException(
                "Greedy einsum planning failed to select a contraction pair.");

            // Labels contracted (summed) by this step = combined − result.
            var contracted = new HashSet<char>(live[bestI]);
            contracted.UnionWith(live[bestJ]);
            contracted.ExceptWith(selectedResult);

            var step = new EinsumPathStep(
                leftIndex: bestI,
                rightIndex: bestJ,
                resultLabels: selectedResult.ToArray(),
                contractedLabels: contracted.ToArray(),
                estimatedFlops: bestCost);
            steps.Add(step);
            choices.Add(new EinsumContractionPair(bestI, bestJ));
            total = checked(total + bestCost);

            // Replace the pair with the intermediate.
            // Remove higher index first so lower-index removal does not shift.
            live.RemoveAt(bestJ);
            live.RemoveAt(bestI);
            live.Add(selectedResult);
        }

        return new EinsumPath(
            steps,
            total,
            new EinsumContractionOrder(choices),
            EinsumPathStrategy.Greedy);
    }

    private static long ProductOfSizes(HashSet<char> labels, Dictionary<char, long> sizes)
    {
        long p = 1;
        foreach (var c in labels) p = checked(p * sizes[c]);
        return p;
    }

    /// <summary>
    /// Cache-aware path selection: returns a previously-recorded, fully
    /// validated contraction order or computes and records a greedy path.
    /// </summary>
    /// <remarks>
    /// Cache rows carry the actual typed pair sequence, not merely the name of
    /// the algorithm that produced it. Malformed, stale, or shape-incompatible
    /// rows fail closed and are replaced by a newly computed greedy path.
    /// </remarks>
    public static EinsumPath Optimize(EinsumShapeBinding binding) => Optimize(
        binding,
        KernelTuningDeviceFingerprint.CurrentCpu(),
        new KernelSearchSpaceVersion(CurrentSearchSpaceVersion),
        new KernelBenchmarkProtocolVersion(CurrentBenchmarkProtocolVersion));

    /// <summary>
    /// Selects a path for an exact execution device, search space, and
    /// benchmark protocol without allowing winners to cross those boundaries.
    /// </summary>
    public static EinsumPath Optimize(
        EinsumShapeBinding binding,
        KernelTuningDeviceFingerprint device,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion)
    {
        if (binding is null) throw new ArgumentNullException(nameof(binding));
        KernelTuningIdentity identity = EinsumPathCache.CreateIdentity(
            binding, device, searchSpaceVersion, benchmarkProtocolVersion);

        if (EinsumPathCache.TryLoad(binding, identity, out EinsumPath? cached) && cached is not null)
            return cached;

        EinsumPath path = Greedy(binding);
        EinsumPathCache.TryStore(binding, identity, path);
        return path;
    }

    /// <summary>Builds and validates a path from a typed pair sequence.</summary>
    internal static EinsumPath BuildPath(
        EinsumShapeBinding binding,
        EinsumContractionOrder order,
        EinsumPathStrategy strategy)
    {
        if (binding is null) throw new ArgumentNullException(nameof(binding));
        if (order is null) throw new ArgumentNullException(nameof(order));
        if (!Enum.IsDefined(typeof(EinsumPathStrategy), strategy))
            throw new ArgumentOutOfRangeException(nameof(strategy));

        EinsumEquation equation = binding.Equation;
        int expectedSteps = Math.Max(0, equation.Operands.Count - 1);
        if (order.Pairs.Count != expectedSteps)
            throw new ArgumentException(
                "A contraction order must contain exactly one fewer pair than operands.",
                nameof(order));

        var sizes = new Dictionary<char, long>(binding.LabelSizes.Count + 1);
        foreach (KeyValuePair<char, int> pair in binding.LabelSizes)
            sizes[pair.Key] = pair.Value;
        long batchProduct = 1L;
        foreach (int dimension in binding.BatchDims)
            batchProduct = checked(batchProduct * dimension);
        if (equation.HasEllipsis) sizes[EllipsisMarker] = batchProduct;

        var live = new List<HashSet<char>>(equation.Operands.Count);
        for (int i = 0; i < equation.Operands.Count; i++)
        {
            OperandLabels operand = equation.Operands[i];
            var labels = new HashSet<char>(operand.Labels);
            if (operand.HasEllipsis) labels.Add(EllipsisMarker);
            live.Add(labels);
        }

        var outputLabels = new HashSet<char>(equation.Output.Labels);
        if (equation.Output.HasEllipsis) outputLabels.Add(EllipsisMarker);
        var steps = new List<EinsumPathStep>(expectedSteps);
        long total = 0;

        foreach (EinsumContractionPair pair in order.Pairs)
        {
            if (pair.LeftIndex < 0 || pair.RightIndex <= pair.LeftIndex || pair.RightIndex >= live.Count)
                throw new ArgumentException(
                    "A contraction pair indexes outside the current live operand list.",
                    nameof(order));

            var combined = new HashSet<char>(live[pair.LeftIndex]);
            combined.UnionWith(live[pair.RightIndex]);
            var needed = new HashSet<char>(outputLabels);
            for (int i = 0; i < live.Count; i++)
            {
                if (i == pair.LeftIndex || i == pair.RightIndex) continue;
                needed.UnionWith(live[i]);
            }

            var resultLabels = new HashSet<char>(combined);
            resultLabels.IntersectWith(needed);
            var contractedLabels = new HashSet<char>(combined);
            contractedLabels.ExceptWith(resultLabels);
            long cost = checked(2L * ProductOfSizes(combined, sizes));
            steps.Add(new EinsumPathStep(
                pair.LeftIndex,
                pair.RightIndex,
                resultLabels.ToArray(),
                contractedLabels.ToArray(),
                cost));
            total = checked(total + cost);

            live.RemoveAt(pair.RightIndex);
            live.RemoveAt(pair.LeftIndex);
            live.Add(resultLabels);
        }

        return new EinsumPath(steps, total, order, strategy);
    }
}

/// <summary>Identifies how an einsum contraction order was selected.</summary>
public enum EinsumPathStrategy
{
    /// <summary>The deterministic greedy cost heuristic selected the order.</summary>
    Greedy = 0,

    /// <summary>An offline evolutionary benchmark selected the order.</summary>
    Evolutionary = 1,

    /// <summary>An external caller supplied the order.</summary>
    External = 2
}

/// <summary>One typed pair selection in the current live operand list.</summary>
public readonly record struct EinsumContractionPair
{
    /// <summary>Creates a canonical pair whose left index is smaller than its right index.</summary>
    public EinsumContractionPair(int leftIndex, int rightIndex)
    {
        if (leftIndex < 0) throw new ArgumentOutOfRangeException(nameof(leftIndex));
        if (rightIndex <= leftIndex) throw new ArgumentOutOfRangeException(nameof(rightIndex));
        LeftIndex = leftIndex;
        RightIndex = rightIndex;
    }

    /// <summary>Index of the first live operand.</summary>
    public int LeftIndex { get; }

    /// <summary>Index of the second live operand.</summary>
    public int RightIndex { get; }
}

/// <summary>An immutable, typed sequence of pairwise contraction choices.</summary>
public sealed class EinsumContractionOrder
{
    private readonly IReadOnlyList<EinsumContractionPair> _pairs;

    /// <summary>Creates an immutable snapshot of a pair sequence.</summary>
    public EinsumContractionOrder(IEnumerable<EinsumContractionPair> pairs)
    {
        if (pairs is null) throw new ArgumentNullException(nameof(pairs));
        _pairs = Array.AsReadOnly(pairs.ToArray());
    }

    /// <summary>Pair choices in execution order.</summary>
    public IReadOnlyList<EinsumContractionPair> Pairs => _pairs;
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

    /// <summary>Typed pair choices that reproduce this path.</summary>
    public EinsumContractionOrder ContractionOrder { get; }

    /// <summary>How the contraction order was selected.</summary>
    public EinsumPathStrategy Strategy { get; }

    /// <summary>Constructs a path.</summary>
    public EinsumPath(IReadOnlyList<EinsumPathStep> steps, long totalFlops)
        : this(
            steps,
            totalFlops,
            new EinsumContractionOrder((steps ?? throw new ArgumentNullException(nameof(steps)))
                .Select(step => new EinsumContractionPair(step.LeftIndex, step.RightIndex))),
            EinsumPathStrategy.External)
    {
    }

    internal EinsumPath(
        IReadOnlyList<EinsumPathStep> steps,
        long totalFlops,
        EinsumContractionOrder contractionOrder,
        EinsumPathStrategy strategy)
    {
        if (steps is null) throw new ArgumentNullException(nameof(steps));
        if (totalFlops < 0) throw new ArgumentOutOfRangeException(nameof(totalFlops));
        if (contractionOrder is null) throw new ArgumentNullException(nameof(contractionOrder));
        if (!Enum.IsDefined(typeof(EinsumPathStrategy), strategy))
            throw new ArgumentOutOfRangeException(nameof(strategy));
        Steps = Array.AsReadOnly(steps.ToArray());
        TotalFlops = totalFlops;
        ContractionOrder = contractionOrder;
        Strategy = strategy;
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
        if (leftIndex < 0) throw new ArgumentOutOfRangeException(nameof(leftIndex));
        if (rightIndex <= leftIndex) throw new ArgumentOutOfRangeException(nameof(rightIndex));
        if (resultLabels is null) throw new ArgumentNullException(nameof(resultLabels));
        if (contractedLabels is null) throw new ArgumentNullException(nameof(contractedLabels));
        if (estimatedFlops < 0) throw new ArgumentOutOfRangeException(nameof(estimatedFlops));
        LeftIndex = leftIndex;
        RightIndex = rightIndex;
        ResultLabels = Array.AsReadOnly(resultLabels.ToArray());
        ContractedLabels = Array.AsReadOnly(contractedLabels.ToArray());
        EstimatedFlops = estimatedFlops;
    }
}
