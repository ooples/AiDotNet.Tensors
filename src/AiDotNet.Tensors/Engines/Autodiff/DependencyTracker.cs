using System.Collections;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Tracks which tape entries depend on which parameters via BitArray maps.
/// Used by <see cref="SelectiveReplayContext{T}"/> to determine which entries
/// need re-execution when specific parameters change.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para><b>How this beats PyTorch:</b></para>
/// <list type="bullet">
/// <item>PyTorch's <c>create_graph=True</c> keeps ALL intermediates alive (massive memory)</item>
/// <item>Our approach: BitArray per parameter marks affected entries (12KB for 1000-entry × 100-param tape)</item>
/// <item>Computing dirty set: <c>BitArray.Or</c> across changed params (sub-microsecond)</item>
/// </list>
/// </remarks>
public sealed class DependencyTracker<T>
{
    private readonly Dictionary<Tensor<T>, BitArray> _paramToEntries;
    private readonly int _entryCount;

    /// <summary>
    /// Builds a dependency map from a tape's entries.
    /// Forward-scans the tape to find which parameters each entry transitively depends on.
    /// </summary>
    public DependencyTracker(IReadOnlyList<TapeEntry<T>> entries, IReadOnlyList<Tensor<T>> parameters)
    {
        _entryCount = entries.Count;
        _paramToEntries = new Dictionary<Tensor<T>, BitArray>(
            parameters.Count, ReferenceEqualityComparer<Tensor<T>>.Instance);

        // Initialize empty BitArrays for each parameter
        foreach (var param in parameters)
            _paramToEntries[param] = new BitArray(_entryCount);

        // Track which tensors depend on which parameters
        var tensorDeps = new Dictionary<Tensor<T>, HashSet<Tensor<T>>>(
            ReferenceEqualityComparer<Tensor<T>>.Instance);

        // Parameters depend on themselves
        foreach (var param in parameters)
            tensorDeps[param] = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance) { param };

        // Forward scan: propagate dependencies through the tape
        for (int i = 0; i < entries.Count; i++)
        {
            var entry = entries[i];
            var deps = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);

            foreach (var input in entry.Inputs)
            {
                if (tensorDeps.TryGetValue(input, out var inputDeps))
                    deps.UnionWith(inputDeps);
            }

            if (deps.Count > 0)
            {
                tensorDeps[entry.Output] = deps;

                // Mark this entry as dependent on each parameter in its dependency set
                foreach (var dep in deps)
                {
                    if (_paramToEntries.TryGetValue(dep, out var bits))
                        bits.Set(i, true);
                }
            }
        }
    }

    /// <summary>
    /// Computes the set of tape entry indices affected by changes to the given parameters.
    /// Uses BitArray.Or for sub-microsecond performance.
    /// </summary>
    public BitArray ComputeDirtyEntries(IReadOnlyList<Tensor<T>> changedParams)
    {
        var dirty = new BitArray(_entryCount);
        foreach (var param in changedParams)
        {
            if (_paramToEntries.TryGetValue(param, out var bits))
                dirty.Or(bits);
        }
        return dirty;
    }

    /// <summary>
    /// Gets the number of entries this tracker covers.
    /// </summary>
    public int EntryCount => _entryCount;
}
