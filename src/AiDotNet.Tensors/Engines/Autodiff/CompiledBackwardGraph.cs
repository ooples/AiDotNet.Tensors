using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Pre-compiled backward execution plan for persistent gradient tapes.
/// Caches the tape traversal order and reachability analysis to avoid
/// recomputing it on every backward pass.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class CompiledBackwardGraph<T>
{
    /// <summary>
    /// Indices of tape entries that are reachable from the loss tensor,
    /// in reverse order (ready for backward traversal).
    /// </summary>
    private readonly int[] _reachableEntryIndices;

    /// <summary>
    /// The loss tensor this graph was compiled for.
    /// </summary>
    private readonly Tensor<T> _loss;

    /// <summary>
    /// Optional source filter — if non-null, only these tensors get gradients.
    /// </summary>
    private readonly Tensor<T>[]? _sources;

    /// <summary>
    /// The tape entries this graph operates on.
    /// </summary>
    private readonly List<TapeEntry<T>> _entries;

    private readonly IEngine _engine;

    /// <summary>
    /// Compiles a backward graph by analyzing which tape entries are reachable
    /// from the loss tensor. Dead entries are eliminated from the execution plan.
    /// </summary>
    public CompiledBackwardGraph(
        List<TapeEntry<T>> entries,
        Tensor<T> loss,
        Tensor<T>[]? sources,
        IEngine engine)
    {
        _entries = entries;
        _loss = loss;
        _sources = sources;
        _engine = engine;

        // Dead node elimination: find which entries are reachable from loss
        var reachable = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        reachable.Add(loss);

        var indices = new List<int>();
        for (int i = entries.Count - 1; i >= 0; i--)
        {
            if (reachable.Contains(entries[i].Output))
            {
                indices.Add(i);
                foreach (var input in entries[i].Inputs)
                    reachable.Add(input);
            }
        }

        _reachableEntryIndices = indices.ToArray();
    }

    /// <summary>
    /// Executes the compiled backward graph. Faster than uncompiled because
    /// dead entries are skipped and the traversal order is pre-computed.
    /// </summary>
    public Dictionary<Tensor<T>, Tensor<T>> Execute()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var grads = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);

        var onesData = new T[_loss.Length];
        for (int j = 0; j < onesData.Length; j++)
            onesData[j] = numOps.One;
        grads[_loss] = new Tensor<T>(onesData, _loss.Shape.ToArray());

        // Execute only reachable entries (dead node elimination applied)
        foreach (int i in _reachableEntryIndices)
        {
            var entry = _entries[i];

            if (!grads.TryGetValue(entry.Output, out var gradOutput))
                continue;

            entry.Backward(gradOutput, entry.Inputs, entry.Output,
                entry.SavedState ?? Array.Empty<object>(), _engine, grads);
        }

        if (_sources is not null)
        {
            var filtered = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
            foreach (var source in _sources)
            {
                if (grads.TryGetValue(source, out var grad))
                    filtered[source] = grad;
            }
            return filtered;
        }

        return grads;
    }

    /// <summary>
    /// Gets the number of entries that will be executed (after dead node elimination).
    /// </summary>
    public int ReachableEntryCount => _reachableEntryIndices.Length;

    /// <summary>
    /// Gets the number of entries that were eliminated (not reachable from loss).
    /// </summary>
    public int EliminatedEntryCount => _entries.Count - _reachableEntryIndices.Length;
}
