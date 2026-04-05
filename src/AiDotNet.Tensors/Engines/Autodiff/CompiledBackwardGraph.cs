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
    private readonly TapeEntryArena<T> _entries;

    private readonly IEngine _engine;

    /// <summary>
    /// Compiles a backward graph by analyzing which tape entries are reachable
    /// from the loss tensor. Dead entries are eliminated from the execution plan.
    /// </summary>
    internal CompiledBackwardGraph(
        TapeEntryArena<T> entries,
        Tensor<T> loss,
        Tensor<T>[]? sources,
        IEngine engine)
    {
        // Arena reference is shared with the owning tape. Safe because:
        // 1. CompiledBackwardGraph is created inside ComputeGradients which holds the tape
        // 2. The tape's arena is not Reset() until tape.Dispose() which is after Execute()
        // 3. For persistent tapes, the arena is not cleared between ComputeGradients calls
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
                var e = entries[i];
                if (e.InputsOverflow is not null)
                {
                    foreach (var input in e.InputsOverflow) reachable.Add(input);
                }
                else
                {
                    reachable.Add(e.Input0);
                    if (e.InputCount >= 2 && e.Input1 is not null) reachable.Add(e.Input1);
                    if (e.InputCount >= 3 && e.Input2 is not null) reachable.Add(e.Input2);
                }
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

        // Assign grad indices for O(1) lookup (same as GradientTape.ComputeGradients)
        int gradIndexCount = 0;
        foreach (int i in _reachableEntryIndices)
        {
            ref var e = ref _entries[i];
            if (e.Output._gradIndex < 0) e.Output._gradIndex = gradIndexCount++;
            if (e.Input0 != null && e.Input0._gradIndex < 0) e.Input0._gradIndex = gradIndexCount++;
            if (e.InputCount >= 2 && e.Input1 != null && e.Input1._gradIndex < 0) e.Input1._gradIndex = gradIndexCount++;
            if (e.InputCount >= 3 && e.Input2 != null && e.Input2._gradIndex < 0) e.Input2._gradIndex = gradIndexCount++;
            if (e.InputsOverflow != null)
                foreach (var inp in e.InputsOverflow)
                    if (inp._gradIndex < 0) inp._gradIndex = gradIndexCount++;
        }

        var indexedGrads = new object?[gradIndexCount];
        DifferentiableOps.SetIndexedGrads(indexedGrads);

        var grads = new Dictionary<Tensor<T>, Tensor<T>>(
            Math.Min(gradIndexCount + 1, 1024),
            ReferenceEqualityComparer<Tensor<T>>.Instance);

        // Seed gradient
        Tensor<T> seedGrad;
        if (_loss.Length == 1)
        {
            seedGrad = new Tensor<T>(new[] { numOps.One }, new[] { 1 });
        }
        else
        {
            var onesData = new T[_loss.Length];
            var one = numOps.One;
            for (int j = 0; j < onesData.Length; j++) onesData[j] = one;
            seedGrad = new Tensor<T>(onesData, _loss._shape);
        }
        grads[_loss] = seedGrad;
        if (_loss._gradIndex >= 0) indexedGrads[_loss._gradIndex] = seedGrad;

        try
        {
            // Execute only reachable entries (dead node elimination applied)
            foreach (int i in _reachableEntryIndices)
            {
                ref var entry = ref _entries[i];

                if (!grads.TryGetValue(entry.Output, out var gradOutput))
                    continue;

                entry.ValidateInputVersions();
                var inputsArray = entry.GetInputsArray();
                entry.Backward(gradOutput, inputsArray, entry.Output,
                    entry.SavedState ?? Array.Empty<object>(), _engine, grads);
            }
        }
        finally
        {
            DifferentiableOps.ClearIndexedGrads();
            foreach (int i in _reachableEntryIndices)
            {
                ref var e = ref _entries[i];
                e.Output._gradIndex = -1;
                if (e.Input0 != null) e.Input0._gradIndex = -1;
                if (e.InputCount >= 2 && e.Input1 != null) e.Input1._gradIndex = -1;
                if (e.InputCount >= 3 && e.Input2 != null) e.Input2._gradIndex = -1;
                if (e.InputsOverflow != null)
                    foreach (var inp in e.InputsOverflow) inp._gradIndex = -1;
            }
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
