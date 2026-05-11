using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Thread-local cache of forward-pattern → reverse-topo entry-index
/// arrays. Closes the fresh-tape backward-walk gap surfaced by issue
/// #327: when a consumer's training loop creates a new
/// <see cref="GradientTape{T}"/> on every step and runs the same forward
/// pattern each time, the first call computes the topological backward
/// order via DFS and stores it here; subsequent calls hit the cache,
/// skip DFS + step-build, and replay by walking the current tape's
/// entries by index.
///
/// <para>The cached data is purely structural — <c>int[]</c> of entry
/// indices in reverse-execution order — and contains no tensor
/// references. Every tensor used during replay comes from the live tape
/// arena, so disposing the original tape doesn't invalidate the cached
/// plan. Pattern hash + recorded entry count form the cache key; either
/// mismatch falls back to the full DFS path.</para>
///
/// <para><b>Critical sequencing</b>: the cache must be populated AFTER
/// <see cref="CompiledDelegateChain{T}.Execute"/> completes — building it
/// before Execute mutates state that breaks higher-order AD
/// (Hvp / Hessian / vmap). See the bisection notes in
/// <see cref="GradientTape{T}.ComputeGradientsViaGraphCore"/>.</para>
/// </summary>
internal static class RebindablePlanCache<T>
{
    [ThreadStatic]
    private static long _cachedPatternHash;

    [ThreadStatic]
    private static int _cachedRecordedEntryCount;

    [ThreadStatic]
    private static int[]? _cachedReverseTopoIndices;

    /// <summary>Quick-check for cache presence without a full lookup.</summary>
    internal static bool IsEmpty => _cachedReverseTopoIndices is null;

    /// <summary>Validates the cached signature against the caller's pattern.</summary>
    internal static bool TrySignature(long patternHash, int currentEntryCount)
        => _cachedReverseTopoIndices is not null
           && _cachedPatternHash == patternHash
           && _cachedRecordedEntryCount == currentEntryCount;

    /// <summary>
    /// Stores the reverse-topo entry-index sequence for the most recent
    /// forward pattern on this thread.
    /// </summary>
    internal static void Store(long patternHash, int recordedEntryCount, int[] reverseTopoIndices)
    {
        _cachedPatternHash = patternHash;
        _cachedRecordedEntryCount = recordedEntryCount;
        _cachedReverseTopoIndices = reverseTopoIndices;
    }

    /// <summary>
    /// Replays the cached backward order on the given tape's live entry
    /// arena. Returns the gradient dictionary (filtered to
    /// <paramref name="sources"/> when non-null), or <c>null</c> when
    /// the cache miss-signature does not match.
    /// </summary>
    internal static Dictionary<Tensor<T>, Tensor<T>>? TryExecute(
        long patternHash,
        TapeEntryArena<T> currentEntries,
        Tensor<T> loss,
        IReadOnlyList<Tensor<T>>? sources,
        IEngine engine)
    {
        if (_cachedReverseTopoIndices is null) return null;
        if (_cachedPatternHash != patternHash) return null;
        if (_cachedRecordedEntryCount != currentEntries.Count) return null;

        var numOps = MathHelper.GetNumericOperations<T>();

        var grads = new Dictionary<Tensor<T>, Tensor<T>>(
            _cachedReverseTopoIndices.Length + 1,
            ReferenceEqualityComparer<Tensor<T>>.Instance);

        // Seed gradient — fresh per call, sized to the current iteration's
        // loss shape. The cached plan never holds the recording-time seed
        // (different tensor object per call), so this is always a new
        // allocation. Sub-microsecond cost; not pool-worthy.
        Tensor<T> seedGrad;
        if (loss.Length == 1)
        {
            seedGrad = new Tensor<T>(new[] { numOps.One }, (int[])loss._shape.Clone());
        }
        else
        {
            var onesData = new T[loss.Length];
            var one = numOps.One;
            for (int j = 0; j < onesData.Length; j++) onesData[j] = one;
            seedGrad = new Tensor<T>(onesData, loss._shape);
        }
        grads[loss] = seedGrad;

        // Walk the cached reverse-topo indices, reading entries from the
        // CURRENT arena. entry.Output is this iteration's tensor, not the
        // recording-time tensor — that's the rebinding semantics.
        var indices = _cachedReverseTopoIndices;
        for (int i = 0; i < indices.Length; i++)
        {
            int idx = indices[i];
            if (idx < 0 || idx >= currentEntries.Count) continue;
            ref var entry = ref currentEntries[idx];

            if (!grads.TryGetValue(entry.Output, out var gradOutput))
                continue;

            entry.Backward(
                gradOutput,
                entry.GetInputsArray(),
                entry.Output,
                entry.SavedState ?? Array.Empty<object>(),
                engine,
                grads);
        }

        // Source filter — same semantics as CompiledDelegateChain.Execute.
        if (sources is not null)
        {
            var filtered = new Dictionary<Tensor<T>, Tensor<T>>(
                sources.Count, ReferenceEqualityComparer<Tensor<T>>.Instance);
            foreach (var source in sources)
                if (grads.TryGetValue(source, out var grad))
                    filtered[source] = grad;
            return filtered;
        }

        return grads;
    }

    /// <summary>Clears the thread-local cache (test isolation).</summary>
    internal static void ResetForTests()
    {
        _cachedPatternHash = 0;
        _cachedRecordedEntryCount = 0;
        _cachedReverseTopoIndices = null;
    }
}
