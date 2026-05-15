using System.Diagnostics;
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

    /// <summary>
    /// Thread-local per-arity inputs buffers (length 1 / 2 / 3) reused
    /// across every entry in a single <see cref="TryExecute"/> walk.
    /// <see cref="TapeEntry{T}.GetInputsArrayInto"/> returns the buffer
    /// whose <c>.Length</c> matches the entry's <c>InputCount</c>, or
    /// the entry's <see cref="TapeEntry{T}.InputsOverflow"/> array for
    /// ≥4 inputs. The arity-matched length preserves the existing
    /// <c>BackwardFunction&lt;T&gt;</c> contract that <c>inputs.Length
    /// == InputCount</c> (many backward functions branch on
    /// <c>inputs.Length</c>). Allocated lazily on first call per thread.
    /// </summary>
    [ThreadStatic] private static Tensor<T>[]? s_inputsBuffer1;
    [ThreadStatic] private static Tensor<T>[]? s_inputsBuffer2;
    [ThreadStatic] private static Tensor<T>[]? s_inputsBuffer3;

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
    /// <para>
    /// When <see cref="CompiledBackwardWalk{T}.Enabled"/> is true, also
    /// captures each entry's static backward <see cref="System.Reflection.MethodInfo"/>
    /// from <paramref name="entries"/> and registers a per-op-specialised
    /// IL walker for the pattern (issue #338 Item 3). The captured methods
    /// are baked into the walker's emitted IL as direct <c>call</c> targets
    /// — no <c>BackwardFunction&lt;T&gt;.Invoke</c> dispatch indirection
    /// at replay time.
    /// </para>
    /// </summary>
    internal static void Store(long patternHash, int recordedEntryCount, int[] reverseTopoIndices, TapeEntryArena<T>? entries = null)
    {
        _cachedPatternHash = patternHash;
        _cachedRecordedEntryCount = recordedEntryCount;
        _cachedReverseTopoIndices = reverseTopoIndices;

        // Issue #338 Item 3: also register a compiled-backward walker
        // keyed by the same pattern hash so the
        // GradientTape.ComputeGradientsViaGraphCore IL-walker fast path
        // (gated on AIDOTNET_COMPILED_BACKWARD) finds it.
        if (CompiledBackwardWalk<T>.Enabled)
        {
            System.Reflection.MethodInfo[]? methods = null;
            if (entries is not null)
                methods = ExtractBackwardMethods(entries, reverseTopoIndices);

            CompiledBackwardWalk<T>.Register(
                patternHash,
                CompiledBackwardWalk<T>.Compile(reverseTopoIndices, methods));
        }
    }

    /// <summary>
    /// Walks the entry arena along the reverse-topo index sequence and
    /// extracts each entry's backward <see cref="System.Reflection.MethodInfo"/>
    /// (from its <c>Backward</c> delegate's <see cref="System.Delegate.Method"/>).
    /// Returns null when ANY entry's backward delegate is non-static —
    /// the IL specialisation path requires static methods, and a null
    /// return tells <see cref="CompiledBackwardWalk{T}.Compile(int[], System.Reflection.MethodInfo[]?)"/>
    /// to take the non-specialised helper-dispatch path.
    /// </summary>
    private static System.Reflection.MethodInfo[]? ExtractBackwardMethods(
        TapeEntryArena<T> entries, int[] reverseTopoIndices)
    {
        var methods = new System.Reflection.MethodInfo[reverseTopoIndices.Length];
        for (int i = 0; i < reverseTopoIndices.Length; i++)
        {
            int idx = reverseTopoIndices[i];
            if ((uint)idx >= (uint)entries.Count) return null;
            ref var entry = ref entries[idx];
            if (entry.Backward is null) return null;
            var m = entry.Backward.Method;
            if (!m.IsStatic) return null;
            methods[i] = m;
        }
        return methods;
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
        //
        // Reuse a single per-walk Tensor<T>[3] for the inputs argument so
        // the steady-state replay doesn't allocate a fresh 1-3 element
        // Tensor<T>[] per op (entry.GetInputsArray() does that, and on a
        // 100-op tape that's 100 small-array allocs per Train step —
        // exactly the GC churn that motivated the cached-replay path in
        // the first place). Safe because (a) this replay path is only
        // reached when !_isBackwardCreateGraph (no re-entrant backward
        // can clobber the buffer mid-walk) and (b) `Backward` consumes
        // inputs synchronously — no backward function in the codebase
        // retains the inputs[] reference past its own return.
        var inputsBuffer1 = s_inputsBuffer1 ??= new Tensor<T>[1];
        var inputsBuffer2 = s_inputsBuffer2 ??= new Tensor<T>[2];
        var inputsBuffer3 = s_inputsBuffer3 ??= new Tensor<T>[3];
        bool timing = BackwardTiming.Enabled;
        var indices = _cachedReverseTopoIndices;
        try
        {
            for (int i = 0; i < indices.Length; i++)
            {
                int idx = indices[i];
                // Out-of-range index signals a stale cache — entry count
                // dropped between Store and TryExecute. The pre-loop
                // signature check should have caught this; if we reach
                // here, the cache key is no longer valid and silent skip
                // would produce wrong gradients. Bail out so the caller
                // falls back to the fresh-walk path.
                if (idx < 0 || idx >= currentEntries.Count) return null;
                ref var entry = ref currentEntries[idx];

                if (!grads.TryGetValue(entry.Output, out var gradOutput))
                    continue;

                long start = timing ? Stopwatch.GetTimestamp() : 0;
                entry.Backward(
                    gradOutput,
                    entry.GetInputsArrayInto(inputsBuffer1, inputsBuffer2, inputsBuffer3),
                    entry.Output,
                    entry.SavedState ?? Array.Empty<object>(),
                    engine,
                    grads);
                if (timing)
                {
                    long ticks = Stopwatch.GetTimestamp() - start;
                    BackwardTiming.Record(entry.Backward.Method.Name, ticks);
                }
            }
        }
        finally
        {
            // Clear every per-arity buffer's references so a forward
            // intermediate that just rode through here (as an Input
            // field of some entry) doesn't get pinned in the thread-
            // static buffer for the rest of the worker thread's
            // lifetime. The GradientTapeLeakTests gradient-cleanup
            // assertions catch exactly this kind of cross-call retention.
            inputsBuffer1[0] = null!;
            inputsBuffer2[0] = null!;
            inputsBuffer2[1] = null!;
            inputsBuffer3[0] = null!;
            inputsBuffer3[1] = null!;
            inputsBuffer3[2] = null!;
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
