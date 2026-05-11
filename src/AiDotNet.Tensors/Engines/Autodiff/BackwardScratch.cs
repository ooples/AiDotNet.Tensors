using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Thread-local scratch buffers used by the backward dispatch path
/// in <see cref="GradientTape{T}.ComputeGradientsViaGraphCore"/> and
/// <see cref="CompiledDelegateChain{T}.Execute"/>.
/// </summary>
/// <remarks>
/// <para>
/// Issue #319 — the per-iter backward setup allocated five fresh
/// objects per call: visited <c>HashSet</c>, topo-order <c>List</c>,
/// source <c>HashSet</c>, gradient <c>Dictionary</c>, and a
/// <see cref="BackwardStep{T}"/>-array. On 30-iter training loops
/// at ViT-Base scale that's ~150 collection allocations / second of
/// pure Gen-0 pressure independent of the actual compute. This class
/// reuses one of each per (thread, T) across backward calls.
/// </para>
/// <para>
/// <b>Recursion safety:</b> nested backwards (Hessian-vector
/// products via <c>createGraph: true</c>) cannot reuse the same
/// buffers — the inner call would clobber the outer call's state.
/// The thread-local <see cref="_inUse"/> flag forces the inner call
/// to fall back to fresh allocation. The pool retains its capacity
/// either way; only the inner call eats one extra alloc.
/// </para>
/// </remarks>
internal static class BackwardScratch<T>
{
    [ThreadStatic] private static HashSet<GradNode<T>>? _visited;
    [ThreadStatic] private static List<GradNode<T>>? _topoOrder;
    [ThreadStatic] private static BackwardStep<T>[]? _steps;
    // Live range of _steps populated by the most recent RentSteps caller.
    // Release() clears only [0, _stepsLive) instead of the full array,
    // preserving the headroom-grow benefit when the next backward needs
    // a similarly-sized buffer. PR #322 review #16 + #18.
    [ThreadStatic] private static int _stepsLive;
    [ThreadStatic] private static HashSet<Tensor<T>>? _sourceSet;
    [ThreadStatic] private static Dictionary<Tensor<T>, Tensor<T>>? _grads;
    [ThreadStatic] private static bool _inUse;

    /// <summary>
    /// Cached seed gradient (ones tensor) keyed by Length. Only one
    /// entry is retained at a time — most training loops repeatedly
    /// compute backward on a loss of the same shape, so a single-
    /// element cache covers the steady state. Mismatches drop the
    /// cache and re-allocate.
    /// </summary>
    [ThreadStatic] private static Tensor<T>? _cachedSeed;
    [ThreadStatic] private static int _cachedSeedLength;

    /// <summary>
    /// Acquires the scratch pool for the current backward call.
    /// Returns false when a nested backward is already in flight on
    /// this thread (caller must fall back to fresh allocation).
    /// </summary>
    internal static bool TryAcquire()
    {
        if (_inUse) return false;
        _inUse = true;
        return true;
    }

    /// <summary>
    /// Releases the scratch lock and clears all reference-holding
    /// buffers. Clearing on release (not on rent) is what makes
    /// <c>GradientTapeLeakTests</c> happy — those tests assert that
    /// after backward returns, no forward-pass tensor remains
    /// reachable. If we deferred clearing to the next Rent, the pool
    /// would hold those references between backwards.
    /// </summary>
    internal static void Release()
    {
        // Capacity is retained — Clear() does not shrink the backing
        // arrays — so the next backward gets the warm cache without
        // tape-tensor refs surviving across calls.
        _visited?.Clear();
        _topoOrder?.Clear();
        _sourceSet?.Clear();
        _grads?.Clear();
        // Cached reshape view holds a small Tensor wrapper — fine to
        // retain across calls when the loss shape is stable (the
        // typical training-loop pattern). RentSeedGradient invalidates
        // this whenever a fresh ones-array is created or the requested
        // shape no longer matches, so no stale-view risk.
        if (_steps is not null && _stepsLive > 0)
        {
            // Only clear the live range. Avoids the O(capacity) cost
            // that would otherwise erase the headroom-grow win (PR #322
            // review #16, #18). ClearStepsRange may already have
            // cleared a subset; clearing again here is idempotent.
            Array.Clear(_steps, 0, Math.Min(_stepsLive, _steps.Length));
            _stepsLive = 0;
        }
        _inUse = false;
    }

    internal static HashSet<GradNode<T>> RentVisited()
    {
        return _visited ??= new HashSet<GradNode<T>>();
    }

    internal static List<GradNode<T>> RentTopoOrder()
    {
        return _topoOrder ??= new List<GradNode<T>>();
    }

    /// <summary>
    /// Returns a <see cref="BackwardStep{T}"/> array of length
    /// >= <paramref name="needed"/>. Caller must treat indices
    /// <c>[0, needed)</c> as the live range; the trailing slots may
    /// contain stale references from a previous backward, so caller
    /// must overwrite them before reading or clear them on release.
    /// </summary>
    internal static BackwardStep<T>[] RentSteps(int needed)
    {
        if (_steps is null || _steps.Length < needed)
        {
            // Grow with headroom so subsequent calls with similar
            // topology don't re-allocate on a fencepost shift.
            int newCap = Math.Max(needed, _steps?.Length * 2 ?? 16);
            _steps = new BackwardStep<T>[newCap];
        }
        // Track how many slots the caller will populate. Release()
        // uses this to clear ONLY the live range, not the full
        // capacity. PR #322 review #16, #18.
        _stepsLive = needed;
        return _steps;
    }

    internal static HashSet<Tensor<T>> RentSourceSet()
    {
        return _sourceSet ??= new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
    }

    internal static Dictionary<Tensor<T>, Tensor<T>> RentGrads(int capacityHint)
    {
        if (_grads is null)
        {
            _grads = new Dictionary<Tensor<T>, Tensor<T>>(
                Math.Max(16, capacityHint),
                ReferenceEqualityComparer<Tensor<T>>.Instance);
        }
#if NET5_0_OR_GREATER
        else
        {
            // EnsureCapacity grows the internal arrays without
            // discarding them. Cleared on Release(), so the dict
            // arrives empty here.
            _grads.EnsureCapacity(Math.Max(16, capacityHint));
        }
#endif
        return _grads;
    }

    /// <summary>
    /// Returns a seed gradient (ones tensor) of the requested
    /// length. Cached per-thread so repeated backwards on a loss of
    /// the same shape skip the ones-array allocation.
    /// </summary>
    internal static Tensor<T> RentSeedGradient(int[] lossShape)
    {
        int length = 1;
        for (int i = 0; i < lossShape.Length; i++) length *= lossShape[i];

        // PR #322 review #19, #21: the cache holds a 1-D ones tensor;
        // matching it to lossShape requires a Reshape. Reshape returns
        // a view (zero-copy on data) but does allocate a Tensor<T>
        // wrapper object. We avoid that wrapper alloc on the steady-
        // state path by also caching the last-reshaped view per shape.
        //
        // Reshape's GradFn-attaching behavior is safe here:
        //   * Common backward (createGraph=false): ComputeGradientsViaGraph
        //     suspends the current tape before calling chain.Execute, so
        //     Tensor.Reshape sees Current==null and does not set GradFn.
        //   * Higher-order AD (createGraph=true): the tape is NOT
        //     suspended; the seed legitimately needs a recorded GradFn
        //     so the outer Hvp/Hessian pass can walk back through it.
        // The cached view is invalidated whenever lossShape changes,
        // so a stale GradFn from a prior createGraph=true call cannot
        // bleed into a subsequent createGraph=false call (the latter
        // would re-Reshape and the suspended tape produces a clean
        // GradFn-free view).
        if (_cachedSeed is null || _cachedSeedLength != length)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            var data = new T[length];
            T one = numOps.One;
            for (int i = 0; i < length; i++) data[i] = one;
            _cachedSeed = new Tensor<T>(data, new[] { length });
            _cachedSeedLength = length;
            _cachedSeedReshapedView = null;
            _cachedSeedReshapedShape = null;
        }

        // 1-D direct match: return the underlying cached tensor.
        if (lossShape.Length == 1 && lossShape[0] == length)
            return _cachedSeed;

        // Higher-dim: try to reuse the prior reshape if the shape matches.
        if (_cachedSeedReshapedView is not null && _cachedSeedReshapedShape is not null
            && ShapeMatches(_cachedSeedReshapedShape, lossShape))
        {
            return _cachedSeedReshapedView;
        }

        // First call for this shape (or shape changed): reshape and
        // cache. Use NoGradScope to suspend any active tape across the
        // Reshape call so the cached view does NOT carry a GradFn —
        // RentSeedGradient must produce a graph-free constant. The
        // outer caller (ComputeGradientsViaGraph) already suspends the
        // tape in the createGraph=false case; this is a defensive
        // belt-and-braces for the createGraph=true path or any
        // non-standard caller.
        using (GradientTape<T>.NoGrad())
        {
            _cachedSeedReshapedView = _cachedSeed.Reshape(lossShape);
        }
        _cachedSeedReshapedShape = (int[])lossShape.Clone();
        return _cachedSeedReshapedView;
    }

    [ThreadStatic] private static Tensor<T>? _cachedSeedReshapedView;
    [ThreadStatic] private static int[]? _cachedSeedReshapedShape;

    private static bool ShapeMatches(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>
    /// Clears the steps array's live range so retained references
    /// don't extend tensor lifetimes past the backward call. Should
    /// be invoked once the chain is no longer needed (i.e. when the
    /// chain wrapper is dropped on the non-persistent path).
    /// </summary>
    internal static void ClearStepsRange(int count)
    {
        if (_steps is null) return;
        // Array.Clear is the cheapest way to zero a value-type span;
        // it null-writes the reference fields inside the BackwardStep
        // struct, releasing the captured Tensor<T> and BackwardFunction
        // references.
        Array.Clear(_steps, 0, Math.Min(count, _steps.Length));
    }
}
