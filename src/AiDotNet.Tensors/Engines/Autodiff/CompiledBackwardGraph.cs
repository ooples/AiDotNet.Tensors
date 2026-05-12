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
    /// The loss tensor this graph was compiled for. Nullable because
    /// long-lived holders of the plan (e.g. the thread-static
    /// <c>AutoTrainingState</c> cache) can call
    /// <see cref="ReleaseCompilationTimeLoss"/> to drop this reference once
    /// they're committed to always passing a current-step loss into
    /// <see cref="Execute(Tensor{T})"/>. Strong references here pin the
    /// compilation-time forward chain (loss.GradFn → upstream
    /// intermediates) for the whole plan lifetime, which (combined with
    /// tensor-arena pool reuse of <see cref="Tensor{T}"/> instances across
    /// training iterations) causes every later step's intermediates to
    /// appear leaked under <see cref="GradientTapeLeakTests"/> /
    /// Issue #283. Production replay always feeds in the current-step
    /// loss; this field exists only for the parameterless
    /// <see cref="Execute()"/> overload used by tests that call
    /// <c>tape.CompileBackward(loss)</c> + <c>compiled.Execute()</c>
    /// inline (where the original loss is still in caller scope).
    /// </summary>
    private Tensor<T>? _loss;

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
    /// Optional reference to the owning tape's RetainGrad set. When non-null,
    /// the cleanup phase of <see cref="Execute"/> preserves <c>.Grad</c> on any
    /// tensor the user has explicitly marked with
    /// <see cref="GradientTape{T}.RetainGrad"/>, matching the behavior of
    /// <see cref="GradientTape{T}.CleanupTapeEntryGrad"/> and
    /// <see cref="GradientTape{T}.ComputeGradientsViaGraphCore"/>.
    /// </summary>
    private readonly HashSet<Tensor<T>>? _retainGrad;

    /// <summary>
    /// Compiles a backward graph by analyzing which tape entries are reachable
    /// from the loss tensor. Dead entries are eliminated from the execution plan.
    /// </summary>
    internal CompiledBackwardGraph(
        TapeEntryArena<T> entries,
        Tensor<T> loss,
        Tensor<T>[]? sources,
        IEngine engine,
        HashSet<Tensor<T>>? retainGrad = null)
    {
        // Arena reference is shared with the owning tape. Safe because:
        // 1. CompiledBackwardGraph is created inside ComputeGradients which holds the tape
        // 2. The tape's arena is not Reset() until tape.Dispose() which is after Execute()
        // 3. For persistent tapes, the arena is not cleared between ComputeGradients calls
        _entries = entries;
        _loss = loss;
        _sources = sources;
        _engine = engine;
        _retainGrad = retainGrad;

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
    /// When TensorCodec algebraic optimization is enabled and beneficial,
    /// delegates to OptimizedBackwardPlan for CSE + transposed BLAS.
    /// </summary>
    public Dictionary<Tensor<T>, Tensor<T>> Execute() => Execute(null);

    /// <summary>
    /// Executes the compiled backward graph with an optional current loss tensor.
    /// When currentLoss is provided, it replaces the compilation-time loss for gradient
    /// seeding. This is necessary because persistent tapes create new loss tensors each
    /// forward pass (via Reset + re-record), but the compiled graph's reachability indices
    /// were computed at compilation time.
    /// </summary>
    internal Dictionary<Tensor<T>, Tensor<T>> Execute(Tensor<T>? currentLoss)
    {
        var loss = currentLoss ?? _loss;
        if (loss is null)
            throw new InvalidOperationException(
                "CompiledBackwardGraph.Execute requires a current-step loss. " +
                "The compilation-time loss was released — call Execute(loss) " +
                "with the current step's loss tensor (this is the production " +
                "replay path; only the inline-test usage relies on the cached " +
                "compilation-time loss, which is still present at that call site).");

        // Phase C: try optimized backward with CSE + algebraic simplification.
        // OptimizedBackwardPlan applies the same RetainGrad-aware intermediate
        // .Grad cleanup as the non-optimized path below, so taking this branch
        // does not bypass the leak fix.
        if (Optimization.TensorCodecOptions.Current.EnableAlgebraicBackward)
        {
            var optimized = OptimizedBackwardPlan<T>.TryCreate(
                _entries, _reachableEntryIndices, loss, _sources, _engine, _retainGrad);
            if (optimized is not null)
                return optimized.Execute();
        }

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

        // Seed gradient — use current loss (may differ from compilation-time _loss
        // when persistent tapes Reset + re-record between steps)
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
        if (loss._gradIndex >= 0) indexedGrads[loss._gradIndex] = seedGrad;

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

            // Persistent-tape parity with the GradientTape.ComputeGradientsViaGraphCore
            // cleanup: clear .GradFn AND .Grad on forward intermediates so they don't
            // get pinned across iterations. CompiledBackwardGraph walks the tape via
            // the arena's entry array (`_entries[i]` for `i in _reachableEntryIndices`)
            // — it does NOT follow GradFn pointers — so nulling GradFn here doesn't
            // break the next Execute call's traversal. The next forward will re-set
            // GradFn on every intermediate before the next backward runs, restoring
            // any back-pointers a consumer of the public `tensor.GradFn` API might
            // expect. Leaving GradFn live across Execute calls is what previously
            // made every step's intermediates appear leaked under
            // GradientTapeLeakTests — the GradFn back-chain kept compilation-step
            // intermediates (which are the same Tensor instances as later iters'
            // intermediates after tensor-arena pool reuse) reachable from any
            // consumer that holds even one downstream tensor.
            HashSet<Tensor<T>>? sourceSet = null;
            if (_sources is not null)
            {
                sourceSet = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
                foreach (var s in _sources) sourceSet.Add(s);
            }

            foreach (int i in _reachableEntryIndices)
            {
                ref var e = ref _entries[i];
                e.Output._gradIndex = -1;
                if (e.Input0 != null) e.Input0._gradIndex = -1;
                if (e.InputCount >= 2 && e.Input1 != null) e.Input1._gradIndex = -1;
                if (e.InputCount >= 3 && e.Input2 != null) e.Input2._gradIndex = -1;
                if (e.InputsOverflow != null)
                    foreach (var inp in e.InputsOverflow) inp._gradIndex = -1;

                // .GradFn / .Grad cleanup on this entry's output (every output is an
                // intermediate — graph leaves never appear as outputs in the entries
                // array). Inputs are NOT cleared because they may be graph leaves
                // (parameters), and the consumer relies on `param.Grad` being
                // populated after a sources=null Execute.
                //
                // Matches the parity rule used by
                // GradientTape.CleanupTapeEntryGrad (line 635) and
                // ComputeGradientsViaGraphCore (line 719): a tensor explicitly
                // marked with RetainGrad() must keep its .Grad even though it
                // is a graph intermediate.
                bool keepGrad =
                    (sourceSet?.Contains(e.Output) == true)
                    || (_retainGrad is not null && _retainGrad.Contains(e.Output));
                e.Output.GradFn = null;
                if (!keepGrad) e.Output.Grad = null;
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
    /// Drops the strong reference to the compilation-time loss tensor.
    /// Long-lived plan holders (the thread-static
    /// <c>AutoTrainingCompiler</c> cache) call this immediately after
    /// storing the plan so the compilation-time forward chain can be GCed
    /// once user code drops it. Subsequent <see cref="Execute(Tensor{T})"/>
    /// calls must pass a current-step loss — production replay always does.
    /// The parameterless <see cref="Execute()"/> overload throws after this
    /// is called, which is fine: production never uses it.
    /// </summary>
    internal void ReleaseCompilationTimeLoss() => _loss = null;

    /// <summary>
    /// Gets the number of entries that will be executed (after dead node elimination).
    /// </summary>
    public int ReachableEntryCount => _reachableEntryIndices.Length;

    /// <summary>
    /// Gets the number of entries that were eliminated (not reachable from loss).
    /// </summary>
    public int EliminatedEntryCount => _entries.Count - _reachableEntryIndices.Length;
}
