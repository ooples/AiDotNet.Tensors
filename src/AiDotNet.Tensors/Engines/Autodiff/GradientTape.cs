using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Records tensor operations for reverse-mode automatic differentiation.
/// When a GradientTape is active (via <c>using</c>), engine operations automatically
/// record their forward computation and backward functions, enabling gradient computation.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
/// <remarks>
/// <para>Follows the ambient context pattern (like PyTorch's autograd). Operations check
/// <see cref="Current"/> and record only when a tape is active — zero overhead otherwise.</para>
/// <para>Supports nesting: inner tapes record independently and restore the parent on dispose.</para>
/// <para><b>Usage:</b></para>
/// <code>
/// using (var tape = new GradientTape&lt;float&gt;())
/// {
///     var z = engine.TensorMatMul(x, w);    // Recorded
///     var y = engine.ReLU(z);                // Recorded
///     var grads = tape.ComputeGradients(y, sources: new[] { w });
///     var dW = grads[w];
/// }
/// </code>
/// </remarks>
public sealed class GradientTape<T> : IDisposable
{
    /// <summary>
    /// Thread-local current gradient tape for ambient context pattern.
    /// When a tape is active, engine operations record to it automatically.
    /// </summary>
    [ThreadStatic]
    private static GradientTape<T>? _current;

    /// <summary>
    /// Gets the current gradient tape for this thread, if any.
    /// Engine operations check this to decide whether to record.
    /// </summary>
    public static GradientTape<T>? Current => _current;

    /// <summary>
    /// Sets the current tape for this thread. Used by instance methods
    /// to modify the thread-local static field.
    /// </summary>
    private static void SetCurrentTape(GradientTape<T>? tape) => _current = tape;

    /// <summary>
    /// Test-isolation hook: clears this thread's current-tape reference for
    /// <typeparamref name="T"/>. An undisposed tape pins <see cref="_current"/>
    /// (a ThreadStatic strong ref the finalizer cannot reach), so the harness
    /// clears it between tests. Not part of the public API.
    /// </summary>
    internal static void ResetCurrentForTests() => _current = null;

    private readonly GradientTape<T>? _parent;
    private readonly TapeEntryArena<T> _entries;
    private readonly GradientTapeOptions _options;
    private IEngine _engine;
    private bool _engineExplicitlyBound;
    private readonly bool _savedReplayMode; // Saved ReplayMode from outer scope for nested tapes
    private bool _disposed;
    // Deterministic per-step activation lifetime (GPU): the DirectGpu activation
    // cache that an outermost tape's forward populates is released wholesale on
    // Dispose via EvictActivationsCreatedAfter(_activationSnapshot), giving ~one-step
    // steady-state memory (PyTorch/JAX semantics) instead of leaning on the LRU byte
    // cap to drain a backlog. Only the engine instance captured at construction is
    // touched, and only entries newer than the snapshot — pre-existing cross-tape
    // intermediates are preserved. Null engine / non-GPU / nested tape => no-op.
    private readonly Engines.DirectGpuTensorEngine? _snapshotEngine;
    private readonly long _activationSnapshot;


    /// <summary>
    /// Gets the number of operations recorded on this tape.
    /// </summary>
    public int EntryCount => _entries.Count;

    /// <summary>Internal access to tape entries for compiled backward hash validation.</summary>
    internal TapeEntryArena<T> Entries => _entries;

    /// <summary>
    /// Removes the last entry from the tape. Used by fused operations to replace
    /// individual entries with a single fused entry.
    /// </summary>
    internal void RemoveLastEntry()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GradientTape<T>));
        _entries.RemoveLast();
    }

    /// <summary>
    /// Gets the options for this tape.
    /// </summary>
    public GradientTapeOptions Options => _options;

    /// <summary>
    /// Gets the engine captured at tape construction time.
    /// This ensures backward passes use the same engine as the forward pass.
    /// </summary>
    public IEngine Engine => _engine;

    /// <summary>
    /// Binds this tape to <paramref name="engine"/> if no engine has yet been
    /// explicitly bound. Idempotent — once bound, subsequent calls are
    /// no-ops. Engine recording paths (e.g. <see cref="CpuEngine.BatchNorm{T}"/>)
    /// call this with <c>this</c> so the backward walk dispatches to the same
    /// engine instance the user invoked the forward op on. Closes #350.
    /// </summary>
    public void BindEngineIfUnset(IEngine engine)
    {
        if (_engineExplicitlyBound) return;
        if (engine is null) return;
        _engine = engine;
        _engineExplicitlyBound = true;
    }

    /// <summary>
    /// Gets whether this tape has been disposed.
    /// </summary>
    public bool IsDisposed => _disposed;

    /// <summary>
    /// Creates a new gradient tape and sets it as the current tape for this thread.
    /// The previous tape (if any) is saved and restored when this tape is disposed.
    /// </summary>
    /// <param name="options">Optional configuration. Uses <see cref="GradientTapeOptions.Default"/> if null.</param>
    // Thread-local arena cache — reused across training steps to avoid per-step allocation.
    // The arena's backing array grows once during warmup, then reuses indefinitely.
    [ThreadStatic]
    private static TapeEntryArena<T>? _cachedArena;

    // Cached scalar seed gradient (ones tensor of shape [1]) — reused across training steps
    [ThreadStatic]
    private static Tensor<T>? _cachedScalarSeed;

    // #1624: release each node's activation references during the streaming
    // backward as the reverse walk consumes them, bounding the live activation set
    // to the frontier instead of the whole forward (the deep-model OOM). On by
    // default; AIDOTNET_STREAMING_RELEASE_ACTIVATIONS=0 disables it as a safety
    // hatch. Only affects ComputeGradientsStreaming (the memory-bounded path).
    internal static bool ReleaseStreamingActivations { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_STREAMING_RELEASE_ACTIVATIONS") != "0";

    public GradientTape(GradientTapeOptions? options = null)
    {
        _options = options ?? GradientTapeOptions.Default;
        // Reuse cached arena if available, otherwise create new one
        _entries = _cachedArena ?? new TapeEntryArena<T>();
        _cachedArena = null; // Take ownership — will return on Dispose
        _entries.Reset();
        // Default to AiDotNetEngine.Current; the first op recorded onto the
        // tape rebinds via BindEngineIfUnset to the engine instance the user
        // actually invoked the op on. Same rationale as LazyTensorScope's
        // engine binding (see issue #350): on auto-detect-GPU systems
        // AiDotNetEngine.Current is DirectGpuTensorEngine, but tests or
        // production code that explicitly creates a new CpuEngine() and
        // calls operations on it expects backward to also run on CPU. Without
        // this rebind the eager-tape backward and compiled backward dispatch
        // to different engines, producing per-element double-precision
        // divergences in the 1e-7 range that look like FMA noise but are
        // actually two distinct backward kernels racing on different devices.
        _engine = AiDotNetEngine.Current;
        _engineExplicitlyBound = false;
        _parent = _current;
        _savedReplayMode = Compilation.AutoTrainingCompiler.ReplayMode;

        // Capture the activation-cache baseline for the OUTERMOST tape on a GPU engine.
        // On Dispose, every activation cached at a higher timestamp (i.e. produced by
        // this forward+backward) is released deterministically — see _snapshotEngine.
        // Captured from the construction-time engine; only used at Dispose if the tape
        // still bound to that same instance (guards the rare CPU<->GPU rebind, #350).
        if (_parent is null && _engine is Engines.DirectGpuTensorEngine snapEngine)
        {
            _snapshotEngine = snapEngine;
            _activationSnapshot = snapEngine.ActivationCacheTimestampSnapshot();

            // Atomic-step semantics for EAGER training (PR #586 follow-up). The
            // compiled-plan StepEager already brackets its forward+backward with
            // SuspendActivationEviction so the byte-cap eviction path can't run
            // mid-step and free buffers a still-queued kernel will reference
            // (#226-class CUDA error 700). Eager training under a tape needs the
            // same atomicity: the OUTERMOST tape on a GPU engine matches the
            // compiled plan's "one step" granularity — every activation it caches
            // is consumed by its own backward, and freeing them mid-step is the
            // race the PR body documents (4 GB cap exposes it; 6 GB workaround
            // hides it). The deterministic per-step eviction
            // (EvictActivationsCreatedAfter, above-Dispose) takes over the
            // memory-bounding role that mid-step eviction was trying to play.
            snapEngine.SuspendActivationEviction();
        }

        if (_options.EnableHooks)
        {
            _hooks = new Dictionary<Tensor<T>, List<Func<Tensor<T>, Tensor<T>>>>(
                ReferenceEqualityComparer<Tensor<T>>.Instance);
            _retainGrad = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        }

        SetCurrentTape(this);
        System.Threading.Interlocked.Increment(ref DifferentiableOps._anyTapeActive);
        // Per-thread counter — used to suppress AutoTracer for THIS thread's
        // tape lifecycle without affecting unrelated inference threads. The
        // ThreadStatic field is incremented before backward starts and stays
        // > 0 until Dispose, including the backward walk where _current
        // is temporarily null. See DifferentiableOps._threadTapeDepth.
        DifferentiableOps._threadTapeDepth++;
    }

    /// <summary>
    /// Records a forward operation to the tape. Called by engine operations
    /// via <see cref="DifferentiableOps.RecordIfActive{T}"/>.
    /// </summary>
    /// <param name="entry">The tape entry describing the operation and its backward function.</param>
    /// <exception cref="ObjectDisposedException">Thrown if the tape has been disposed.</exception>
    public void Record(TapeEntry<T> entry)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GradientTape<T>));
        }

        // MaxEntries: drops new entries when at capacity (not evict-oldest).
        // Arena-based storage doesn't support efficient front-eviction (would require O(n) shift).
        // For bounded tapes used in gradient checkpointing, this is acceptable since
        // the checkpoint segments are replayed with fresh tapes.
        if (_options.MaxEntries > 0 && _entries.Count >= _options.MaxEntries)
        {
            return; // Drop new entries when at capacity
        }

        _entries.Add(entry);
    }

    /// <summary>
    /// Returns a ref to the next arena slot for direct field writes.
    /// This eliminates the 80-byte struct copy that Record(entry) requires,
    /// reducing per-op recording overhead by ~50ns.
    /// </summary>
    // Sentinel slot used when MaxEntries is reached — writes are discarded
    private TapeEntry<T> _discardSlot;

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    internal ref TapeEntry<T> RecordSlot()
    {
        // Drop new entries when at capacity (bounded tape)
        if (_options.MaxEntries > 0 && _entries.Count >= _options.MaxEntries)
        {
            _discardSlot = default;
            return ref _discardSlot;
        }
        return ref _entries.AllocateSlot();
    }

    /// <summary>
    /// Computes gradients of <paramref name="loss"/> with respect to <paramref name="sources"/>
    /// by walking the tape in reverse order (reverse-mode automatic differentiation).
    /// </summary>
    /// <param name="loss">The scalar loss tensor to differentiate. Should have a single element.</param>
    /// <remarks>Hot path — uses AggressiveOptimization to skip tiered JIT compilation.</remarks>
    /// <param name="sources">Optional set of tensors to compute gradients for.
    /// If null, computes gradients for all input tensors on the tape.</param>
    /// <param name="createGraph">If true, keeps the tape recording during backward so gradient
    /// ops are recorded for higher-order differentiation (gradient of gradient).
    /// Required for WGAN-GP gradient penalty, MAML, Hessian computation.</param>
    /// <returns>A dictionary mapping each source tensor to its gradient tensor.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the tape has been disposed.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the tape has no entries.</exception>
#if !NETFRAMEWORK
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
#endif
    /// <summary>
    /// Identical to <see cref="ComputeGradients"/> but returns the
    /// gradient dictionary wrapped in a <see cref="GradientsScope{T}"/>
    /// disposable. When the caller disposes the scope (typically at
    /// the end of a training step), every gradient tensor is returned
    /// to <see cref="AutoTensorCache"/> so the next iteration's
    /// backward can reuse the buffers instead of allocating fresh.
    /// Closes the per-step gradient-tensor allocation cost reported
    /// in issue #327.
    /// </summary>
    /// <param name="loss">The scalar loss tensor to differentiate.</param>
    /// <param name="sources">Optional source tensors whose gradients
    /// the caller will read. Required for safe pooling — the scope's
    /// dispose path pools every dictionary value, so unfiltered
    /// callers would observe corrupted tensors on the next iter.</param>
    /// <param name="createGraph">Same semantics as
    /// <see cref="ComputeGradients"/>. createGraph=true is unusual
    /// here; consumers using higher-order AD should hold the
    /// gradient dict themselves rather than scope-pool it.</param>
    public GradientsScope<T> ComputeGradientsScope(
        Tensor<T> loss,
        IReadOnlyList<Tensor<T>> sources,
        bool createGraph = false)
    {
        if (sources is null) throw new ArgumentNullException(nameof(sources));
        var grads = ComputeGradients(loss, sources, createGraph);
        return new GradientsScope<T>(grads);
    }

    /// <summary>
    /// Memory-bounded streaming backward. Computes the gradient of
    /// <paramref name="loss"/> w.r.t. each tensor in <paramref name="sources"/>
    /// (the model parameters) and hands each one to
    /// <paramref name="onSourceGradient"/> at the exact backward step where its
    /// last contribution lands — its provably-earliest safe release point — then
    /// drops the reference so the gradient set is reclaimed incrementally. The
    /// full parameter-gradient set is therefore NEVER resident at once, which is
    /// what lets a model whose gradients exceed RAM still take a training step
    /// (the per-parameter optimizer update happens inside the callback).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the single-process, deterministic analogue of PyTorch's
    /// optimizer-in-backward (<c>_apply_optimizer_in_backward</c>) / FSDP
    /// CPU-offload, but it frees each gradient at its topological last-use rather
    /// than at parameter-registration granularity, minimizing the peak resident
    /// gradient set. Gradients are accumulated in the same reverse-topological
    /// order the standard <see cref="ComputeGradients"/> path uses, so the values
    /// handed to the callback are bit-identical to the non-streaming result.
    /// </para>
    /// <para>
    /// The callback MUST consume its gradient synchronously (apply the optimizer
    /// step) and must NOT retain the reference — the gradient buffer is released
    /// for collection as soon as the callback returns. A source that receives no
    /// gradient contribution gets no callback (mirrors <see cref="ComputeGradients"/>
    /// omitting it from the returned dictionary).
    /// </para>
    /// </remarks>
    /// <param name="loss">The scalar loss tensor to differentiate. Must be
    /// tape-connected (non-null <c>GradFn</c>).</param>
    /// <param name="sources">Parameter tensors whose gradients to stream. Each is
    /// emitted exactly once, when complete.</param>
    /// <param name="onSourceGradient">Invoked as <c>(source, gradient)</c> the
    /// moment <paramref name="loss"/>'s gradient w.r.t. that source is final.</param>
    public void ComputeGradientsStreaming(
        Tensor<T> loss,
        IReadOnlyList<Tensor<T>> sources,
        Action<Tensor<T>, Tensor<T>> onSourceGradient)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GradientTape<T>));
        if (sources is null) throw new ArgumentNullException(nameof(sources));
        if (onSourceGradient is null) throw new ArgumentNullException(nameof(onSourceGradient));
        if (_entries.Count == 0)
            throw new InvalidOperationException("Cannot compute gradients: the tape has no recorded operations.");
        if (loss.GradFn is null)
            throw new InvalidOperationException(
                "Streaming backward requires a tape-connected loss (loss.GradFn is null).");

        var engine = _engine;
        var numOps = Helpers.MathHelper.GetNumericOperations<T>();

        // Suspend recording so the backward ops invoked below don't append fresh
        // tape entries (parity with the non-streaming graph path, which also
        // suspends — see ComputeGradients line ~306).
        var savedCurrent = _current;
        SetCurrentTape(null);
        try
        {
            // Forward topological order; reverse is the backward execution order.
            var visited = new HashSet<GradNode<T>>();
            var topoOrder = new List<GradNode<T>>();
            TopologicalSort(loss.GradFn!, visited, topoOrder);
            int stepCount = topoOrder.Count;

            var steps = new BackwardStep<T>[stepCount];
            // #1624: parallel node handles so each step's backward can release its
            // activation references the moment they are consumed (see below).
            var nodes = new GradNode<T>[stepCount];
            for (int i = 0; i < stepCount; i++)
            {
                var node = topoOrder[stepCount - 1 - i]; // reverse for backward
                nodes[i] = node;
                var nodeOutput = node.Output ?? throw new InvalidOperationException(
                    "Streaming backward: GradNode.Output was null during plan setup (node already released).");
                var nodeBackward = node.Backward ?? throw new InvalidOperationException(
                    "Streaming backward: GradNode.Backward was null during plan setup (node already released).");
                steps[i] = new BackwardStep<T>
                {
                    Output = nodeOutput,
                    Inputs = node.GetInputsArray(),
                    Backward = nodeBackward,
                    SavedState = node.SavedState,
                };
            }

            // Last backward step that contributes to each source's gradient.
            // A source is a leaf parameter: it only ever appears as an op INPUT,
            // so its gradient is only written (accumulated), never read as a
            // step output — making the last write its safe release point.
            var lastUse = new Dictionary<Tensor<T>, int>(
                ReferenceEqualityComparer<Tensor<T>>.Instance);
            foreach (var s in sources)
                if (s is not null && !lastUse.ContainsKey(s)) lastUse[s] = -1;
            for (int i = 0; i < stepCount; i++)
            {
                var inputs = steps[i].Inputs;
                if (inputs is null) continue;
                for (int j = 0; j < inputs.Length; j++)
                {
                    var inp = inputs[j];
                    if (inp is not null && lastUse.ContainsKey(inp))
                        lastUse[inp] = i; // later step overwrites → keeps the last
                }
            }
            // stepIndex -> sources to emit+release right after that step runs.
            var emitAt = new Dictionary<int, List<Tensor<T>>>();
            foreach (var kv in lastUse)
            {
                if (kv.Value < 0) continue; // source never used → no gradient
                if (!emitAt.TryGetValue(kv.Value, out var list))
                    emitAt[kv.Value] = list = new List<Tensor<T>>();
                list.Add(kv.Key);
            }

            var grads = new Dictionary<Tensor<T>, Tensor<T>>(
                stepCount + 1, ReferenceEqualityComparer<Tensor<T>>.Instance);

            // Seed gradient (dL/dL = 1), same construction as ComputeGradients:
            // use the loss's ACTUAL shape (which may be [] for a 0-dim scalar or
            // [1]), not a hardcoded [1] — a hardcoded rank-1 seed mismatches a
            // 0-dim scalar loss and can break shape-checked backward ops.
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

            for (int i = 0; i < stepCount; i++)
            {
                ref var step = ref steps[i];
                if (grads.TryGetValue(step.Output, out var gradOutput))
                {
                    // Weight-streaming integration: rehydrate any paged-out input
                    // weights before the backward reads them. The forward path does
                    // this through StreamingAutogradHook.OnInputAccessed; the plain
                    // streaming-backward walk bypasses that hook, so materialize
                    // explicitly here. Materialize is a no-op for tensors that are
                    // resident or not registered with the streaming pool, so this is
                    // free when weight streaming is off. The pool evicts other LRU
                    // weights to stay under its resident cap, keeping peak bounded —
                    // and the just-materialized weight is resident when its gradient
                    // completes (its last use), so the optimizer epilogue can update
                    // it in place; eviction later writes the update back to disk.
                    var stepInputs = step.Inputs;
                    if (stepInputs is not null)
                    {
                        for (int j = 0; j < stepInputs.Length; j++)
                        {
                            var inp = stepInputs[j];
                            if (inp is not null) WeightRegistry.Materialize(inp);
                        }
                    }

                    step.Backward(gradOutput, step.Inputs, step.Output,
                        step.SavedState ?? Array.Empty<object>(), engine, grads);

                    // Release this node-output's gradient now that its backward has
                    // consumed it. In the reverse topo walk every consumer of this
                    // output ran earlier, so its gradient is fully accumulated and is
                    // never read again. Leaving it in `grads` ROOTS it for the rest of
                    // the walk — and with every intermediate kept, the dict pins the
                    // FULL backward's worth of gradient tensors at once, so GC can't
                    // reclaim any of them and a deep net (ResNet50) OOMs. Dropping the
                    // reference here unroots it so GC bounds the streaming peak to the
                    // live frontier. Sources are leaves (never a step Output) and are
                    // emitted + released separately below; the lastUse guard keeps this
                    // from touching them.
                    if (step.Output is not null && !lastUse.ContainsKey(step.Output))
                    {
                        grads.Remove(step.Output);
                        step.Output.Grad = null;
                    }
                }

                // Emit + release every source whose gradient just completed.
                if (emitAt.TryGetValue(i, out var ready))
                {
                    for (int k = 0; k < ready.Count; k++)
                    {
                        var src = ready[k];
                        if (grads.TryGetValue(src, out var g))
                        {
                            onSourceGradient(src, g);
                            grads.Remove(src);   // drop the dict reference …
                            src.Grad = null;     // … and the per-tensor mirror → reclaimable
                        }
                    }
                }

                // #1624: free this step's ACTIVATION references now that its
                // backward has consumed them. The forward builds the FULL
                // activation set; under a persistent tape the node graph otherwise
                // keeps every activation resident through the whole backward, so
                // the backward's own buffer allocations (AutoTensorCache) pile on
                // top and OOM a deep model (the SimCSE #1624 failure throws here).
                // Releasing each node's Output / SavedState / backward-closure as
                // the reverse walk consumes it — combined with layers no longer
                // pinning activations (consumer-side layer-cache skip) — bounds the
                // live set to the backward frontier instead of the whole forward.
                // The node Output is never a source (sources are leaves, emitted +
                // released above), so this never drops a gradient the optimizer
                // still needs.
                if (ReleaseStreamingActivations)
                {
                    var n = nodes[i];
                    var nodeOutput = n?.Output;
                    if (n is not null && nodeOutput is not null && !lastUse.ContainsKey(nodeOutput))
                    {
                        n.Output = null;
                        n.SavedState = null;
                        n.Backward = null;
                    }
                    // Reset the whole struct slot (Output/Inputs/Backward/SavedState)
                    // in one assignment — releases every reference without a
                    // null-forgiving cast. Safe: this is the end of the iteration and
                    // steps[i] is never read again.
                    step = default;
                }
            }

            // Clear .Grad on forward intermediates (parity with the non-persistent
            // cleanup in ComputeGradientsViaGraphCore) so they don't pin a full
            // backward's worth of gradient tensors after we return.
            foreach (var node in topoOrder)
            {
                var outp = node.Output;
                if (outp is not null && !lastUse.ContainsKey(outp))
                    outp.Grad = null;
            }
        }
        finally
        {
            SetCurrentTape(savedCurrent);
            if (!_options.Persistent) _entries.Reset();
        }
    }

    public Dictionary<Tensor<T>, Tensor<T>> ComputeGradients(
        Tensor<T> loss,
        IReadOnlyList<Tensor<T>>? sources = null,
        bool createGraph = false)
        => ComputeGradients(loss, sources, createGraph, seedOverride: null);

    /// <summary>
    /// Backward with an optional custom gradient seed. When <paramref name="seedOverride"/>
    /// is null this is byte-identical to the public <see cref="ComputeGradients(Tensor{T},IReadOnlyList{Tensor{T}},bool)"/>
    /// (ones-at-loss seeding, compiled/graph fast paths eligible). When it is supplied, the
    /// listed (tensor, grad) pairs are seeded into the backward instead of ones-at-loss, and
    /// the slow tape walk is forced (the compiled/graph fast paths assume a single ones seed at
    /// <paramref name="loss"/>). This is the mixed-precision bridge entry point: the FP16 sub-tape's
    /// backward is seeded with the FP32 grad cast down at each up-cast boundary (see
    /// <see cref="MixedPrecisionTape"/>). Not public — mixed-dtype autograd is the only caller.
    /// </summary>
    internal Dictionary<Tensor<T>, Tensor<T>> ComputeGradients(
        Tensor<T> loss,
        IReadOnlyList<Tensor<T>>? sources,
        bool createGraph,
        IReadOnlyList<KeyValuePair<Tensor<T>, Tensor<T>>>? seedOverride)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GradientTape<T>));
        }

        if (_entries.Count == 0)
        {
            throw new InvalidOperationException("Cannot compute gradients: the tape has no recorded operations.");
        }

        // Auto-training compiler: highest priority — use compiled backward if available.
        // Must be checked BEFORE the graph path because DifferentiableOps always records
        // (GradFn is always set), so the graph path would always win otherwise.
        // Tensor- or node-level hooks both rewrite gradients during the
        // tape walk; either kind forces the slower tape path because
        // ComputeGradientsViaGraph (the fast path below) doesn't visit
        // tape entries and would silently bypass our hook calls.
        bool hasHooksRegistered = (_hooks is not null && _hooks.Count > 0)
            || (_nodeHooks is not null && _nodeHooks.Count > 0)
            || (_nodePredicateHooks is not null && _nodePredicateHooks.Count > 0);
        if (_options.Persistent && !createGraph && seedOverride is null)
        {
            var compiledBwd = Compilation.AutoTrainingCompiler.TryGetCompiledBackward(this, loss, sources?.ToArray());
            if (compiledBwd is not null)
            {
                // Issue #283: suspend the tape during compiled-backward
                // replay. Without this, engine ops invoked from BackwardFunctions
                // (TensorMatMul/TensorTranspose/etc.) see Current != null and
                // record fresh tape entries whose GradNodes hold the FORWARD
                // intermediates as inputs. The compiled plan's cleanup only
                // visits the forward _reachableEntryIndices — backward-
                // recorded entries' GradFn chains are NEVER cleared, pinning
                // ~7 forward intermediates × ~40KB each per iter (the
                // 133KB-1MB/call signature in the GradientTapeLeakTests
                // transformer-scale probes). The non-compiled path
                // (ComputeGradientsViaGraph) already suspends — this is the
                // missing parity. Same gating: createGraph=true intentionally
                // keeps recording so higher-order ops (Hvp/Hessian) land in
                // the outer tape, but the compiled-replay path is gated on
                // !createGraph (line 249 above) so we always suspend here.
                var savedCompiledReplayCurrent = _current;
                SetCurrentTape(null);
                try
                {
                    return compiledBwd.Execute(loss);
                }
                finally
                {
                    SetCurrentTape(savedCompiledReplayCurrent);
                }
            }
        }

        // Graph-based backward: walk GradFn pointers instead of tape.
        // This is faster because it skips tape traversal, dict lookups, and relevance checks.
        // Skip graph path when anomaly detection or hooks are enabled — the tape path handles those.
        // The graph path bypasses the NaN/Inf check — force the slower
        // tape path whenever anomaly detection is on, either via the
        // per-tape flag or an ambient AnomalyModeScope. Same reasoning
        // that exists for DetectAnomaly extends to the scope.
        if (loss.GradFn is not null && !createGraph && !DetectAnomaly && !AnomalyModeScope.IsActive && !hasHooksRegistered && seedOverride is null)
        {
            // Record step pattern BEFORE backward — backward ops get recorded on the tape
            // (since DifferentiableOps always records), which would make the hash nondeterministic.
            // The pattern hash should only reflect the forward pass ops.
            int forwardEntryCount = _entries.Count;
            if (_options.Persistent)
            {
                Compilation.AutoTrainingCompiler.RecordStep(_entries, forwardEntryCount, loss);
            }

            var result = ComputeGradientsViaGraph(loss, sources);

            // Try to compile AFTER backward (needs tape entries for CompileBackward)
            if (_options.Persistent)
            {
                Compilation.AutoTrainingCompiler.TryCompileBackward(this, loss, sources?.ToArray());
            }

            if (!_options.Persistent) _entries.Reset();
            return result;
        }

        var engine = _engine;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Assign gradient indices to all tensors that appear in the tape.
        // This enables O(1) array access in AccumulateGrad instead of dictionary hash lookup.
        int gradIndexCount = 0;
        for (int i = 0; i < _entries.Count; i++)
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

        // Flat gradient array: indexed by _gradIndex for O(1) access.
        var indexedGrads = new object?[gradIndexCount];
        DifferentiableOps.SetIndexedGrads(indexedGrads);

        // Parallel sparse-gradient array for embedding-table parameters. Same _gradIndex
        // basis as indexedGrads; each slot holds a List<SparseEmbeddingGradient<T>>.
        // Sparse-aware optimizers (Adam/AdamW with the sparse-scatter step) check this
        // first to skip the dense [vocab, dim] gradient that dominates per-step alloc
        // on token-embedding-bearing models — LayoutXLM 768 MB, Amphion 1045 MB,
        // AVHuBERT 900 MB, AST 877 MB, AudioFlamingo2 610 MB per backward without it.
        var indexedSparseGrads = new object?[gradIndexCount];
        DifferentiableOps.SetIndexedSparseGrads(indexedSparseGrads);

        // Dictionary facade: backward functions still receive Dictionary<Tensor<T>, Tensor<T>>.
        // AccumulateGrad writes to both the array and dictionary.
        var grads = new Dictionary<Tensor<T>, Tensor<T>>(
            Math.Min(gradIndexCount + 1, 1024),
            ReferenceEqualityComparer<Tensor<T>>.Instance);

        if (seedOverride is not null)
        {
            // Mixed-precision bridge: seed the supplied (tensor, grad) pairs directly instead
            // of ones-at-loss. Each tensor is an interior node of this (sub-)tape that received
            // its gradient from the other-dtype tape across a differentiable cast boundary.
            for (int s = 0; s < seedOverride.Count; s++)
            {
                var t = seedOverride[s].Key;
                var g = seedOverride[s].Value;
                if (grads.TryGetValue(t, out var existing))
                {
                    grads[t] = engine.TensorAdd(existing, g);
                }
                else
                {
                    grads[t] = g;
                }
                if (t._gradIndex >= 0 && t._gradIndex < indexedGrads.Length)
                    indexedGrads[t._gradIndex] = grads[t];
            }
        }
        else
        {
            // Seed: gradient of loss w.r.t. itself is ones with the same shape.
            // Fast path for scalar loss (the overwhelmingly common case in training).
            // Reuse cached scalar seed across training steps to avoid per-backward allocation.
            Tensor<T> seedGrad;
            if (loss.Length == 1)
            {
                // Use loss's actual shape (could be [1] or [] for 0-dim scalar)
                seedGrad = _cachedScalarSeed ??= new Tensor<T>(new[] { numOps.One }, (int[])loss._shape.Clone());
            }
            else
            {
                var onesData = new T[loss.Length];
                var one = numOps.One;
                for (int j = 0; j < onesData.Length; j++)
                    onesData[j] = one;
                seedGrad = new Tensor<T>(onesData, loss._shape);
            }
            grads[loss] = seedGrad;
            if (loss._gradIndex >= 0 && loss._gradIndex < indexedGrads.Length)
                indexedGrads[loss._gradIndex] = seedGrad;
        }

        // When createGraph=false (default): suspend recording so backward engine calls
        // don't append to this tape — they'd corrupt persistent tapes and shift bounded tapes.
        // When createGraph=true: KEEP recording so backward ops are on the tape for
        // higher-order differentiation (gradient of gradient).
        var savedCurrent = _current;
        if (!createGraph)
        {
            SetCurrentTape(null);
        }
        // Signal AccumulateGrad to use out-of-place TensorAdd so the
        // second backward pass sees a connected graph (see the field
        // doc on DifferentiableOps._isBackwardCreateGraph).
        var savedBackwardCreateGraph = DifferentiableOps._isBackwardCreateGraph;
        if (createGraph)
        {
            DifferentiableOps._isBackwardCreateGraph = true;
        }

        try
        {
            // Pre-compute boolean guards to skip expensive checks in the hot loop.
            // These avoid dictionary lookups and iteration when features are not used.
            bool hasHooks = _hooks is not null && _hooks.Count > 0;
            bool hasRetainGrad = _retainGrad is not null && _retainGrad.Count > 0;
            bool profileEnabled = ProfileBackward;

            // Walk tape in reverse (reverse-mode AD)
            // Anomaly detection fires when EITHER the per-tape flag is set
            // (local opt-in) OR an AnomalyModeScope is active on this thread
            // (process-wide opt-in, mirroring torch.autograd.set_detect_anomaly).
            bool anomalyActive = DetectAnomaly || AnomalyModeScope.IsActive;
            var numOpsForAnomaly = anomalyActive ? MathHelper.GetNumericOperations<T>() : null;

            // Tape backward pruning: when sources are specified, forward-walk to find all tensors
            // downstream of those sources. Skip entries whose output is not reachable.
            // For persistent tapes, cache the relevance set since sources are typically the same
            // across training steps (always the model parameters).
            HashSet<Tensor<T>>? relevantTensors = null;
            // Only build relevance set for large tapes — the O(n) forward-walk cost
            // exceeds the backward pruning benefit for small tapes (< 100 entries).
            if (sources is not null && sources.Count > 0 && _entries.Count >= 100)
            {
                relevantTensors = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
                foreach (var s in sources)
                    relevantTensors.Add(s);

                // Forward pass: mark tensors reachable from sources
                // Uses inline input fields to avoid per-entry array allocation
                for (int i = 0; i < _entries.Count; i++)
                {
                    ref var entry = ref _entries[i];
                    bool inputRelevant = false;

                    if (entry.InputsOverflow is not null)
                    {
                        foreach (var inp in entry.InputsOverflow)
                        {
                            if (relevantTensors.Contains(inp)) { inputRelevant = true; break; }
                        }
                    }
                    else
                    {
                        if (relevantTensors.Contains(entry.Input0)) inputRelevant = true;
                        else if (entry.InputCount >= 2 && entry.Input1 is not null && relevantTensors.Contains(entry.Input1)) inputRelevant = true;
                        else if (entry.InputCount >= 3 && entry.Input2 is not null && relevantTensors.Contains(entry.Input2)) inputRelevant = true;
                    }

                    if (inputRelevant)
                    {
                        relevantTensors.Add(entry.Output);
                    }
                }
            }

            for (int i = _entries.Count - 1; i >= 0; i--)
            {
                ref var entry = ref _entries[i];

                // Skip if we don't have a gradient for this entry's output
                if (!grads.TryGetValue(entry.Output, out var gradOutput))
                {
                    continue;
                }

                // Tape backward pruning: skip entries that don't contribute to requested sources.
                if (relevantTensors is not null && !relevantTensors.Contains(entry.Output))
                {
                    continue;
                }

                // Construct input array once for this entry's backward pass
                var inputsArray = entry.GetInputsArray();

                // Issue #276 streaming-pool prefetch: rehydrate any input
                // that's tagged WeightLifetime.Streaming before the kernel
                // reads it. No-op for default-lifetime tensors (the hot
                // path), so the indexed-grad fast loop pays only one
                // branch per input.
                StreamingAutogradHook.OnForwardRecorded(inputsArray);

                // Apply tensor hooks (skip dictionary lookup entirely when no hooks registered)
                if (hasHooks && _hooks!.TryGetValue(entry.Output, out var hookList))
                {
                    foreach (var hook in hookList)
                        gradOutput = hook(gradOutput);
                    grads[entry.Output] = gradOutput;
                }

                // Apply graph-node hooks: name-keyed and predicate-based.
                // These fire after tensor hooks so a user can chain
                // tensor → node modifications. Each set is lazily
                // populated; the null checks below short-circuit the
                // common no-hook case at the cost of one branch.
                if (_nodeHooks is not null
                    && _nodeHooks.TryGetValue(entry.OperationName, out var nodeHookList))
                {
                    foreach (var hook in nodeHookList)
                        gradOutput = hook(gradOutput);
                    grads[entry.Output] = gradOutput;
                }
                if (_nodePredicateHooks is not null)
                {
                    foreach (var (match, hook) in _nodePredicateHooks)
                    {
                        if (match(entry))
                            gradOutput = hook(gradOutput);
                    }
                    grads[entry.Output] = gradOutput;
                }

                // Validate that no input tensor was mutated after recording
                entry.ValidateInputVersions();

                // Invoke the backward function. Phase A (#338) timing
                // wrapper records per-op ticks when AIDOTNET_BWD_TIMING=1.
                long _bwdStart = BackwardTiming.Enabled ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                entry.Backward(gradOutput, inputsArray, entry.Output, entry.SavedState ?? Array.Empty<object>(), engine, grads);
                if (BackwardTiming.Enabled)
                    BackwardTiming.Record(entry.Backward.Method.Name, System.Diagnostics.Stopwatch.GetTimestamp() - _bwdStart);

                // Performance profiling (only when explicitly enabled)
                // Timing wraps the backward call above — the Stopwatch overhead
                // is negligible relative to backward computation cost.
                if (profileEnabled)
                {
                    System.Console.WriteLine($"  backward[{entry.OperationName}]");
                }

                // Anomaly detection (only when explicitly enabled — either
                // tape.DetectAnomaly or an AnomalyModeScope). Throws the
                // dedicated AnomalyDetectedException so callers can catch
                // autograd anomalies without a broader ArithmeticException
                // match.
                if (numOpsForAnomaly is not null)
                {
                    foreach (var input in inputsArray)
                    {
                        if (grads.TryGetValue(input, out var inputGrad))
                        {
                            for (int k = 0; k < inputGrad.Length; k++)
                            {
                                double val = numOpsForAnomaly.ToDouble(inputGrad[k]);
                                if (double.IsNaN(val) || double.IsInfinity(val))
                                {
                                    // Emit an instant marker on the active profiler
                                    // so a chrome-trace export pinpoints the failing
                                    // op visually. The marker carries the op name
                                    // and the input index so users can localize the
                                    // NaN to the exact backward edge — #220 anomaly
                                    // localization scope.
                                    Profiling.Profiler.RecordInstant(
                                        $"anomaly:{entry.OperationName}",
                                        category: "autograd",
                                        args: new System.Collections.Generic.Dictionary<string, string>
                                        {
                                            ["op"] = entry.OperationName,
                                            ["element"] = k.ToString(System.Globalization.CultureInfo.InvariantCulture),
                                            ["value"] = val.ToString(System.Globalization.CultureInfo.InvariantCulture),
                                        });
                                    throw new AnomalyDetectedException(entry.OperationName);
                                }
                            }
                        }
                    }
                }

                // Retain gradient (skip entirely when not requested — common case)
                if (hasRetainGrad)
                {
                    foreach (var input in inputsArray)
                    {
                        if (_retainGrad!.Contains(input) && grads.TryGetValue(input, out var retained))
                        {
                            // Hook into input tensors that requested gradient retention
                            if (_hooks is not null && _hooks.TryGetValue(input, out var inputHooks))
                            {
                                foreach (var hook in inputHooks)
                                    grads[input] = hook(grads[input]);
                            }
                        }
                    }
                }
            }
        }
        finally
        {
            if (!createGraph)
            {
                SetCurrentTape(savedCurrent);
            }
            DifferentiableOps._isBackwardCreateGraph = savedBackwardCreateGraph;
            // Clear indexed gradient array and reset tensor grad indices
            DifferentiableOps.ClearIndexedGrads();
            DifferentiableOps.ClearIndexedSparseGrads();
            for (int i = 0; i < _entries.Count; i++)
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

        // Tape-walk parity for the .Grad / .GradFn cleanup that ComputeGradientsViaGraph does.
        // AccumulateGrad sets `tensor.Grad = stored` for every tensor that receives a
        // gradient contribution — including forward intermediates — and each tensor's
        // GradFn pointer chains back through the entire forward graph via captured
        // BackwardFunction closures. Without explicit cleanup, any external code that
        // retains even ONE forward intermediate (e.g. a layer's `_lastInput` cache)
        // keeps the whole graph alive across Train calls, surviving Gen2 GC.
        //
        // The previous guard `sources is not null` skipped cleanup whenever the caller
        // didn't specify a source filter. In practice every long-running training loop
        // calls ComputeGradients(loss, sources: null) — including AiDotNet's
        // NeuralNetworkBase.TrainWithTape (passes null and consumes the returned Dictionary
        // directly) — so the cleanup never ran and the heap grew linearly at the per-call
        // saved-for-backward footprint. ooples/AiDotNet#1227 documented this at ~1.5 MB/call
        // on a 4-encoder-layer Transformer; the same model on the equivalent raw-tape
        // probe in this repo measured 0 B/call because the test always passed explicit
        // sources.
        //
        // Fix: run the cleanup whether or not `sources` was provided. When sources is
        // null we still clear .GradFn on every tape-recorded tensor (breaks the graph
        // back-pointer chain) and clear .Grad on tensors that aren't pinned via
        // RetainGrad. The returned `grads` Dictionary already holds the gradient
        // references the caller needs — clearing the per-tensor .Grad field doesn't
        // drop any data the caller can still access through the dictionary.
        //
        // SKIP cleanup when createGraph=true: the outer caller is computing higher-
        // order derivatives (Hvp / Hessian / double-backward), and forward
        // intermediates' .Grad / .GradFn fields are part of the higher-order graph
        // that the next backward pass needs to walk.
        if (!_options.Persistent && !createGraph)
        {
            HashSet<Tensor<T>>? sourceSet = null;
            if (sources is not null)
            {
                sourceSet = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
                foreach (var s in sources) sourceSet.Add(s);
            }

            // Pre-collect every output tensor in the arena: these are the
            // intermediates produced by tape-recorded ops. Graph leaves
            // (parameters / user-supplied roots) never appear as Output in
            // an entry. We need this distinction so the BC contract
            // "param.Grad stays populated after a sources=null backward"
            // holds in the tape-walk path the same way it holds in
            // ComputeGradientsViaGraphCore (line 776+). Without this,
            // calling CleanupTapeEntryGrad on a leaf input would null
            // param.Grad and silently break any consumer reading it.
            var intermediates = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
            for (int i = 0; i < _entries.Count; i++)
            {
                intermediates.Add(_entries[i].Output);
            }

            for (int i = 0; i < _entries.Count; i++)
            {
                ref var e = ref _entries[i];
                CleanupTapeEntryGrad(e.Output, sourceSet, intermediates);
                if (e.Input0 != null) CleanupTapeEntryGrad(e.Input0, sourceSet, intermediates);
                if (e.InputCount >= 2 && e.Input1 != null) CleanupTapeEntryGrad(e.Input1, sourceSet, intermediates);
                if (e.InputCount >= 3 && e.Input2 != null) CleanupTapeEntryGrad(e.Input2, sourceSet, intermediates);
                if (e.InputsOverflow != null)
                {
                    foreach (var inp in e.InputsOverflow)
                    {
                        if (inp != null) CleanupTapeEntryGrad(inp, sourceSet, intermediates);
                    }
                }
            }
        }

        // If sources specified, filter to only those
        if (sources is not null)
        {
            var filtered = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
            foreach (var source in sources)
            {
                if (grads.TryGetValue(source, out var grad))
                {
                    filtered[source] = grad;
                }
            }

            // Auto-training compiler: record step pattern for both filtered and unfiltered paths
            if (_options.Persistent)
            {
                Compilation.AutoTrainingCompiler.RecordStep(_entries, _entries.Count, loss);
                Compilation.AutoTrainingCompiler.TryCompileBackward(this, loss, sources?.ToArray());
            }

            if (!_options.Persistent)
            {
                _entries.Reset();
            }

            return filtered;
        }

        // Auto-training compiler for unfiltered path (no sources specified)
        if (_options.Persistent)
        {
            Compilation.AutoTrainingCompiler.RecordStep(_entries, _entries.Count, loss);
            Compilation.AutoTrainingCompiler.TryCompileBackward(this, loss, sources?.ToArray());
        }

        if (!_options.Persistent)
        {
            _entries.Reset();
        }

        return grads;
    }

    private void CleanupTapeEntryGrad(Tensor<T> t, HashSet<Tensor<T>>? sourceSet, HashSet<Tensor<T>> intermediates)
    {
        // Only null GradFn on tensors THIS tape owns. A tensor whose GradFn
        // doesn't belong to this tape's graph (e.g. an output of an outer
        // tape, fed in as input to this tape's ops) must keep its back-
        // pointer intact — otherwise the outer tape's later backward
        // would see a severed graph and either crash or silently drop
        // gradients. The ownership signal is membership in `intermediates`,
        // which holds every Output produced by this tape's entries.
        bool ownedByThisTape = intermediates.Contains(t);
        if (ownedByThisTape)
        {
            t.GradFn = null;
        }
        // Issue #338: clear the tape-pinning flag set by
        // DifferentiableOps.Record* — the backward walk has completed and
        // the tensor is safe to pool again. Cleared regardless of source /
        // RetainGrad status because the pin guards against pool reuse, not
        // .Grad lifecycle.
        t._pinnedByTape = false;
        // Preserve .Grad when the caller explicitly listed this tensor as a source
        // (they're going to read `tensor.Grad` rather than the returned Dictionary).
        if (sourceSet?.Contains(t) == true) return;
        // Preserve .Grad on tensors the user explicitly marked with RetainGrad().
        if (_retainGrad is not null && _retainGrad.Contains(t)) return;
        // Distinguish intermediate vs leaf when sources is null:
        //   sourceSet != null + non-source: caller's explicit filter said
        //     "only keep what I listed" — clear regardless of ownership.
        //     Foreign-leaf .Grad wasn't populated by this backward pass
        //     anyway (this tape only accumulates to its own reachable
        //     graph), so clearing it is effectively a no-op on the typical
        //     nested-tape case.
        //   sourceSet == null + this-tape intermediate: clear (returned
        //     dict still holds the gradient; .Grad is only the per-tensor
        //     mirror).
        //   sourceSet == null + leaf or foreign tensor:
        //     preserve .Grad — for true leaves (params), the BC contract
        //     that AiDotNet.NeuralNetworkBase.TrainWithTape and the graph-
        //     walk path (ComputeGradientsViaGraphCore line 776+) honor;
        //     for foreign tensors, their .Grad belongs to the outer tape
        //     and we must not touch it.
        if (sourceSet is not null || ownedByThisTape)
        {
            t.Grad = null;
        }
    }

    /// <summary>
    /// Graph-based backward: walks GradFn pointers on tensors instead of the tape.
    /// Eliminates tape traversal, dictionary lookups, and relevance checks.
    /// </summary>
    // Cached delegate chain for persistent tapes — avoids topological sort on repeat backward
    private CompiledDelegateChain<T>? _cachedDelegateChain;

    private Dictionary<Tensor<T>, Tensor<T>> ComputeGradientsViaGraph(
        Tensor<T> loss,
        IReadOnlyList<Tensor<T>>? sources)
    {
        // Suspend recording while backward runs so engine ops invoked from
        // backward funcs don't append to *this* tape (would shift bounded
        // tapes / corrupt persistent ones), and so backward fast paths that
        // gate on `Current is null` (parallel SimdGemm in MatMulBackward et
        // al.) are actually reachable on the graph path. The tape-walk path
        // does the same just before its main loop; this is the parity fix
        // for the graph path, which is the common case (createGraph=false,
        // no anomaly detection, no hooks).
        //
        // We DON'T suspend when this graph backward is itself running inside
        // an outer createGraph=true backward (DifferentiableOps._isBackwardCreateGraph
        // signals that), because the outer tape wants to keep recording so
        // each gradient computation lands in the higher-order graph for Hvp.
        var savedCurrent = _current;
        bool suspendTape = !DifferentiableOps._isBackwardCreateGraph;
        if (suspendTape)
        {
            SetCurrentTape(null);
        }

        try
        {
            return ComputeGradientsViaGraphCore(loss, sources);
        }
        finally
        {
            if (suspendTape)
            {
                SetCurrentTape(savedCurrent);
            }
        }
    }

    private Dictionary<Tensor<T>, Tensor<T>> ComputeGradientsViaGraphCore(
        Tensor<T> loss,
        IReadOnlyList<Tensor<T>>? sources)
    {
        // Rebindable plan cache: fresh-tape repeat-pattern fast path.
        if (!_options.Persistent
            && !DifferentiableOps._isBackwardCreateGraph
            && _entries.Count > 0)
        {
            long lookupHash = AutoTrainingCompiler.ComputeStructureHash(_entries, _entries.Count);
            // Quick signature check first — empty cache or hash mismatch
            // both return false at the cost of three thread-static field
            // reads, much cheaper than the full TryExecute call.
            if (RebindablePlanCache<T>.TrySignature(lookupHash, _entries.Count))
            {
                var savedCurrent = _current;
                SetCurrentTape(null);
                try
                {
                    // Issue #338 Item 3: compiled-IL backward walker. When
                    // AIDOTNET_COMPILED_BACKWARD is enabled, route through
                    // the pattern-keyed walker cache. The walker today is
                    // a passthrough (same dispatch logic, just hoisted out
                    // of the cache module); future commits replace each
                    // walker with a DynamicMethod whose IL bakes in the
                    // index sequence + backward delegate signatures.
                    if (CompiledBackwardWalk<T>.Enabled)
                    {
                        var walker = CompiledBackwardWalk<T>.TryGetWalker(lookupHash);
                        if (walker is not null)
                        {
                            var compiledResult = walker(_entries, loss, sources, _engine);
                            if (compiledResult is not null)
                            {
                                CleanupAfterCachedReplay(compiledResult, sources);
                                return compiledResult;
                            }
                        }
                    }

                    var cached = RebindablePlanCache<T>.TryExecute(lookupHash, _entries, loss, sources, _engine);
                    if (cached is not null)
                    {
                        // Per-call cleanup parity with the fresh-walk path
                        // below — clear GradFn / .Grad on intermediates so
                        // they don't pin one full backward's worth of
                        // gradient tensors across successive calls.
                        CleanupAfterCachedReplay(cached, sources);
                        return cached;
                    }
                }
                finally { SetCurrentTape(savedCurrent); }
            }
        }

        // Issue #319 Phase 1: thread-local scratch buffers for the
        // backward dispatch path. Acquire once at the top of the call;
        // nested backwards (createGraph=true Hessian path) fall back to
        // fresh allocation everywhere — the inner call would otherwise
        // clobber the outer call's buffers.
        bool acquiredScratch = BackwardScratch<T>.TryAcquire();
        try
        {
            // If we have a cached delegate chain from a previous
            // backward, replay it directly. The chain itself is
            // immutable; only the grads dict + seed gradient are
            // rented, both inside chain.Execute. Apply the same
            // per-intermediate .Grad cleanup the fresh-walk path does
            // (the persistent-tape replay would otherwise pin one full
            // backward's worth of gradient tensors on every
            // intermediate between Execute calls — reopened-#1227 with
            // the consumer-side cache pattern). We CANNOT clear GradFn
            // here: the chain reuses each step's Output.GradFn on the
            // next Execute, so it has to stay live. .Grad gets re-set
            // by AccumulateGrad on the next forward+backward anyway,
            // so clearing here only severs the per-tensor mirror — the
            // returned grads dict carries the data either way.
            if (_cachedDelegateChain is not null)
            {
                var cachedResult = _cachedDelegateChain.Execute(loss, sources, _engine, useScratch: acquiredScratch);
                HashSet<Tensor<T>>? cachedSourceSet = null;
                if (sources is not null)
                {
                    cachedSourceSet = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
                    foreach (var s in sources) cachedSourceSet.Add(s);
                }
                var cachedSteps = _cachedDelegateChain.Steps;
                for (int i = 0; i < cachedSteps.Length; i++)
                {
                    ref var step = ref cachedSteps[i];
                    if (cachedSourceSet?.Contains(step.Output) == true) continue;
                    if (_retainGrad is not null && _retainGrad.Contains(step.Output)) continue;
                    step.Output.Grad = null;
                }
                return cachedResult;
            }

            var engine = _engine;
            HashSet<GradNode<T>> visited;
            List<GradNode<T>> topoOrder;
            if (acquiredScratch)
            {
                visited = BackwardScratch<T>.RentVisited();
                topoOrder = BackwardScratch<T>.RentTopoOrder();
            }
            else
            {
                visited = new HashSet<GradNode<T>>();
                topoOrder = new List<GradNode<T>>();
            }

            // Topological sort via DFS from loss.GradFn — build the execution order.
            // The walk CAN cross tape boundaries — that's how higher-order AD
            // (createGraph=true → Hvp) reaches recorded backward ops from a
            // prior tape. Cross-tape cleanup is gated below by OwningTape
            // identity instead of restricting the walk itself.
            TopologicalSort(loss.GradFn!, visited, topoOrder);

            // Build delegate chain from topological order (capture for replay).
            // Persistent tapes need a private array — the chain is cached on the
            // tape and re-executed on later backwards, so we can't share the
            // scratch buffer with the next call. Non-persistent tapes rent
            // from the scratch pool and clear on release.
            int stepCount = topoOrder.Count;
            BackwardStep<T>[] steps;
            if (_options.Persistent || !acquiredScratch)
            {
                steps = new BackwardStep<T>[stepCount];
            }
            else
            {
                steps = BackwardScratch<T>.RentSteps(stepCount);
            }
            for (int i = 0; i < stepCount; i++)
            {
                var node = topoOrder[stepCount - 1 - i]; // reverse for backward order
                var nodeOutput = node.Output ?? throw new InvalidOperationException(
                    "Backward: GradNode.Output was null during delegate-chain build (node already released).");
                var nodeBackward = node.Backward ?? throw new InvalidOperationException(
                    "Backward: GradNode.Backward was null during delegate-chain build (node already released).");
                steps[i] = new BackwardStep<T>
                {
                    Output = nodeOutput,
                    Inputs = node.GetInputsArray(),
                    Backward = nodeBackward,
                    SavedState = node.SavedState
                };
            }

            var chain = new CompiledDelegateChain<T>(steps, stepCount);

            // Cache for persistent tapes (same network structure every step)
            if (_options.Persistent)
                _cachedDelegateChain = chain;

            // Execute the chain
            var result = chain.Execute(loss, sources, engine, useScratch: acquiredScratch);

            // Build rebindable plan for fresh-tape callers (closes the
            // consumer fresh-tape gap from issue #327). MUST run AFTER
            // chain.Execute — building the plan before Execute mutated
            // state that broke higher-order AD (Hvp/Hessian); see the
            // bisection comment in RebindablePlanCache for details.
            //
            // After Execute, we know:
            //   * The backward pass completed and produced `result`.
            //   * tape entries are stable (recording suspended during
            //     non-createGraph backward; for createGraph, rebindEligible
            //     is false).
            // Building + storing the plan here lets the NEXT fresh tape
            // with the same forward pattern hit the cache at the top of
            // this method (TryGet) and skip DFS + step-build entirely.
            if (!_options.Persistent
                && !DifferentiableOps._isBackwardCreateGraph
                && _entries.Count > 0)
            {
                // Detect cross-tape walks (Hvp / higher-order AD nests tapes;
                // the topoOrder crosses into the inner tape's nodes). The
                // cache only knows about THIS tape's entries — a cached plan
                // would silently skip the inner tape's gradient contributions
                // and produce wrong gradients on every replay. Punt to the
                // DFS path on every call when any cross-tape node appears.
                bool crossTape = false;
                for (int i = 0; i < stepCount; i++)
                {
                    if (!ReferenceEquals(topoOrder[i].OwningTape, this))
                    {
                        crossTape = true;
                        break;
                    }
                }

                if (!crossTape)
                {
                    long patternHash = AutoTrainingCompiler.ComputeStructureHash(_entries, _entries.Count);

                    // Map each GradNode in topoOrder to its tape entry index.
                    // Tape entries are recorded in forward (topological) order;
                    // entry.Output.GradFn IS the GradNode for that op. One O(N)
                    // walk gives us the map for the per-node lookup below.
                    var nodeToEntryIndex = new Dictionary<GradNode<T>, int>(_entries.Count);
                    for (int e = 0; e < _entries.Count; e++)
                    {
                        ref var rec = ref _entries[e];
                        var fn = rec.Output?.GradFn;
                        if (fn is not null && !nodeToEntryIndex.ContainsKey(fn))
                            nodeToEntryIndex[fn] = e;
                    }

                    // reverseTopoIndices[i] = entry index of the i-th step
                    // we'd dispatch in backward (topoOrder traversed in reverse).
                    var reverseTopoIndices = new int[stepCount];
                    int writeIdx = 0;
                    for (int i = stepCount - 1; i >= 0; i--)
                    {
                        var node = topoOrder[i];
                        if (nodeToEntryIndex.TryGetValue(node, out int entryIdx))
                            reverseTopoIndices[writeIdx++] = entryIdx;
                    }
                    if (writeIdx < reverseTopoIndices.Length)
                        Array.Resize(ref reverseTopoIndices, writeIdx);

                    RebindablePlanCache<T>.Store(patternHash, _entries.Count, reverseTopoIndices, _entries);
                }
            }

        // Clear GradFn to release graph memory (non-persistent tapes
        // get new graphs each step). Walk inputs inline rather than
        // via GetInputsArray() — that helper allocates a fresh
        // Tensor<T>[] every call and we'd burn one allocation per
        // node solely to traverse refs we already have direct field
        // access to. Issue #279: the per-train allocation bloat
        // compounds the leak signature.
        //
        // ALSO clear .Grad on intermediate tensors. AccumulateGrad
        // sets `tensor.Grad = stored` for every tensor that receives
        // a gradient contribution, including forward intermediates
        // (Q, K, V, attention scores, layernorm outputs, …). Those
        // intermediates have no caller-visible reason to keep .Grad
        // populated after backward returns — only the `sources`
        // (model parameters) are read by the optimizer. Leaving
        // .Grad set chains the gradient tensor's lifetime to the
        // intermediate's lifetime; when ANY external reference holds
        // the intermediate (a pooled output buffer, a weak-ref
        // tracker, debugging hook, …) the gradient stays live too.
        // Source tensors keep .Grad — that's the optimizer's input.
        // Tensors registered via RetainGrad() also keep .Grad — that's the
        // user's explicit request to retain gradients on a non-leaf tensor.
            if (!_options.Persistent)
            {
                // Build a set of source tensors for O(1) sourcehood check.
                //   sources != null (even if empty): caller specified the
                //     protected set explicitly — keep .Grad on listed
                //     sources, clear on everything else (including the
                //     empty-set case).
                //   sources == null: caller is consuming gradients via
                //     the returned dictionary (the standard pattern in
                //     AiDotNet.NeuralNetworkBase.TrainWithTape). We MUST
                //     still clear .Grad on forward intermediates — a
                //     consumer-side cache that retains a single
                //     intermediate (a layer's `_lastInput`, a streaming
                //     pool retention, …) would otherwise pin one full
                //     backward's worth of gradient tensors across every
                //     successive Train call (reopened-#1227 measured this
                //     at ~1.5 MB/call on a 4-encoder-layer Transformer).
                //     The dict returned to the caller keeps the gradient
                //     references by value — clearing per-tensor .Grad
                //     just severs the back-pointer from intermediates,
                //     not the dict's mapping.
                HashSet<Tensor<T>>? sourceSet = null;
                if (sources is not null)
                {
                    // Reuse the thread-local source set when we own the scratch.
                    sourceSet = acquiredScratch
                        ? BackwardScratch<T>.RentSourceSet()
                        : new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
                    foreach (var s in sources) sourceSet.Add(s);
                }
                // Local helper: should this tensor keep its .Grad?
                //   - Explicit source (caller asked for its grad by reference): yes
                //   - RetainGrad-marked non-leaf: yes
                //   - Otherwise: no — even when sources was null, because the
                //     gradient is preserved in the returned dictionary
                //     regardless of whether .Grad stays set on the tensor.
                //     (See the long comment above for why this is required
                //     to close reopened-#1227.)
                bool ShouldKeepGrad(Tensor<T> t)
                {
                    if (sourceSet?.Contains(t) == true) return true;
                    if (_retainGrad is not null && _retainGrad.Contains(t)) return true;
                    return false;
                }
                // Per-tensor ownership guard (PR #322 review #1): the node-level
                // OwningTape check below ensures we don't process a foreign
                // tape's NODE, but a this-tape node can still have a foreign
                // tape's tensor as INPUT (e.g. checkpoint outputs flowing
                // into the inner tape). Cleaning .GradFn / .Grad on those
                // inputs would corrupt the outer tape's state. Gate per-input
                // cleanup on whether the input's GradFn (its producer)
                // belongs to this tape.
                bool InputOwnedByThisTape(Tensor<T> t)
                {
                    var fn = t.GradFn;
                    if (fn is null) return false;  // leaf — leave it alone
                    return ReferenceEquals(fn.OwningTape, this);
                }
                // Issue #319 Phase 3: while clearing GradFn references on
                // tape tensors, also return the GradNode itself to the
                // pool. Safe only when:
                //   - this isn't a createGraph=true backward (the
                //     higher-order recorder added new ops that
                //     reference these same nodes)
                //   - the tape isn't persistent (the chain is cached
                //     and may be re-executed on the next backward)
                bool canPoolNodes = !DifferentiableOps._isBackwardCreateGraph;

                bool canPoolIntermediates = canPoolNodes;
                foreach (var node in topoOrder)
                {
                    // Cross-tape guard (PR #322 review #17): the topo
                    // sort follows GradFn pointers across tape boundaries
                    // when a shared tensor's GradFn points back to an
                    // outer tape's node (GradientCheckpointing pattern).
                    // We must NOT null GradFn/Grad on those outer-owned
                    // tensors — the outer's cleanup still depends on
                    // them. The graph walk itself is allowed to cross
                    // tape boundaries (higher-order AD / Hvp needs that);
                    // only the destructive cleanup is restricted.
                    bool nodeOwnedByThisTape = ReferenceEquals(node.OwningTape, this);
                    if (!nodeOwnedByThisTape) continue;

                    // node.Output is owned by this tape (verified above). It may
                    // already be null when the streaming backward released it after
                    // consuming it — nothing left to clean in that case.
                    var nodeOutput = node.Output;
                    if (nodeOutput is not null)
                    {
                        nodeOutput.GradFn = null;
                        // Issue #338: clear the tape-pinning flag on every
                        // tape-owned tensor we touch — the backward walk has
                        // consumed it and pooling is safe again.
                        nodeOutput._pinnedByTape = false;
                        if (!ShouldKeepGrad(nodeOutput))
                            nodeOutput.Grad = null;
                    }
                    node.Input0._pinnedByTape = false;
                    if (node.Input1 is not null) node.Input1._pinnedByTape = false;
                    if (node.Input2 is not null) node.Input2._pinnedByTape = false;
                    if (node.InputsOverflow is not null)
                        foreach (var inp in node.InputsOverflow)
                            inp._pinnedByTape = false;

                    // Input0 is non-nullable on GradNode<T>; the recorder
                    // always populates it. But it CAN be a leaf (no GradFn)
                    // or a foreign-tape intermediate (GradFn.OwningTape != this).
                    //   leaf: preserve .Grad — that's the BC contract for
                    //     param.Grad-reading consumers.
                    //   foreign: preserve both .GradFn and .Grad — those
                    //     belong to the outer tape.
                    //   this-tape intermediate: clear both, gated by
                    //     ShouldKeepGrad for the source / RetainGrad cases.
                    if (InputOwnedByThisTape(node.Input0))
                    {
                        node.Input0.GradFn = null;
                        if (!ShouldKeepGrad(node.Input0))
                            node.Input0.Grad = null;
                    }
                    else if (sourceSet is not null && !ShouldKeepGrad(node.Input0))
                    {
                        // Caller's explicit "keep what I listed" contract:
                        // foreign-leaf .Grad wasn't populated by this
                        // backward anyway, so clearing here is a no-op for
                        // nested-tape inputs — but matches the behavior of
                        // the non-Persistent leak-fix path.
                        node.Input0.Grad = null;
                    }

                    if (node.Input1 is not null)
                    {
                        if (InputOwnedByThisTape(node.Input1))
                        {
                            node.Input1.GradFn = null;
                            if (!ShouldKeepGrad(node.Input1))
                                node.Input1.Grad = null;
                        }
                        else if (sourceSet is not null && !ShouldKeepGrad(node.Input1))
                        {
                            node.Input1.Grad = null;
                        }
                    }
                    if (node.Input2 is not null)
                    {
                        if (InputOwnedByThisTape(node.Input2))
                        {
                            node.Input2.GradFn = null;
                            if (!ShouldKeepGrad(node.Input2))
                                node.Input2.Grad = null;
                        }
                        else if (sourceSet is not null && !ShouldKeepGrad(node.Input2))
                        {
                            node.Input2.Grad = null;
                        }
                    }
                    if (node.InputsOverflow is not null)
                    {
                        foreach (var inp in node.InputsOverflow)
                        {
                            if (inp is null) continue;
                            if (InputOwnedByThisTape(inp))
                            {
                                inp.GradFn = null;
                                if (!ShouldKeepGrad(inp))
                                    inp.Grad = null;
                            }
                            else if (sourceSet is not null && !ShouldKeepGrad(inp))
                            {
                                inp.Grad = null;
                            }
                        }
                    }

                    // Pool return is also gated on createGraph mode —
                    // higher-order AD keeps the recorded ops alive past
                    // this cleanup. Already-gated by OwningTape above.
                    if (canPoolNodes)
                    {
                        GradNodePool<T>.Return(node);
                    }

                    // node.Output is intentionally NOT pooled — user code
                    // may still hold it (loss, logged intermediates). Safe
                    // forward-intermediate pooling needs an opt-in API or
                    // Tensor-side liveness tracking.
                }
            }
            else
            {
                // Issue #283 / #338: persistent-tape path. Pre-fix this branch
                // only cleared _pinnedByTape flags; .GradFn / .Grad were left
                // set, so any consumer holding an intermediate tensor (the
                // layer-cache pattern: MultiHeadAttentionLayer's _lastInput
                // etc.) pinned the entire backward graph through that
                // intermediate's GradFn pointer. The chain.Execute uses
                // BackwardStep[] which stores Output / Inputs directly — it
                // does NOT follow tensor.GradFn at execute time — so nulling
                // GradFn here is safe even though the chain is cached for
                // replay. The next forward will re-set GradFn on every
                // intermediate before the next backward runs.
                //
                // sourceSet collection mirrors the !Persistent branch above
                // so the keepGrad rule (explicit sources / RetainGrad)
                // applies consistently.
                HashSet<Tensor<T>>? persistentSourceSet = null;
                if (sources is not null)
                {
                    persistentSourceSet = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
                    foreach (var s in sources) persistentSourceSet.Add(s);
                }
                bool PersistentShouldKeepGrad(Tensor<T> t)
                {
                    if (persistentSourceSet?.Contains(t) == true) return true;
                    if (_retainGrad is not null && _retainGrad.Contains(t)) return true;
                    return false;
                }

                foreach (var node in topoOrder)
                {
                    if (!ReferenceEquals(node.OwningTape, this)) continue;
                    // node.Output may already be null when the streaming backward
                    // released it after consuming it — skip its cleanup in that case.
                    var nodeOutput = node.Output;
                    if (nodeOutput is not null) nodeOutput._pinnedByTape = false;
                    node.Input0._pinnedByTape = false;
                    if (node.Input1 is not null) node.Input1._pinnedByTape = false;
                    if (node.Input2 is not null) node.Input2._pinnedByTape = false;
                    if (node.InputsOverflow is not null)
                        foreach (var inp in node.InputsOverflow)
                            inp._pinnedByTape = false;

                    // Issue #283 fix: destructive cleanup on Output AND on
                    // intermediate Inputs. Output is always an intermediate
                    // (leaves never appear as Output of an entry). Inputs
                    // may be leaves (params / raw inputs) — preserve their
                    // .Grad and don't touch their .GradFn (leaves have
                    // GradFn=null anyway, or it belongs to an outer tape).
                    if (nodeOutput is not null)
                    {
                        nodeOutput.GradFn = null;
                        if (!PersistentShouldKeepGrad(nodeOutput))
                            nodeOutput.Grad = null;
                    }

                    static void ClearIfIntermediate(Tensor<T>? t, GradientTape<T> self,
                        Func<Tensor<T>, bool> shouldKeep, bool canPool)
                    {
                        if (t is null) return;
                        var fn = t.GradFn;
                        if (fn is null) return;
                        if (!ReferenceEquals(fn.OwningTape, self)) return;
                        t.GradFn = null;
                        if (!shouldKeep(t)) t.Grad = null;
                        if (canPool) GradNodePool<T>.Return(fn);
                    }
                    bool canPoolPersist = !DifferentiableOps._isBackwardCreateGraph;
                    ClearIfIntermediate(node.Input0, this, PersistentShouldKeepGrad, canPoolPersist);
                    if (node.InputCount >= 2) ClearIfIntermediate(node.Input1, this, PersistentShouldKeepGrad, canPoolPersist);
                    if (node.InputCount >= 3) ClearIfIntermediate(node.Input2, this, PersistentShouldKeepGrad, canPoolPersist);
                    if (node.InputsOverflow is not null)
                    {
                        foreach (var inp in node.InputsOverflow)
                            ClearIfIntermediate(inp, this, PersistentShouldKeepGrad, canPoolPersist);
                    }

                    // Pool-return this node too (already nulled GradFn on output above).
                    if (canPoolPersist) GradNodePool<T>.Return(node);
                }
            }

            // Drop captured refs in the steps array's live range so they
            // don't extend tensor lifetimes through the scratch pool.
            // Skip for persistent tapes — the chain caches `steps` for
            // re-execution on the next backward call.
            if (!_options.Persistent && acquiredScratch)
            {
                BackwardScratch<T>.ClearStepsRange(stepCount);
            }

            return result;
        }
        finally
        {
            if (acquiredScratch)
            {
                BackwardScratch<T>.Release();
            }
        }
    }


    // Per-call cleanup invoked when the rebindable plan cache hit fires.
    // Walks tape entries, clears GradFn / .Grad on intermediates the
    // caller didn't ask to keep, and returns GradNodes to the pool.
    // Mirrors BOTH the tape-walk path's input cleanup (lines 611-625)
    // AND the graph-walk path's pool-return (lines 1047-1148) — without
    // this, every cache hit pins one backward's worth of gradient
    // tensors AND leaves intermediate inputs' .Grad / .GradFn pointing
    // at recycled GradNodes, which on Linux Server GC manifests as
    // 7-of-14 forward intermediates surviving Gen2 GC (issue #283
    // CI repro: xNorm, q, k, attn-softmax, ctx, residual, h2-relu).
    private void CleanupAfterCachedReplay(
        Dictionary<Tensor<T>, Tensor<T>> result,
        IReadOnlyList<Tensor<T>>? sources)
    {
        if (DifferentiableOps._isBackwardCreateGraph) return;
        HashSet<Tensor<T>>? sourceSet = null;
        if (sources is not null)
        {
            sourceSet = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
            foreach (var s in sources) sourceSet.Add(s);
        }

        // Pre-collect the set of tensors that are outputs of THIS tape's
        // entries — those are the intermediates we own. Inputs that don't
        // appear in this set are leaves (sources, externally-supplied
        // values) and must be left alone: their .Grad is the BC contract
        // the optimizer-step pattern reads, and their .GradFn is either
        // null (leaf) or owned by an OUTER tape (nested-tape pattern).
        var intermediates = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        for (int i = 0; i < _entries.Count; i++)
        {
            var o = _entries[i].Output;
            if (o is not null) intermediates.Add(o);
        }

        for (int i = 0; i < _entries.Count; i++)
        {
            ref var entry = ref _entries[i];
            var output = entry.Output;
            if (output is null) continue;
            var fn = output.GradFn;
            if (fn is not null && ReferenceEquals(fn.OwningTape, this))
            {
                output.GradFn = null;
                // Issue #338 tape-pinning: backward replay has consumed
                // this output (and the entry above re-registered it via
                // Record*). Cleared regardless of source/retain — the pin
                // guards the pool, not .Grad.
                output._pinnedByTape = false;
                bool keepGrad = sourceSet?.Contains(output) == true
                    || (_retainGrad is not null && _retainGrad.Contains(output));
                if (!keepGrad) output.Grad = null;
                GradNodePool<T>.Return(fn);
            }

            // Also clear .GradFn / .Grad on every INPUT that this tape
            // owns. The graph-walk cleanup at the bottom of
            // ComputeGradientsViaGraphCore does this via topoOrder; the
            // tape-walk path does it via _entries (lines 611-625). The
            // cached-replay path was visiting Outputs only — leaving each
            // intermediate's .Grad mirror (set during backward by
            // AccumulateGrad) pinned through the GradNode that was about
            // to be pool-recycled. On Linux Server GC the timing exposed
            // this as 7 surviving intermediates; on Workstation GC the
            // promotion order hid it.
            CleanupCachedReplayInput(entry.Input0, sourceSet, intermediates);
            if (entry.InputCount >= 2) CleanupCachedReplayInput(entry.Input1, sourceSet, intermediates);
            if (entry.InputCount >= 3) CleanupCachedReplayInput(entry.Input2, sourceSet, intermediates);
            if (entry.InputsOverflow is not null)
            {
                foreach (var inp in entry.InputsOverflow)
                    CleanupCachedReplayInput(inp, sourceSet, intermediates);
            }
        }
    }

    private void CleanupCachedReplayInput(
        Tensor<T>? t, HashSet<Tensor<T>>? sourceSet, HashSet<Tensor<T>> intermediates)
    {
        if (t is null) return;
        // Issue #338: clear the tape-pinning flag set by DifferentiableOps.Record*
        // — backward replay has consumed this tensor and pooling is safe again.
        t._pinnedByTape = false;
        bool ownedByThisTape = intermediates.Contains(t);
        if (ownedByThisTape)
        {
            // Idempotent — the Output-loop already cleared this for entries
            // visited earlier in the walk. Cheap to repeat; cheap to skip.
            t.GradFn = null;
        }
        if (sourceSet?.Contains(t) == true) return;
        if (_retainGrad is not null && _retainGrad.Contains(t)) return;
        // Same gating as CleanupTapeEntryGrad: clear .Grad only when
        // the caller explicitly filtered (sourceSet != null) OR the
        // tensor is one of this tape's intermediates. Leaves with no
        // source filter keep their .Grad — that's the optimizer-step
        // BC contract.
        if (sourceSet is not null || ownedByThisTape)
        {
            t.Grad = null;
        }
    }

    private static void TopologicalSort(GradNode<T> node, HashSet<GradNode<T>> visited, List<GradNode<T>> result)
    {
        if (!visited.Add(node)) return;

        // Visit children first (inputs)
        if (node.Input0?.GradFn is not null) TopologicalSort(node.Input0.GradFn, visited, result);
        if (node.Input1?.GradFn is not null) TopologicalSort(node.Input1.GradFn, visited, result);
        if (node.Input2?.GradFn is not null) TopologicalSort(node.Input2.GradFn, visited, result);
        if (node.InputsOverflow is not null)
            foreach (var inp in node.InputsOverflow)
                if (inp.GradFn is not null) TopologicalSort(inp.GradFn, visited, result);

        // Add after children (reverse post-order)
        result.Add(node);
    }

    /// <summary>
    /// Clears all recorded entries from the tape without disposing it.
    /// </summary>
    public void Reset()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GradientTape<T>));
        }

        _entries.Reset();
    }

    /// <summary>
    /// Gets or sets whether anomaly detection is enabled.
    /// When true, each backward function's output is checked for NaN/Inf.
    /// </summary>
    public bool DetectAnomaly { get; set; }

    /// <summary>
    /// When true, logs per-backward-function timing to Console for performance profiling.
    /// </summary>
    public bool ProfileBackward { get; set; }

    /// <summary>
    /// Tensor-specific backward hooks. When a tensor's gradient is computed during backward,
    /// all registered hooks for that tensor are called with the gradient.
    /// Like PyTorch's tensor.register_hook().
    /// </summary>
    private readonly Dictionary<Tensor<T>, List<Func<Tensor<T>, Tensor<T>>>>? _hooks;

    /// <summary>
    /// Set of tensors that should retain their gradients even if they are non-leaf tensors.
    /// Like PyTorch's tensor.retain_grad().
    /// </summary>
    private readonly HashSet<Tensor<T>>? _retainGrad;

    /// <summary>
    /// Hooks keyed by op name — fired during backward for every tape
    /// entry whose OperationName matches. Lazily allocated on first
    /// RegisterNodeHook call so tapes that don't use node hooks pay
    /// nothing.
    /// </summary>
    private Dictionary<string, List<Func<Tensor<T>, Tensor<T>>>>? _nodeHooks;

    /// <summary>
    /// Predicate-based node hooks. Each (match, hook) pair fires for
    /// every entry where match returns true. Linear scan during
    /// backward — keep predicates cheap.
    /// </summary>
    private List<(Func<TapeEntry<T>, bool> Match, Func<Tensor<T>, Tensor<T>> Hook)>? _nodePredicateHooks;

    /// <summary>
    /// Registers a hook on a tensor that will be called with its gradient during backward.
    /// The hook can modify the gradient by returning a new tensor.
    /// </summary>
    /// <param name="tensor">The tensor to hook.</param>
    /// <param name="hook">Function receiving gradient, returning (possibly modified) gradient.</param>
    public void RegisterHook(Tensor<T> tensor, Func<Tensor<T>, Tensor<T>> hook)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GradientTape<T>));
        var hooks = _hooks ?? throw new InvalidOperationException(
            "Hooks require GradientTapeOptions with EnableHooks=true");
        if (!hooks.TryGetValue(tensor, out var list))
        {
            list = new List<Func<Tensor<T>, Tensor<T>>>();
            hooks[tensor] = list;
        }
        list.Add(hook);
    }

    /// <summary>
    /// Registers a hook against a graph node identified by the
    /// recorded operation name. Fires during backward for every tape
    /// entry whose <see cref="TapeEntry{T}.OperationName"/> matches
    /// <paramref name="opName"/>, with the gradient flowing into that
    /// entry's output. Matches the spirit of
    /// <c>torch.autograd.graph.Node.register_hook</c> and survives
    /// fusion better than tensor-keyed hooks because the user
    /// registers against the op identity, not a specific
    /// <see cref="Tensor{T}"/> reference that may be eliminated by
    /// graph rewrites.
    /// </summary>
    /// <param name="opName">The operation name to match (case
    /// sensitive) — typically the value passed to
    /// <see cref="DifferentiableOps.RecordIfActive"/>.</param>
    /// <param name="hook">Function receiving the gradient flowing
    /// into the node's output, returning a (possibly modified) one.</param>
    /// <exception cref="ObjectDisposedException">Tape was disposed.</exception>
    /// <exception cref="InvalidOperationException">Hooks are not
    /// enabled on the tape options.</exception>
    /// <exception cref="ArgumentNullException">Either argument is null.</exception>
    public void RegisterNodeHook(string opName, Func<Tensor<T>, Tensor<T>> hook)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GradientTape<T>));
        if (opName is null) throw new ArgumentNullException(nameof(opName));
        if (hook is null) throw new ArgumentNullException(nameof(hook));
        if (_hooks is null) throw new InvalidOperationException(
            "Hooks require GradientTapeOptions with EnableHooks=true");
        _nodeHooks ??= new Dictionary<string, List<Func<Tensor<T>, Tensor<T>>>>(StringComparer.Ordinal);
        if (!_nodeHooks.TryGetValue(opName, out var list))
        {
            list = new List<Func<Tensor<T>, Tensor<T>>>();
            _nodeHooks[opName] = list;
        }
        list.Add(hook);
    }

    /// <summary>
    /// Predicate-based variant of <see cref="RegisterNodeHook(string, Func{Tensor{T}, Tensor{T}})"/>.
    /// Fires for every entry where <paramref name="match"/> returns
    /// <c>true</c>. Lets callers attach to fused replacements whose
    /// op name they don't know up front (e.g. "any entry whose
    /// inputs include this Linear weight").
    /// </summary>
    public void RegisterNodeHook(
        Func<TapeEntry<T>, bool> match,
        Func<Tensor<T>, Tensor<T>> hook)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GradientTape<T>));
        if (match is null) throw new ArgumentNullException(nameof(match));
        if (hook is null) throw new ArgumentNullException(nameof(hook));
        if (_hooks is null) throw new InvalidOperationException(
            "Hooks require GradientTapeOptions with EnableHooks=true");
        _nodePredicateHooks ??= new List<(Func<TapeEntry<T>, bool>, Func<Tensor<T>, Tensor<T>>)>();
        _nodePredicateHooks.Add((match, hook));
    }

    /// <summary>
    /// Marks a tensor to retain its gradient after backward, even if it's a non-leaf tensor.
    /// Like PyTorch's tensor.retain_grad().
    /// </summary>
    public void RetainGrad(Tensor<T> tensor)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GradientTape<T>));
        var retain = _retainGrad ?? throw new InvalidOperationException(
            "RetainGrad requires GradientTapeOptions with EnableHooks=true");
        retain.Add(tensor);
    }

    /// <summary>
    /// Compiles the backward graph for repeated execution on persistent tapes.
    /// Dead node elimination removes entries not reachable from the loss tensor.
    /// </summary>
    /// <param name="loss">The loss tensor to differentiate.</param>
    /// <param name="sources">Optional tensors to compute gradients for.</param>
    /// <returns>A compiled backward graph that can be executed multiple times efficiently.</returns>
    public CompiledBackwardGraph<T> CompileBackward(Tensor<T> loss, Tensor<T>[]? sources = null)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GradientTape<T>));
        if (!_options.Persistent)
            throw new InvalidOperationException("CompileBackward requires a persistent tape.");
        if (_entries.Count == 0)
            throw new InvalidOperationException("Cannot compile: the tape has no recorded operations.");
        if (loss.Length != 1)
            throw new ArgumentException($"CompileBackward requires a scalar loss tensor (length 1), got length {loss.Length}.", nameof(loss));

        return new CompiledBackwardGraph<T>(_entries, loss, sources, _engine, _retainGrad);
    }

    /// <summary>
    /// Computes gradients and applies them in a single pass.
    /// Avoids the overhead of building a gradient dictionary then iterating it.
    /// </summary>
    /// <param name="loss">The loss tensor.</param>
    /// <param name="parameters">Parameter tensors to update.</param>
    /// <param name="learningRate">SGD learning rate.</param>
    public void GradientAndUpdate(Tensor<T> loss, Tensor<T>[] parameters, T learningRate)
    {
        var grads = ComputeGradients(loss, sources: parameters);
        var numOps = Helpers.MathHelper.GetNumericOperations<T>();

        foreach (var param in parameters)
        {
            if (grads.TryGetValue(param, out var grad))
            {
                if (param.Length != grad.Length)
                    throw new InvalidOperationException(
                        $"Gradient length ({grad.Length}) does not match parameter length ({param.Length}). " +
                        "This indicates a shape mismatch in the backward pass.");
                // In-place SGD: param -= lr * grad
                for (int i = 0; i < param.Length; i++)
                    param[i] = numOps.Subtract(param[i], numOps.Multiply(learningRate, grad[i]));
            }
        }

        // Note: ComputeGradients() already handles auto-compilation recording
        // (RecordStep + TryCompileBackward) — no need to duplicate here.
    }

    /// <summary>
    /// Finalizer backstop for the process-wide
    /// <see cref="DifferentiableOps._anyTapeActive"/> counter, which the ctor
    /// increments and only <see cref="Dispose"/> decrements. A stuck-positive
    /// count forces every op on every thread down the tape-recording slow path
    /// and can flip tape-gated dispatch (e.g. the BlasManaged GEMM packed path
    /// keys on <see cref="Current"/> being null), so we decrement the global on
    /// GC. Only the Interlocked global is touched: the ThreadStatic
    /// <c>_current</c> / <c>_threadTapeDepth</c> belong to the originating thread
    /// and must never be poked from the finalizer thread.
    ///
    /// <para><b>Scope of this backstop.</b> A tape is only eligible for GC (and
    /// hence finalization) once nothing roots it — and an undisposed tape stays
    /// rooted by this thread's <c>[ThreadStatic] _current</c> slot until that
    /// thread either clears <c>_current</c> or exits. So this finalizer heals
    /// the realistic production leak: an undisposed tape on a <i>transient</i>
    /// worker/request thread, whose TLS is released when the thread ends,
    /// letting the tape collect and the global count self-correct. It does
    /// <i>not</i> rescue an undisposed tape pinned by a <i>long-lived</i>
    /// thread's <c>_current</c> — there the global stays elevated until that
    /// thread is reused (each <see cref="GradientTape{T}"/> ctor overwrites
    /// <c>_current</c> with the new tape) or exits. The real fix for that case
    /// is disposal: always wrap tape usage in <c>using</c>/try-finally. In the
    /// test harness, <c>TapeIsolationGuard</c> clears <c>_current</c> after every
    /// test, which both unroots any leaked tape (so this finalizer can run) and
    /// prevents same-thread cross-test contamination.</para>
    /// </summary>
    ~GradientTape()
    {
        if (!_disposed)
            System.Threading.Interlocked.Decrement(ref DifferentiableOps._anyTapeActive);
    }

    /// <summary>
    /// Disposes the tape and restores the parent tape (if any) as the current tape.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        GC.SuppressFinalize(this);
        // Restore the previous ReplayMode instead of hard-resetting.
        // This is critical for nested tapes: an inner tape disposing must not
        // clear replay suppression that an outer tape still needs.
        Compilation.AutoTrainingCompiler.ReplayMode = _savedReplayMode;
        System.Threading.Interlocked.Decrement(ref DifferentiableOps._anyTapeActive);
        DifferentiableOps._threadTapeDepth--;
        SetCurrentTape(_parent);
        // AiDotNet#1340: explicitly clear the cached delegate chain's
        // per-step references (Output / Inputs / SavedState / Backward)
        // BEFORE nulling the chain pointer. Without this, if any external
        // reference path retains the chain instance (finalizer ordering,
        // async continuations, BackwardScratch pool entries), the
        // `_steps[]` array continues to pin 30+ tensor references per
        // backward step — measured at ~79 KB/call retention on a 164k-param
        // Transformer L=2 chain, projecting to ~3.8 GB at 50k Train calls.
        // The Clear() walk zeros each BackwardStep<T> in-place; the chain
        // becomes safe-to-not-execute (no-op) once cleared.
        _cachedDelegateChain?.Clear();
        _cachedDelegateChain = null;

        // Issue #283 fix: invalidate ONLY this tape's forward-recorded
        // outputs from the GPU activation cache BEFORE resetting the
        // entries (reset zeros Output refs, leaving nothing to walk).
        // The cache is designed for inference-side layer chaining, but
        // during TRAINING each forward op registers a deferred-download
        // materializer keyed on its result array. Those registrations
        // strong-ref the result arrays — and once user code holds an
        // intermediate externally (the layer-cache pattern: AiDotNet's
        // MultiHeadAttentionLayer _lastQueryInput etc.), the array →
        // cache-entry → GPU-buffer chain pins ~16-32 KB per cached
        // activation for the rest of the process. Measured 547 KB/call
        // retention on the 4L Transformer NullSources probe; this fix
        // drops it to 0 B/call.
        //
        // Why per-tape-entry instead of ClearActivationCache: a tape's
        // forward might run AFTER a separate inference-chain that
        // populated the cache with entries the outer scope still wants
        // (e.g. cached weights/biases the next inference reuses).
        // Wholesale clearing breaks ~64 unrelated test scenarios that
        // share the cache across tape boundaries. Walking _entries
        // touches only the recorded outputs of THIS tape — exactly
        // what's at risk of being pinned across our Dispose.
        //
        // Only the OUTERMOST tape invalidates — nested tapes
        // (Hvp/Hessian) must not clear their parent's pending
        // materializers mid-backward. Detect outermost via
        // _parent == null.
        //
        // 2026-06-05: this used to walk _entries and invalidate only each recorded
        // op's `entry.Output`. That was INCOMPLETE — activations cached under keys
        // that were never a recorded tape Output (e.g. the [B,H,ctx,ctx] attention
        // scores re-uploaded during backward) leaked: gcroot traced 36GB+ of live
        // float[] back to DirectGpuTensorEngine._activationCache surviving across
        // training steps. The deterministic fix releases EVERY activation this
        // forward+backward produced (timestamp > the snapshot captured at ctor),
        // which is a strict superset of the old per-Output walk and bounds memory
        // to ~one step. Entries created before the tape are preserved, so the
        // cross-tape inference-reuse scenarios the old walk protected still hold.
        // Guard on the SAME engine instance the snapshot came from (CPU<->GPU
        // rebind, #350); the byte/managed caps remain as a backstop either way.
        if (_parent is null && _snapshotEngine is not null
            && ReferenceEquals(_snapshotEngine, _engine))
        {
            // Pair the SuspendActivationEviction taken in the ctor BEFORE the
            // deterministic eviction below. ResumeActivationEviction first so
            // EvictActivationsCreatedAfter (which respects the suspend flag)
            // actually runs — otherwise the step's activations would linger
            // until the next non-suspended insert and the steady-state memory
            // bound (~one step) would not hold for back-to-back tape scopes.
            _snapshotEngine.ResumeActivationEviction();
            _snapshotEngine.EvictActivationsCreatedAfter(_activationSnapshot);
        }

        // Return arena to thread-local cache for reuse by next GradientTape
        _entries.Reset();
        _cachedArena = _entries;
    }

    // ──────────────────────────────────────────────────────────────
    // NoGradScope: suppress tape recording during inference
    // ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Suppresses gradient tape recording within its scope.
    /// Like PyTorch's torch.no_grad() — zero overhead for inference.
    /// </summary>
    /// <example>
    /// using (GradientTape&lt;float&gt;.NoGrad())
    /// {
    ///     var output = engine.TensorMatMul(x, w); // NOT recorded
    /// }
    /// </example>
    public static NoGradScope<T> NoGrad() => new();

    /// <summary>
    /// Enters an <see cref="InferenceModeScope{T}"/> — strictly stronger than
    /// <see cref="NoGrad"/>. Inference mode suppresses tape recording AND
    /// tells the engine that tensor version counters will not be consulted,
    /// so in-place ops that would normally be blocked by autograd's
    /// version-check invariant become legal. PyTorch distinguishes
    /// <c>no_grad</c> from <c>inference_mode</c> for the same reason.
    /// </summary>
    /// <example>
    /// using (GradientTape&lt;float&gt;.InferenceMode())
    /// {
    ///     var y = engine.TensorMatMul(x, w);     // NOT recorded
    ///     engine.TensorAddInPlace(y, bias);      // in-place allowed
    /// }
    /// </example>
    public static InferenceModeScope<T> InferenceMode() => new();
}

/// <summary>
/// Suppresses gradient tape recording. Operations performed while this scope
/// is active will not be recorded to any GradientTape, enabling zero-overhead inference.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class NoGradScope<T> : IDisposable
{
    [ThreadStatic]
    private static int _suppressionCount;

    private bool _disposed;

    /// <summary>
    /// Gets whether tape recording is currently suppressed on this thread.
    /// Checked by DifferentiableOps.RecordIfActive.
    /// </summary>
    public static bool IsSuppressed => _suppressionCount > 0;

    /// <summary>
    /// Test-isolation hook: clears this thread's suppression count. A test that
    /// enters a <see cref="NoGradScope{T}"/> and never disposes it (exception
    /// path) otherwise leaves recording suppressed for every later test on the
    /// thread, silently zeroing their gradients. Not part of the public API.
    /// </summary>
    internal static void ResetForTests() => _suppressionCount = 0;

    /// <summary>
    /// Creates a new NoGradScope, incrementing the suppression counter.
    /// Supports nesting: multiple scopes can be active simultaneously.
    /// </summary>
    public NoGradScope()
    {
        _suppressionCount++;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _suppressionCount--;
    }

    /// <summary>
    /// Increments the suppression counter without allocating a scope.
    /// Used by <see cref="InferenceModeScope{T}"/> so entering inference
    /// mode also counts as entering no-grad (since inference mode is
    /// strictly stronger). Matched exactly once by
    /// <see cref="DecrementSuppressionCount"/>.
    /// </summary>
    internal static void IncrementSuppressionCount() => _suppressionCount++;

    /// <summary>
    /// Pair to <see cref="IncrementSuppressionCount"/>. Must not be
    /// called without a matching increment or the counter will go
    /// negative and <see cref="IsSuppressed"/> will report the wrong
    /// value.
    /// </summary>
    internal static void DecrementSuppressionCount() => _suppressionCount--;
}

/// <summary>
/// Strictly stronger than <see cref="NoGradScope{T}"/>: suppresses tape
/// recording AND marks that tensor version counters will not be consulted
/// while the scope is active, so in-place ops are legal. Matches PyTorch's
/// <c>torch.inference_mode()</c>.
/// </summary>
/// <remarks>
/// <para><b>Relationship to <see cref="NoGradScope{T}"/>:</b></para>
/// <para>
/// An active inference mode scope counts as "no-grad suppression" for
/// <see cref="DifferentiableOps.IsRecording{T}"/> so the hot path fast-exits
/// identically. On top of that, <see cref="IsActive"/> lets engine code
/// (or in-place op implementations) skip the version-counter bump that
/// would otherwise fire for tensor mutation, which is the feature that
/// unlocks in-place arithmetic on inference inputs.
/// </para>
/// <para><b>Nesting:</b> multiple scopes may be open simultaneously;
/// the deepest scope wins and all enclosing scopes remain effective.</para>
/// <para><b>Thread-local:</b> the flag is <see cref="ThreadStaticAttribute"/>
/// so one thread entering inference mode does not affect other threads.</para>
/// </remarks>
/// <typeparam name="T">The numeric type the tape is generic over.</typeparam>
public sealed class InferenceModeScope<T> : IDisposable
{
    [ThreadStatic]
    private static int _activeCount;

    private bool _disposed;

    /// <summary>
    /// Gets whether an inference-mode scope is active on this thread.
    /// Checked by in-place op implementations to skip version-counter
    /// bumping and any related autograd bookkeeping.
    /// </summary>
    public static bool IsActive => _activeCount > 0;

    /// <summary>
    /// Test-isolation hook: clears this thread's inference-scope count. A test
    /// that enters an <see cref="InferenceModeScope{T}"/> and never disposes it
    /// (exception path) otherwise leaves in-place mutation legal and recording
    /// suppressed for every later test on the thread. Not part of the public API.
    /// </summary>
    internal static void ResetForTests() => _activeCount = 0;

    /// <summary>
    /// Creates a new scope. Increments three counters: the NoGrad
    /// suppression counter (InferenceMode is strictly stronger so
    /// <see cref="NoGradScope{T}.IsSuppressed"/> reports true while
    /// active), the typed inference-mode counter (so
    /// <see cref="IsActive"/> reports true), and the type-erased
    /// counter consulted by non-generic <see cref="LinearAlgebra.TensorBase"/>
    /// in-place mutators that don't know <typeparamref name="T"/> at
    /// the call site.
    /// </summary>
    public InferenceModeScope()
    {
        NoGradScope<T>.IncrementSuppressionCount();
        _activeCount++;
        InferenceModeFlag.Enter();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _activeCount--;
        InferenceModeFlag.Exit();
        NoGradScope<T>.DecrementSuppressionCount();
    }
}

/// <summary>
/// Type-erased thread-local counter shared by every
/// <see cref="InferenceModeScope{T}"/>, regardless of <c>T</c>.
/// Read by non-generic in-place op implementations
/// (<see cref="LinearAlgebra.TensorBase.IncrementVersion"/>,
/// <see cref="TapeEntry{T}.ValidateInputVersions"/>) that need to
/// know whether <i>any</i> inference scope is active without
/// knowing the element type at the call site.
/// </summary>
/// <remarks>
/// Maintained as a sibling to <see cref="InferenceModeScope{T}"/>
/// rather than replacing the typed counter — typed callers can
/// still query the <c>T</c>-specific scope when they need to
/// distinguish (e.g. to apply a dtype-specific optimisation) and
/// the type-erased flag covers the cross-cutting in-place-mutation
/// hot path.
/// </remarks>
public static class InferenceModeFlag
{
    [ThreadStatic]
    private static int _activeCount;

    /// <summary>
    /// True when at least one <see cref="InferenceModeScope{T}"/> is
    /// active on the calling thread for any <c>T</c>. Read by
    /// in-place tensor mutation paths to skip version-counter bumps
    /// (mutation is legal under inference mode) and tape-entry
    /// version validation (no autograd recording took place, so
    /// nothing to validate).
    /// </summary>
    public static bool IsActive => _activeCount > 0;

    internal static void Enter() => _activeCount++;

    internal static void Exit() => _activeCount--;

    /// <summary>
    /// Test-isolation hook: clears the type-erased inference-mode count for
    /// this thread. Paired with <see cref="InferenceModeScope{T}.ResetForTests"/>
    /// so a leaked inference scope cannot contaminate later tests. Not public.
    /// </summary>
    internal static void ResetForTests() => _activeCount = 0;
}

/// <summary>
/// Typed reference equality comparer for gradient dictionaries.
/// Ensures tensor identity (not value equality) is used for gradient mapping.
/// </summary>
internal sealed class ReferenceEqualityComparer<TItem> : IEqualityComparer<TItem> where TItem : class
{
    public static readonly ReferenceEqualityComparer<TItem> Instance = new();

    private ReferenceEqualityComparer() { }

    public bool Equals(TItem? x, TItem? y) => ReferenceEquals(x, y);

    public int GetHashCode(TItem obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
}
