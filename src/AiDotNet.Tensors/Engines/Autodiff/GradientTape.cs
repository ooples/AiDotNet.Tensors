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

    private readonly GradientTape<T>? _parent;
    private readonly TapeEntryArena<T> _entries;
    private readonly GradientTapeOptions _options;
    private readonly IEngine _engine;
    private readonly bool _savedReplayMode; // Saved ReplayMode from outer scope for nested tapes
    private bool _disposed;


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

    public GradientTape(GradientTapeOptions? options = null)
    {
        _options = options ?? GradientTapeOptions.Default;
        // Reuse cached arena if available, otherwise create new one
        _entries = _cachedArena ?? new TapeEntryArena<T>();
        _cachedArena = null; // Take ownership — will return on Dispose
        _entries.Reset();
        _engine = AiDotNetEngine.Current;
        _parent = _current;
        _savedReplayMode = Compilation.AutoTrainingCompiler.ReplayMode;

        if (_options.EnableHooks)
        {
            _hooks = new Dictionary<Tensor<T>, List<Func<Tensor<T>, Tensor<T>>>>(
                ReferenceEqualityComparer<Tensor<T>>.Instance);
            _retainGrad = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        }

        SetCurrentTape(this);
        System.Threading.Interlocked.Increment(ref DifferentiableOps._anyTapeActive);
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
    public Dictionary<Tensor<T>, Tensor<T>> ComputeGradients(
        Tensor<T> loss,
        IReadOnlyList<Tensor<T>>? sources = null,
        bool createGraph = false)
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
        bool hasHooksRegistered = _hooks is not null && _hooks.Count > 0;
        if (_options.Persistent && !createGraph)
        {
            var compiledBwd = Compilation.AutoTrainingCompiler.TryGetCompiledBackward(this, loss, sources?.ToArray());
            if (compiledBwd is not null)
            {
                return compiledBwd.Execute(loss);
            }
        }

        // Graph-based backward: walk GradFn pointers instead of tape.
        // This is faster because it skips tape traversal, dict lookups, and relevance checks.
        // Skip graph path when anomaly detection or hooks are enabled — the tape path handles those.
        if (loss.GradFn is not null && !createGraph && !DetectAnomaly && !hasHooksRegistered)
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

        // Dictionary facade: backward functions still receive Dictionary<Tensor<T>, Tensor<T>>.
        // AccumulateGrad writes to both the array and dictionary.
        var grads = new Dictionary<Tensor<T>, Tensor<T>>(
            Math.Min(gradIndexCount + 1, 1024),
            ReferenceEqualityComparer<Tensor<T>>.Instance);

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
            var numOpsForAnomaly = DetectAnomaly ? MathHelper.GetNumericOperations<T>() : null;

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

                // Apply tensor hooks (skip dictionary lookup entirely when no hooks registered)
                if (hasHooks && _hooks!.TryGetValue(entry.Output, out var hookList))
                {
                    foreach (var hook in hookList)
                        gradOutput = hook(gradOutput);
                    grads[entry.Output] = gradOutput;
                }

                // Validate that no input tensor was mutated after recording
                entry.ValidateInputVersions();

                // Invoke the backward function
                entry.Backward(gradOutput, inputsArray, entry.Output, entry.SavedState ?? Array.Empty<object>(), engine, grads);

                // Performance profiling (only when explicitly enabled)
                // Timing wraps the backward call above — the Stopwatch overhead
                // is negligible relative to backward computation cost.
                if (profileEnabled)
                {
                    System.Console.WriteLine($"  backward[{entry.OperationName}]");
                }

                // Anomaly detection (only when explicitly enabled)
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
                                    throw new ArithmeticException(
                                        $"Op '{entry.OperationName}' backward produced {(double.IsNaN(val) ? "NaN" : "Inf")} " +
                                        $"at input gradient index {k}. Check forward inputs for numerical issues.");
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
        // If we have a cached delegate chain from a previous backward, replay it directly
        if (_cachedDelegateChain is not null)
        {
            return _cachedDelegateChain.Execute(loss, sources, _engine);
        }

        var engine = _engine;

        // Topological sort via DFS from loss.GradFn — build the execution order
        var visited = new HashSet<GradNode<T>>();
        var topoOrder = new List<GradNode<T>>();
        TopologicalSort(loss.GradFn!, visited, topoOrder);

        // Build delegate chain from topological order (capture for replay)
        var steps = new BackwardStep<T>[topoOrder.Count];
        for (int i = 0; i < topoOrder.Count; i++)
        {
            var node = topoOrder[topoOrder.Count - 1 - i]; // reverse for backward order
            steps[i] = new BackwardStep<T>
            {
                Output = node.Output,
                Inputs = node.GetInputsArray(),
                Backward = node.Backward,
                SavedState = node.SavedState
            };
        }

        var chain = new CompiledDelegateChain<T>(steps);

        // Cache for persistent tapes (same network structure every step)
        if (_options.Persistent)
            _cachedDelegateChain = chain;

        // Execute the chain
        var result = chain.Execute(loss, sources, engine);

        // Clear GradFn to release graph memory (non-persistent tapes get new graphs each step)
        if (!_options.Persistent)
        {
            foreach (var node in topoOrder)
            {
                node.Output.GradFn = null;
                foreach (var inp in node.GetInputsArray())
                    inp.GradFn = null;
            }
        }

        return result;
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

        return new CompiledBackwardGraph<T>(_entries, loss, sources, _engine);
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
    /// Disposes the tape and restores the parent tape (if any) as the current tape.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Restore the previous ReplayMode instead of hard-resetting.
        // This is critical for nested tapes: an inner tape disposing must not
        // clear replay suppression that an outer tape still needs.
        Compilation.AutoTrainingCompiler.ReplayMode = _savedReplayMode;
        System.Threading.Interlocked.Decrement(ref DifferentiableOps._anyTapeActive);
        SetCurrentTape(_parent);
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
    /// Creates a new scope. Increments both the NoGrad suppression
    /// counter (so <see cref="NoGradScope{T}.IsSuppressed"/> reports
    /// true — InferenceMode is strictly stronger) and the inference-mode
    /// counter (so <see cref="IsActive"/> reports true).
    /// </summary>
    public InferenceModeScope()
    {
        NoGradScope<T>.IncrementSuppressionCount();
        _activeCount++;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _activeCount--;
        NoGradScope<T>.DecrementSuppressionCount();
    }
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
