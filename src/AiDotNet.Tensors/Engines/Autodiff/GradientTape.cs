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
    private bool _disposed;

    /// <summary>
    /// Gets the number of operations recorded on this tape.
    /// </summary>
    public int EntryCount => _entries.Count;


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

        if (_options.EnableHooks)
        {
            _hooks = new Dictionary<Tensor<T>, List<Func<Tensor<T>, Tensor<T>>>>(
                ReferenceEqualityComparer<Tensor<T>>.Instance);
            _retainGrad = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        }

        SetCurrentTape(this);
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

        var engine = _engine;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Gradient accumulator: maps each tensor (by reference identity) to its accumulated gradient.
        // Pre-size to entry count to avoid 5-6 dictionary resizes during backward walk.
        var grads = new Dictionary<Tensor<T>, Tensor<T>>(
            Math.Min(_entries.Count + 1, 1024),
            ReferenceEqualityComparer<Tensor<T>>.Instance);

        // Seed: gradient of loss w.r.t. itself is ones with the same shape.
        // Fast path for scalar loss (the overwhelmingly common case in training).
        // Reuse cached scalar seed across training steps to avoid per-backward allocation.
        Tensor<T> seedGrad;
        if (loss.Length == 1)
        {
            seedGrad = _cachedScalarSeed ??= new Tensor<T>(new[] { numOps.One }, new[] { 1 });
        }
        else
        {
            var onesData = new T[loss.Length];
            var one = numOps.One;
            for (int j = 0; j < onesData.Length; j++)
                onesData[j] = one;
            seedGrad = new Tensor<T>(onesData, loss.Shape.ToArray());
        }
        grads[loss] = seedGrad;

        // When createGraph=false (default): suspend recording so backward engine calls
        // don't append to this tape — they'd corrupt persistent tapes and shift bounded tapes.
        // When createGraph=true: KEEP recording so backward ops are on the tape for
        // higher-order differentiation (gradient of gradient).
        var savedCurrent = _current;
        if (!createGraph)
        {
            SetCurrentTape(null);
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
            // downstream of those sources. During the backward walk, skip entries whose output
            // is not in this reachable set — their gradients don't contribute to any requested
            // source gradient. This prunes subgraphs that don't depend on the sources.
            HashSet<Tensor<T>>? relevantTensors = null;
            if (sources is not null && sources.Count > 0)
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
                if (profileEnabled)
                {
                    // Profiling is rare — accept the overhead of Stopwatch only when active
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

            if (!_options.Persistent)
            {
                _entries.Reset();
            }

            return filtered;
        }

        if (!_options.Persistent)
        {
            _entries.Reset();
        }

        return grads;
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
    }

    /// <summary>
    /// Disposes the tape and restores the parent tape (if any) as the current tape.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
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
