using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Manages a lazy computation graph. When active, tensor operations create LazyNodes
/// instead of executing. On Realize(), the graph is optimized (fused) and executed.
///
/// Lifecycle: Enable() → record ops → Realize() or Dispose() → compiled execution
///
/// For training: CompileTraining() produces a CompiledTrainingPlan that caches the
/// compiled forward+backward for zero-overhead replay across training steps.
/// </summary>
internal sealed class LazyTensorScope : IDisposable
{
    private readonly LazyTensorScope? _parent;
    private readonly List<ILazyNode> _nodes = new();
    private readonly TensorStorageLeaseSet _storageLeases = new();
    private IEngine _engine;
    private bool _engineExplicitlyBound;
    private bool _disposed;
    private bool _realized;
    private readonly GraphTraceKind _traceKind;

    internal LazyTensorScope(
        LazyTensorScope? parent,
        GraphTraceKind traceKind = GraphTraceKind.Compatibility)
    {
        _parent = parent;
        _traceKind = traceKind;
        // Default to AiDotNetEngine.Current; the first op recorded into the
        // scope will rebind via BindEngineIfUnset to the engine instance the
        // user actually invoked the op on. This matters because the global
        // ambient engine on auto-detect-GPU systems is DirectGpuTensorEngine,
        // but a test or production code that explicitly creates a new
        // CpuEngine() and calls operations on it expects the compiled plan to
        // replay on CPU — not on the (potentially mismatched) ambient engine.
        // See issue #350: T=double rank-3 BatchNorm compile-replay diverged
        // on hosts where the auto-detected GPU engine doesn't have a faithful
        // CPU-tensor BatchNorm path.
        _engine = AiDotNetEngine.Current;
        _engineExplicitlyBound = false;
    }

    /// <summary>
    /// Binds this scope to <paramref name="engine"/> if no engine has yet
    /// been explicitly bound. Called by each engine's GraphMode-recording
    /// branch (e.g. <see cref="CpuEngine.BatchNorm{T}"/>) so the compiled
    /// plan replays on the engine the user actually invoked the op on,
    /// not on the global ambient engine. Idempotent — subsequent calls with
    /// any engine are no-ops once the first binding lands.
    /// </summary>
    internal void BindEngineIfUnset(IEngine engine)
    {
        if (_engineExplicitlyBound) return;
        if (engine is null) return;
        _engine = engine;
        _engineExplicitlyBound = true;
    }

    /// <summary>Number of lazy operations recorded.</summary>
    internal int NodeCount => _nodes.Count;

    /// <summary>
    /// The explicit purpose of this trace. Composite kernels use this to distinguish fused
    /// inference replay from differentiable training/compatibility capture.
    /// </summary>
    internal GraphTraceKind TraceKind => _traceKind;

    /// <summary>
    /// Records a unary operation as a lazy node. Returns a tensor whose data
    /// is not yet computed — it will be materialized during Realize().
    /// </summary>
    internal Tensor<T> RecordUnary<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T> input,
        int[] outputShape,
        Action<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        // Allocate output tensor with correct shape but uninitialized data
        var output = TensorAllocator.RentUninitialized<T>(outputShape);

        var node = new LazyNode<T>(opType, opName, input, output, execute, backwardFn, savedState);
        output.LazySource = node;
        AddNode(node);

        return output;
    }

    /// <summary>Records a cross-type operation (input type differs from output type).</summary>
    internal Tensor<TOut> RecordCrossType<TIn, TOut>(
        LazyNodeType opType,
        string opName,
        Tensor<TIn> input,
        int[] outputShape,
        Action<IEngine, Tensor<TOut>> execute)
    {
        GraphMode.ThrowIfInferenceUnsupported(GraphCaptureLimitation.MixedElementTypes);

        var output = TensorAllocator.RentUninitialized<TOut>(outputShape);
        var node = new CrossTypeLazyNode<TIn, TOut>(opType, opName, input, output, execute);
        output.LazySource = node;
        AddNode(node);
        return output;
    }

    /// <summary>
    /// Records a cross-type operation WITH a backward — the mixed-precision (#555) cast node. Identical
    /// to <see cref="RecordCrossType{TIn,TOut}"/> for the forward, but the node carries a
    /// <see cref="CrossTypeBackwardFunction{TIn,TOut}"/> so the compiled backward walk can bridge the
    /// gradient from the output's dtype grad space into the input's.
    /// </summary>
    internal Tensor<TOut> RecordCrossTypeWithBackward<TIn, TOut>(
        LazyNodeType opType,
        string opName,
        Tensor<TIn> input,
        int[] outputShape,
        Action<IEngine, Tensor<TOut>> execute,
        CrossTypeBackwardFunction<TIn, TOut> backwardFn,
        object[]? savedState = null)
    {
        var output = TensorAllocator.RentUninitialized<TOut>(outputShape);
        var node = new CrossTypeLazyNode<TIn, TOut>(opType, opName, input, output, execute, backwardFn, savedState);
        output.LazySource = node;
        AddNode(node);
        return output;
    }

    /// <summary>Records a binary operation as a lazy node.</summary>
    internal Tensor<T> RecordBinary<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T> input0,
        Tensor<T> input1,
        int[] outputShape,
        Action<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        var output = TensorAllocator.RentUninitialized<T>(outputShape);

        var node = new LazyNode<T>(opType, opName, input0, input1, output, execute, backwardFn, savedState);
        output.LazySource = node;
        AddNode(node);

        return output;
    }

    /// <summary>
    /// Records a pure metadata-view op (contiguous <c>Reshape</c>, <c>Squeeze</c>,
    /// <c>Unsqueeze</c>, <c>Permute</c> on already-contiguous tensors, ...).
    /// Unlike <see cref="RecordUnary"/>, the output tensor is supplied by the
    /// caller and shares storage with the input — a fresh output buffer would
    /// defeat the whole point of a view. The recorded execute delegate is a
    /// no-op because writes to the producer's buffer are live-visible through
    /// the view.
    ///
    /// This exists so that a <c>Func&lt;Tensor&lt;T&gt;&gt;</c> forward that
    /// ends in a pure-view op still hands the compiler a tensor with a
    /// <see cref="Tensor{T}.LazySource"/>, which lets
    /// <see cref="CompiledInferencePlan{T}.Compile"/> pick up the caller's
    /// actual return value as <c>_finalOutput</c> instead of guessing via
    /// the last-step heuristic.
    /// </summary>
    internal Tensor<T> RecordView<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T> input,
        Tensor<T> view,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        // The caller already built the view with shared storage — we just
        // attach it to the graph so the compile step sees a producing node.
        var node = new LazyNode<T>(opType, opName, input, view,
            execute: (_, _) => { /* storage is shared; nothing to compute */ },
            backwardFn,
            savedState);
        view.LazySource = node;
        AddNode(node);
        return view;
    }

    /// <summary>
    /// Records a unary operation whose caller has already materialized the output tensor during
    /// tracing. Replay still executes <paramref name="execute"/> because the output owns storage
    /// independent of the input. This is the materializing counterpart to <see cref="RecordView"/>.
    /// </summary>
    internal Tensor<T> RecordMaterializedUnary<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T> input,
        Tensor<T> output,
        Action<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        var node = new LazyNode<T>(
            opType, opName, input, output, execute, backwardFn, savedState);
        output.LazySource = node;
        AddNode(node);
        return output;
    }

    /// <summary>
    /// Records an operation whose output was materialized once to establish its exact shape during
    /// tracing, while retaining every public tensor operand as a graph dependency. This is the
    /// variadic counterpart to <see cref="RecordMaterializedUnary{T}"/> and is used by opaque
    /// inference nodes for kernels that do not have a primitive graph decomposition yet.
    /// </summary>
    internal Tensor<T> RecordMaterializedVariadic<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T>[] inputs,
        Tensor<T> output,
        Action<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length == 0)
            throw new ArgumentException("A materialized graph operation must have at least one tensor input.", nameof(inputs));

        var node = new LazyNode<T>(
            opType, opName, inputs, output, execute, backwardFn, savedState);
        output.LazySource = node;
        AddNode(node);
        return output;
    }

    /// <summary>
    /// Records an IN-PLACE operation that mutates an already-existing tensor.
    /// Unlike <see cref="RecordUnary"/> / <see cref="RecordBinary"/> /
    /// <see cref="RecordVariadic"/>, the "output" of the lazy node IS the
    /// target tensor itself (shared storage, no allocation). The recorded
    /// execute action mutates <paramref name="target"/>'s data buffer
    /// in-place during replay so subsequent forward replays see the
    /// accumulated effect of all earlier in-place ops on the same tensor —
    /// the property that makes EMAs (BatchNorm running stats), accumulators,
    /// and other stateful update patterns produce the right values across
    /// repeated <c>plan.Step()</c> calls.
    ///
    /// <para>CRITICAL semantic note: the caller MUST NOT have mutated
    /// <paramref name="target"/> at trace time before invoking this method.
    /// The replay action depends on <paramref name="target"/>'s STATE AT
    /// THE START OF REPLAY 1 being identical to its state when the trace
    /// captured it. If a trace-time eager mutation pollutes
    /// <paramref name="target"/>'s data, the first replay starts from the
    /// polluted state instead of the original (initial-condition) state, and
    /// every replay-N onward inherits that pollution. For per-step state
    /// accumulators (BN running mean/variance, optimizer momentums, etc.)
    /// the consequence is initial-condition drift that compounds into a
    /// substantial offset — manifests as test flakiness (40% pass) on
    /// GraFPrint Training_ShouldReduceLoss because the running stats end up
    /// some pool-state-dependent fraction off from the true EMA, and BN
    /// inference's <c>1/sqrt(var+eps)</c> amplifies the offset.</para>
    ///
    /// <para>Issue #350 v3: in-place ops (TensorAddInPlace,
    /// TensorMultiplyScalarInPlace, TensorMultiplyInPlace,
    /// TensorSubtractInPlace, TensorBroadcastAddInPlace, the in-place
    /// activations Swish/GELU/Tanh/Mish/LeakyReLU/Sigmoid/ReLU) previously
    /// had ZERO GraphMode handling — every one of them executed eagerly
    /// during compile-time trace and its effect was NEVER replayed. This
    /// silently broke any caller that mixed in-place updates with the
    /// rest of the lazy graph (most visibly: BatchNormalizationLayer's
    /// running-stat EMA when invoked under CompiledTrainingPlan, which
    /// is what made GraFPrint Training_ShouldReduceLoss diverge after
    /// the engine-side savedState fix landed).</para>
    /// </summary>
    internal void RecordInPlace<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T> target,
        Tensor<T>[] otherInputs,
        Action<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        // Build inputs array with target as the first input so the dependency
        // graph correctly orders this node after whatever produced target.
        var inputs = new Tensor<T>[1 + otherInputs.Length];
        inputs[0] = target;
        for (int i = 0; i < otherInputs.Length; i++) inputs[i + 1] = otherInputs[i];
        // CRITICAL: snapshot the prior producer of `target` BEFORE we
        // overwrite target.LazySource on line below. Without this snapshot,
        // GetInputNodes()/RealizeInputs() would resolve Input0.LazySource
        // back to THIS node (self-loop), and:
        //   - DCE would count phantom self-consumption
        //   - OperationReorderingPass would compute inDegree=1 for this node,
        //     never schedule it, and silently DROP the in-place op from the
        //     compiled list
        //   - subsequent in-place ops on the same target would resolve all
        //     earlier in-place node's "input producer" to the LATEST one
        // The LazyNode's externalPrerequisite slot captures the real prior
        // producer so the graph compiler sees a correct linear chain.
        var prevProducer = target.LazySource;
        // Output = target itself (shared storage). Subsequent ops that
        // consume target see this node as the source, ensuring topological
        // order keeps the in-place mutation before its consumers.
        var node = new LazyNode<T>(
            opType, opName, inputs, target, execute,
            backwardFn: backwardFn,
            savedState: savedState,
            externalPrerequisite: prevProducer);
        target.LazySource = node;
        AddNode(node);
    }

    /// <summary>Records a variadic operation as a lazy node.</summary>
    internal Tensor<T> RecordVariadic<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T>[] inputs,
        int[] outputShape,
        Action<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        var output = TensorAllocator.RentUninitialized<T>(outputShape);

        var node = new LazyNode<T>(opType, opName, inputs, output, execute, backwardFn, savedState);
        output.LazySource = node;
        AddNode(node);

        return output;
    }

    /// <summary>
    /// Materializes all lazy nodes in the graph. Walks in topological order
    /// (inputs before outputs) and executes each node that hasn't been realized.
    /// In the future, this will run the fusion pass and compiled plan instead.
    /// </summary>
    internal void Realize()
    {
        if (_realized) return;
        _realized = true;

        // Run graph compiler: optimization passes (fusion, DCE) + topological sort
        var compiler = new LazyGraphCompiler();
        var optimized = compiler.Compile(_nodes);

        // Suspend graph mode so execute delegates call eager paths (no re-entry)
        var savedScope = GraphMode.Current;
        GraphMode.SetCurrent(null);
        try
        {
            foreach (var node in optimized)
            {
                if (!node.IsRealized)
                    node.Realize(_engine);
            }
        }
        finally
        {
            GraphMode.SetCurrent(savedScope);
        }

        // Clear lazy source references so tensors behave normally after realization
        foreach (var node in optimized)
        {
            node.ClearOutputLazySource();
        }
    }

    /// <summary>Gets all recorded nodes (for graph compiler).</summary>
    internal IReadOnlyList<ILazyNode> Nodes => _nodes;

    private void AddNode(ILazyNode node)
    {
        // Acquire ownership before publishing the node. A composite is free to dispose a
        // temporary tensor immediately after its operation returns; the graph must therefore
        // hold an independent storage reference from the moment the node becomes visible.
        node.AddStorageLeases(_storageLeases);
        _nodes.Add(node);
    }

    /// <summary>
    /// Releases the trace-time lease snapshot after a successfully compiled plan has acquired
    /// its own post-optimization snapshot. This also releases storages abandoned by memory
    /// planning rebinds instead of retaining them for the plan's lifetime.
    /// </summary>
    internal void ReleaseStorageLeases() => _storageLeases.Dispose();

    /// <summary>
    /// Compiles the lazy graph into an inference plan for zero-overhead replay.
    /// Call this instead of Realize() when you want to cache and replay the plan.
    /// </summary>
    /// <remarks>
    /// Falls back to the last-step heuristic for <c>_finalOutput</c>. Prefer
    /// <see cref="CompileInference{T}(Tensor{T})"/> when you have the caller's
    /// returned tensor — the explicit path is correct even when the forward
    /// ends in a pure-view op or when optimization passes reorder steps.
    /// </remarks>
    internal CompiledInferencePlan<T> CompileInference<T>()
    {
        return CompilePlan(
            () => CompiledInferencePlan<T>.Compile(this, _engine, explicitOutput: null));
    }

    /// <summary>
    /// Compiles the lazy graph with an explicit output tensor. The plan's
    /// <c>Execute()</c> returns <paramref name="explicitOutput"/> instead of
    /// the last optimized step's output buffer — fixes issue #228.
    /// </summary>
    internal CompiledInferencePlan<T> CompileInference<T>(Tensor<T> explicitOutput)
    {
        return CompilePlan(
            () => CompiledInferencePlan<T>.Compile(this, _engine, explicitOutput));
    }

    /// <summary>
    /// Compiles the lazy graph while binding the caller-supplied tensor as
    /// the plan's mutable input slot. Other captured leaf tensors are treated
    /// as constants for replay.
    /// </summary>
    internal CompiledInferencePlan<T> CompileInference<T>(Tensor<T> explicitOutput, Tensor<T> input)
    {
        return CompilePlan(
            () => CompiledInferencePlan<T>.Compile(this, _engine, explicitOutput, input));
    }

    /// <summary>
    /// Compiles the lazy graph while selecting the first traced leaf tensor
    /// matching <paramref name="inputShape"/> as the mutable input slot.
    /// </summary>
    internal CompiledInferencePlan<T> CompileInference<T>(Tensor<T> explicitOutput, int[] inputShape)
    {
        return CompilePlan(
            () => CompiledInferencePlan<T>.Compile(this, _engine, explicitOutput, inputShape));
    }

    /// <summary>
    /// Compiles the lazy graph while binding every tensor in <paramref name="inputs"/> as a
    /// mutable input slot the plan re-binds on each replay. Use this when more than one captured
    /// leaf varies per call (e.g. a diffusion denoiser's noisy sample plus its per-step timestep
    /// embedding); all other captured leaves stay frozen as constants.
    /// </summary>
    internal CompiledInferencePlan<T> CompileInference<T>(Tensor<T> explicitOutput, Tensor<T>[] inputs)
    {
        return CompilePlan(
            () => CompiledInferencePlan<T>.Compile(this, _engine, explicitOutput, inputs));
    }

    /// <summary>
    /// Compiles the lazy graph into a training plan with forward + backward steps.
    /// The plan can be replayed for zero-overhead training iterations.
    /// </summary>
    /// <remarks>
    /// Falls back to the last forward step's output as the loss tensor. Prefer
    /// the explicit-loss overload when you have the caller's returned loss —
    /// a forward+loss lambda ending in a view op (e.g. <c>loss.Reshape([])</c>
    /// to scalarize) is only correct via the explicit path.
    /// </remarks>
    internal CompiledTrainingPlan<T> CompileTraining<T>(Tensor<T>[] parameters)
    {
        EnsureTrainingParametersPrepared(parameters);
        return CompilePlan(
            () => CompiledTrainingPlan<T>.Compile(this, _engine, parameters, explicitLoss: null));
    }

    /// <summary>
    /// Compiles the lazy graph into a training plan, threading the caller's
    /// returned loss tensor through as the explicit loss output (issue #228).
    /// </summary>
    internal CompiledTrainingPlan<T> CompileTraining<T>(Tensor<T>[] parameters, Tensor<T> explicitLoss)
    {
        EnsureTrainingParametersPrepared(parameters);
        return CompilePlan(
            () => CompiledTrainingPlan<T>.Compile(this, _engine, parameters, explicitLoss));
    }

    /// <summary>
    /// Completes the terminal transition from a trace to a compiled plan. Any compiler failure
    /// abandons the trace before the original exception continues to the caller.
    /// </summary>
    private TPlan CompilePlan<TPlan>(Func<TPlan> compile)
    {
        try
        {
            MarkCompiled();
            return compile();
        }
        catch
        {
            Abandon();
            throw;
        }
    }

    /// <summary>
    /// Privatizes COW-shared parameters before compilation specializes graph actions or an
    /// optimizer retains their live backing arrays. Lazy/off-loss-path parameters are included:
    /// they are valid registered parameters even when this trace does not produce a gradient.
    /// </summary>
    private void EnsureTrainingParametersPrepared<T>(Tensor<T>[] parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        for (int i = 0; i < parameters.Length; i++)
        {
            if (parameters[i] is null)
                throw new ArgumentException("Training parameters cannot contain null tensors.", nameof(parameters));

            // A plain graph scope remains valid for unshared tensors (including
            // existing internal tests). A COW parameter, however, must have
            // detached before tracing: doing it here is too late because graph
            // nodes may already retain parameter-derived storage views.
            if (_traceKind != GraphTraceKind.Training && parameters[i].IsCowShared)
            {
                throw new InvalidOperationException(
                    "Copy-on-write parameters must be prepared before a training graph is traced. " +
                    "Create the scope with GraphMode.EnableTraining(parameters), not GraphMode.Enable().");
            }

            // Defensive idempotent guard for unshared tensors and callers that
            // entered through EnableTraining. This also protects future backing
            // capture added during compilation itself.
            parameters[i].PrepareForInPlaceWrite();
        }
    }

    /// <summary>
    /// Marks this scope as compiled so Dispose() won't auto-realize the graph.
    /// Called by CompileInference/CompileTraining which handle the graph themselves.
    /// </summary>
    internal void MarkCompiled()
    {
        _realized = true;

        // Compilation is a terminal transition for this trace. Leaving the compiled scope
        // ambient until the lexical using statement unwinds lets plan replay (or an optimizer
        // call made immediately after compilation) append nodes to a scope whose storage
        // leases have already been transferred and released.
        if (ReferenceEquals(GraphMode.Current, this))
            GraphMode.SetCurrent(_parent);
    }

    /// <summary>
    /// Discards an incomplete explicit compilation trace without executing it.
    /// Inference and training scopes are opened to build plans, so replaying a
    /// partial graph while an exception unwinds would introduce work and user-
    /// visible side effects that the failed operation never requested.
    /// </summary>
    private void Abandon()
    {
        // Undo in reverse recording order. Several in-place nodes can successively replace the
        // same target's LazySource, so each one must expose its predecessor before that predecessor
        // can restore the producer from an enclosing scope.
        for (int i = _nodes.Count - 1; i >= 0; i--)
            _nodes[i].RestoreOutputLazySource();

        _realized = true;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        try
        {
            // Compatibility scopes retain the historical lazy-evaluation
            // contract. Explicit inference/training scopes exist only to
            // compile a plan; if compilation did not complete, abandon the
            // partial trace instead of executing callbacks during Dispose.
            if (!_realized)
            {
                if (_traceKind == GraphTraceKind.Compatibility)
                    Realize();
                else
                    Abandon();
            }
        }
        finally
        {
            // MarkCompiled may already have restored the parent. Only unwind this scope when
            // it is still current; otherwise a later nested scope must remain untouched.
            if (ReferenceEquals(GraphMode.Current, this))
                GraphMode.SetCurrent(_parent);
            _storageLeases.Dispose();
        }
    }
}
