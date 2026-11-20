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
    private readonly bool _isProbe;

    internal CompiledOpShapeCache CompiledOpShapes { get; set; } = new();

    internal LazyTensorScope(
        LazyTensorScope? parent,
        GraphTraceKind traceKind = GraphTraceKind.Compatibility,
        bool isProbe = false)
    {
        _parent = parent;
        _traceKind = traceKind;
        _isProbe = isProbe;
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

    /// <summary>
    /// <see cref="BindEngineIfUnset"/> usable inside a condition: binds and returns true.
    /// </summary>
    internal bool BindEngine(IEngine engine)
    {
        BindEngineIfUnset(engine);
        return true;
    }

    private readonly HashSet<object> _retainedOutputs = new(ReferenceEqualityComparer<object>.Instance);
    private readonly Dictionary<string, object> _namedRetainedOutputs = new(StringComparer.Ordinal);

    internal IReadOnlyDictionary<string, object> NamedRetainedOutputs => _namedRetainedOutputs;

    internal void RetainOutput<T>(string key, Tensor<T> output)
    {
        if (string.IsNullOrEmpty(key)) throw new ArgumentException("A retained output key must be nonempty.", nameof(key));
        if (_namedRetainedOutputs.ContainsKey(key))
            throw new InvalidOperationException($"The retained output key '{key}' is already registered in this forward.");
        RetainOutput(output);
        _namedRetainedOutputs.Add(key, output);
    }

    internal bool HasRetainedOutputs => _retainedOutputs.Count != 0;

    /// <summary>Declares an observable output while the forward is still being recorded.</summary>
    internal void RetainOutput<T>(Tensor<T> output)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(LazyTensorScope));
        if (_realized) throw new InvalidOperationException("Outputs must be retained before the trace is compiled or realized.");
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (_traceKind == GraphTraceKind.Inference)
            throw new InvalidOperationException("RetainOutput requires a training trace.");
        if (!_nodes.OfType<LazyNode<T>>().Any(node => ReferenceEquals(node.Output, output)))
            throw new ArgumentException("The retained output must be produced by the active training trace.", nameof(output));
        _retainedOutputs.Add(output);
    }

    /// <summary>
    /// Outputs and their storage aliases whose values must survive the step. Shape/view operations can borrow their
    /// input's device buffer even when their lazy output has separate storage during tracing, so close over both sides.
    /// This changes lifetime only; the existing recorded in-place read/write ordering remains authoritative.
    /// </summary>
    internal HashSet<Tensor<T>> RetainedOutputClosure<T>()
    {
        var retained = new HashSet<Tensor<T>>(_retainedOutputs.OfType<Tensor<T>>(), ReferenceEqualityComparer<Tensor<T>>.Instance);
        if (retained.Count != _retainedOutputs.Count)
            throw new NotSupportedException("Retained outputs must have the compiled training plan's element type.");
        if (retained.Count == 0) return retained;
        var nodes = _nodes.OfType<LazyNode<T>>().ToArray();
        bool changed;
        do
        {
            changed = false;
            foreach (var node in nodes)
            {
                var inputs = node.GetInputsArray();
                bool view = node.OpType is LazyNodeType.Reshape or LazyNodeType.Transpose
                    or LazyNodeType.TensorPermute or LazyNodeType.Slice or LazyNodeType.Expand
                    || node.OpName is "StopGradient" or "TensorStopGradient";
                if (retained.Contains(node.Output) && view)
                    foreach (var input in inputs) changed |= retained.Add(input);
                if ((view && inputs.Any(retained.Contains))
                    || retained.Any(output => ReferenceEquals(output._storage, node.Output._storage)))
                    changed |= retained.Add(node.Output);
            }
        } while (changed);
        return retained;
    }


    /// <summary>Number of lazy operations recorded.</summary>
    internal int NodeCount => _nodes.Count;

    /// <summary>
    /// The explicit purpose of this trace. Composite kernels use this to distinguish fused
    /// inference replay from differentiable training/compatibility capture.
    /// </summary>
    internal GraphTraceKind TraceKind => _traceKind;

    /// <summary>
    /// Whether this is a storage-free probe trace (<see cref="GraphMode.EnableProbe"/>): recorded outputs are
    /// shape-only tensors and no node ever executes. Any realization aborts the probe with
    /// <see cref="ObjectiveProbeAbortedException"/>.
    /// </summary>
    internal bool IsProbe => _isProbe;

    /// <summary>
    /// The output tensor for a recorded node. A probe gives a shape-only tensor whose storage is never created
    /// (reading it realizes the node, which a probe forbids); every other trace rents uninitialized storage the
    /// node's Execute will overwrite.
    /// </summary>
    private Tensor<TOut> RentOutput<TOut>(int[] outputShape)
        => _isProbe ? Tensor<TOut>.CreateDeferred(outputShape) : TensorAllocator.RentUninitialized<TOut>(outputShape);

    /// <summary>
    /// Whether an op of element type <typeparamref name="T"/> recorded now is inside a <see cref="NoGradScope{T}"/> of a
    /// training trace. The eager tape records nothing there; the trace still records the op (each replay must refresh
    /// its value, as <c>StopGradient</c>'s node does) but without a backward, so a compiled plan's gradient stops at the
    /// same place the tape's does. Before this, a frozen sub-network run under no-grad inside a compiled objective
    /// (a T5 encoder under a decoder-only loss) received the full gradient and the optimizer moved it. FP16
    /// activation-storage nodes (<see cref="Half"/>) stand in for a float forward, so the float scope governs them too.
    /// A compatibility trace (<see cref="GraphMode.Enable"/>) counts as well: <c>CompileTraining</c> accepts one whose
    /// parameters are unshared, and its no-grad ops must stop the gradient there just the same. Inference traces carry
    /// no gradient anyway.
    /// </summary>
    private bool RecordsWithoutGradient<T>()
        => _traceKind is GraphTraceKind.Training or GraphTraceKind.Compatibility
           && (NoGradScope<T>.IsSuppressed || (typeof(T) == typeof(Half) && NoGradScope<float>.IsSuppressed));

    /// <summary>The backward a node recorded now carries: none inside a no-grad scope (see <see cref="RecordsWithoutGradient{T}"/>).</summary>
    private BackwardFunction<T>? GradientEdge<T>(BackwardFunction<T>? backwardFn, out bool noGrad)
    {
        noGrad = RecordsWithoutGradient<T>();
        return noGrad ? null : backwardFn;
    }

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
        var output = RentOutput<T>(outputShape);

        backwardFn = GradientEdge(backwardFn, out bool noGrad);
        var node = new LazyNode<T>(opType, opName, input, output, execute, backwardFn, savedState) { RecordedUnderNoGrad = noGrad };
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

        var output = RentOutput<TOut>(outputShape);
        var node = new CrossTypeLazyNode<TIn, TOut>(opType, opName, input, output, execute)
        {
            RecordedUnderNoGrad = RecordsWithoutGradient<TIn>() || RecordsWithoutGradient<TOut>(),
        };
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
        var output = RentOutput<TOut>(outputShape);
        // A cast across a no-grad boundary of either element type bridges no gradient.
        bool noGrad = RecordsWithoutGradient<TIn>() || RecordsWithoutGradient<TOut>();
        var node = new CrossTypeLazyNode<TIn, TOut>(opType, opName, input, output, execute, noGrad ? null : backwardFn, savedState)
        {
            RecordedUnderNoGrad = noGrad,
        };
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
        var output = RentOutput<T>(outputShape);

        backwardFn = GradientEdge(backwardFn, out bool noGrad);
        var node = new LazyNode<T>(opType, opName, input0, input1, output, execute, backwardFn, savedState) { RecordedUnderNoGrad = noGrad };
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
        backwardFn = GradientEdge(backwardFn, out bool noGrad);
        var node = new LazyNode<T>(opType, opName, input, view,
            execute: (_, _) => { /* storage is shared; nothing to compute */ },
            backwardFn,
            savedState) { RecordedUnderNoGrad = noGrad };
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
        backwardFn = GradientEdge(backwardFn, out bool noGrad);
        var node = new LazyNode<T>(
            opType, opName, input, output, execute, backwardFn, savedState) { RecordedUnderNoGrad = noGrad };
        output.LazySource = node;
        AddNode(node);
        return output;
    }

    // Auxiliary outputs (argmax indices) of recorded nodes, keyed by the object the caller holds, mapped to the
    // node's primary output. A node that consumes the auxiliary output depends on that output, so a schedule cannot
    // run it before the producer refreshes the indices.
    private readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, object> _auxiliaryProducers = new();

    /// <summary>Records that each replay of the node producing <paramref name="producerOutput"/> refreshes <paramref name="auxiliary"/>.</summary>
    internal void RegisterAuxiliaryOutput(object auxiliary, object producerOutput)
    {
        _auxiliaryProducers.Remove(auxiliary);
        _auxiliaryProducers.Add(auxiliary, producerOutput);
    }

    /// <summary>The primary output of the node whose replay refreshes <paramref name="auxiliary"/>, if one does.</summary>
    internal bool TryGetAuxiliaryProducer(object auxiliary, out object? producerOutput)
    {
        bool found = _auxiliaryProducers.TryGetValue(auxiliary, out var output);
        producerOutput = output;
        return found;
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

        backwardFn = GradientEdge(backwardFn, out bool noGrad);
        var node = new LazyNode<T>(
            opType, opName, inputs, output, execute, backwardFn, savedState) { RecordedUnderNoGrad = noGrad };
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
        backwardFn = GradientEdge(backwardFn, out bool noGrad);
        var node = new LazyNode<T>(
            opType, opName, inputs, target, execute,
            backwardFn: backwardFn,
            savedState: savedState,
            externalPrerequisite: prevProducer) { RecordedUnderNoGrad = noGrad };
        // Every node that read the target since its prior producer must run before this write: an in-place write of
        // another tensor that reads it (y += x before x *= 2), a reader of a view sharing its storage (copy(view(x))),
        // and a cross-type reader. Only the target's own chain of writes is ordered by the prior-producer edge instead.
        // Enclosing scopes count too: a nested scope realizes its write when it closes, before the enclosing scope's
        // pending readers of the target run. Record order is process-wide and grows along each scope's list, so each
        // list is scanned from its end and stops at the prior producer instead of walking the whole trace.
        long since = prevProducer?.RecordOrder ?? 0;
        var readers = new List<ILazyNode>();
        for (var scope = this; scope is not null; scope = scope._parent)
            CollectReadersSince(scope._nodes, target, since, readers);
        node.SetEarlierReaders(readers.ToArray());
        target.LazySource = node;
        AddNode(node);
    }

    private static void CollectReadersSince<T>(List<ILazyNode> nodes, Tensor<T> target, long since, List<ILazyNode> readers)
    {
        for (int i = nodes.Count - 1; i >= 0; i--)
        {
            var recorded = nodes[i];
            if (recorded.RecordOrder <= since) break;
            if (recorded.IsRealized) continue;
            if (recorded is LazyNode<T> { WritesInPlace: true } writer && ReferenceEquals(writer.Output, target)) continue;
            if (recorded.ReadsStorageOf(target)) readers.Add(recorded);
        }
    }

    /// <summary>
    /// Records an op that returns a new output AND writes <paramref name="inPlaceTarget"/> (one of its inputs) in place,
    /// as one kernel (a recurrent decode step: the output and the advanced state from one pass over the state). The op's
    /// node runs after every node that read the target before it; a no-op in-place node on the target, after the op,
    /// becomes the target's producer, so later readers see the written value and a compiled plan sees the target as a
    /// leaf written in place. Forward only.
    /// </summary>
    internal Tensor<T> RecordVariadicWritingInPlace<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T>[] inputs,
        int[] outputShape,
        Tensor<T> inPlaceTarget,
        Action<IEngine, Tensor<T>> execute)
        => RecordVariadicWritingInPlace(opType, opName, inputs, outputShape, new[] { inPlaceTarget }, execute);

    /// <summary>
    /// <see cref="RecordVariadicWritingInPlace{T}(LazyNodeType, string, Tensor{T}[], int[], Tensor{T}, Action{IEngine, Tensor{T}})"/>
    /// for an op that writes several of its inputs in place (each gets its own write node, after the op).
    /// </summary>
    internal Tensor<T> RecordVariadicWritingInPlace<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T>[] inputs,
        int[] outputShape,
        Tensor<T>[] inPlaceTargets,
        Action<IEngine, Tensor<T>> execute)
    {
        foreach (var target in inPlaceTargets)
            if (Array.IndexOf(inputs, target) < 0)
                throw new ArgumentException("Every in-place target must be one of the op's inputs.", nameof(inPlaceTargets));
        // These ops are forward only (decoding) and record no backward. A training trace that still expects a gradient
        // here would compile a plan that silently drops it for every input, so the trace is refused instead; inside a
        // no-grad scope nothing is expected and the op records as usual.
        if (_traceKind == GraphTraceKind.Training && !RecordsWithoutGradient<T>())
            throw new NotSupportedException(
                $"{opName} is forward only and records no gradient, so it cannot be part of a training trace. "
                + "Run it inside a no-grad scope, or train through the eager tape.");
        var output = RentOutput<T>(outputShape);
        var node = new LazyNode<T>(opType, opName, inputs, output, execute) { RecordedUnderNoGrad = true };
        SnapshotTraceStateTargets(node, inPlaceTargets);
        // The op itself overwrites the targets, so it is ordered after each target's earlier readers (as RecordInPlace
        // orders its write).
        var readers = new List<ILazyNode>();
        foreach (var target in inPlaceTargets)
        {
            long since = target.LazySource?.RecordOrder ?? 0;
            for (var scope = this; scope is not null; scope = scope._parent)
                CollectReadersSince(scope._nodes, target, since, readers);
        }
        node.SetEarlierReaders(readers.ToArray());
        output.LazySource = node;
        AddNode(node);
        foreach (var target in inPlaceTargets)
            RecordInPlace<T>(opType, opName + ".StateWrite", target, new[] { output }, static (_, _) => { });
        return output;
    }

    // A trace may realize part of its graph while it is still being recorded: an op whose graph-mode path computes its
    // value eagerly (to capture its outputs) reads its input, which runs every node upstream of it, including a
    // state-writing decode op, whose state would then be advanced by the trace itself and once more by every execution.
    // So each such op's targets are snapshotted when it is recorded (only a target's first write in the trace: that
    // snapshot is the pre-trace state), and compiling or abandoning the trace writes the snapshot back into any target
    // the trace advanced. Kept on the root scope, which is the one that compiles.
    private List<(ILazyNode Node, ISignatureTensor Target, ISignatureTensor Snapshot)>? _traceStateSnapshots;

    private void SnapshotTraceStateTargets<T>(ILazyNode node, Tensor<T>[] targets)
    {
        var root = this;
        while (root._parent is not null) root = root._parent;
        foreach (var target in targets)
        {
            if (target.LazySource is not null) continue;
            if (root._traceStateSnapshots?.Exists(s => ReferenceEquals(s.Target, target)) == true) continue;
            var snapshot = new Tensor<T>((int[])target._shape.Clone(), new Vector<T>(target.ToArray()));
            (root._traceStateSnapshots ??= new()).Add((node, target, snapshot));
        }
    }

    private void RestoreTraceStateTargets()
    {
        if (_traceStateSnapshots is not { } snapshots) return;
        _traceStateSnapshots = null;
        foreach (var (node, target, snapshot) in snapshots)
            if (node.IsRealized) target.CopyLeafFrom(snapshot);
    }

    /// <summary>Records a variadic operation as a lazy node.</summary>

    internal Tensor<T> RecordVariadic<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T>[] inputs,
        int[] outputShape,
        Action<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null,
        int[]? differentiableInputIndices = null)
    {
        var output = RentOutput<T>(outputShape);

        backwardFn = GradientEdge(backwardFn, out bool noGrad);
        var node = new LazyNode<T>(opType, opName, inputs, output, execute, backwardFn, savedState) { RecordedUnderNoGrad = noGrad };
        node.SetDifferentiableInputIndices(differentiableInputIndices);
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

    /// <summary>
    /// What the forward first read on the host during this trace (a traced operation whose value it read, or an
    /// engine operation that returned a host value), or null. An inference plan cannot represent what the forward
    /// computed from that read, so compiling refuses the trace.
    /// </summary>
    internal string? HostEscape { get; private set; }

    internal void MarkHostEscape(ILazyNode node)
    {
        HostEscape ??= $"the value of traced operation {node.OpType}";
    }

    // An engine operation that returns a host value (TensorSum, TensorVecDot, ...) read its operand on the host, so
    // whatever the forward computes from that value would replay as the trace-time constant, even when the operand is
    // a plan input whose values change on every replay.
    internal void MarkHostEscape(string operationName)
    {
        HostEscape ??= $"the result of {operationName}";
    }

    /// <summary>The engine the recorded nodes replay on.</summary>
    internal IEngine Engine => _engine;

    /// <summary>
    /// Adds a node built from a recorded one rather than through a Record* call: a batch-polymorphic
    /// template instantiates its trace at a new batch this way instead of tracing the forward again.
    /// </summary>
    internal void AddInstantiatedNode(ILazyNode node) => AddNode(node);

    /// <summary>Gets all recorded nodes (for graph compiler).</summary>
    internal IReadOnlyList<ILazyNode> Nodes => _nodes;

    /// <summary>
    /// The first op that recorded a gradient-tape entry while this scope was the thread's trace (an op that ran eagerly,
    /// or a custom backward recorded with DifferentiableOps.RecordUnary/RecordBinary/RecordIfActive), or null. The trace
    /// holds no node for it, so a plan compiled from the trace cannot replay it; a closure objective that does this is
    /// unsignable (<see cref="ClosureObjectiveGuard{T}"/>).
    /// </summary>
    internal string? TapeRecordedOp { get; set; }

    /// <summary>
    /// The first traced op whose output was computed while this training scope was still tracing (a host read such as
    /// <c>ToArray()</c> of a traced value), or null. A probe of the same objective aborts at that read, so a closure
    /// objective that does this is unsignable (<see cref="ClosureObjectiveGuard{T}"/>).
    /// </summary>
    internal string? HostReadOp { get; set; }

    // Process-wide, so record order also holds between nodes of nested or successive traces.
    private static long s_recordOrder;

    private void AddNode(ILazyNode node)
    {
        if (node is ISignableLazyNode signable) signable.RecordedInProbe = IsProbe;
        // Acquire ownership before publishing the node. A composite is free to dispose a
        // temporary tensor immediately after its operation returns; the graph must therefore
        // hold an independent storage reference from the moment the node becomes visible.
        node.AddStorageLeases(_storageLeases);
        node.RecordingEngine = _engine;
        node.RecordOrder = System.Threading.Interlocked.Increment(ref s_recordOrder);
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
    /// Compiles a stateful inference plan: as the multi-input compile, plus <paramref name="stateUpdates"/>, pairs of
    /// a traced leaf and the value the graph computes to replace it. Each execution ends by writing every next value
    /// into its state's storage, so a recurrent or KV-cached decode step advances its state on every replay
    /// (CompiledInferencePlan's CUDA graph included) without host work.
    /// </summary>
    internal CompiledInferencePlan<T> CompileInference<T>(Tensor<T> explicitOutput, Tensor<T>[] inputs,
        IReadOnlyList<(Tensor<T> State, Tensor<T> Next)> stateUpdates)
    {
        return CompilePlan(
            () => CompiledInferencePlan<T>.Compile(this, _engine, explicitOutput, inputs, stateUpdates));
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
        // Signed before compilation, which may rewrite the node list (fusion, DCE).
        ClosureObjectiveGuard<T>.OnCompileTraining(this, parameters, explicitLoss: null);
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
        // Signed before compilation, which may rewrite the node list (fusion, DCE).
        ClosureObjectiveGuard<T>.OnCompileTraining(this, parameters, explicitLoss);
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
        finally
        {
            RestoreTraceStateTargets();
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
