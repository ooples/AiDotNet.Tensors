using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Lets an ordinary compiled training plan replay a closure objective: a <c>Func&lt;Tensor&lt;T&gt;&gt;</c> that builds
/// its per-step data (a batch, noise, labels, a sampled timestep's coefficient) as new tensors and scalars inside
/// itself. A plan traced once would replay the first step's values forever; before each replay the guard re-traces
/// the objective as a storage-free probe, compares its <see cref="ObjectiveSignature"/> with the compiled one, and on
/// a match copies the new step's external-leaf data into the plan's leaves.
/// </summary>
/// <remarks>
/// <para>Attach the guard the way a compile-time recorder is attached: open <see cref="BeginRecording"/> around the
/// plan's compilation, then <see cref="TakeRecorded"/> returns the guard for the plan compiled inside it (null when
/// the objective was unsignable, or when more than one training plan compiled inside the recording).</para>
/// <para>Correct by construction: a step whose objective differs from the compiled one in anything the plan would
/// freeze (an op, a shape, a captured scalar) gets <see cref="ClosureGuardResult.Mismatched"/> and must not replay
/// that plan.</para>
/// </remarks>
internal sealed class ClosureObjectiveGuard<T>
{
    [ThreadStatic]
    private static Recording? _recording;

    private readonly object[] _leaves;
    private readonly CompiledOpShapeCache _shapes;

    private ClosureObjectiveGuard(ObjectiveSignature signature, object[] leaves, CompiledOpShapeCache shapes)
    {
        Signature = signature;
        _leaves = leaves;
        _shapes = shapes;
        foreach (var leaf in leaves)
            if (leaf is ISignatureTensor st && st.ElementType != typeof(T))
                HasForeignLeaves = true;
    }

    /// <summary>The compiled trace's signature.</summary>
    internal ObjectiveSignature Signature { get; }

    /// <summary>
    /// Whether an external leaf has another element type than the plan (integer labels, say). A captured CUDA step
    /// graph re-uploads only the plan's own element type per replay, so a plan with such leaves must replay
    /// uncaptured (<see cref="CompiledTrainingPlan{T}.DisableGraphStep"/>); <see cref="AttachTo"/> does that.
    /// </summary>
    internal bool HasForeignLeaves { get; }

    /// <summary>Why the last <see cref="TryRebind"/> did not match, for a diagnostic; null after a match.</summary>
    internal string? LastReason { get; private set; }

    /// <summary>Number of external leaves the plan reads.</summary>
    internal int LeafCount => _leaves.Length;

    /// <summary>
    /// Starts recording the next training-plan compilation on this thread. Dispose to stop; recordings nest (the
    /// inner one wins while open).
    /// </summary>
    internal static IDisposable BeginRecording()
    {
        var recording = new Recording(_recording);
        _recording = recording;
        return recording;
    }

    /// <summary>
    /// The guard for the plan compiled inside the innermost open recording, or null when nothing signable was
    /// compiled (the objective was unsignable, or several plans compiled). <paramref name="reason"/> says why.
    /// </summary>
    internal static ClosureObjectiveGuard<T>? TakeRecorded(out string? reason)
    {
        var recording = _recording;
        if (recording is null) { reason = "no recording is open"; return null; }
        reason = recording.Reason;
        var guard = recording.Guard;
        recording.Guard = null;
        return guard;
    }

    /// <summary>Called by <see cref="LazyTensorScope.CompileTraining{T}(Tensor{T}[])"/> before it compiles.</summary>
    internal static void OnCompileTraining(LazyTensorScope scope, Tensor<T>[] parameters, Tensor<T>? explicitLoss)
    {
        var nodes = scope.Nodes;
        var recording = _recording;
        if (recording is null) return;
        recording.Compilations++;
        if (recording.Compilations > 1)
        {
            recording.Guard = null;
            recording.Reason = "more than one training plan compiled inside the recording";
            return;
        }

        if (UnreplayableReason(scope, explicitLoss) is { } unreplayable)
        {
            recording.Reason = unreplayable;
            return;
        }

        var signature = ObjectiveSignature.Build(nodes, parameters, out var leaves);
        if (!signature.Signable)
        {
            recording.Reason = signature.UnsignableReason;
            return;
        }
        if (DisconnectedReason(scope, explicitLoss, parameters) is { } disconnected)
        {
            recording.Reason = disconnected;
            return;
        }

        recording.Guard = new ClosureObjectiveGuard<T>(signature, leaves, scope.CompiledOpShapes.Snapshot());
        recording.Reason = null;
    }

    /// <summary>
    /// Prepares <paramref name="plan"/> for guarded replay: a plan whose leaves include another element type keeps
    /// to the uncaptured step (see <see cref="HasForeignLeaves"/>).
    /// </summary>
    internal void AttachTo(CompiledTrainingPlan<T> plan)
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));
        if (HasForeignLeaves) plan.DisableGraphStep();
    }

    /// <summary>
    /// Re-traces <paramref name="objective"/> as a probe and, when its signature equals the compiled one, copies its
    /// external-leaf data into the compiled leaves so the plan's next step trains on this step's data.
    /// </summary>
    /// <param name="objective">The closure objective for this step. It runs once, under a probe trace.</param>
    /// <param name="parameters">The plan's parameters, in the order it was compiled with.</param>
    /// <param name="probed">The probe's signature (null when the probe aborted); a caller keying several plans by
    /// signature looks it up on <see cref="ClosureGuardResult.Mismatched"/>.</param>
    internal ClosureGuardResult TryRebind(Func<Tensor<T>> objective, Tensor<T>[] parameters, out ObjectiveSignature? probed)
    {
        var result = Probe(objective, parameters, out probed, out var leaves, out string? reason, _shapes);
        if (result != ClosureGuardResult.Matched || probed is null)
        {
            LastReason = reason;
            return result;
        }

        if (!probed.Equals(Signature))
        {
            LastReason = Signature.DescribeDifference(probed) ?? "signatures differ";
            return ClosureGuardResult.Mismatched;
        }

        if (!TryCopyLeaves(leaves, out reason))
        {
            LastReason = reason;
            return ClosureGuardResult.Mismatched;
        }

        LastReason = null;
        return ClosureGuardResult.Matched;
    }

    /// <summary>
    /// Copies the leaves of a probe whose signature equals <see cref="Signature"/> into the compiled leaves. For a
    /// caller that probed once (<see cref="Probe"/>) and then picked this guard's plan from several by signature.
    /// </summary>
    internal bool TryCopyLeaves(object[] probedLeaves, out string? reason)
    {
        if (probedLeaves is null) throw new ArgumentNullException(nameof(probedLeaves));
        if (probedLeaves.Length != _leaves.Length)
        {
            reason = $"the probe has {probedLeaves.Length} leaves, the plan {_leaves.Length}";
            return false;
        }

        // Preflight the entire copy order: swapping existing leaves must not overwrite a later source.
        var written = new HashSet<object>(Autodiff.ReferenceEqualityComparer<object>.Instance);
        for (int i = 0; i < _leaves.Length; i++)
        {
            if (probedLeaves[i] is ISignatureTensor source && written.Contains(source.SignatureStorageIdentity))
            {
                reason = $"leaf {i} aliases a compiled leaf overwritten earlier in the rebind";
                return false;
            }
            if (!ReferenceEquals(_leaves[i], probedLeaves[i]) && _leaves[i] is ISignatureTensor target)
                written.Add(target.SignatureStorageIdentity);
        }

        for (int i = 0; i < _leaves.Length; i++)
        {
            if (ReferenceEquals(_leaves[i], probedLeaves[i])) continue;   // a persistent buffer the caller refills
            if (_leaves[i] is not ISignatureTensor target || probedLeaves[i] is not ISignatureTensor source)
            {
                reason = $"leaf {i} is not a tensor";
                return false;
            }

            try
            {
                target.CopyLeafFrom(source);
            }
            catch (Exception ex) when (ex is ObjectDisposedException || ex is InvalidOperationException || ex is ArgumentException)
            {
                // The caller disposed or recycled a tensor the plan was compiled against: this plan cannot take new
                // data any more, so the caller compiles a fresh one.
                reason = $"leaf {i} can no longer be written ({ex.GetType().Name}: {ex.Message})";
                return false;
            }
        }

        reason = null;
        return true;
    }

    /// <summary>
    /// Runs <paramref name="objective"/> under a probe trace and signs it. Returns
    /// <see cref="ClosureGuardResult.Matched"/> when a signable signature was produced (the caller still compares
    /// it), <see cref="ClosureGuardResult.Aborted"/> when the objective read a value or threw, and
    /// <see cref="ClosureGuardResult.Unsignable"/> when it captured uncomparable state.
    /// </summary>
    internal static ClosureGuardResult Probe(
        Func<Tensor<T>> objective,
        Tensor<T>[] parameters,
        out ObjectiveSignature? signature,
        out object[] leaves,
        out string? reason,
        CompiledOpShapeCache? shapes = null)
    {
        if (objective is null) throw new ArgumentNullException(nameof(objective));
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        signature = null;
        leaves = Array.Empty<object>();
        try
        {
            using var scope = GraphMode.EnableProbe(shapes);
            var loss = objective();
            if (UnreplayableReason(scope, loss) is { } unreplayable)
            {
                reason = unreplayable;
                return ClosureGuardResult.Unsignable;
            }
            signature = ObjectiveSignature.Build(scope.Nodes, parameters, out leaves);
            if (signature.Signable && DisconnectedReason(scope, loss, parameters) is { } disconnected)
            {
                reason = disconnected;
                return ClosureGuardResult.Unsignable;
            }
        }
        catch (ObjectiveProbeAbortedException ex)
        {
            reason = $"the objective reads '{ex.OpName}' on the host";
            return ClosureGuardResult.Aborted;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            // An op that cannot trace without data (or the objective's own failure, which the eager step will
            // surface): either way this step cannot replay a plan.
            // Name where it threw: an op that indexes a value the storage-free probe never computes shows up here.
            var site = ThrowSite(ex) is { } method ? $" in {method.DeclaringType?.Name}.{method.Name}" : "";
            reason = $"the objective could not be probed ({ex.GetType().Name}{site}: {ex.Message})";
            return ClosureGuardResult.Aborted;
        }

        if (!signature.Signable)
        {
            reason = signature.UnsignableReason;
            return ClosureGuardResult.Unsignable;
        }

        reason = null;
        return ClosureGuardResult.Matched;
    }

    /// <summary>
    /// Why a plan compiled from <paramref name="scope"/>'s trace could not replay the objective whatever its signature,
    /// or null: an op recorded only on the gradient tape (a custom backward closing over this step's host values, such
    /// as a loss computed on the host and recorded with DifferentiableOps.RecordUnary), or a loss no traced op produced.
    /// </summary>
    private static string? UnreplayableReason(LazyTensorScope scope, Tensor<T>? loss)
    {
        if (scope.TapeRecordedOp is { } op)
            return $"the objective records '{op}' on the gradient tape, outside the trace (a custom backward a plan cannot replay)";
        if (scope.HostReadOp is { } read)
            return $"the objective reads '{read}' on the host while it is traced";
        if (loss is null) return null;
        bool traced = false;
        if (loss.LazySource is { } source)
            foreach (var node in scope.Nodes)
                if (ReferenceEquals(node, source)) { traced = true; break; }
        return traced ? null : "the objective's loss is not computed by a traced op (a value computed on the host, say)";
    }

    /// <summary>Why a signable trace still cannot train: its loss reaches no parameter; null when it does.</summary>
    private static string? DisconnectedReason(LazyTensorScope scope, Tensor<T>? loss, Tensor<T>[] parameters)
        => loss is not null && parameters.Length > 0 && !ReachesParameter(scope.Nodes, loss, parameters)
            ? "the objective's loss reaches no trained parameter through traced ops (a loss computed on the host and then scaled, say)"
            : null;

    /// <summary>
    /// Whether <paramref name="loss"/> depends on any of <paramref name="parameters"/> through the traced nodes: a walk
    /// from the loss back through each node's inputs, where an input that shares a parameter's storage (the parameter
    /// or a view of it) is a hit, and an input no node produced is a leaf.
    /// </summary>
    private static bool ReachesParameter(IReadOnlyList<ILazyNode> nodes, Tensor<T> loss, Tensor<T>[] parameters)
    {
        var paramStorage = new HashSet<object>(Autodiff.ReferenceEqualityComparer<object>.Instance);
        foreach (var p in parameters)
            if (p is ISignatureTensor st) paramStorage.Add(st.SignatureStorageIdentity);
        // Every node that writes a tensor (an in-place op writes one an earlier node produced): all are its producers.
        var producerOf = new Dictionary<object, List<ISignableLazyNode>>(Autodiff.ReferenceEqualityComparer<object>.Instance);
        foreach (var node in nodes)
            if (node is ISignableLazyNode signable)
            {
                if (!producerOf.TryGetValue(signable.OutputTensor, out var writers))
                    producerOf[signable.OutputTensor] = writers = new List<ISignableLazyNode>(1);
                writers.Add(signable);
            }
        var visited = new HashSet<object>(Autodiff.ReferenceEqualityComparer<object>.Instance);
        var pending = new Stack<object>();
        pending.Push(loss);
        while (pending.Count > 0)
        {
            var t = pending.Pop();
            if (!visited.Add(t)) continue;
            if (t is ISignatureTensor st && paramStorage.Contains(st.SignatureStorageIdentity)) return true;
            if (!producerOf.TryGetValue(t, out var producers)) continue;
            foreach (var producer in producers)
                for (int i = 0; i < producer.SignatureInputCount; i++)
                    if (producer.GetSignatureInput(i) is { } input) pending.Push(input);
        }
        return false;
    }

    private sealed class Recording : IDisposable
    {
        private readonly Recording? _parent;
        private bool _disposed;

        internal Recording(Recording? parent) => _parent = parent;

        internal ClosureObjectiveGuard<T>? Guard { get; set; }
        internal string? Reason { get; set; } = "no training plan compiled inside the recording";
        internal int Compilations { get; set; }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (ReferenceEquals(_recording, this)) _recording = _parent;
        }
    }

    // The innermost frame outside the BCL: TargetSite is often a throw helper (ThrowHelpers.ThrowIndexOutOfRange...).
    private static System.Reflection.MethodBase? ThrowSite(Exception ex)
    {
        var frames = new System.Diagnostics.StackTrace(ex, false).GetFrames();
        if (frames is not null)
            foreach (var frame in frames)
                if (frame?.GetMethod() is { DeclaringType: { } type } method
                    && type.Assembly != typeof(object).Assembly
                    && !(type.Namespace ?? "").StartsWith("System", StringComparison.Ordinal))
                    return method;
        return ex.TargetSite;
    }
}
