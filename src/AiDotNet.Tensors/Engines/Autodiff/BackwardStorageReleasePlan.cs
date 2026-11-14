using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// The error a read of a tensor freed at its last backward use throws: which forward op made it, which backward step
/// freed it, and how to keep it. Built once per plan candidate and formatted only if something reads the tensor.
/// </summary>
internal sealed class LastUseReleaseNote : Helpers.ReleaseNote
{
    private readonly int[] _shape;
    private readonly System.Reflection.MethodInfo? _producer;
    private readonly int _producerStep;
    private readonly System.Reflection.MethodInfo? _freer;
    private readonly int _freeingStep;
    private readonly long _key;

    internal LastUseReleaseNote(int[] shape, System.Reflection.MethodInfo? producer, int producerStep,
        System.Reflection.MethodInfo? freer, int freeingStep, long key)
    {
        _key = key;
        _shape = shape;   // a tensor's shape array is never mutated in place
        _producer = producer;
        _producerStep = producerStep;
        _freer = freer;
        _freeingStep = freeingStep;
    }

    // "MatMulBackward" -> "MatMul": the forward op a backward function belongs to.
    internal static string OpName(System.Reflection.MethodInfo? method)
    {
        string name = method?.Name ?? "?";
        int lambda = name.IndexOf('>');   // a lambda's compiler name: <Outer>b__12_0
        if (name.StartsWith("<", StringComparison.Ordinal) && lambda > 1) name = name.Substring(1, lambda - 1);
        return name.EndsWith("Backward", StringComparison.Ordinal) ? name.Substring(0, name.Length - "Backward".Length) : name;
    }

    // A failed read still teaches the probe: the next run of this graph keeps this output.
    internal override void OnRead() => AutoRetainProbe.Retain(_key);

    internal override string Describe() =>
        $"This tensor [{string.Join(", ", _shape)}] is the output of forward op '{OpName(_producer)}' (backward step " +
        $"#{_producerStep}). Its GPU memory was freed, without a host copy, right after backward step #{_freeingStep} " +
        $"('{OpName(_freer)}'), its last use in the backward graph, as PyTorch frees a graph's saved tensors during backward. " +
        "To read it after backward, call GradientTape.Retain(tensor) on it before backward runs " +
        "(or TensorScope.Keep(tensor) inside a TensorScope). This read was recorded: later backward runs of this graph " +
        "keep this op's output (AutoRetainProbe), as they keep anything read within one step of the graph's first run.";
}

/// <summary>
/// Whether the backward running on this thread frees device intermediates at their last use: the running tape's
/// <see cref="GradientTapeOptions.ReleaseDeviceActivationsAtLastUse"/>. Thread scoped, because the backward runners
/// (eager walk, cached delegate chain, rebindable plan) are shared across tapes; an inner tape (a checkpoint
/// recompute) sets its own and restores the outer one.
/// </summary>
internal static class ReleaseScope
{
    [ThreadStatic] private static bool t_freeDeviceActivations;

    internal static bool FreeDeviceActivations => t_freeDeviceActivations;

    /// <summary>Sets the flag for this thread's backward and returns the previous value, to pass to <see cref="Exit"/>.</summary>
    internal static bool Enter(bool freeDeviceActivations)
    {
        bool previous = t_freeDeviceActivations;
        t_freeDeviceActivations = freeDeviceActivations;
        return previous;
    }

    internal static void Exit(bool previous) => t_freeDeviceActivations = previous;
}

/// <summary>
/// Precomputed, storage-aware release schedule for progressive GPU activation reclamation.
/// Tensor objects are not allocation identities: reshape/transpose/slice views can all share
/// one storage, and a saved tensor can outlive an earlier alias's backward step. The schedule
/// therefore releases an output storage only after its final declared backward use.
/// </summary>
internal sealed class BackwardStorageReleasePlan<T>
{
    private readonly struct ReleaseCandidate
    {
        internal readonly object StorageIdentity;
        internal readonly Tensor<T> Tensor;
        internal readonly LastUseReleaseNote Note;
        internal readonly long Key;   // the producer's structural identity (AutoRetainProbe)

        internal ReleaseCandidate(object storageIdentity, Tensor<T> tensor, LastUseReleaseNote note, long key)
        {
            StorageIdentity = storageIdentity;
            Tensor = tensor;
            Note = note;
            Key = key;
        }
    }

    private sealed class StorageUsage
    {
        internal readonly object Identity;
        internal Tensor<T>? ReleaseCandidate;
        internal int ProducerStep = -1;
        internal int LastUse;

        internal StorageUsage(object identity, int lastUse)
        {
            Identity = identity;
            LastUse = lastUse;
        }
    }

    private sealed class Builder
    {
        private readonly Dictionary<object, StorageUsage> _usage =
            new(ReferenceEqualityComparer<object>.Instance);

        // Tensor objects some step produced, and the tensors steps take as inputs. An input no step produced is a
        // leaf (a parameter, a caller's input); a recorded op's output can alias its storage (a reshape view of a
        // weight), and that storage belongs to the leaf, never to the graph, so it is never released. Saved state is
        // not an input: a forward op may save its own intermediate's base tensor next to a view it outputs.
        private readonly HashSet<Tensor<T>> _outputs = new(ReferenceEqualityComparer<Tensor<T>>.Instance);
        private readonly List<Tensor<T>> _reads = new();

        internal void Observe(Tensor<T>? tensor, int step, bool canRelease, bool savedState = false)
        {
            if (tensor is null) return;
            if (canRelease) _outputs.Add(tensor); else if (!savedState) _reads.Add(tensor);
            object identity = tensor.StorageIdentity;
            if (!_usage.TryGetValue(identity, out var usage))
            {
                usage = new StorageUsage(identity, step);
                _usage.Add(identity, usage);
            }
            else if (step > usage.LastUse)
            {
                usage.LastUse = step;
            }

            // Only graph-produced outputs are owned by progressive activation reclamation.
            // Inputs that are leaves must never become releasable merely because they were read.
            if (canRelease && usage.ReleaseCandidate is null)
            {
                usage.ReleaseCandidate = tensor;
                usage.ProducerStep = step;
            }
        }

        internal BackwardStorageReleasePlan<T> Build(int stepCount, Func<int, System.Reflection.MethodInfo?> method)
        {
            foreach (var read in _reads)
                if (!_outputs.Contains(read) && _usage.TryGetValue(read.StorageIdentity, out var leafOwned))
                    leafOwned.ReleaseCandidate = null;
            var counts = new int[stepCount];
            foreach (var usage in _usage.Values)
            {
                if (usage.ReleaseCandidate is not null)
                    counts[usage.LastUse]++;
            }

            var releases = new ReleaseCandidate[]?[stepCount];
            for (int i = 0; i < stepCount; i++)
            {
                if (counts[i] != 0)
                    releases[i] = new ReleaseCandidate[counts[i]];
            }

            Array.Clear(counts, 0, counts.Length);
            // A producer's structural key: its backward function, output shape and ordinal among same-keyed outputs,
            // stable across the per-step rebuilds of the same graph. The graph signature combines them all.
            var ordinals = new Dictionary<long, int>();
            long signature = stepCount;
            foreach (var usage in _usage.Values)
            {
                if (usage.ReleaseCandidate is null) continue;
                int step = usage.LastUse;
                var producer = method(usage.ProducerStep);
                long key;
                unchecked
                {
                    long shapeKey = (producer?.MetadataToken ?? 0) * 1_000_003L + (producer?.Module.GetHashCode() ?? 0);
                    foreach (int d in usage.ReleaseCandidate._shape) shapeKey = shapeKey * 31 + d;
                    ordinals.TryGetValue(shapeKey, out int ordinal);
                    ordinals[shapeKey] = ordinal + 1;
                    key = shapeKey * 1_000_033L + ordinal;
                    signature = (signature * 1_099_511_628_211L) ^ key;
                }
                if (releases[step] is { } slot)
                    slot[counts[step]++] = new ReleaseCandidate(usage.Identity, usage.ReleaseCandidate,
                        new LastUseReleaseNote(usage.ReleaseCandidate._shape, producer, usage.ProducerStep, method(step), step, key), key);
            }

            return new BackwardStorageReleasePlan<T>(releases, signature);
        }
    }

    private struct SavedUseVisitor : ISavedStateTensorVisitor
    {
        private readonly Builder _builder;
        private readonly int _step;

        internal SavedUseVisitor(Builder builder, int step)
        {
            _builder = builder;
            _step = step;
        }

        public void Visit(ITensorStorageLeaseSource tensor)
        {
            if (tensor is Tensor<T> typed)
                _builder.Observe(typed, _step, canRelease: false, savedState: true);
        }
    }

    private readonly ReleaseCandidate[]?[] _releasesAfterStep;
    private readonly long _signature;
    private int _lastStep = int.MaxValue;   // a step index not above the last one starts a new backward run
    private bool _probing;

    private BackwardStorageReleasePlan(ReleaseCandidate[]?[] releasesAfterStep, long signature)
    {
        _releasesAfterStep = releasesAfterStep;
        _signature = signature;
    }

    internal static BackwardStorageReleasePlan<T> Create(BackwardStep<T>[] steps, int count)
    {
        var builder = new Builder();
        for (int i = 0; i < count; i++)
        {
            ref var step = ref steps[i];
            builder.Observe(step.Output, i, canRelease: true);
            if (step.Inputs is not null)
            {
                for (int j = 0; j < step.Inputs.Length; j++)
                    builder.Observe(step.Inputs[j], i, canRelease: false);
            }

            var visitor = new SavedUseVisitor(builder, i);
            SavedStateTensorTraversal.Visit(step.SavedState, ref visitor);
        }
        return builder.Build(count, i => steps[i].Backward?.Method);
    }

    internal static BackwardStorageReleasePlan<T> Create(
        TapeEntryArena<T> entries,
        int[] executionOrder)
    {
        var builder = new Builder();
        for (int step = 0; step < executionOrder.Length; step++)
        {
            ref var entry = ref entries[executionOrder[step]];
            builder.Observe(entry.Output, step, canRelease: true);
            if (entry.InputsOverflow is not null)
            {
                for (int j = 0; j < entry.InputsOverflow.Length; j++)
                    builder.Observe(entry.InputsOverflow[j], step, canRelease: false);
            }
            else
            {
                builder.Observe(entry.Input0, step, canRelease: false);
                if (entry.InputCount >= 2)
                    builder.Observe(entry.Input1, step, canRelease: false);
                if (entry.InputCount >= 3)
                    builder.Observe(entry.Input2, step, canRelease: false);
            }

            var visitor = new SavedUseVisitor(builder, step);
            SavedStateTensorTraversal.Visit(entry.SavedState, ref visitor);
        }
        return builder.Build(executionOrder.Length, i => entries[executionOrder[i]].Backward?.Method);
    }

    /// <summary>
    /// Invalidates storages whose final backward use is the completed step. A changed storage
    /// identity means replay rebinding invalidated the precomputed proof; fail closed by retaining
    /// that activation rather than risking premature release.
    /// </summary>
    internal void ReleaseAfterStep(
        int step,
        DirectGpuTensorEngine engine,
        HashSet<object> protectedStorages)
    {
        // Higher-order AD (createGraph: true) records new differentiable ops during backward whose own backward pass
        // may read forward activations saved by this first backward; keep them intact until the outer tape finishes.
        if (DifferentiableOps._isBackwardCreateGraph) return;
        var candidates = _releasesAfterStep[step];
        if (!ReleaseScope.FreeDeviceActivations)
        {
            // The tape did not opt in (GradientTapeOptions.ReleaseDeviceActivationsAtLastUse): a device-only
            // intermediate stays readable while it is held, as PyTorch keeps a tensor Python still references, and goes
            // back to the collector's reclaim; one with a host copy is only uncached.
            if (candidates is null) return;
            for (int i = 0; i < candidates.Length; i++)
            {
                var candidate = candidates[i];
                if (protectedStorages.Contains(candidate.StorageIdentity)) continue;
                if (!ReferenceEquals(candidate.Tensor.StorageIdentity, candidate.StorageIdentity)) continue;
                engine.ReleaseDeadActivation(candidate.Tensor, candidate.Note, freeDeviceStorage: false);
            }
            return;
        }
        if (step <= _lastStep)
        {
            // A nested opted-in backward walk (e.g. a custom recompute inside an outer step's Backward delegate)
            // frees its private recomputed activations immediately rather than opening a quarantine window.
            _probing = DifferentiableOps.IsInsideBackwardStep ? false : AutoRetainProbe.BeginRun(_signature);
        }
        _lastStep = step;
        if (candidates is null) return;
        for (int i = 0; i < candidates.Length; i++)
        {
            var candidate = candidates[i];
            if (protectedStorages.Contains(candidate.StorageIdentity)) continue;
            if (!ReferenceEquals(candidate.Tensor.StorageIdentity, candidate.StorageIdentity)) continue;
            if (AutoRetainProbe.IsRetained(candidate.Key)) continue;   // read after backward on a probe run: kept
            // Dead after this step: never download it (materializing moved ~0.6 GB device-to-host per training step
            // on a 26.8M-parameter LM).
            if (_probing)
            {
                var tensor = candidate.Tensor;
                var note = candidate.Note;
                var weak = new WeakReference<Tensor<T>>(tensor);   // the quarantine must not keep it alive
                AutoRetainProbe.Quarantine(_signature, tensor.DataVector, tensor.GetBackingArrayForCacheLookupUnsafe(),
                    candidate.Key, () => { if (weak.TryGetTarget(out var held)) engine.ReleaseDeadActivation(held, note); });
                continue;
            }
            engine.ReleaseDeadActivation(candidate.Tensor, candidate.Note);
        }
    }

    internal bool IsStorageScheduledAfterStep(Tensor<T> tensor, int step)
    {
        var candidates = _releasesAfterStep[step];
        if (candidates is null) return false;
        object identity = tensor.StorageIdentity;
        for (int i = 0; i < candidates.Length; i++)
        {
            if (ReferenceEquals(candidates[i].StorageIdentity, identity)) return true;
        }
        return false;
    }
}
