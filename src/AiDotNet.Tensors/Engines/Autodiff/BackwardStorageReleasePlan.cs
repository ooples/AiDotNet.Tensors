using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

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

        internal ReleaseCandidate(object storageIdentity, Tensor<T> tensor)
        {
            StorageIdentity = storageIdentity;
            Tensor = tensor;
        }
    }

    private sealed class StorageUsage
    {
        internal readonly object Identity;
        internal Tensor<T>? ReleaseCandidate;
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

        internal void Observe(Tensor<T>? tensor, int step, bool canRelease)
        {
            if (tensor is null) return;
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
                usage.ReleaseCandidate = tensor;
        }

        internal BackwardStorageReleasePlan<T> Build(int stepCount)
        {
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
            foreach (var usage in _usage.Values)
            {
                if (usage.ReleaseCandidate is null) continue;
                int step = usage.LastUse;
                releases[step]![counts[step]++] =
                    new ReleaseCandidate(usage.Identity, usage.ReleaseCandidate);
            }

            return new BackwardStorageReleasePlan<T>(releases);
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
                _builder.Observe(typed, _step, canRelease: false);
        }
    }

    private readonly ReleaseCandidate[]?[] _releasesAfterStep;

    private BackwardStorageReleasePlan(ReleaseCandidate[]?[] releasesAfterStep)
        => _releasesAfterStep = releasesAfterStep;

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
        return builder.Build(count);
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
        return builder.Build(executionOrder.Length);
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
        var candidates = _releasesAfterStep[step];
        if (candidates is null) return;
        for (int i = 0; i < candidates.Length; i++)
        {
            var candidate = candidates[i];
            if (protectedStorages.Contains(candidate.StorageIdentity)) continue;
            if (!ReferenceEquals(candidate.Tensor.StorageIdentity, candidate.StorageIdentity)) continue;
            engine.InvalidateGpuCacheForTensor(candidate.Tensor);
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
