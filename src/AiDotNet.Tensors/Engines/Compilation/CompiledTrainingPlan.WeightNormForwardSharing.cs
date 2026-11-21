using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private readonly struct WeightNormForwardKey : IEquatable<WeightNormForwardKey>
    {
        internal readonly Tensor<T> Direction;
        internal readonly Tensor<T> Gain;
        private readonly long _epsilon;
        internal WeightNormForwardKey(Tensor<T> direction, Tensor<T> gain, double epsilon)
        { Direction = direction; Gain = gain; _epsilon = BitConverter.DoubleToInt64Bits(epsilon); }
        public bool Equals(WeightNormForwardKey other) => ReferenceEquals(Direction, other.Direction)
            && ReferenceEquals(Gain, other.Gain) && _epsilon == other._epsilon;
        public override bool Equals(object? obj) => obj is WeightNormForwardKey other && Equals(other);
        public override int GetHashCode() => unchecked((RuntimeHelpers.GetHashCode(Direction) * 397
            ^ RuntimeHelpers.GetHashCode(Gain)) * 397 ^ _epsilon.GetHashCode());
    }

    private readonly struct WeightNormSavedStorageVisitor : ISavedStateTensorVisitor
    {
        private readonly HashSet<object> _storages;
        private readonly HashSet<Tensor<T>> _tensors;
        internal WeightNormSavedStorageVisitor(HashSet<object> storages, HashSet<Tensor<T>> tensors)
        { _storages = storages; _tensors = tensors; }
        public void Visit(ITensorStorageLeaseSource tensor)
        {
            _storages.Add(tensor.StorageIdentity);
            if (tensor is Tensor<T> typed) _tensors.Add(typed);
        }
    }

    /// <summary>
    /// Shares only the storage of repeated pure normalizations. Both forward executions and all distinct VJP nodes
    /// remain in the schedule. This runs before any specialization captures an array or establishes GPU residency.
    /// </summary>
    internal static int ShareWeightNormForwardStorage(IReadOnlyList<CompiledStep<T>> steps,
        Tensor<T>[] parameters, Tensor<T> loss, HashSet<Tensor<T>> retained, IEngine engine,
        LazyTensorScope? traceScope = null)
    {
        var unwrapped = engine.Unwrap();
        if (Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_WEIGHTNORM_FORWARD_SHARE") == "0"
            || !ReferenceEquals(engine, unwrapped)
            || (engine.Unwrap().GetType() != typeof(CpuEngine) && engine.Unwrap().GetType() != typeof(DirectGpuTensorEngine))
            || traceScope?.HostReadOp is not null || ForwardStepObserver is not null || SanitizeForwardBuffers)
            return 0;
        int normalizations = 0;
        foreach (var step in steps)
            if (step.OpType == OpType.TensorWeightNorm && ++normalizations == 2) break;
        if (normalizations < 2) return 0;

        var tensors = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        var produced = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        var writers = new Dictionary<Tensor<T>, int>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        var protectedStorage = new HashSet<object>(ReferenceEqualityComparer<object>.Instance);
        var savedArrays = new HashSet<object>(ReferenceEqualityComparer<object>.Instance);
        var savedVisitor = new WeightNormSavedStorageVisitor(protectedStorage, tensors);
        foreach (var parameter in parameters) { tensors.Add(parameter); protectedStorage.Add(parameter.StorageIdentity); }
        tensors.Add(loss);
        protectedStorage.Add(loss.StorageIdentity);
        foreach (var output in retained) { tensors.Add(output); protectedStorage.Add(output.StorageIdentity); }
        foreach (var step in steps)
        {
            tensors.Add(step.OutputBuffer);
            produced.Add(step.OutputBuffer);
            writers[step.OutputBuffer] = writers.TryGetValue(step.OutputBuffer, out int count) ? count + 1 : 1;
            foreach (var input in step.Inputs) tensors.Add(input);
            SavedStateTensorTraversal.Visit(step.SavedState, ref savedVisitor);
            CollectWeightNormSavedArrays(step.SavedState, savedArrays);
        }
        foreach (var tensor in tensors)
            if (!produced.Contains(tensor)) protectedStorage.Add(tensor.StorageIdentity);

        var storageUsers = new Dictionary<object, int>(ReferenceEqualityComparer<object>.Instance);
        var vectorUsers = new Dictionary<object, int>(ReferenceEqualityComparer<object>.Instance);
        var arrayUsers = new Dictionary<object, int>(ReferenceEqualityComparer<object>.Instance);
        foreach (var tensor in tensors)
        {
            Count(storageUsers, tensor.StorageIdentity);
            Count(vectorUsers, tensor.GraphVectorIdentity);
            if (tensor.GraphBackingArrayIdentity is { } array) Count(arrayUsers, array);
        }

        // Decide eligibility before rebinds change reference counts. External tensor aliases are caught by the exact
        // tensor-owner + known trace-lease refcount; independent wrappers by vector/array identity and LiveStorages.
        var eligible = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        foreach (var step in steps)
        {
            var output = step.OutputBuffer;
            if (!IsPureWeightNorm(step) || protectedStorage.Contains(output.StorageIdentity)
                || writers[output] != 1 || storageUsers[output.StorageIdentity] != 1
                || vectorUsers[output.GraphVectorIdentity] != 1
                || output.GraphBackingArrayIdentity is { } array && (arrayUsers[array] != 1 || savedArrays.Contains(array))
                || !output.CanShareUnboundGraphStorage(traceScope?.OwnsStorageLease(output.StorageIdentity) == true))
                continue;
            eligible.Add(output);
        }

        var canonical = new Dictionary<WeightNormForwardKey, Tensor<T>>();
        var invalidated = new List<WeightNormForwardKey>();
        int shared = 0;
        foreach (var step in steps)
        {
            if (step.ReplaysHostState) canonical.Clear();
            // A metadata-only view does not write its source. Every actual declared source write starts a fresh
            // group: later duplicates must never overwrite a pre-write normalization still needed by backward.
            if (!IsWeightNormMetadataView(step))
            {
                invalidated.Clear();
                foreach (var pair in canonical)
                    if (SameWeightNormPhysicalStorage(step.OutputBuffer, pair.Key.Direction)
                        || SameWeightNormPhysicalStorage(step.OutputBuffer, pair.Key.Gain)
                        || SameWeightNormPhysicalStorage(step.OutputBuffer, pair.Value))
                        invalidated.Add(pair.Key);
                foreach (var key in invalidated) canonical.Remove(key);
            }
            if (!eligible.Contains(step.OutputBuffer)) continue;
            var keyForStep = new WeightNormForwardKey(step.Inputs[0], step.Inputs[1], (double)step.SavedState![0]);
            if (!canonical.TryGetValue(keyForStep, out var first)) canonical.Add(keyForStep, step.OutputBuffer);
            else
            {
                step.OutputBuffer.RebindStorageFromGraph(first);
                shared++;
            }
        }
        return shared;

        static void Count(Dictionary<object, int> counts, object identity)
            => counts[identity] = counts.TryGetValue(identity, out int count) ? count + 1 : 1;
    }

    private static bool IsPureWeightNorm(CompiledStep<T> step) => step.OpType == OpType.TensorWeightNorm
        && step.Inputs.Length == 2 && !step.ReplaysHostState && step.DifferentiableInputIndices is null
        && step.OutputBuffer._shape.AsSpan().SequenceEqual(step.Inputs[0]._shape)
        && step.SavedState is { Length: 1 } saved && saved[0] is double epsilon
        && !double.IsNaN(epsilon) && !double.IsInfinity(epsilon) && epsilon >= 0
        && (step.BackwardFn is null || step.BackwardFn == BackwardFunctions<T>.WeightNormBackward);

    private static bool IsWeightNormMetadataView(CompiledStep<T> step) => step.OutputBuffer.IsView
        && (step.OpName == "TensorReshape" || step.OpName == "Reshape" || step.OpName == "TensorTranspose"
            || step.OpName == "TensorPermute" || step.OpName == "StopGradient");

    private static void CollectWeightNormSavedArrays(object? saved, HashSet<object> arrays)
    {
        if (saved is null) return;
        var pending = new Stack<object>();
        var seen = new HashSet<object>(ReferenceEqualityComparer<object>.Instance);
        pending.Push(saved);
        while (pending.Count != 0)
        {
            var value = pending.Pop();
            if (!seen.Add(value)) continue;
            if (value is T[] values) arrays.Add(values);
            else if (value is Array array && array.GetType().GetElementType() is { IsValueType: false })
            {
                foreach (var item in array) if (item is not null) pending.Push(item);
            }
            else if (value is ISavedStateTensorContainer container)
                foreach (var item in container.SavedStateValues) if (item is not null) pending.Push(item);
        }
    }

    private static bool SameWeightNormPhysicalStorage(Tensor<T> left, Tensor<T> right) =>
        ReferenceEquals(left.StorageIdentity, right.StorageIdentity)
        || ReferenceEquals(left.GraphVectorIdentity, right.GraphVectorIdentity)
        || left.GraphBackingArrayIdentity is { } array && ReferenceEquals(array, right.GraphBackingArrayIdentity);
}
