using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    // Channel-bias backward uses shape and upstream gradients. LeakyReLU with a finite
    // nonnegative slope has the same x > 0 derivative mask after its forward overwrites x.
    // A private, single-reader bias result may therefore also hold its activated result.
    internal static int ShareCpuLeakyBiasStorage(IReadOnlyList<CompiledStep<T>> steps, Tensor<T>[] parameters,
        Tensor<T> loss, HashSet<Tensor<T>> retained, IEngine engine, LazyTensorScope scope)
    {
        if (engine.GetType() != typeof(CpuEngine) || (typeof(T) != typeof(float) && typeof(T) != typeof(double))
            || scope.HostReadOp is not null || ForwardStepObserver is not null || SanitizeForwardBuffers
            || !steps.Any(step => step.OpType == OpType.LeakyReLU))
            return 0;

        var comparer = ReferenceEqualityComparer<Tensor<T>>.Instance;
        var tensors = new HashSet<Tensor<T>>(parameters, comparer) { loss };
        tensors.UnionWith(retained);
        var producers = new Dictionary<Tensor<T>, CompiledStep<T>?>(comparer);
        var reshapedViews = new Dictionary<Tensor<T>, Tensor<T>?>(comparer);
        var readers = new Dictionary<Tensor<T>, int>(comparer);
        var protectedStorage = new HashSet<object>(ReferenceEqualityComparer<object>.Instance);
        foreach (var tensor in tensors) protectedStorage.Add(tensor.StorageIdentity);
        var saved = new WeightNormSavedStorageVisitor(protectedStorage, tensors);
        var savedArrays = new HashSet<object>(ReferenceEqualityComparer<object>.Instance);
        foreach (var step in steps)
        {
            tensors.Add(step.OutputBuffer);
            producers[step.OutputBuffer] = producers.ContainsKey(step.OutputBuffer) ? null : step;
            if (step.OpName == "Reshape" && step.BackwardFn == BackwardFunctions<T>.ReshapeBackward
                && !step.ReplaysHostState && step.Inputs.Length == 1 && step.OutputBuffer.SharesStorageWith(step.Inputs[0]))
            {
                var parent = step.Inputs[0];
                reshapedViews[parent] = reshapedViews.ContainsKey(parent) ? null : step.OutputBuffer;
            }
            foreach (var input in step.Inputs)
            {
                if (input is null) continue;
                tensors.Add(input);
                readers.TryGetValue(input, out int count);
                readers[input] = count + 1;
            }
            SavedStateTensorTraversal.Visit(step.SavedState, ref saved);
            CollectWeightNormSavedArrays(step.SavedState, savedArrays);
        }

        var storages = new Dictionary<object, int>(ReferenceEqualityComparer<object>.Instance);
        var vectors = new Dictionary<object, int>(ReferenceEqualityComparer<object>.Instance);
        var arrays = new Dictionary<object, int>(ReferenceEqualityComparer<object>.Instance);
        foreach (var tensor in tensors)
        {
            if (!producers.ContainsKey(tensor)) protectedStorage.Add(tensor.StorageIdentity);
            Count(storages, tensor.StorageIdentity);
            Count(vectors, tensor.GraphVectorIdentity);
            if (tensor.GraphBackingArrayIdentity is { } array) Count(arrays, array);
        }

        bool Private(Tensor<T> tensor, Tensor<T>? allowedView)
        {
            bool shared = allowedView is not null && tensor.SharesStorageWith(allowedView);
            int owners = shared ? 2 : 1;
            int live = (tensor.IsDisposed ? 0 : 1) + (shared && !allowedView!.IsDisposed ? 1 : 0);
            return !protectedStorage.Contains(tensor.StorageIdentity)
                && storages[tensor.StorageIdentity] == owners && vectors[tensor.GraphVectorIdentity] == owners
                && (tensor.GraphBackingArrayIdentity is not { } data || arrays[data] == owners && !savedArrays.Contains(data))
                && tensor.CanShareUnboundGraphStorage(scope.OwnsStorageLease(tensor.StorageIdentity), live);
        }

        int sharedCount = 0;
        foreach (var activation in steps)
        {
            if (activation.OpType != OpType.LeakyReLU || activation.BackwardFn != BackwardFunctions<T>.LeakyReLUBackward
                || activation.Inputs.Length != 1 || activation.ReplaysHostState
                || activation.SavedState is not { Length: 1 } args || args[0] is not double slope
                || !(slope >= 0.0) || double.IsInfinity(slope) || typeof(T) == typeof(float) && float.IsInfinity((float)slope))
                continue;
            var input = activation.Inputs[0];
            var output = activation.OutputBuffer;
            if (readers[input] != 1 || producers[output] != activation || output.IsView
                || !input._shape.AsSpan().SequenceEqual(output._shape)
                || !producers.TryGetValue(input, out var source) || source is null)
                continue;
            Tensor<T>? biasViewSource = null;
            if (source.OpName == "Reshape" && source.Inputs.Length == 1 && source.BackwardFn == BackwardFunctions<T>.ReshapeBackward)
            {
                biasViewSource = source.Inputs[0];
                if (readers[biasViewSource] != 1 || !producers.TryGetValue(biasViewSource, out source) || source is null)
                    continue;
            }
            if (source.OpName != "TensorChannelBiasAdd" || source.Inputs.Length != 2
                || source.BackwardFn != BackwardFunctions<T>.ChannelBiasAddBackward || source.OutputBuffer.IsView
                || !Private(input, biasViewSource)
                || biasViewSource is not null && input.SharesStorageWith(biasViewSource) && !Private(biasViewSource, input))
                continue;
            reshapedViews.TryGetValue(output, out var outputView);
            if (!Private(output, outputView) || outputView is not null && !Private(outputView, output))
                continue;
            // Rebind the complete result and its recorded reshape together. A caller-held
            // view, retained state, or an unrelated storage wrapper fails the ownership checks.
            protectedStorage.Add(input.StorageIdentity);
            output.RebindStorageFromGraph(input);
            outputView?.RebindGraphReshapeStorageFrom(output);
            sharedCount++;
        }
        return sharedCount;

        static void Count(Dictionary<object, int> counts, object identity)
        {
            counts.TryGetValue(identity, out int count);
            counts[identity] = count + 1;
        }
    }
}
