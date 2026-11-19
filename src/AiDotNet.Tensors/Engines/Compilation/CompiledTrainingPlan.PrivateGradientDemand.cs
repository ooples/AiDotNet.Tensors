using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T> : IInferencePrivateGradientWorkspaceOwner
{
    // Weak keys observe allocations without extending the lifetime of a buffer replaced by another owner.
    private readonly ConditionalWeakTable<IGpuBuffer, object> _privateGradientAllocations = new();
    private static readonly object s_privateGradientAllocation = new();
    private readonly HashSet<object> _privateProtectedVectors = new(ReferenceEqualityComparer<object>.Instance);
    private readonly HashSet<object> _privateProtectedArrays = new(ReferenceEqualityComparer<object>.Instance);
    private IDirectGpuBackend? _privateGradientBackend;
    private int _privateGradientThread;
    private bool _privateGradientRegistered;
    private bool _privateGradientCompletedStep;
    internal long PrivateGradientNativeReleaseRequestedBytes { get; private set; }
    internal long PrivateGradientNativeReleaseRequestedBuffers { get; private set; }
    internal int FailPrivateGradientResidencyAtForTesting { get; set; } = -1;
    bool IInferencePrivateGradientWorkspaceOwner.PrivateGradientWorkspaceDemandPending => _privateGradientCompletedStep && !_disposed;

    private static void TrackFreshPrivateGradientTensor(Tensor<T> tensor)
    {
        if (tensor.GraphBackingArrayIdentity is T[] array)
            ((Vector<T>)tensor.GraphVectorIdentity).ResultOwner = ResultBufferTracker.TrackFreshPrivateArray(array, 1);
    }

    internal void RecordPrivateGradientAllocation(IGpuBuffer buffer)
    {
        if (typeof(T) == typeof(float) && ReferenceEquals(GpuBufferViewRoot.Of(buffer), buffer))
            _privateGradientAllocations.GetValue(buffer, static _ => s_privateGradientAllocation);
    }

    internal void RegisterPrivateGradientDemandOwner(DirectGpuTensorEngine engine, IDirectGpuBackend backend)
    {
        if (!CanReturnPrivateGradientWorkspace(engine)
            || Environment.GetEnvironmentVariable("AIDOTNET_INFERENCE_PRIVATE_GRADIENT_RETURN") == "0") return;
        _privateGradientBackend = backend;
        _privateGradientThread = Environment.CurrentManagedThreadId;
        _privateGradientCompletedStep = true;
        engine.MarkPrivateGradientWorkspaceReady();
        if (_privateGradientRegistered) return;
        engine.RegisterPrivateGradientWorkspace(this);
        _privateGradientRegistered = true;
    }

    private bool CanReturnPrivateGradientWorkspace(DirectGpuTensorEngine engine) => !_disposed
        && typeof(T) == typeof(float) && ReferenceEquals(_engine, engine)
        && _engine.Unwrap().GetType() == typeof(DirectGpuTensorEngine)
        && _workspaceExplicitGraphDisable && !_graphPreparationStarted && _graphStepDisabled
        && _graphStepEligible && !HasLiveStepGraph && _residentBodyRanLast && !_residentStepFailed
        && _dataForwardReplay is null && _fp16HeteroOrder is null && _checkpointing is null
        && !_isFrozenWeights && _lastBackwardReads is not null && _forwardSteps is not null
        && _scratchPool is not null && !_scratchPool.HasOutstandingWorkspaceLeases
        && ForwardStepObserver is null && StepProbe is null && !SanitizeForwardBuffers;

    void IInferencePrivateGradientWorkspaceOwner.ReturnPrivateGradientWorkspaceForInference(
        DirectGpuTensorEngine engine, IDirectGpuBackend backend)
        => ReturnPrivateGradientWorkspaceForInference(engine, backend);

    internal void ReturnPrivateGradientWorkspaceForInference(DirectGpuTensorEngine engine, IDirectGpuBackend backend)
    {
        if (!_privateGradientCompletedStep || !CanReturnPrivateGradientWorkspace(engine)
            || Environment.GetEnvironmentVariable("AIDOTNET_INFERENCE_PRIVATE_GRADIENT_RETURN") == "0"
            || !ReferenceEquals(_privateGradientBackend, backend)
            || _privateGradientThread != Environment.CurrentManagedThreadId
            || engine.ResidentStepActive || TrainingStep.InStep || engine.InferenceCaptureActive
            || GradientRecording.IsRecording<float>() || GraphMode.IsActive || Gpu.DeferredScope.Current is not null
            || Gpu.AutocastScope.IsEnabled
            || backend is IGpuGraphCapture { SupportsGraphCapture: true } capture && capture.IsStreamCapturing())
            return;
        // One inspection per completed step, including a step whose buffers all escaped. Repeated inference allocations
        // must not rebuild this metadata, and another successful training step publishes a fresh opportunity.
        _privateGradientCompletedStep = false;
        if (InferenceWorkspaceDemandScope.HandoffIdleWorkspaceEnabled)
        {
            _scratchPool!.ReturnIdleLeasesToBackendCache(backend);
            _firstWriteRedirects.Clear();
        }
        _workspaceProtectedRoots.Clear();
        _workspaceProtectedHandles.Clear();
        _workspaceProtectedStorage.Clear();
        _privateProtectedVectors.Clear();
        _privateProtectedArrays.Clear();
        _workspaceProduced.Clear();
        _workspaceTensors.Clear();
        _workspaceCandidates.Clear();
        _workspaceDetach.Clear();

        void ProtectBuffer(IGpuBuffer? buffer)
        {
            if (buffer is null) return;
            var root = GpuBufferViewRoot.Of(buffer);
            _workspaceProtectedRoots.Add(root);
            if (root.Handle != IntPtr.Zero) _workspaceProtectedHandles.Add(root.Handle);
        }
        void Protect(Tensor<T>? tensor)
        {
            if (tensor is null) return;
            _workspaceTensors.Add(tensor);
            _workspaceProtectedStorage.Add(tensor.StorageIdentity);
            _privateProtectedVectors.Add(tensor.GraphVectorIdentity);
            if (tensor.GraphBackingArrayIdentity is { } array) _privateProtectedArrays.Add(array);
            ProtectBuffer(tensor._gpuBuffer);
        }
        bool InspectSaved(object? initial)
        {
            if (initial is null) return true;
            var pending = new Stack<object>();
            var seen = new HashSet<object>(ReferenceEqualityComparer<object>.Instance);
            pending.Push(initial);
            while (pending.Count != 0)
            {
                var value = pending.Pop();
                if (!seen.Add(value)) continue;
                if (value is Tensor<T> tensor) { Protect(tensor); Protect(tensor.Grad); continue; }
                if (value is Tensor<int> ints) { ProtectBuffer(ints._gpuBuffer); continue; }
                if (value is Tensor<long> longs) { ProtectBuffer(longs._gpuBuffer); continue; }
                if (value is IGpuBuffer buffer) { ProtectBuffer(buffer); continue; }
                if (value is T[] array) { _privateProtectedArrays.Add(array); continue; }
                if (value is object[] nested)
                {
                    foreach (var item in nested) if (item is not null) pending.Push(item);
                    continue;
                }
                if (value is ISavedStateTensorContainer container)
                {
                    foreach (var item in container.SavedStateValues) if (item is not null) pending.Push(item);
                    continue;
                }
                // Unknown owners may hide a native pointer or array shared with a public result.
                var type = value.GetType();
                if (!(type.IsPrimitive || type.IsEnum || value is string || value is int[] || value is long[]
                    || value is float[] || value is double[] || value is bool[] || value is int[][])) return false;
            }
            return true;
        }
        foreach (var gradient in _preAllocatedGrads) _workspaceTensors.Add(gradient);
        if (_liveGradientMap is { } live)
            foreach (var gradient in live.Values) _workspaceTensors.Add(gradient);
        if (_aliasedGradViews is { } aliases)
            foreach (var (view, target) in aliases) { _workspaceTensors.Add(view); _workspaceTensors.Add(target); }
        foreach (var step in _forwardSteps!) _workspaceProduced.Add(step.OutputBuffer);
        foreach (var step in _forwardSteps!)
        {
            _workspaceTensors.Add(step.OutputBuffer);
            ProtectBuffer(step.OutputBuffer._gpuBuffer); // Primal bindings have a separate replay/retirement lifecycle.
            foreach (var input in step.Inputs)
            {
                _workspaceTensors.Add(input);
                if (!_workspaceProduced.Contains(input)) { Protect(input); Protect(input.Grad); }
            }
            if (step.SavedState is { } saved && !InspectSaved(saved)) return;
        }
        foreach (var parameter in _parameters) { Protect(parameter); Protect(parameter.Grad); }
        foreach (var result in ResultGradients()) Protect(result);
        foreach (var output in _retainedOutputs)
        {
            Protect(output); Protect(output.Grad);
            if (_liveGradientMap?.TryGetValue(output, out var gradient) == true) Protect(gradient);
        }
        foreach (var output in _namedRetainedOutputs.Values) { Protect(output); Protect(output.Grad); }
        foreach (var state in InPlaceStateTensors()) { Protect(state); Protect(state.Grad); }
        Protect(_compiledInputTensor); Protect(_graphRefreshTensor);
        Protect(_lossOutput); Protect(_lossGradDest); Protect(_lossGradSeed);
        foreach (var buffer in _gpuOptimizerBuffers) ProtectBuffer(buffer);
        foreach (var buffer in _ownedLeafBuffers.Values) ProtectBuffer(buffer);
        foreach (var buffer in _foreignParameterLeaves.Values) ProtectBuffer(buffer);
        if (_residentGradBuffers is { } resident) foreach (var buffer in resident) ProtectBuffer(buffer);
        if (_optimizerRuntimeState is { } runtime)
            foreach (var moments in GpuMomentSlots(runtime)) foreach (var buffer in moments) ProtectBuffer(buffer);

        var knownOwners = new Dictionary<object, int>(ReferenceEqualityComparer<object>.Instance);
        var knownVectorStorages = new Dictionary<object, HashSet<object>>(ReferenceEqualityComparer<object>.Instance);
        var knownArrayVectors = new Dictionary<object, HashSet<object>>(ReferenceEqualityComparer<object>.Instance);
        foreach (var tensor in _workspaceTensors)
        {
            object storage = tensor.StorageIdentity;
            if (!tensor.IsDisposed) knownOwners[storage] = knownOwners.TryGetValue(storage, out int n) ? n + 1 : 1;
            if (!knownVectorStorages.TryGetValue(tensor.GraphVectorIdentity, out var storages))
                knownVectorStorages[tensor.GraphVectorIdentity] = storages = new HashSet<object>(ReferenceEqualityComparer<object>.Instance);
            storages.Add(storage);
            if (tensor.GraphBackingArrayIdentity is { } array)
            {
                if (!knownArrayVectors.TryGetValue(array, out var vectors))
                    knownArrayVectors[array] = vectors = new HashSet<object>(ReferenceEqualityComparer<object>.Instance);
                vectors.Add(tensor.GraphVectorIdentity);
            }
        }
        foreach (var tensor in _workspaceTensors)
        {
            int expected = (knownOwners.TryGetValue(tensor.StorageIdentity, out int n) ? n : 0)
                + (_storageLeases?.ContainsStorage(tensor.StorageIdentity) == true ? 1 : 0);
            var vector = (Vector<T>)tensor.GraphVectorIdentity;
            if (tensor._storage.RefCount != expected || vector.LiveStorages != knownVectorStorages[vector].Count
                || tensor.HasPendingCopyOnWritePeer || tensor._ownsGpuBuffer || tensor._storage.HasGpuBufferOwner
                || tensor.GraphBackingArrayIdentity is { } array && (vector.ResultOwner is not { } owner
                    || owner.State.Escaped || System.Threading.Volatile.Read(ref owner.State.Wrappers) != knownArrayVectors[array].Count))
                Protect(tensor);
        }
        foreach (var gradient in _preAllocatedGrads)
        {
            if (gradient._gpuBuffer is not { } buffer || !ReferenceEquals(gradient._gpuBackend, backend)) continue;
            var root = GpuBufferViewRoot.Of(buffer);
            if (root.Handle != IntPtr.Zero && root is IPoolableGpuBuffer && !StepScratchPool.IsPooled(root)
                && _privateGradientAllocations.TryGetValue(root, out _)) _workspaceCandidates.Add(root);
        }
        // A public alias can lack a native binding altogether, or have a separate GPU root over the same host array.
        // Removing that array/vector's materializer would lose the public value even if native root identities differ.
        foreach (var tensor in _workspaceTensors)
            if (tensor._gpuBuffer is { } buffer && (_workspaceProtectedStorage.Contains(tensor.StorageIdentity)
                || _privateProtectedVectors.Contains(tensor.GraphVectorIdentity)
                || tensor.GraphBackingArrayIdentity is { } array && _privateProtectedArrays.Contains(array)))
                ProtectBuffer(buffer);
        _workspaceCandidates.RemoveWhere(root => _workspaceProtectedRoots.Contains(root)
            || _workspaceProtectedHandles.Contains(root.Handle));
        if (_workspaceCandidates.Count == 0) return;
        foreach (var tensor in _workspaceTensors)
            if (tensor._gpuBuffer is { } buffer && _workspaceCandidates.Contains(GpuBufferViewRoot.Of(buffer)))
                _workspaceDetach.Add(tensor);

        _firstWriteRedirects.Clear(); // These targets are physical buffers, not live gradient tensor lookups.
        _gradHeldZero = null;
        _unwrittenGradIndices = null;
        _pooledGradZeroAtStart = null;
        _lendSchedule = null;
        if (_pooledGradDeviceBuffers is { } pooled)
            foreach (var pair in pooled.Where(pair => _workspaceCandidates.Contains(GpuBufferViewRoot.Of(pair.Value))).ToArray())
                pooled.Remove(pair.Key);
        // Snapshot all aliases before a shared VectorDeviceState binding is cleared by the first detach.
        // Cache invalidation normally disposes an entry's buffer into the backend cache. For this short ownership
        // transfer the plan must keep the root, then issue its one native Release; the existing pooled marker tells
        // ActivationCacheEntry.Dispose that its entry is only a borrower. No slot or loan is created or counted.
        foreach (var root in _workspaceCandidates) _scratchPool!.Track(root);
        try
        {
            foreach (var tensor in _workspaceDetach) engine.DetachCompiledWorkspaceTensor(tensor);
        }
        finally
        {
            foreach (var root in _workspaceCandidates) StepScratchPool.Untrack(root);
        }
        foreach (var root in _workspaceCandidates)
        {
            long bytes = root.SizeInBytes;
            ((IPoolableGpuBuffer)root).Release();
            _privateGradientAllocations.Remove(root);
            PrivateGradientNativeReleaseRequestedBytes += bytes;
            PrivateGradientNativeReleaseRequestedBuffers++;
        }
    }
}
