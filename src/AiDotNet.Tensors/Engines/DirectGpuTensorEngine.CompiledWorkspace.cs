using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    private readonly List<WeakReference<IInferencePrivateGradientWorkspaceOwner>> _privateGradientOwners = new();
    private volatile bool _privateGradientDemandPending;

    internal void MarkPrivateGradientWorkspaceReady() => _privateGradientDemandPending = true;

    internal void RegisterPrivateGradientWorkspace(IInferencePrivateGradientWorkspaceOwner owner)
    {
        // Called only at a completed step while CompiledStepLock is held. The registry must not keep a model alive.
        for (int i = _privateGradientOwners.Count - 1; i >= 0; i--)
            if (!_privateGradientOwners[i].TryGetTarget(out _)) _privateGradientOwners.RemoveAt(i);
        _privateGradientOwners.Add(new WeakReference<IInferencePrivateGradientWorkspaceOwner>(owner));
    }

    private CompiledTransientWorkspaceBroker? _compiledWorkspaceBroker;
    internal CompiledTransientWorkspaceBroker CompiledWorkspaceBroker =>
        _compiledWorkspaceBroker ??= new CompiledTransientWorkspaceBroker(this);

    // An inference demand can exceed every idle compiled cell (e.g. a larger batch). Return only that broker's
    // already-detached, permanently uncaptured backing so the native allocator can satisfy the new size regime.
    // Parameter/result/optimizer/captured roots and other plans' outstanding loans never enter the broker's free list.
    internal void ReturnIncompatibleInferenceWorkspace(IDirectGpuBackend backend, long requestedBytes)
    {
        if ((_compiledWorkspaceBroker is not { RetainedBytes: > 0 } && !_privateGradientDemandPending)
            || ResidentStepActive || InferenceCaptureActive || TrainingStep.InStep
            || Autodiff.GradientRecording.IsRecording<float>()
            || Compilation.GraphMode.IsActive || Gpu.DeferredScope.Current?.IsRecording == true
            || Gpu.AutocastScope.IsEnabled)
            return;
        using var operation = EnterGpuOp();
        if (!operation.Held) return;
        lock (CompiledStepLock)
        {
            if (ResidentStepActive || InferenceCaptureActive || TrainingStep.InStep
                || backend is IGpuGraphCapture { SupportsGraphCapture: true } capture && capture.IsStreamCapturing())
                return;
            _compiledWorkspaceBroker?.ReturnIncompatibleCellsToNativeAllocator(backend, requestedBytes);
            bool pending = false;
            for (int i = _privateGradientOwners.Count - 1; i >= 0; i--)
            {
                if (!_privateGradientOwners[i].TryGetTarget(out var owner)) _privateGradientOwners.RemoveAt(i);
                else
                {
                    owner.ReturnPrivateGradientWorkspaceForInference(this, backend);
                    pending |= owner.PrivateGradientWorkspaceDemandPending;
                }
            }
            _privateGradientDemandPending = pending;
        }
    }

    /// <summary>Step scratch this engine's compiled plans co-own (see <see cref="SharedStepScratch"/>).</summary>
    internal SharedStepScratch SharedStepScratch { get; } = new();

    // Only called for private, dead compiled intermediates while their scratch owner still owns the native buffer.
    // Reuse never materializes host data; a stale internal read fails rather than reading another plan's new value.
    internal void DetachCompiledWorkspaceTensor<T>(Tensor<T> tensor)
    {
        var array = tensor.GetBackingArrayForCacheLookupUnsafe();
        HostSync.Release(tensor.DataVector, ReleasedIntermediateMessage);
        if (array is not null) HostSync.Release(array, ReleasedIntermediateMessage);
        if (tensor._gpuMaterializerKey is { } key) HostSync.Release(key, ReleasedIntermediateMessage);
        tensor._gpuMaterializerKey = null;
        // Retain the reusable callback object; a later BindResidentBuffer rebinds it and registers the current key.
        if (array is not null) InvalidateActivationCacheEntry(array);
        InvalidateActivationCacheEntry(tensor.DataVector);
        tensor._gpuBuffer = null;
        tensor._gpuBackend = null;
        tensor._gpuBufferVersion = -1;
    }
}
