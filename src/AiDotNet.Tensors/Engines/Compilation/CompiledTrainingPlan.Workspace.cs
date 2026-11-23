using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    internal static bool ShareTransientWorkspaceForTests { get; set; } = true;
    private readonly HashSet<IGpuBuffer> _workspaceProtectedRoots = new(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
    private readonly HashSet<IntPtr> _workspaceProtectedHandles = new();
    private readonly HashSet<object> _workspaceProtectedStorage = new(Autodiff.ReferenceEqualityComparer<object>.Instance);
    private readonly HashSet<Tensor<T>> _workspaceProduced = new(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
    private readonly HashSet<Tensor<T>> _workspaceTensors = new(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
    private readonly HashSet<IGpuBuffer> _workspaceCandidates = new(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
    private readonly List<Tensor<T>> _workspaceDetach = new();
    private bool _workspaceSharingBodyAllowed;
    private bool _workspaceExplicitGraphDisable;
    private bool _graphPreparationStarted;
    internal long SharedWorkspaceBytesLastStep { get; private set; }
    internal int SharedWorkspaceSlotsLastStep { get; private set; }

    // Capturable plans keep private stable pointers for their complete lifetime. The foreign-leaf guard's sticky
    // DisableGraphStep decision is authoritative; a transient capture failure or warmup is not permission to share.
    private bool CanShareTransientWorkspace(DirectGpuTensorEngine engine) => ShareTransientWorkspaceForTests
        && typeof(T) == typeof(float) && _workspaceExplicitGraphDisable && !_graphPreparationStarted
        && _graphStepDisabled && _graphStepEligible && !HasLiveStepGraph
        && !_residentStepFailed && _dataForwardReplay is null && _fp16HeteroOrder is null && _checkpointing is null
        && _checkpoint is null
        && !_isFrozenWeights && _lastBackwardReads is not null && _forwardSteps is not null
        && ForwardStepObserver is null && StepProbe is null && !SanitizeForwardBuffers
        && !engine.InferenceCaptureActive && !Gpu.AutocastScope.IsEnabled && Gpu.DeferredScope.Current?.IsRecording != true;

    private void ReturnTransientWorkspace(DirectGpuTensorEngine engine, IDirectGpuBackend backend)
    {
        SharedWorkspaceBytesLastStep = 0;
        SharedWorkspaceSlotsLastStep = 0;
        if (!_workspaceSharingBodyAllowed || !CanShareTransientWorkspace(engine) || _scratchPool is not { } pool) return;
        pool.WithdrawLent();
        if (pool.HasOutstandingWorkspaceLeases) return;
        if (!BuildWorkspaceProtection()) return;
        pool.CollectTransferableForwardSlots(_forwardActions.Length, IsUnprotectedRoot, _workspaceCandidates);
        if (_workspaceCandidates.Count == 0) return;
        // Snapshot aliases before detaching one shared vector clears the binding visible through its siblings.
        foreach (var tensor in _workspaceTensors)
            if (!_workspaceProtectedStorage.Contains(tensor.StorageIdentity) && tensor._gpuBuffer is { } buffer
                && _workspaceCandidates.Contains(GpuBufferViewRoot.Of(buffer))) _workspaceDetach.Add(tensor);
        foreach (var tensor in _workspaceDetach) engine.DetachCompiledWorkspaceTensor(tensor);
        TransferWorkspace(pool, backend, engine);
    }

    /// <summary>
    /// Collects what outlives a step in this plan's buffers: parameters, gradients, the loss and its seed, inputs and
    /// step inputs not produced by a forward step, retained outputs, in-place state, optimizer buffers and everything a
    /// forward step saved that it did not produce. False when a saved state is opaque (it may hide native pointers).
    /// </summary>
    private bool BuildWorkspaceProtection()
    {
        _workspaceProtectedRoots.Clear();
        _workspaceProtectedHandles.Clear();
        _workspaceProtectedStorage.Clear();
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
            _workspaceProtectedStorage.Add(tensor.StorageIdentity);
            ProtectBuffer(tensor._gpuBuffer);
        }
        bool InspectSaved(object? saved)
        {
            if (saved is null) return true;
            if (saved is Tensor<T> tensor)
            {
                _workspaceTensors.Add(tensor);
                if (!_workspaceProduced.Contains(tensor)) Protect(tensor);
                return true;
            }
            if (saved is Tensor<int> ints) { ProtectBuffer(ints._gpuBuffer); return true; }
            if (saved is Tensor<long> longs) { ProtectBuffer(longs._gpuBuffer); return true; }
            if (saved is IGpuBuffer buffer) { ProtectBuffer(buffer); return true; }
            if (saved is object[] nested)
            {
                foreach (var item in nested) if (!InspectSaved(item)) return false;
                return true;
            }
            // An opaque saved-state owner may hide native pointers. Refuse it rather than presume its buffers died.
            var type = saved.GetType();
            return type.IsPrimitive || type.IsEnum || saved is string || saved is int[] || saved is long[]
                || saved is float[] || saved is double[] || saved is bool[] || saved is int[][];
        }
        foreach (var step in _forwardSteps!) _workspaceProduced.Add(step.OutputBuffer);
        foreach (var parameter in _parameters) { Protect(parameter); Protect(parameter.Grad); }
        Protect(_compiledInputTensor);
        Protect(_graphRefreshTensor);
        Protect(_lossOutput);
        Protect(_lossGradDest);
        Protect(_lossGradSeed);
        foreach (var gradient in _gradients) Protect(gradient);
        foreach (var gradient in _preAllocatedGrads) Protect(gradient);
        if (_liveGradientMap is { } live) foreach (var gradient in live.Values) Protect(gradient);
        foreach (var leaf in _requestedInputGradientLeaves) { Protect(leaf); Protect(leaf.Grad); }
        if (_aliasedGradViews is { } views)
            foreach (var (view, target) in views) { Protect(view); Protect(target); }
        foreach (var output in _retainedOutputs) Protect(output);
        foreach (var output in _namedRetainedOutputs.Values) Protect(output);
        foreach (var state in InPlaceStateTensors()) Protect(state);
        foreach (var buffer in _gpuOptimizerBuffers) ProtectBuffer(buffer);
        if (_residentGradBuffers is { } resident) foreach (var buffer in resident) ProtectBuffer(buffer);
        if (_optimizerRuntimeState is { } runtime)
            foreach (var moments in GpuMomentSlots(runtime)) foreach (var buffer in moments) ProtectBuffer(buffer);
        foreach (var step in _forwardSteps!)
        {
            _workspaceTensors.Add(step.OutputBuffer);
            foreach (var input in step.Inputs)
            {
                _workspaceTensors.Add(input);
                if (!_workspaceProduced.Contains(input)) Protect(input);
            }
            if (step.SavedState is { } saved)
                foreach (var item in saved) if (!InspectSaved(item)) return false;
        }
        // Step-I/O inputs: a replay skips the copy into a buffer that still holds the same input.
        if (_stepIoLeaves is { } ioLeaves) foreach (var leaf in ioLeaves) Protect(leaf);
        if (_stepIoBuffers is { } ioBuffers) foreach (var buffer in ioBuffers) ProtectBuffer(buffer);
        // Owning tensor buffers are outside this loan contract even if a pool happens to name them.
        foreach (var tensor in _workspaceTensors)
            if (tensor._gpuBuffer is { } buffer && (tensor._ownsGpuBuffer || tensor._storage.OwnsGpuBuffer(buffer)))
                Protect(tensor);
        // A partial/strided public view may have no direct binding while its whole-storage sibling does.
        // Storage identity still makes that sibling's native root observable through the view.
        foreach (var tensor in _workspaceTensors)
            if (_workspaceProtectedStorage.Contains(tensor.StorageIdentity)) ProtectBuffer(tensor._gpuBuffer);
        return true;
    }

    private bool IsUnprotectedRoot(IGpuBuffer buffer)
    {
        var root = GpuBufferViewRoot.Of(buffer);
        return ReferenceEquals(root, buffer) && buffer.Handle != IntPtr.Zero
            && !_workspaceProtectedRoots.Contains(root) && !_workspaceProtectedHandles.Contains(root.Handle);
    }

    private void TransferWorkspace(StepScratchPool pool, IDirectGpuBackend backend, DirectGpuTensorEngine engine)
    {
        var broker = engine.CompiledWorkspaceBroker;
        SharedWorkspaceBytesLastStep = pool.TransferForwardSlots(_workspaceCandidates, backend, broker,
            out int transferred);
        SharedWorkspaceSlotsLastStep = transferred;
        if (transferred > 0) _lendSchedule = null; // Cached schedules contain physical buffers, not logical slots.
        if (Environment.GetEnvironmentVariable("AIDOTNET_GRAD_POOL_DIAG") == "1")
        {
            var buckets = new SortedDictionary<int, int>();
            foreach (var buffer in _workspaceCandidates)
                buckets[buffer.Size] = buckets.TryGetValue(buffer.Size, out int count) ? count + 1 : 1;
            try
            {
                AiDotNetEngine.LogDiagnostic($"[PLAN-WORKSPACE] plan={_planId} slots={transferred} bytes={SharedWorkspaceBytesLastStep} " +
                $"broker_bytes={broker.RetainedBytes} reuse_count={broker.ReuseCount} reuse_bytes={broker.ReuseBytes} " +
                $"buckets=[{string.Join(",", buckets)}]");
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                // A diagnostic sink cannot roll back the completed optimizer update.
            }
        }
    }

    // Idle step memory lent to the backend between captured steps (IIdleDeviceMemoryLender).
    private IIdleDeviceMemoryLender? _idleLender;
    private IGpuBuffer[]? _idleBuffers;
    private (IntPtr Exec, IntPtr Odd, int Assignment, int Count) _idleBuffersFor;

    /// <summary>Bytes this plan lent after its last step (tests and diagnostics).</summary>
    internal long IdleLentBytesLastStep { get; private set; }

    /// <summary>
    /// After a captured step: lends the step buffers nothing reads before the next replay writes them to the backend,
    /// so validation and eager inference between steps allocate from them instead of growing the device footprint.
    /// They come back (moving whatever still uses them) when the next step begins (<see cref="ReclaimIdleStepMemory"/>).
    /// </summary>
    private void LendIdleStepMemory()
    {
        IdleLentBytesLastStep = 0;
        if (!LendIdleStepMemoryEnabled || !HasLiveStepGraph || _scratchPool is not { } pool || _forwardSteps is null
            || !_engine.TryGet<DirectGpuTensorEngine>(out var engine)
            || engine.GetBackend() is not IIdleDeviceMemoryLender lender)
            return;
        var key = (_stepGraphExec, _stepGraphExecOdd, pool.PinnedCount, pool.BufferCount);
        if (_idleBuffers is null || _idleBuffersFor != key)
        {
            _idleBuffers = null;
            if (!BuildWorkspaceProtection()) return;
            _workspaceCandidates.Clear();
            pool.CollectIdleBetweenRuns(_forwardActions.Length, IsUnprotectedRoot, _workspaceCandidates);
            _idleBuffers = new IGpuBuffer[_workspaceCandidates.Count];
            _workspaceCandidates.CopyTo(_idleBuffers);
            _workspaceCandidates.Clear();
            _idleBuffersFor = key;
        }
        if (_idleBuffers.Length == 0) return;
        lender.LendIdle(this, _idleBuffers);
        _idleLender = lender;
        long bytes = 0;
        foreach (var buffer in _idleBuffers) bytes += buffer.SizeInBytes;
        IdleLentBytesLastStep = bytes;
    }

    /// <summary>Takes back the memory <see cref="LendIdleStepMemory"/> lent: before a step, an invalidation or disposal.</summary>
    private void ReclaimIdleStepMemory()
    {
        if (System.Threading.Interlocked.Exchange(ref _idleLender, null) is { } lender) lender.ReclaimIdle(this);
    }

    /// <summary>Process switch for tests that need the backend's own allocations between steps.</summary>
    internal static bool LendIdleStepMemoryEnabled { get; set; } = true;
}
