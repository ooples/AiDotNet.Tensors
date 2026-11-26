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
        && !_residentStepFailed && _dataForwardReplay is null && _checkpointing is null
        && _checkpoint is null
        && !_isFrozenWeights && _lastBackwardReads is not null && _forwardSteps is not null
        && ForwardStepObserver is null && StepProbe is null && !SanitizeForwardBuffers
        && !engine.InferenceCaptureActive && !Gpu.AutocastScope.IsEnabled && Gpu.DeferredScope.Current?.IsRecording != true;

    // The scratch bytes and buffer count this plan last offered to the engine's co-owned step scratch.
    private long _publishedScratchBytes = -1;
    private int _publishedScratchCount = -1;

    /// <summary>
    /// Whether this plan's step scratch may be co-owned with the engine's other plans (SharedStepScratch): a plan the
    /// cross-plan workspace broker does not serve -- one that keeps stable pointers, captured or capturable -- whose
    /// step-persistent buffers <see cref="TryProtectStepPersistentBuffers"/> can see.
    /// </summary>
    /// <remarks>An eager plan whose graph step was disabled explicitly is the broker's, even when autocast keeps the
    /// broker from serving it this run: co-owning would bind its tensors to memory another plan writes, which the broker
    /// prevents by detaching them (DetachCompiledWorkspaceTensor).</remarks>
    private bool CanCoOwnStepScratch(DirectGpuTensorEngine engine) => SharedStepScratch.Enabled
        && typeof(T) == typeof(float) && !_workspaceSharingBodyAllowed
        && !(_workspaceExplicitGraphDisable && _graphStepDisabled)
        && _dataForwardReplay is null && _checkpointing is null && _checkpoint is null && _forwardSteps is not null
        && ForwardStepObserver is null && StepProbe is null && !SanitizeForwardBuffers
        && !engine.InferenceCaptureActive && Gpu.DeferredScope.Current?.IsRecording != true;

    /// <summary>
    /// After a whole executed run: offers this plan's step-transient scratch to the engine's other plans, when the pool
    /// has changed since the last offer (the first runs allocate it; later runs reuse it).
    /// </summary>
    private void PublishStepScratch(DirectGpuTensorEngine engine, IDirectGpuBackend backend, StepScratchPool pool)
    {
        long bytes = pool.BufferBytes;
        int count = pool.BufferCount;
        if (bytes == _publishedScratchBytes && count == _publishedScratchCount) return;
        _publishedScratchBytes = bytes;
        _publishedScratchCount = count;
        if (!TryProtectStepPersistentBuffers()) return;
        var offer = new List<IGpuBuffer>();
        foreach (var buffer in pool.AllBuffers()) if (IsStepTransient(buffer)) offer.Add(buffer);
        engine.SharedStepScratch.Publish(pool, backend, offer);
        if (Environment.GetEnvironmentVariable("AIDOTNET_GRAD_POOL_DIAG") == "1")
        {
            long offered = 0;
            foreach (var buffer in offer) offered += buffer.SizeInBytes;
            var shared = engine.SharedStepScratch;
            try
            {
                AiDotNetEngine.LogDiagnostic($"[PLAN-SHARED-SCRATCH] plan={_planId} pool_bytes={bytes} pool_buffers={count} " +
                    $"offered_bytes={offered} offered_buffers={offer.Count} engine_shared={shared.Count} " +
                    $"takes={shared.Takes} taken_bytes={shared.TakenBytes} revocations={shared.Revocations}");
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                // A diagnostic sink cannot undo the completed run.
            }
        }
    }

    private void ValidateTakenStepScratch(SharedStepScratch registry, StepScratchPool pool, bool completed)
    {
        var taken = pool.TakeTakenThisRun();
        if (taken.Count == 0) return;
        bool known = completed && TryProtectStepPersistentBuffers();
        foreach (var buffer in taken)
            if (!known || !IsStepTransient(buffer)) registry.Revoke(buffer, pool);
        // The next offer must re-read the pool: what was revoked is no longer shared.
        _publishedScratchBytes = -1;
    }

    private void ReturnTransientWorkspace(DirectGpuTensorEngine engine, IDirectGpuBackend backend)
    {
        SharedWorkspaceBytesLastStep = 0;
        SharedWorkspaceSlotsLastStep = 0;
        if (!_workspaceSharingBodyAllowed || !CanShareTransientWorkspace(engine) || _scratchPool is not { } pool) return;
        pool.WithdrawLent();
        if (pool.HasOutstandingWorkspaceLeases) return;
        if (!TryProtectStepPersistentBuffers()) return;
        pool.CollectTransferableForwardSlots(_forwardActions.Length, IsStepTransient, _workspaceCandidates);
        if (_workspaceCandidates.Count == 0) return;
        // Snapshot aliases before detaching one shared vector clears the binding visible through its siblings.
        foreach (var tensor in _workspaceTensors)
            if (!_workspaceProtectedStorage.Contains(tensor.StorageIdentity) && tensor._gpuBuffer is { } buffer
                && _workspaceCandidates.Contains(GpuBufferViewRoot.Of(buffer))) _workspaceDetach.Add(tensor);
        foreach (var tensor in _workspaceDetach) engine.DetachCompiledWorkspaceTensor(tensor);
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

    /// <summary>
    /// Records every device buffer something reads outside a step of this plan -- the parameters and their gradients,
    /// the inputs, the loss and its seed, retained outputs, in-place state, optimizer and resident gradient buffers, and
    /// any tensor a step saves that no forward step produces -- so that <see cref="IsStepTransient"/> accepts only
    /// buffers whose values live within one step. False when a step saves state whose buffers cannot be seen.
    /// </summary>
    private bool TryProtectStepPersistentBuffers()
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
            // A fused-attention node saves the row log-sum-exp its forward wrote for its backward.
            if (saved is DirectGpuTensorEngine.FusedAttentionRecordedNode attention)
            {
                ProtectBuffer(attention.Lse?._gpuBuffer);
                return true;
            }
            // A fused readout's mutation guard holds the readout weight and bias.
            if (saved is CpuEngine.ReadoutMutationGuard<T> readout)
            {
                Protect(readout.Weight);
                Protect(readout.Bias);
                return true;
            }
            // A fused node's saved inner backward (ConvChannelEpilogueBackward keeps the convolution's): a static method
            // captures nothing. A closure may hold buffers, so it stays opaque below.
            if (saved is Delegate { Target: null }) return true;
            // A channels-last fused node's layouts: its geometry, and the convolution input it reads in the backward.
            if (saved is ChannelsLastSlot slot) return InspectSaved(slot.Input);
            // An opaque saved-state owner may hide native pointers. Refuse it rather than presume its buffers died.
            var type = saved.GetType();
            bool plain = type.IsPrimitive || type.IsEnum || saved is string || saved is int[] || saved is long[]
                || saved is float[] || saved is double[] || saved is bool[] || saved is int[][];
            if (!plain) _workspaceOpaqueSavedType = type.FullName;
            return plain;
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

    /// <summary>Whether <paramref name="buffer"/> holds only values one step writes and reads (after
    /// <see cref="TryProtectStepPersistentBuffers"/>): a whole allocation no protected tensor is bound to.</summary>
    private bool IsStepTransient(IGpuBuffer buffer)
    {
        var root = GpuBufferViewRoot.Of(buffer);
        return ReferenceEquals(root, buffer) && buffer.Handle != IntPtr.Zero
            && !_workspaceProtectedRoots.Contains(root) && !_workspaceProtectedHandles.Contains(root.Handle);
    }

    // Idle step memory lent to the backend between captured steps (IIdleDeviceMemoryLender).
    private IIdleDeviceMemoryLender? _idleLender;

    /// <summary>Bytes this plan lent after its last step, once an allocation asked for them (tests and diagnostics).</summary>
    internal long IdleLentBytesLastStep { get; private set; }

    /// <summary>
    /// After a captured step: lends the step buffers nothing reads before the next replay writes them to the backend,
    /// so validation and eager inference between steps allocate from them instead of growing the device footprint.
    /// They come back (moving whatever still uses them) when the next step begins (<see cref="ReclaimIdleStepMemory"/>).
    /// They are found only when an allocation between steps needs memory the backend's pool cannot serve: finding them
    /// walks every step of the plan (about 2 ms a step for a 16-layer LM), host time that was spent after every step,
    /// while the device waited, even when nothing allocated before the next one.
    /// </summary>
    private void LendIdleStepMemory()
    {
        IdleLentBytesLastStep = 0;
        if (s_idleLendDiag)
            AiDotNetEngine.LogDiagnostic($"[IDLE-LEND] plan={_planId} enabled={LendIdleStepMemoryEnabled} graph={HasLiveStepGraph} " +
                $"pool={_scratchPool is not null} forward_steps={_forwardSteps is not null}");
        if (!LendIdleStepMemoryEnabled || _idleLendingFailed || !HasLiveStepGraph || _scratchPool is not { } pool || _forwardSteps is null
            || !_engine.TryGet<DirectGpuTensorEngine>(out var engine)
            || engine.GetBackend() is not IIdleDeviceMemoryLender lender)
            return;
        _collectIdleStepMemory ??= CollectIdleStepMemory;
        lender.LendIdleOnDemand(this, _collectIdleStepMemory);
        _idleLender = lender;
    }

    private Func<IReadOnlyCollection<IGpuBuffer>?>? _collectIdleStepMemory;

    // The lender asks (an allocation between this plan's steps found nothing pooled). Rebuilt on every ask: what
    // outlives a step (a rebased readout, a retained output) can change between steps. The ask runs on the allocating
    // thread, under the lender's lock: it takes the plan's step lock without waiting (the plan's own thread may hold it
    // while it changes what is protected) and lends nothing when that fails, or when collecting throws, rather than
    // fail an unrelated allocation.
    private IReadOnlyCollection<IGpuBuffer>? CollectIdleStepMemory()
    {
        if (!System.Threading.Monitor.TryEnter(_stepSync)) return null;
        try
        {
            if (_disposed) return null;
            return CollectIdleStepMemoryCore();
        }
        catch (Exception ex) when (ex is InvalidOperationException or ObjectDisposedException)
        {
            if (s_idleLendDiag) AiDotNetEngine.LogDiagnostic($"[IDLE-LEND] plan={_planId} refused: {ex.GetType().Name}: {ex.Message}");
            System.Diagnostics.Trace.TraceWarning(
                $"Compiled plan {_planId} could not lend its idle step memory ({ex.GetType().Name}); the allocation is served without it.");
            return null;
        }
        finally
        {
            System.Threading.Monitor.Exit(_stepSync);
        }
    }

    private IReadOnlyCollection<IGpuBuffer>? CollectIdleStepMemoryCore()
    {
        if (_scratchPool is not { } pool || _forwardSteps is null) return null;
        if (!TryProtectStepPersistentBuffers())
        {
            if (s_idleLendDiag) AiDotNetEngine.LogDiagnostic($"[IDLE-LEND] plan={_planId} refused: an opaque saved state {_workspaceOpaqueSavedType}");
            return null;
        }
        _workspaceCandidates.Clear();
        // Scratch another plan co-owns (SharedStepScratch) is never lent: that plan's replay writes it. What is lent is
        // marked so that no other plan takes it until the loan ends (ReclaimIdleStepMemory).
        var shared = _engine.As<DirectGpuTensorEngine>()?.SharedStepScratch;
        pool.CollectIdleBetweenRuns(_forwardActions.Length,
            buffer => IsStepTransient(buffer) && shared?.HasOtherOwner(buffer, pool) != true, _workspaceCandidates);
        var idle = new IGpuBuffer[_workspaceCandidates.Count];
        _workspaceCandidates.CopyTo(idle);
        _workspaceCandidates.Clear();
        var marked = _idleMarkedLent ??= new HashSet<IGpuBuffer>(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
        foreach (var buffer in idle) if (marked.Add(buffer)) SharedStepScratch.MarkLent(buffer);
        long bytes = 0;
        foreach (var buffer in idle) bytes += buffer.SizeInBytes;
        IdleLentBytesLastStep = bytes;
        return idle;
    }

    /// <summary>Test hook: runs first in <see cref="ReclaimIdleStepMemory"/>, so a test can make it fail.</summary>
    internal Action? TestFaultInIdleReclaim;

    /// <summary>Takes back the memory <see cref="LendIdleStepMemory"/> lent: before a step, an invalidation or disposal.</summary>
    /// <remarks>A loan that cannot be moved (no memory for it, or it was baked into a graph) still reads memory the next
    /// replay overwrites: the plan then stops replaying its graph and lending, and trains eagerly from here on.</remarks>
    private void ReclaimIdleStepMemory()
    {
        TestFaultInIdleReclaim?.Invoke();
        if (System.Threading.Interlocked.Exchange(ref _idleLender, null) is not { } lender) return;
        try
        {
            lender.ReclaimIdle(this);
            if (System.Threading.Interlocked.Exchange(ref _idleMarkedLent, null) is { } lent)
                foreach (var buffer in lent) SharedStepScratch.ClearLent(buffer);
        }
        catch (Exception ex)
        {
            _idleLendingFailed = true;
            InvalidateCapturedStepGraph();
            _graphStepDisabled = true;
            RecordGraphCaptureFailure("a buffer borrowed from this plan's idle step memory could not be moved back ("
                + ex.GetType().Name + "); this plan trains eagerly from now on");
            throw;
        }
    }

    // Set once a reclaim failed: the plan lends nothing again (and what it lent stays marked, so no plan takes it).
    private bool _idleLendingFailed;
    private HashSet<IGpuBuffer>? _idleMarkedLent;

    /// <summary>Tests: the buffers this plan has lent since its last step began, and its step scratch.</summary>
    internal IReadOnlyCollection<IGpuBuffer> IdleLentBuffersForTest
        => (IReadOnlyCollection<IGpuBuffer>?)_idleMarkedLent ?? Array.Empty<IGpuBuffer>();

    internal StepScratchPool? ScratchPoolForTest => _scratchPool;

    private string? _workspaceOpaqueSavedType;
    private static readonly bool s_idleLendDiag = Environment.GetEnvironmentVariable("AIDOTNET_IDLE_LOAN_DIAG") == "1";

    /// <summary>Process switch for tests that need the backend's own allocations between steps.</summary>
    internal static bool LendIdleStepMemoryEnabled { get; set; } = true;
}
