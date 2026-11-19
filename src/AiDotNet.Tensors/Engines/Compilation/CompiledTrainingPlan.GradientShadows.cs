using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private readonly bool _gradientShadowsEnabled =
        Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRADIENT_SHADOWS") == "1";
    private HashSet<Tensor<T>>? _shadowGradientSet;
    private readonly HashSet<IGpuBuffer> _shadowProtectedRoots = new(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
    private readonly HashSet<IGpuBuffer> _shadowMoved = new(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
    private readonly Dictionary<Tensor<T>, IGpuBuffer> _shadowPreviousBuffers = new(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
    private readonly Dictionary<IGpuBuffer, IGpuBuffer> _shadowTargetReplacements = new(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
    private readonly Dictionary<Tensor<T>, (IGpuBuffer Buffer, IDirectGpuBackend Backend, int Thread, long Key)> _shadowLoans =
        new(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
    private StepScratchPool? _shadowLoanOwner;
    private long _nextShadowLoan;
    internal long GradientShadowBytes { get; private set; }
    internal long GradientShadowCompletedSteps { get; private set; }
    internal long GradientShadowReuses { get; private set; }

    private bool CanUseGradientShadows(DirectGpuTensorEngine engine) => _gradientShadowsEnabled
        && _workspaceSharingBodyAllowed && CanShareTransientWorkspace(engine);

    private CompiledPointerTableScope.Lease EnterShadowPointerTableScope(DirectGpuTensorEngine engine, IDirectGpuBackend backend)
        => CanUseGradientShadows(engine)
            && !(backend is IGpuGraphCapture { SupportsGraphCapture: true } capture && capture.IsStreamCapturing())
                ? CompiledPointerTableScope.Enter(_planId) : CompiledPointerTableScope.Enter(0);

    private void RefreshShadowGradientRedirects()
    {
        _shadowTargetReplacements.Clear();
        foreach (var pair in _shadowPreviousBuffers)
            if (pair.Key._gpuBuffer is { } current && current.Handle != IntPtr.Zero)
                _shadowTargetReplacements[pair.Value] = current;
        _firstWriteRedirects.RemapTargets(_shadowTargetReplacements);
        _shadowPreviousBuffers.Clear();
    }

    private bool TryBindShadowGradientLoan(DirectGpuTensorEngine engine, IDirectGpuBackend backend, Tensor<T> gradient)
    {
        if (!CanUseGradientShadows(engine) || !engine.CanUseCompiledGradientShadow(gradient)
            || !(_shadowGradientSet ??= new HashSet<Tensor<T>>(_gradients,
                Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance)).Contains(gradient)) return false;
        if (gradient._gpuBuffer is { Handle: var handle } && handle != IntPtr.Zero) return false;
        int thread = Environment.CurrentManagedThreadId;
        if (_shadowLoans.TryGetValue(gradient, out var held))
        {
            if (!ReferenceEquals(held.Backend, backend) || held.Thread != thread || held.Buffer.Handle == IntPtr.Zero)
                return false;
            engine.BindResidentBuffer(gradient, held.Buffer, backend);
            return true;
        }
        var buffer = engine.CompiledWorkspaceBroker.TryRent(backend, gradient.Length, exactSize: true);
        if (buffer is null) return false;
        var owner = _shadowLoanOwner ??= engine.CreateScratchPool();
        long key = ++_nextShadowLoan;
        owner.Track(buffer);
        owner.Slots.Add(key, buffer);
        _shadowLoans.Add(gradient, (buffer, backend, thread, key));
        engine.BindResidentBuffer(gradient, buffer, backend);
        GradientShadowReuses++;
        return true;
    }

    private void ShadowCompletedParameterGradients(DirectGpuTensorEngine engine, IDirectGpuBackend backend)
    {
        if (!CanUseGradientShadows(engine) || _scratchPool is not { } pool
            || backend is IGpuGraphCapture { SupportsGraphCapture: true } graph && graph.IsStreamCapturing()) return;
        var protectedRoots = _shadowProtectedRoots;
        protectedRoots.Clear();
        void Protect(IGpuBuffer? buffer) { if (buffer is not null) protectedRoots.Add(GpuBufferViewRoot.Of(buffer)); }
        foreach (var parameter in _parameters) Protect(parameter._gpuBuffer);
        foreach (var buffer in _gpuOptimizerBuffers) Protect(buffer);
        foreach (var buffer in _ownedLeafBuffers.Values) Protect(buffer);
        foreach (var buffer in _foreignParameterLeaves.Values) Protect(buffer);
        foreach (var buffer in pool.Slots.Values) Protect(buffer);
        foreach (var buffer in pool.Transient.Values) Protect(buffer);
        foreach (var output in _retainedOutputs) Protect(output._gpuBuffer);
        foreach (var output in _namedRetainedOutputs.Values) Protect(output._gpuBuffer);
        Protect(_lossOutput._gpuBuffer); Protect(_lossGradDest?._gpuBuffer); Protect(_lossGradSeed?._gpuBuffer);
        var moved = _shadowMoved;
        moved.Clear();
        long bytes = 0;
        foreach (var gradient in _gradients)
        {
            if (gradient is null || gradient._gpuBuffer is not { } buffer || !ReferenceEquals(GpuBufferViewRoot.Of(buffer), buffer)
                || protectedRoots.Contains(buffer) || moved.Contains(buffer)) continue;
            bool loan = _shadowLoans.TryGetValue(gradient, out var held) && ReferenceEquals(held.Buffer, buffer);
            if (!loan && (StepScratchPool.IsPooled(buffer) || !_privateGradientAllocations.TryGetValue(buffer, out _))) continue;
            if (!engine.TryCopyCompiledGradientToHost(gradient, buffer, backend)) continue;
            pool.Track(buffer); // prevent activation-cache borrowers disposing backing during this ownership transfer
            engine.DetachHostCurrentCompiledGradient(gradient);
            if (loan)
            {
                _shadowLoanOwner!.Slots.Remove(held.Key);
                _shadowLoans.Remove(gradient);
            }
            _privateGradientAllocations.Remove(buffer);
            _shadowPreviousBuffers[gradient] = buffer;
            engine.CompiledWorkspaceBroker.Return(backend, buffer);
            moved.Add(buffer);
            bytes += (long)gradient.Length * sizeof(float);
        }
        if (bytes == 0) return;
        _gradHeldZero = null;
        _unwrittenGradIndices = null;
        _lendSchedule = null;
        GradientShadowBytes += bytes;
        GradientShadowCompletedSteps++;
        if (Environment.GetEnvironmentVariable("AIDOTNET_GRAD_POOL_DIAG") == "1")
            Console.Error.WriteLine($"[GRADIENT-SHADOW] plan={_planId} steps={GradientShadowCompletedSteps} bytes={bytes} total_bytes={GradientShadowBytes} reuses={GradientShadowReuses}");
    }
}
