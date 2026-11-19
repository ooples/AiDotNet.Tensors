using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private readonly bool _gradientPhaseReuseEnabled =
        Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRADIENT_PHASE_REUSE") == "1";
    private List<Tensor<T>>?[]? _shadowFirstTouches;
    private readonly HashSet<Tensor<T>> _shadowDelayEligible = new(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
    private readonly HashSet<Tensor<T>> _shadowPending = new(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
    private readonly HashSet<IGpuBuffer> _shadowPhaseProtectedRoots = new(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
    private bool _shadowPhaseActive;
    private readonly Dictionary<IGpuBuffer, List<Tensor<T>>> _shadowForwardAliases =
        new(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
    internal long GradientPhaseReuseBytes { get; private set; }
    internal long GradientPhaseDelayedBindings { get; private set; }

    private void PrepareShadowFirstTouches(List<int[]?> coverage, List<CompiledStep<T>> steps,
        Dictionary<Tensor<T>, Tensor<T>> gradMap)
    {
        if (!_gradientShadowsEnabled || !_gradientPhaseReuseEnabled || coverage.Count != _backwardActions.Length) return;
        foreach (var action in coverage) if (action is null) return;
        var primalStorage = new HashSet<object>(Autodiff.ReferenceEqualityComparer<object>.Instance);
        var seen = new HashSet<object>(Autodiff.ReferenceEqualityComparer<object>.Instance);
        bool Inspect(object? value)
        {
            if (value is null || !seen.Add(value)) return true;
            if (value is Tensor<T> tensor) { primalStorage.Add(tensor.StorageIdentity); return true; }
            if (value is Tensor<int> || value is Tensor<long>) return true;
            if (value is object[] nested) { foreach (var item in nested) if (!Inspect(item)) return false; return true; }
            // Native and opaque owners require a separate alias proof; decline the entire delayed schedule.
            var type = value.GetType();
            return type.IsPrimitive || type.IsEnum || value is string || value is int[] || value is long[]
                || value is float[] || value is double[] || value is bool[] || value is int[][];
        }
        foreach (var step in steps)
        {
            Inspect(step.OutputBuffer);
            foreach (var input in step.Inputs) Inspect(input);
            if (!Inspect(step.SavedState)) return;
        }
        var candidates = new HashSet<Tensor<T>>(_gradients, Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
        var storageCounts = new Dictionary<object, int>(Autodiff.ReferenceEqualityComparer<object>.Instance);
        foreach (var gradient in _preAllocatedGrads)
            storageCounts[gradient.StorageIdentity] = storageCounts.TryGetValue(gradient.StorageIdentity, out int n) ? n + 1 : 1;
        var first = new Dictionary<Tensor<T>, int>(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
        var firstAny = new Dictionary<Tensor<T>, int>(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
        void Touch(Tensor<T> primal, int action)
        {
            if (!gradMap.TryGetValue(primal, out var gradient)) return;
            if (!firstAny.ContainsKey(gradient)) firstAny.Add(gradient, action);
            if (!candidates.Contains(gradient)
                || primalStorage.Contains(gradient.StorageIdentity)
                || !storageCounts.TryGetValue(gradient.StorageIdentity, out int count) || count != 1) return;
            if (!first.ContainsKey(gradient)) first.Add(gradient, action);
        }
        for (int a = 0; a < coverage.Count; a++)
            foreach (int s in coverage[a]!)
            {
                Touch(steps[s].OutputBuffer, a);
                foreach (var input in steps[s].Inputs) Touch(input, a);
            }
        _shadowFirstTouches = new List<Tensor<T>>?[coverage.Count];
        foreach (var item in first)
        {
            (_shadowFirstTouches[item.Value] ??= new List<Tensor<T>>()).Add(item.Key);
            _shadowDelayEligible.Add(item.Key);
        }
        PrepareShadowBankFirstTouches(firstAny, primalStorage, coverage.Count);
    }

    private void BeginShadowGradientPhase(DirectGpuTensorEngine engine)
    {
        _shadowPending.Clear();
        _shadowPendingBanks.Clear();
        _shadowForwardAliases.Clear();
        _shadowPhaseProtectedRoots.Clear();
        _shadowPhaseActive = false;
        if (CanUseGradientShadows(engine)) _firstWriteRedirects.CaptureRemapSources();
        if (!CanUseGradientShadows(engine) || _shadowFirstTouches is null) return;
        if (!ProtectShadowSavedRoots(_shadowPhaseProtectedRoots)) return;
        _shadowPhaseActive = true;
        BeginShadowBankPhase(engine);
        void Add(Tensor<T> tensor)
        {
            if (tensor._gpuBuffer is not { } buffer) return;
            var root = GpuBufferViewRoot.Of(buffer);
            if (!_shadowForwardAliases.TryGetValue(root, out var aliases))
                _shadowForwardAliases[root] = aliases = new List<Tensor<T>>();
            if (!aliases.Contains(tensor)) aliases.Add(tensor);
        }
        foreach (var step in _forwardSteps!)
        {
            Add(step.OutputBuffer);
            foreach (var input in step.Inputs) Add(input);
            if (step.SavedState is { } saved) foreach (var value in saved) if (value is Tensor<T> tensor) Add(tensor);
        }
    }

    private bool DelayShadowGradient(DirectGpuTensorEngine engine, Tensor<T> gradient)
    {
        if (!_shadowPhaseActive || !CanUseGradientShadows(engine) || !_shadowDelayEligible.Contains(gradient)
            || gradient._gpuBuffer is { Handle: var handle } && handle != IntPtr.Zero
            || !engine.CanUseCompiledGradientShadow(gradient)) return false;
        _shadowPending.Add(gradient);
        return true;
    }

    private void BindShadowGradientsStartingAt(int action, DirectGpuTensorEngine engine, IDirectGpuBackend backend, int elementSize)
    {
        BindShadowGradientBanksStartingAt(action, engine, backend, elementSize);
        if (_shadowFirstTouches?[action] is not { } gradients) return;
        foreach (var gradient in gradients)
        {
            if (!_shadowPending.Contains(gradient)) continue;
            if (!TryBindShadowGradientLoan(engine, backend, gradient))
            {
                var buffer = engine.EnsureResidentBufferWithAllocationInfo(gradient, out bool allocated);
                if (allocated && buffer is not null) RecordPrivateGradientAllocation(buffer);
            }
            if (gradient._gpuBuffer is not { } bound) throw new InvalidOperationException("Delayed gradient has no resident buffer.");
            ZeroDeviceBuffer(backend, bound, gradient.Length, elementSize);
            engine.BindResidentBuffer(gradient, bound, backend);
            _shadowPending.Remove(gradient);
            GradientPhaseDelayedBindings++;
        }
        RefreshShadowGradientRedirects();
    }

    private IGpuBuffer? TryTakePhaseGradientBacking(DirectGpuTensorEngine engine, Tensor<T> gradient, int length = 0)
    {
        if (!_shadowPhaseActive || !(_shadowPending.Contains(gradient) || DelayShadowGradientBank(gradient))
            || _scratchPool is not { } pool
            || !pool.TryTakeGradientBacking(length == 0 ? gradient.Length : length, out var buffer, _shadowPhaseProtectedRoots)) return null;
        if (_shadowForwardAliases.TryGetValue(buffer, out var aliases))
            foreach (var alias in aliases)
                if (alias._gpuBuffer is { } current && ReferenceEquals(GpuBufferViewRoot.Of(current), buffer))
                    engine.DetachCompiledWorkspaceTensor(alias);
        GradientPhaseReuseBytes += buffer.SizeInBytes;
        _lendSchedule = null;
        return buffer;
    }
}
