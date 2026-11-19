using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private List<Array>?[]? _shadowBankFirstTouches;
    private readonly Dictionary<Array, List<Tensor<T>>> _shadowBankAliases = new(Autodiff.ReferenceEqualityComparer<Array>.Instance);
    private readonly HashSet<Array> _shadowPendingBanks = new(Autodiff.ReferenceEqualityComparer<Array>.Instance);
    private readonly HashSet<Array> _shadowBankDelayEligible = new(Autodiff.ReferenceEqualityComparer<Array>.Instance);
    internal long GradientBankDelayedBindings { get; private set; }

    private void PrepareShadowBankFirstTouches(Dictionary<Tensor<T>, int> firstTouches,
        HashSet<object> primalStorage, int actions)
    {
        if (!_gradientBankShadowsEnabled) return;
        var excluded = new HashSet<Array>(Autodiff.ReferenceEqualityComparer<Array>.Instance);
        // Loss reseeding happens before the backward loop; its destination cannot wait for an action's first touch.
        // Public parameter gradients use their separate first-touch/ownership path.
        if (_lossGradDest?.GraphBackingArrayIdentity is { } lossBank) excluded.Add(lossBank);
        if (_lossGradSeed.GraphBackingArrayIdentity is { } seedBank) excluded.Add(seedBank);
        foreach (var gradient in _gradients)
            if (gradient.GraphBackingArrayIdentity is { } publicBank) excluded.Add(publicBank);
        foreach (var gradient in _preAllocatedGrads)
        {
            if (gradient.GraphBackingArrayIdentity is not T[] array || !ResultBufferTracker.IsUnexposedPrivateArray(array)) continue;
            if (primalStorage.Contains(gradient.StorageIdentity) || primalStorage.Contains(array)) excluded.Add(array);
            if (!_shadowBankAliases.TryGetValue(array, out var aliases))
                _shadowBankAliases[array] = aliases = new List<Tensor<T>>();
            aliases.Add(gradient);
        }
        var first = new Dictionary<Array, int>(Autodiff.ReferenceEqualityComparer<Array>.Instance);
        foreach (var item in firstTouches)
            if (item.Key.GraphBackingArrayIdentity is T[] array && _shadowBankAliases.ContainsKey(array) && !excluded.Contains(array))
                first[array] = first.TryGetValue(array, out int prior) ? Math.Min(prior, item.Value) : item.Value;
        _shadowBankFirstTouches = new List<Array>?[actions];
        foreach (var item in first)
        {
            (_shadowBankFirstTouches[item.Value] ??= new List<Array>()).Add(item.Key);
            _shadowBankDelayEligible.Add(item.Key);
        }
    }

    private void BeginShadowBankPhase(DirectGpuTensorEngine engine)
    {
        if (_shadowBankFirstTouches is null) return;
        foreach (var array in _shadowBankArrays)
        {
            if (!_shadowBankDelayEligible.Contains(array) || !ResultBufferTracker.IsUnexposedPrivateArray(array)
                || !_shadowBankAliases.TryGetValue(array, out var aliases)) continue;
            bool eligible = true;
            foreach (var alias in aliases)
                if (!ReferenceEquals(alias.GraphBackingArrayIdentity, array)
                    || alias._gpuBuffer is { Handle: var handle } && handle != IntPtr.Zero
                    || !engine.CanUseCompiledGradientShadow(alias)) { eligible = false; break; }
            if (eligible) _shadowPendingBanks.Add(array);
        }
    }

    private bool DelayShadowGradientBank(Tensor<T> gradient) => gradient.GraphBackingArrayIdentity is { } array
        && _shadowPendingBanks.Contains(array);

    private void BindShadowGradientBanksStartingAt(int action, DirectGpuTensorEngine engine, IDirectGpuBackend backend, int elementSize)
    {
        if (_shadowBankFirstTouches?[action] is not { } arrays) return;
        bool rebound = false;
        foreach (var array in arrays)
        {
            if (!_shadowPendingBanks.Contains(array)) continue;
            var aliases = _shadowBankAliases[array];
            if (!TryBindShadowGradientBank(engine, backend, aliases[0], out var buffer))
                throw new InvalidOperationException("Delayed private gradient bank cannot be made resident.");
            ZeroDeviceBuffer(backend, buffer, array.Length, elementSize);
            foreach (var alias in aliases) engine.BindResidentBuffer(alias, buffer, backend);
            if (_aliasedGradViews is { } views)
                foreach (var (view, target) in views)
                    if (ReferenceEquals(target.GraphBackingArrayIdentity, array)) engine.BindResidentBuffer(view, buffer, backend);
            if (_pooledGradDeviceBuffers is { } pooled) pooled[array] = buffer;
            _shadowPendingBanks.Remove(array);
            GradientBankDelayedBindings++;
            rebound = true;
        }
        if (rebound) RefreshShadowGradientRedirects();
    }
}
