using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private readonly bool _gradientBankLastUseEnabled =
        Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRADIENT_BANK_LAST_USE") == "1";
    private List<Array>?[]? _shadowBankLastTouches;
    private readonly Dictionary<Tensor<T>, (int First, int Last)> _shadowRemapRanges =
        new(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
    private readonly Dictionary<Tensor<T>, IGpuBuffer> _shadowActionRemapSources =
        new(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
    private readonly HashSet<IGpuBuffer> _shadowAmbiguousRemaps = new(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
    internal long GradientBankEarlyShadowBytes { get; private set; }
    internal long GradientBankEarlyShadows { get; private set; }

    private void PrepareShadowBankLastTouches(Dictionary<Tensor<T>, int> first,
        Dictionary<Tensor<T>, int> last, int actions)
    {
        if (!_gradientBankLastUseEnabled || _shadowBankFirstTouches is null) return;
        foreach (var item in first) _shadowRemapRanges[item.Key] = (item.Value, last[item.Key]);
        _shadowBankLastTouches = new List<Array>?[actions];
        foreach (var bank in _shadowBankAliases)
        {
            int begin = actions, end = -1;
            foreach (var alias in bank.Value)
                if (first.TryGetValue(alias, out int a)) { begin = Math.Min(begin, a); end = Math.Max(end, last[alias]); }
            if (end < 0) continue;
            // All tenants of a pooled bank use the bank's entire lifetime for physical-pointer remapping.
            foreach (var alias in bank.Value) _shadowRemapRanges[alias] = (begin, end);
            if (_shadowBankDelayEligible.Contains(bank.Key))
                (_shadowBankLastTouches[end] ??= new List<Array>()).Add(bank.Key);
        }
    }

    private void RefreshShadowGradientRedirectsForAction(int action)
    {
        if (_shadowBankLastTouches is null) return;
        _shadowTargetReplacements.Clear();
        _shadowAmbiguousRemaps.Clear();
        foreach (var pair in _shadowActionRemapSources)
            if (_shadowRemapRanges.TryGetValue(pair.Key, out var range) && range.First <= action && action <= range.Last
                && pair.Key._gpuBuffer is { } current && current.Handle != IntPtr.Zero)
            {
                if (_shadowTargetReplacements.TryGetValue(pair.Value, out var prior) && !ReferenceEquals(prior, current))
                    _shadowAmbiguousRemaps.Add(pair.Value);
                _shadowTargetReplacements[pair.Value] = current;
            }
        _firstWriteRedirects.RemapCapturedTargetsForAction(_forwardActions.Length + action, _shadowTargetReplacements,
            _shadowAmbiguousRemaps);
    }

    private void CaptureGradientBankRemapSources()
    {
        if (_shadowBankLastTouches is null) return;
        _shadowActionRemapSources.Clear();
        // Early retirement records this step's physical backing for the next step. It must not change the
        // sources used to resolve redirects captured at the start of the current step.
        foreach (var pair in _shadowPreviousBuffers) _shadowActionRemapSources.Add(pair.Key, pair.Value);
    }

    private void ShadowGradientBanksEndingAt(int action, DirectGpuTensorEngine engine, IDirectGpuBackend backend)
    {
        // The first executed body identifies unwritten tenants and unpools them after backward. Preserve that
        // initialization pass before lending banks whose layout can still change underneath the schedule.
        if (!_shadowPhaseActive || _unwrittenGradIndices is null || _shadowBankLastTouches?[action] is not { } banks) return;
        foreach (var bank in banks)
        {
            if (!CanRetireShadowBank(bank)) continue;
            long bytes = ShadowCompletedGradientBanks(engine, backend, _shadowPhaseProtectedRoots, bank);
            if (bytes == 0) continue;
            GradientBankEarlyShadowBytes += bytes;
            GradientBankEarlyShadows++;
        }
    }

    private bool CanRetireShadowBank(Array bank)
    {
        if (!_shadowBankAliases.TryGetValue(bank, out var aliases) || aliases[0]._gpuBuffer is not { } root) return false;
        // The completion-time proof permits native aliases in other banks because all backward work is done.
        // Early retirement additionally requires that no other gradient bank or forward value still binds this root.
        foreach (var gradient in _preAllocatedGrads)
            if (gradient._gpuBuffer is { } other && ReferenceEquals(GpuBufferViewRoot.Of(other), root)
                && !ReferenceEquals(gradient.GraphBackingArrayIdentity, bank)) return false;
        if (_shadowForwardAliases.TryGetValue(root, out var forward))
            foreach (var tensor in forward)
                if (tensor._gpuBuffer is { } buffer && ReferenceEquals(GpuBufferViewRoot.Of(buffer), root)) return false;
        return true;
    }
}
