using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private readonly bool _reuseOwnedLeafInputs =
        Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_OWNED_LEAF_INPUT_REUSE") == "1";
    private Dictionary<Tensor<T>, IDirectGpuBackend>? _ownedLeafInputBackends;

    private void RememberOwnedLeafInputBackend(Tensor<T> leaf, IDirectGpuBackend backend)
    {
        if (!_reuseOwnedLeafInputs) return;
        (_ownedLeafInputBackends ??= new Dictionary<Tensor<T>, IDirectGpuBackend>(
            ReferenceEqualityComparer<Tensor<T>>.Instance))[leaf] = backend;
    }

    private bool TryRefreshOwnedLeafInput(DirectGpuTensorEngine engine, Tensor<T> leaf, IDirectGpuBackend backend)
        => _reuseOwnedLeafInputs && leaf._gpuBuffer is null
           && _ownedLeafBuffers.TryGetValue(leaf, out var owned)
           && _ownedLeafInputBackends is not null
           && _ownedLeafInputBackends.TryGetValue(leaf, out var ownerBackend)
           && ReferenceEquals(backend, ownerBackend)
           && engine.TryRefreshPlanOwnedLeafInput(leaf, backend, owned);
}
