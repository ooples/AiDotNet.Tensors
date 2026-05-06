// Copyright (c) AiDotNet. All rights reserved.
// Issue #301 — DDIM sampler-update chain detection for graph-mode auto-fusion.

using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches the DDIM sampler-update elementwise chain and emits a
/// single <see cref="LazyNodeType.FusedDDIMStep"/> node. The kernel
/// itself ships now (see
/// <see cref="AiDotNet.Tensors.Helpers.CpuFusedOperations.FusedDDIMStep"/>);
/// the chain-walker is landing in a follow-up — the pattern is
/// registered now so the registry slot is reserved without touching
/// sibling files at follow-up time.
/// </summary>
internal sealed class DDIMStepFusionPattern : IFusionPattern
{
    public string Name => "DDIMStep";

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        return false;
    }
}
